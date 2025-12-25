"""Integrated simulator combining Social Force, Social-GAN, and Frenet planning.

This is the main simulation module that orchestrates all components.
"""

import numpy as np
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from ..core.data_structures import (
    EgoVehicleState,
    PedestrianState,
    SimulationResult,
    compute_safety_metrics_static,
)
from ..core.coordinate_converter import CoordinateConverter
from ..config import SimulationConfig
from ..planning import CubicSpline2D, FrenetPlanner
# MAX_T is now configurable via config.max_t
from ..pedestrian.observer import PedestrianObserver
from ..prediction.trajectory_predictor import TrajectoryPredictor
from ..core.state_machine import FailSafeStateMachine, VehicleState


import pysocialforce as psf
PYSOCIALFORCE_AVAILABLE = True


class PedestrianSimulator:
    """Pedestrian simulator using Social Force model (via PySocialForce).
    
    Acts as a wrapper around the pysocialforce library.
    """
    
    def __init__(
        self,
        initial_states: np.ndarray,
        groups: Optional[List[List[int]]] = None,
        obstacles: Optional[List] = None,
        dt: float = 0.1,
        config_file: Optional[str] = None,
        ego_radius: float = 1.0,
        social_force_params: Optional[Dict] = None
    ):
        """Initialize simulator.
        
        Args:
            initial_states: Initial state array [N, 6] (x, y, vx, vy, gx, gy)
            groups: List of grouping lists (indices)
            obstacles: List of obstacle specifications
            dt: Simulation time step
            config_file: Path to PySocialForce config file
            ego_radius: Radius of the ego vehicle [m]
            social_force_params: Dictionary of SFM parameters to override
        """
        self.dt = dt
        self.initial_states = initial_states
        self.time = 0.0
        self.ego_agent_index = -1  # Index of the ego agent in the state array
        self.ego_radius = ego_radius
        
        # Check for PySocialForce availability
        if not PYSOCIALFORCE_AVAILABLE:
            raise ImportError(
                "PySocialForce is required for this simulator. "
                "Please install it via `pip install pysocialforce`."
            )
            
        self._init_pysocialforce(
            groups, 
            obstacles, 
            config_file,
            social_force_params=social_force_params
        )

    def _init_pysocialforce(
        self, 
        groups: Optional[List[List[int]]] = None, 
        obstacles: Optional[List] = None,
        config_file: Optional[str] = None,
        social_force_params: Optional[Dict] = None
    ):
        """Initialize PySocialForce simulator."""
        # Convert states to PySocialForce format
        # [N, 6] -> state
        
        # Add a dummy agent for the Ego vehicle at the end
        # Position it far away initially so it doesn't disturb initialization
        n_peds = self.initial_states.shape[0]
        ego_dummy = np.array([[9999.0, 9999.0, 0.0, 0.0, 9999.0, 9999.0]])
        state = np.vstack([self.initial_states, ego_dummy])
        self.ego_agent_index = n_peds
        
        # Initialize simulator
        # PySocialForce expects obstacles as list of line segments or polygons
        # Here we assume obstacles is a list of [x_min, x_max, y_min, y_max] box lists
        # We need to convert them to line segments for PSF
        psf_obstacles = []
        if obstacles:
            for obs in obstacles:
                if len(obs) == 4:  # [x_min, x_max, y_min, y_max]
                    x_min, x_max, y_min, y_max = obs
                    # Box as 4 lines, filtering out zero-length segments
                    # PySocialForce expects (x1, y1, x2, y2)
                    segments = [
                        (x_min, y_min, x_max, y_min), # Bottom edge
                        (x_max, y_min, x_max, y_max), # Right edge
                        (x_max, y_max, x_min, y_max), # Top edge
                        (x_min, y_max, x_min, y_min)  # Left edge
                    ]
                    for s in segments:
                        # Check if length > epsilon (s is x1, y1, x2, y2)
                        # Calculates Euclidean distance squared to be safe, or just check component diffs
                        dx = s[2] - s[0]
                        dy = s[3] - s[1]
                        if (dx*dx + dy*dy) > 1e-12:
                            psf_obstacles.append(s)
        
        self.sim = psf.Simulator(
            state=state,
            groups=groups,
            obstacles=psf_obstacles if psf_obstacles else None,
            config_file=config_file
        )

        # Apply custom parameters if provided
        if social_force_params and hasattr(self.sim, 'config'):
            for key, value in social_force_params.items():
                # Support nested keys with dot notation (e.g. "ped_repulsion.sigma")
                if "." in key:
                    section, subkey = key.split(".", 1)
                    if hasattr(self.sim.config, section):
                        section_obj = getattr(self.sim.config, section)
                        if isinstance(section_obj, dict):
                             section_obj[subkey] = value
                        else:
                             setattr(section_obj, subkey, value)
                else:
                    setattr(self.sim.config, key, value)
                    
        # Also try to update specific force parameters if exposed
        if social_force_params:
             # Basic handling for common PySocialForce parameters if directly exposed
             pass
        
        # Manually set dt (step_width)
        if hasattr(self.sim, 'peds'):
            self.sim.peds.step_width = self.dt
            # Make Ego agent larger to reflect vehicle size
            # PySocialForce defaults to 0.3 or 0.4
            if hasattr(self.sim.peds, 'agent_radius'):
                # Handle varying implementations of agent radius
                # If it's a scalar, we can't change it per agent easily without modifying library
                # If it's an array, we can set the specific index
                if isinstance(self.sim.peds.agent_radius, np.ndarray):
                     self.sim.peds.agent_radius[self.ego_agent_index] = self.ego_radius
                elif isinstance(self.sim.peds.agent_radius, list):
                     self.sim.peds.agent_radius[self.ego_agent_index] = self.ego_radius
                else:
                    logger.debug("Could not set specific radius for Ego agent in PySocialForce")

    def step(self, ego_state: Optional[EgoVehicleState] = None, n: int = 1):
        """Advance simulation by n time steps.
        
        Args:
            ego_state: Current state of the ego vehicle
            n: Number of time steps to simulate
        """
        if self.sim:
            # Update Ego agent state before stepping
            if ego_state is not None and self.ego_agent_index >= 0:
                # Update position and velocity
                # state: [x, y, vx, vy, gx, gy, tau]
                self.sim.peds.state[self.ego_agent_index, 0] = ego_state.x
                self.sim.peds.state[self.ego_agent_index, 1] = ego_state.y
                self.sim.peds.state[self.ego_agent_index, 2] = ego_state.v * np.cos(ego_state.yaw)
                self.sim.peds.state[self.ego_agent_index, 3] = ego_state.v * np.sin(ego_state.yaw)
                
                # Update goal to be far ahead to avoid goal forces pulling it backward
                # Or just keep it as is (far away)
                # Ideally, we want the car to be a moving obstacle, not really influenced by forces
                # But in this step, we mainly care about its influence on OTHERS.
                # Since we overwrite its state next time, its own force update doesn't matter much.
            
            self.sim.step(n)
            self.time += n * self.dt
        else:
            raise RuntimeError("Simulator not initialized correctly.")

    def get_state(self) -> PedestrianState:
        """Get current pedestrian state.
        
        Returns:
            Current pedestrian state (excluding Ego agent)
        """
        if self.sim:
            # PySocialForce state: [N, 7] (x, y, vx, vy, gx, gy, tau)
            full_state = self.sim.peds.state
            
            # Exclude Ego agent
            if self.ego_agent_index >= 0:
                # Assuming Ego is at the end
                state = full_state[:self.ego_agent_index]
            else:
                state = full_state
            
            return PedestrianState(
                positions=state[:, 0:2].copy(),
                velocities=state[:, 2:4].copy(),
                goals=state[:, 4:6].copy(),
                timestamp=self.time
            )
        else:
            raise RuntimeError("Simulator not initialized.")


# Backwards compatibility alias
SimplePedestrianSimulator = PedestrianSimulator


class IntegratedSimulator:
    """Integrated path planning simulator.
    
    This simulator combines:
    1. Social Force Model for pedestrian ground truth
    2. Social-GAN for trajectory prediction
    3. Frenet Optimal Trajectory for path planning
    
    Args:
        config: Simulation configuration
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.time = 0.0
        self.step_count = 0
        self.history: List[SimulationResult] = []
        
        # Initialize components
        logger.info("Initializing integrated simulator...")
        
        # 1. Create reference path
        self.reference_path = CubicSpline2D(
            config.reference_waypoints_x,
            config.reference_waypoints_y
        )
        logger.info(f"Reference path created with "
                   f"{len(config.reference_waypoints_x)} waypoints")

        # Safety parameters
        self.ego_radius = getattr(config, "ego_radius", 1.0)
        self.ped_radius = getattr(config, "ped_radius", 0.3)
        self.obstacle_radius = getattr(config, "obstacle_radius", self.ped_radius)
        
        # 2. Initialize pedestrian simulator
        if len(config.ped_initial_states) > 0:
            ped_states = np.array(config.ped_initial_states)
            self.pedestrian_sim = PedestrianSimulator(
                initial_states=ped_states,
                groups=config.ped_groups,
                obstacles=config.static_obstacles,
                dt=config.dt,
                config_file=getattr(config, "social_force_config", None),
                ego_radius=self.ego_radius,
                social_force_params=getattr(config, "social_force_params", None)
            )
        else:
            self.pedestrian_sim = None
            logger.warning("No pedestrians in scenario")
        
        # 3. Initialize pedestrian observer
        self.observer = PedestrianObserver(
            obs_len=config.obs_len,
            dt=config.dt,
            sgan_dt=0.4  # SGAN expects 0.4s sampling regardless of simulation dt
        )
        
        # 4. Initialize trajectory predictor
        plan_horizon = getattr(config, 'max_t', 5.0)
        self.predictor = TrajectoryPredictor(
            model_path=config.sgan_model_path,
            pred_len=config.pred_len,
            num_samples=getattr(config, 'num_samples', 1),
            device=config.device,
            sgan_dt=self.observer.sgan_dt,
            sim_dt=config.dt,
            plan_horizon=plan_horizon,
            method=getattr(config, 'prediction_method', 'sgan')
        )
        
        # 5. Initialize path planner
        self.planner = FrenetPlanner(
            reference_path=self.reference_path,
            max_speed=config.ego_max_speed,
            max_accel=config.ego_max_accel,
            max_curvature=config.ego_max_curvature,
            dt=config.dt,
            d_road_w=config.d_road_w,
            max_road_width=config.max_road_width,
            robot_radius=self.ego_radius,
            obstacle_radius=config.obstacle_radius,
            min_t=getattr(config, 'min_t', 4.0),
            max_t=getattr(config, 'max_t', 5.0),
            d_t_s=getattr(config, 'd_t_s', 5.0 / 3.6),
            n_s_sample=getattr(config, 'n_s_sample', 1),
            k_j=config.k_j,
            k_t=config.k_t,
            k_d=config.k_d,
            k_s_dot=config.k_s_dot,
            k_lat=config.k_lat,
            k_lon=config.k_lon
        )
        
        # 6. Initialize coordinate converter
        self.coord_converter = CoordinateConverter(self.reference_path)
        
        # 7. Initialize ego vehicle state
        ego_arr = np.array(config.ego_initial_state)
        self.ego_state = EgoVehicleState.from_array(ego_arr, timestamp=0.0)

        # 8. Initialize State Machine
        self.state_machine = FailSafeStateMachine(config)
        self.ego_state.state = self.state_machine.current_state
        self._replan_attempts = 0  # Track re-planning attempts to prevent infinite loops
        self._max_replan_attempts = 3  # Maximum number of re-planning attempts per step

        # Precompute static obstacles (expanded to point set for collision checks)
        self.static_obstacle_points = self._expand_static_obstacles(
            config.static_obstacles, step=0.5
        )
        
        if self.pedestrian_sim is not None:
            self.warmup()
        
        logger.info("Integrated simulator initialization complete")
    
    def warmup(self):
        """Warm up the simulation to fill observer history.
        
        This advances the pedestrian simulation and observer without
        advancing the main simulation clock or recording history,
        so that predictions are available immediately at t=0.
        """
        warmup_steps = int(self.config.obs_len * self.observer.sgan_dt / self.config.dt)
        logger.info(f"Warming up simulation for {warmup_steps} steps...")
        
        for _ in range(warmup_steps):
            # Step pedestrians (Ego is at 0,0,0,0 during warmup as defined in init)
            self.pedestrian_sim.step(self.ego_state)
            ped_state = self.pedestrian_sim.get_state()
            self.observer.update(ped_state)
            
        logger.info("Warmup complete. Observer is ready.")
    
    def step(self) -> SimulationResult:
        """Execute one simulation step.
        
        Returns:
            Simulation result for this step
        """
        # 1. Advance pedestrian simulation
        ped_state = None
        if self.pedestrian_sim is not None:
            self.pedestrian_sim.step(self.ego_state)
            ped_state = self.pedestrian_sim.get_state()
            
            # Update observer
            self.observer.update(ped_state)
        
        # 2. Predict pedestrian trajectories
        predicted_traj = None
        predicted_dist = None
        dynamic_obstacles = np.empty((0, 0, 2))
        static_obstacles = self.static_obstacle_points.copy()
        t_pred = 0.0
        
        if ped_state is not None and self.observer.is_ready:
            try:
                # Get observations
                obs_traj, obs_traj_rel, seq_start_end = self.observer.get_observation()
                
                # Predict
                t_start = time.perf_counter()
                predicted_traj, predicted_dist = self.predictor.predict_single_best(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                t_pred = time.perf_counter() - t_start
                
                # Preserve time dimension for dynamic collision checks - NO CONVERSION
                dynamic_obstacles = self.coord_converter.pass_through_obstacle(
                    predicted_traj
                )
                
                logger.debug(
                    f"Predicted {predicted_traj.shape[0]} pedestrian trajectories "
                    f"for {predicted_traj.shape[1]} steps"
                )
                
            except Exception as e:
                logger.warning(f"Prediction failed: {e}, using constant velocity extrapolation")
                if ped_state is not None:
                    # Create a simple constant velocity prediction for the planning horizon
                    # This ensures we have proper time dimension for collision checking
                    n_peds = ped_state.n_peds
                    plan_horizon = getattr(self.config, 'max_t', 5.0)  # Default to 5.0s if not set
                    plan_horizon_steps = max(1, int(plan_horizon / self.config.dt))
                    
                    # Use current velocities for extrapolation
                    current_positions = ped_state.positions  # [n_peds, 2]
                    current_velocities = ped_state.velocities  # [n_peds, 2]
                    
                    # Generate trajectory: [n_peds, n_steps, 2]
                    dynamic_obstacles = np.zeros((n_peds, plan_horizon_steps, 2))
                    for step in range(plan_horizon_steps):
                        t = (step + 1) * self.config.dt
                        dynamic_obstacles[:, step, :] = current_positions + current_velocities * t
                else:
                    dynamic_obstacles = np.empty((0, 0, 2))
                t_pred = 0.0
        elif ped_state is not None:
            # Not enough observations yet, use current positions
            dynamic_obstacles = ped_state.positions[:, None, :]
            t_pred = 0.0

        # Include current pedestrian positions at time t=0 for collision checks
        if ped_state is not None:
            current_positions = ped_state.positions[:, None, :]
            if dynamic_obstacles.size == 0:
                dynamic_obstacles = current_positions
            else:
                already_has_current = (
                    dynamic_obstacles.shape[1] >= 1
                    and np.allclose(dynamic_obstacles[:, 0, :], current_positions[:, 0, :])
                )
                if not already_has_current:
                    dynamic_obstacles = np.concatenate([current_positions, dynamic_obstacles], axis=1)
        
        # 3. Plan path with State Machine
        
        # Determine constraints based on current state (before planning)
        # In the first step, we are in NORMAL unless initialized otherwise.
        # But we actually want the *result* of the previous step to drive this one's constraints?
        # Or do we want to iterate?
        # Let's try to plan with *current* state.
        
        # Get planner config from state machine
        sm_output = self.state_machine._get_planner_config()
        
        target_speed = sm_output.target_speed_override
        if target_speed is None:
            target_speed = self.config.ego_target_speed
            
        t_start = time.perf_counter()
        planned_path = self.planner.plan(
            self.ego_state,
            static_obstacles,
            dynamic_obstacles,
            target_speed=target_speed,
            constraint_overrides=sm_output.constraint_overrides
        )
        t_plan = time.perf_counter() - t_start
        
        # Update State Machine based on result
        found_path = (planned_path is not None)
        
        # Compute current safety metrics for state machine update (before moving)
        # Use the shared function to avoid code duplication
        if ped_state is not None:
            current_metrics = compute_safety_metrics_static(
                ego_state=self.ego_state,
                ped_state=ped_state,
                ego_radius=self.ego_radius,
                ped_radius=self.ped_radius
            )
        else:
            current_metrics = {'min_distance': float('inf'), 'collision': False, 'ttc': float('inf')}
        
        # Update SM
        new_sm_output = self.state_machine.update(found_path, current_metrics)
        
        # If state CHANGED to a more critical one (NORMAL -> CAUTION or CAUTION -> EMERGENCY)
        # AND we didn't find a path, we should RE-PLAN immediately with the new relaxed constraints
        # to avoid wasting a step doing nothing (or previous emergency stop).
        # Limit re-planning attempts to prevent infinite loops.
        
        if not found_path and new_sm_output.state != sm_output.state and self._replan_attempts < self._max_replan_attempts:
            logger.warning(
                f"Planning failed in {sm_output.state}. Transitioning to {new_sm_output.state} "
                f"and retrying (attempt {self._replan_attempts + 1}/{self._max_replan_attempts})..."
            )
            
            # Update local state variable to reflect new state for logging/recording
            self.ego_state.state = new_sm_output.state
            self._replan_attempts += 1
            
            # Re-plan
            target_speed = new_sm_output.target_speed_override
            if target_speed is None:
                target_speed = self.config.ego_target_speed
                
            planned_path = self.planner.plan(
                self.ego_state,
                static_obstacles,
                dynamic_obstacles,
                target_speed=target_speed,
                constraint_overrides=new_sm_output.constraint_overrides
            )
            
            if planned_path is not None:
                logger.info(f"Re-planning successful in {new_sm_output.state}")
            else:
                logger.error(
                    f"Re-planning failed even in {new_sm_output.state} "
                    f"(attempt {self._replan_attempts}/{self._max_replan_attempts})"
                )
        elif not found_path and self._replan_attempts >= self._max_replan_attempts:
            logger.error(
                f"Maximum re-planning attempts ({self._max_replan_attempts}) reached. "
                f"Proceeding with emergency stop."
            )

        # 4. Update ego vehicle state
        # Storage for old accel
        old_a = self.ego_state.a
        
        if planned_path is not None and len(planned_path) >= 2:
            # Follow path
            try:
                self.ego_state = planned_path.get_state_at_index(1)
                current_jerk = (self.ego_state.a - old_a) / self.config.dt
                self.ego_state.jerk = current_jerk
                self.ego_state.timestamp = self.time + self.config.dt
                self.ego_state.state = self.state_machine.current_state # Update state in object
            except IndexError:
                # Fallback to stop
                self._apply_emergency_stop(old_a)
        else:
            # No path found (even after retry) -> Apply Emergency Stop Logic locally
            # determining how 'hard' to stop based on state
            logger.warning("No valid path found. Applying stop.")
            self._apply_emergency_stop(old_a)
            self.ego_state.state = self.state_machine.current_state
        
        # 5. Create result
        result = SimulationResult(
            time=self.time,
            ego_state=self.ego_state,
            ped_state=ped_state or PedestrianState(
                positions=np.empty((0, 2)),
                velocities=np.empty((0, 2)),
                goals=np.empty((0, 2)),
                timestamp=self.time
            ),
            predicted_trajectories=predicted_traj,
            predicted_distribution=predicted_dist,
            planned_path=planned_path,
            ego_radius=self.ego_radius,
            ped_radius=self.ped_radius,
            processing_times={'prediction': t_pred, 'planning': t_plan}
            # state=self.ego_state.state # Implicitly in ego_state
        )
        
        # Compute metrics
        result.metrics = result.compute_safety_metrics()
        
        # Record history
        self.history.append(result)
        
        # Update time
        self.time += self.config.dt
        self.step_count += 1
        
        # Reset re-planning attempts counter for the next step
        # (This ensures we can retry re-planning in the next step if needed)
        self._replan_attempts = 0
        
        return result

    def _apply_emergency_stop(self, old_a: float):
        """Apply emergency stop dynamics."""
        # Use emergency deceleration
        max_dec = self.config.ego_max_accel * 2.0 # Hard braking
        
        self.ego_state.v = max(0.0, self.ego_state.v - max_dec * self.config.dt)
        new_a = -max_dec if self.ego_state.v > 0 else 0.0
        
        current_jerk = (new_a - old_a) / self.config.dt
        
        self.ego_state.a = new_a
        self.ego_state.jerk = current_jerk
        self.ego_state.timestamp = self.time + self.config.dt

    @staticmethod
    def _expand_static_obstacles(static_obstacles, step: float = 0.5) -> np.ndarray:
        """Expand rectangular static obstacles into boundary points for collision checks."""
        if static_obstacles is None or len(static_obstacles) == 0:
            return np.empty((0, 2))
        
        points = []
        for rect in static_obstacles:
            if len(rect) != 4:
                continue
            x_min, x_max, y_min, y_max = rect
            xs = np.arange(x_min, x_max + step, step)
            ys = np.arange(y_min, y_max + step, step)
            
            # Top and bottom edges
            for x in xs:
                points.append((x, y_min))
                points.append((x, y_max))
            # Left and right edges
            for y in ys:
                points.append((x_min, y))
                points.append((x_max, y))
        
        if len(points) == 0:
            return np.empty((0, 2))
        
        points_arr = np.unique(np.array(points), axis=0)
        return points_arr
    
    def run(self, n_steps: Optional[int] = None) -> List[SimulationResult]:
        """Run simulation for multiple steps.
        
        Args:
            n_steps: Number of steps to run (if None, use config.total_time)
            
        Returns:
            List of simulation results
        """
        if n_steps is None:
            n_steps = int(self.config.total_time / self.config.dt)
        
        logger.info(f"Running simulation for {n_steps} steps "
                   f"(T={n_steps * self.config.dt:.1f}s)")
        
        for i in range(n_steps):
            result = self.step()
            
            if i % 10 == 0:
                logger.info(f"Step {i}/{n_steps}, t={self.time:.1f}s, "
                          f"ego=({self.ego_state.x:.1f}, {self.ego_state.y:.1f}), "
                          f"v={self.ego_state.v:.1f}m/s, "
                          f"min_dist={result.metrics.get('min_distance', float('inf')):.2f}m")
            
            # Check for collision
            if result.metrics.get('collision', False):
                logger.error(f"Collision detected at t={self.time:.1f}s!")
                break
            
            # Check for goal reached
            current_s, _, _, _, _, _ = self.coord_converter.find_nearest_point_on_path(
                self.ego_state.x, self.ego_state.y
            )
            max_s = self.reference_path.s[-1]
            dist_to_goal = max_s - current_s
            
            if dist_to_goal < 2.0:
                logger.success(f"Goal reached at t={self.time:.1f}s! (Dist to goal: {dist_to_goal:.1f}m)")
                break
        
        logger.info(f"Simulation complete: {len(self.history)} steps")
        return self.history
    
    def save_results(self, output_path: Optional[str] = None):
        """Save simulation results to file.
        
        Args:
            output_path: Output directory path
        """
        if output_path is None:
            output_path = self.config.output_path
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory data
        trajectory_file = output_dir / "trajectory.npz"
        
        # Extract data
        times = [r.time for r in self.history]
        ego_x = [r.ego_state.x for r in self.history]
        ego_y = [r.ego_state.y for r in self.history]
        ego_v = [r.ego_state.v for r in self.history]
        ego_jerk = [r.ego_state.jerk for r in self.history]
        min_distances = [r.metrics.get('min_distance', float('inf')) for r in self.history]
        ttc_list = [r.metrics.get('ttc', float('inf')) for r in self.history]
        
        # Processing times
        proc_pred = [r.processing_times.get('prediction', 0.0) for r in self.history]
        proc_plan = [r.processing_times.get('planning', 0.0) for r in self.history]

        ped_positions = [r.ped_state.positions for r in self.history]
        ped_velocities = [r.ped_state.velocities for r in self.history]
        ped_goals = [r.ped_state.goals for r in self.history]

        predicted_list = [
            r.predicted_trajectories if r.predicted_trajectories is not None else np.empty((0,))
            for r in self.history
        ]

        planned_x = [
            np.array(r.planned_path.x) if r.planned_path is not None else np.array([])
            for r in self.history
        ]
        planned_y = [
            np.array(r.planned_path.y) if r.planned_path is not None else np.array([])
            for r in self.history
        ]
        planned_v = [
            np.array(r.planned_path.v) if r.planned_path is not None else np.array([])
            for r in self.history
        ]
        planned_a = [
            np.array(r.planned_path.a) if r.planned_path is not None else np.array([])
            for r in self.history
        ]
        planned_yaw = [
            np.array(r.planned_path.yaw) if r.planned_path is not None else np.array([])
            for r in self.history
        ]
        planned_cost = [
            r.planned_path.cost if r.planned_path is not None else float('inf')
            for r in self.history
        ]
        
        ego_state_enum = [r.ego_state.state.name for r in self.history]
        
        np.savez(
            trajectory_file,
            times=np.array(times),
            ego_x=np.array(ego_x),
            ego_y=np.array(ego_y),
            ego_v=np.array(ego_v),
            ego_jerk=np.array(ego_jerk),
            ego_state=np.array(ego_state_enum), # Store state string
            min_distances=np.array(min_distances),
            ttc=np.array(ttc_list),
            proc_prediction=np.array(proc_pred),
            proc_planning=np.array(proc_plan),
            ped_positions=np.array(ped_positions, dtype=object),
            ped_velocities=np.array(ped_velocities, dtype=object),
            ped_goals=np.array(ped_goals, dtype=object),
            predicted_trajectories=np.array(predicted_list, dtype=object),
            planned_x=np.array(planned_x, dtype=object),
            planned_y=np.array(planned_y, dtype=object),
            planned_v=np.array(planned_v, dtype=object),
            planned_a=np.array(planned_a, dtype=object),
            planned_yaw=np.array(planned_yaw, dtype=object),
            planned_cost=np.array(planned_cost)
        )
        
        
        # --- Metrics Calculation (Always run for export) ---
        try:
            from ..core.metrics import calculate_aggregate_metrics
            metrics = calculate_aggregate_metrics(self.history, self.config.dt)
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            metrics = {}

        # Add processing time stats
        if proc_pred:
            metrics['avg_prediction_time'] = sum(proc_pred) / len(proc_pred)
            metrics['max_prediction_time'] = max(proc_pred)
        if proc_plan:
            metrics['avg_planning_time'] = sum(proc_plan) / len(proc_plan)
            metrics['max_planning_time'] = max(proc_plan)

        # --- save results to CSV and TXT ---
        # Prepare context data
        context = {
            "prediction_method": getattr(self.config, 'prediction_method', 'unknown'),
            "sgan_model": getattr(self.config, 'sgan_model_path', 'none'),
            "ego_target_speed": getattr(self.config, 'ego_target_speed', 0.0),
            "scenario_file": str(getattr(self.config, 'config_path', 'unknown')), # config_path might not be standard, checking
            "total_time": self.time,
            "steps": len(self.history)
        }
        
        # 1. Output CSV (metrics_summary.csv)
        csv_path = output_dir / "metrics_summary.csv"
        
        # Combine metrics and context
        csv_data = context.copy()
        csv_data.update(metrics)
        
        # We also want to record if collision happened as a boolean
        if 'collision' not in csv_data:
            csv_data['collision'] = any(r.metrics.get('collision', False) for r in self.history)

        try:
            file_exists = csv_path.exists()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
            logger.info(f"Saved metrics summary to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics CSV: {e}")

        # 2. Output Text Report (metrics_report.txt)
        txt_path = output_dir / "metrics_report.txt"
        try:
            with open(txt_path, 'w') as f:
                f.write("=" * 40 + "\n")
                f.write("       SIMULATION REPORT\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("--- Configuration ---\n")
                for k, v in context.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                
                f.write("--- Metrics ---\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                
                # Add minimal stats if metrics missing
                if not metrics:
                     f.write("No detailed metrics available.\n")
                     
                f.write("=" * 40 + "\n")
            logger.info(f"Saved metrics report to {txt_path}")
        except Exception as e:
             logger.error(f"Failed to save metrics report: {e}")


        # Calculate aggregated metrics and generate dashboard if visualization is enabled
        if getattr(self.config, 'visualization_enabled', True):
            logger.info("Simulation Metrics:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v}")


            # Generate Dashboard
            try:
                from ..visualization.dashboard import create_dashboard
                dashboard_path = output_dir / "dashboard.png"
                create_dashboard(
                    self.history, 
                    str(dashboard_path), 
                    metrics=metrics,
                    map_config=getattr(self.config, 'map_config', None)
                )
            except Exception as e:
                logger.error(f"Failed to generate dashboard: {e}")

            # Generate Simulation Plot (Restored)
            try:
                from ..visualization.dashboard import create_simulation_plot
                sim_plot_path = output_dir / "simulation.png"
                create_simulation_plot(
                    self.history,
                    str(sim_plot_path),
                    map_config=getattr(self.config, 'map_config', None)
                )
            except Exception as e:
                logger.error(f"Failed to generate simulation plot: {e}")
        else:
            logger.debug("Visualization disabled, skipping dashboard generation.")
            
    def visualize(self, output_path: Optional[str] = None):
        """Deprecated: Use dashboard generation instead."""
        pass
