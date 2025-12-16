"""Integrated simulator combining Social Force, Social-GAN, and Frenet planning.

This is the main simulation module that orchestrates all components.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from ..core.data_structures import (
    EgoVehicleState,
    PedestrianState,
    SimulationResult,
)
from ..core.coordinate_converter import CoordinateConverter
from ..config import SimulationConfig
from ..planning import CubicSpline2D, FrenetPlanner
from ..planning.frenet_planner import MAX_T
from ..pedestrian.observer import PedestrianObserver
from ..prediction.trajectory_predictor import TrajectoryPredictor


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
        ego_radius: float = 1.0
    ):
        """Initialize simulator.
        
        Args:
            initial_states: Initial state array [N, 6] (x, y, vx, vy, gx, gy)
            groups: List of grouping lists (indices)
            obstacles: List of obstacle specifications
            dt: Simulation time step
            config_file: Path to PySocialForce config file
            ego_radius: Radius of the ego vehicle [m]
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
            
        self._init_pysocialforce(groups, obstacles, config_file)

    def _init_pysocialforce(
        self, 
        groups: Optional[List[List[int]]] = None, 
        obstacles: Optional[List] = None,
        config_file: Optional[str] = None
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
        self.safety_buffer = getattr(config, "safety_buffer", 0.0)
        
        # 2. Initialize pedestrian simulator
        if len(config.ped_initial_states) > 0:
            ped_states = np.array(config.ped_initial_states)
            self.pedestrian_sim = PedestrianSimulator(
                initial_states=ped_states,
                groups=config.ped_groups,
                obstacles=config.static_obstacles,
                dt=config.dt,
                config_file=getattr(config, "social_force_config", None),
                ego_radius=self.ego_radius
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
        self.predictor = TrajectoryPredictor(
            model_path=config.sgan_model_path,
            pred_len=config.pred_len,
            num_samples=1,
            device=config.device,
            sgan_dt=self.observer.sgan_dt,
            sim_dt=config.dt,
            plan_horizon=MAX_T,
            method=getattr(config, 'prediction_method', 'sgan')
        )
        
        # 5. Initialize path planner
        self.planner = FrenetPlanner(
            reference_path=self.reference_path,
            max_speed=config.ego_max_speed,
            max_accel=config.ego_max_accel,
            max_curvature=config.ego_max_curvature,
            dt=config.dt,
            robot_radius=self.ego_radius,
            obstacle_radius=self.obstacle_radius,
            safety_buffer=self.safety_buffer,
            k_j=getattr(config, "k_j", None),
            k_t=getattr(config, "k_t", None),
            k_d=getattr(config, "k_d", None),
            k_s_dot=getattr(config, "k_s_dot", None),
            k_lat=getattr(config, "k_lat", None),
            k_lon=getattr(config, "k_lon", None)
        )
        
        # 6. Initialize coordinate converter
        self.coord_converter = CoordinateConverter(self.reference_path)
        
        # 7. Initialize ego vehicle state
        ego_arr = np.array(config.ego_initial_state)
        self.ego_state = EgoVehicleState.from_array(ego_arr, timestamp=0.0)

        # Precompute static obstacles (expanded to point set for collision checks)
        self.static_obstacle_points = self._expand_static_obstacles(
            config.static_obstacles, step=0.5
        )
        
        logger.info("Integrated simulator initialization complete")
    
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
        dynamic_obstacles = np.empty((0, 0, 2))
        static_obstacles = self.static_obstacle_points.copy()
        
        if ped_state is not None and self.observer.is_ready:
            try:
                # Get observations
                obs_traj, obs_traj_rel, seq_start_end = self.observer.get_observation()
                
                # Predict
                predicted_traj = self.predictor.predict_single_best(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                
                # Preserve time dimension for dynamic collision checks - NO CONVERSION
                dynamic_obstacles = self.coord_converter.pass_through_obstacle(
                    predicted_traj
                )
                
                logger.debug(
                    f"Predicted {predicted_traj.shape[0]} pedestrian trajectories "
                    f"for {predicted_traj.shape[1]} steps"
                )
                
            except Exception as e:
                logger.warning(f"Prediction failed: {e}, using current positions")
                if ped_state is not None:
                    dynamic_obstacles = ped_state.positions[:, None, :]
        elif ped_state is not None:
            # Not enough observations yet, use current positions
            dynamic_obstacles = ped_state.positions[:, None, :]

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
        
        # 3. Plan path
        planned_path = self.planner.plan(
            self.ego_state,
            static_obstacles,
            dynamic_obstacles,
            target_speed=self.config.ego_target_speed
        )
        
        # 4. Update ego vehicle state
        # Store old acceleration for jerk calculation
        old_a = self.ego_state.a
        
        if planned_path is not None and len(planned_path) >= 2:
            # Follow the first step of the planned path
            try:
                self.ego_state = planned_path.get_state_at_index(1)
                
                # Calculate jerk: (new_a - old_a) / dt
                current_jerk = (self.ego_state.a - old_a) / self.config.dt
                self.ego_state.jerk = current_jerk
                
                self.ego_state.timestamp = self.time + self.config.dt
                logger.debug(f"Ego vehicle moved to ({self.ego_state.x:.2f}, "
                           f"{self.ego_state.y:.2f})")
            except IndexError as e:
                logger.warning(f"Path index error: {e}, using index 0 instead")
                if len(planned_path) > 0:
                    self.ego_state = planned_path.get_state_at_index(0)
                    
                    # Calculate jerk: (new_a - old_a) / dt
                    current_jerk = (self.ego_state.a - old_a) / self.config.dt
                    self.ego_state.jerk = current_jerk
                    
                    self.ego_state.timestamp = self.time + self.config.dt
                else:
                    logger.warning("No valid path found, emergency stop")
                    self.ego_state.v = max(0.0, self.ego_state.v - self.config.ego_max_accel * self.config.dt)
                    new_a = -self.config.ego_max_accel if self.ego_state.v > 0 else 0.0
                    
                    # Calculate jerk: (new_a - old_a) / dt
                    current_jerk = (new_a - old_a) / self.config.dt
                    
                    self.ego_state.a = new_a
                    self.ego_state.jerk = current_jerk
                    self.ego_state.timestamp = self.time + self.config.dt
        else:
            # No valid path, maintain current state (emergency stop)
            logger.warning("No valid path found, emergency stop")
            self.ego_state.v = max(0.0, self.ego_state.v - self.config.ego_max_accel * self.config.dt)
            new_a = -self.config.ego_max_accel if self.ego_state.v > 0 else 0.0
            
            # Calculate jerk: (new_a - old_a) / dt
            current_jerk = (new_a - old_a) / self.config.dt
            
            self.ego_state.a = new_a
            self.ego_state.jerk = current_jerk
            self.ego_state.timestamp = self.time + self.config.dt
        
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
            planned_path=planned_path,
            ego_radius=self.ego_radius,
            ped_radius=self.ped_radius,
            safety_buffer=self.safety_buffer
        )
        
        # Compute metrics
        result.metrics = result.compute_safety_metrics()
        
        # Record history
        self.history.append(result)
        
        # Update time
        self.time += self.config.dt
        self.step_count += 1
        
        return result

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
        
        np.savez(
            trajectory_file,
            times=np.array(times),
            ego_x=np.array(ego_x),
            ego_y=np.array(ego_y),
            ego_v=np.array(ego_v),
            ego_jerk=np.array(ego_jerk),
            min_distances=np.array(min_distances),
            ttc=np.array(ttc_list),
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
        
        # Calculate aggregated metrics and generate dashboard if visualization is enabled
        if getattr(self.config, 'visualization_enabled', True):
            try:
                from ..core.metrics import calculate_aggregate_metrics
                metrics = calculate_aggregate_metrics(self.history, self.config.dt)
                logger.info("Simulation Metrics:")
                for k, v in metrics.items():
                    logger.info(f"  {k}: {v}")
            except Exception as e:
                logger.error(f"Failed to calculate metrics: {e}")
                metrics = {}

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
