"""Integrated simulator combining Social Force, Social-GAN, and Frenet planning.

This is the main simulation module that orchestrates all components.
"""

import copy
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
from ..core.footprint import footprint_from_config
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
        social_force_params: Optional[Dict] = None,
        v0_randomization: bool = False,
        v0_std: float = 0.19,
        v0_min: float = 0.3
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
            v0_randomization: Add per-agent N(0, v0_std) noise to desired speeds
                (pysocialforce max_speeds), making the ground truth
                distributional across agents. Draws from the global NumPy RNG,
                so runs are reproducible under the benchmark seed; when False
                no random numbers are consumed (behavior preservation).
            v0_std: Standard deviation of the desired-speed noise [m/s]
            v0_min: Floor on the randomized desired speed [m/s]
        """
        self.dt = dt
        self.initial_states = initial_states
        self.time = 0.0
        self.ego_radius = ego_radius
        self.ego_repulsion_sigma = 0.7
        self.ego_repulsion_v0 = 3.5
        self._ego_position: Optional[np.ndarray] = None
        
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

        if v0_randomization:
            # pysocialforce recomputes max_speeds from initial_speeds on every
            # state assignment, so the persistent initial_speeds must carry the
            # randomization: max_speeds = mult * max(init + noise/mult, min/mult)
            #              = max(nominal_v0 + noise, v0_min)
            peds = self.sim.peds
            multiplier = float(peds.max_speed_multiplier)
            noise = np.random.normal(0.0, v0_std, len(peds.initial_speeds))
            peds.initial_speeds = np.maximum(
                peds.initial_speeds + noise / multiplier, v0_min / multiplier
            )
            peds.max_speeds = multiplier * peds.initial_speeds

    @staticmethod
    def _set_nested_config_value(config: Dict, key: str, value) -> None:
        """Set a dotted configuration key in a nested dictionary."""
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    def _apply_social_force_params(self, social_force_params: Optional[Dict]) -> None:
        """Apply wrapper and PySocialForce parameters after initialization."""
        if not social_force_params:
            return

        legacy_aliases = {
            "ped_repulsion.sigma": "ego_repulsion.sigma",
            "ped_repulsion.v0": "ego_repulsion.v0",
        }
        normalized_params = dict(social_force_params)
        for old_key, new_key in legacy_aliases.items():
            if old_key in normalized_params and new_key not in normalized_params:
                logger.warning(f"'{old_key}' is deprecated; use '{new_key}'")
                normalized_params[new_key] = normalized_params[old_key]

        self.ego_repulsion_sigma = float(
            normalized_params.get("ego_repulsion.sigma", self.ego_repulsion_sigma)
        )
        self.ego_repulsion_v0 = float(
            normalized_params.get("ego_repulsion.v0", self.ego_repulsion_v0)
        )
        if self.ego_repulsion_sigma <= 0:
            raise ValueError("ego_repulsion.sigma must be positive")
        if self.ego_repulsion_v0 < 0:
            raise ValueError("ego_repulsion.v0 must be non-negative")

        for key, value in normalized_params.items():
            if key.startswith("ego_repulsion.") or key in legacy_aliases:
                continue
            self._set_nested_config_value(self.sim.config.config, key, value)

        agent_radius = normalized_params.get(
            "agent_radius",
            normalized_params.get("scene.agent_radius"),
        )
        if agent_radius is not None:
            self.sim.peds.agent_radius = float(agent_radius)

    def _overwrite_ego_state(self, ego_state: EgoVehicleState) -> None:
        """Store the externally planned ego state for pedestrian interaction."""
        self._ego_position = np.array([ego_state.x, ego_state.y], dtype=float)

    def _compute_ego_repulsive_force(self) -> np.ndarray:
        """Compute an explicit ego-to-pedestrian repulsive force."""
        forces = np.zeros((self.sim.peds.size(), 2))
        if self._ego_position is None or self.ego_repulsion_v0 == 0:
            return forces

        positions = self.sim.peds.pos()
        deltas = positions - self._ego_position
        distances = np.linalg.norm(deltas, axis=1)
        directions = np.zeros_like(deltas)
        nonzero = distances > 1e-9
        directions[nonzero] = deltas[nonzero] / distances[nonzero, None]

        clearance = np.maximum(
            distances - (self.ego_radius + float(self.sim.peds.agent_radius)),
            0.0,
        )
        magnitudes = self.ego_repulsion_v0 * np.exp(-clearance / self.ego_repulsion_sigma)
        return directions * magnitudes[:, None]

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
                    # PySocialForce expects (x1, x2, y1, y2)
                    segments = [
                        (x_min, x_max, y_min, y_min), # Bottom edge
                        (x_max, x_max, y_min, y_max), # Right edge
                        (x_max, x_min, y_max, y_max), # Top edge
                        (x_min, x_min, y_max, y_min)  # Left edge
                    ]
                    for s in segments:
                        # Check if length > epsilon (s is x1, x2, y1, y2)
                        # Calculates Euclidean distance squared to be safe, or just check component diffs
                        dx = s[1] - s[0]
                        dy = s[3] - s[2]
                        if (dx*dx + dy*dy) > 1e-12:
                            psf_obstacles.append(s)
        
        self.sim = psf.Simulator(
            state=self.initial_states,
            groups=groups,
            obstacles=psf_obstacles if psf_obstacles else None,
            config_file=config_file
        )

        self._apply_social_force_params(social_force_params)
        
        # Manually set dt (step_width)
        if hasattr(self.sim, 'peds'):
            self.sim.peds.step_width = self.dt

    def step(self, ego_state: Optional[EgoVehicleState] = None, n: int = 1):
        """Advance simulation by n time steps.
        
        Args:
            ego_state: Current state of the ego vehicle
            n: Number of time steps to simulate
        """
        if self.sim:
            for _ in range(n):
                if ego_state is not None:
                    self._overwrite_ego_state(ego_state)
                force = self.sim.compute_forces() + self._compute_ego_repulsive_force()
                self.sim.peds.step(force)
                self.time += self.dt
        else:
            raise RuntimeError("Simulator not initialized correctly.")

    def get_state(self) -> PedestrianState:
        """Get current pedestrian state.
        
        Returns:
            Current pedestrian state
        """
        if self.sim:
            # PySocialForce state: [N, 7] (x, y, vx, vy, gx, gy, tau)
            full_state = self.sim.peds.state
            
            return PedestrianState(
                positions=full_state[:, 0:2].copy(),
                velocities=full_state[:, 2:4].copy(),
                goals=full_state[:, 4:6].copy(),
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
        self.ego_footprint = footprint_from_config(config)  # None = legacy single circle
        
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
                social_force_params=getattr(config, "social_force_params", None),
                v0_randomization=getattr(config, "sfm_v0_randomization", False),
                v0_std=getattr(config, "sfm_v0_std", 0.19),
                v0_min=getattr(config, "sfm_v0_min", 0.3)
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
            k_lon=config.k_lon,
            chance_epsilon=getattr(config, 'chance_epsilon', 0.0),
            collision_margin_inflation=getattr(config, 'collision_margin_inflation', 1.0),
            footprint=self.ego_footprint
        )

        # Distribution-aware planning: feed the full prediction distribution to the
        # planner's chance-constrained collision check instead of one sample.
        self.distribution_aware_planning = getattr(config, 'distribution_aware_planning', False)

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

        # Persistent prediction failures must surface instead of silently
        # degrading to constant-velocity fallback on every step (C-2).
        self._consecutive_prediction_failures = 0
        self._max_consecutive_prediction_failures = 5

        # Why the last run() ended: 'collision', 'goal' or 'timeout' (None
        # before/while running and after a crash mid-run). Benchmarks use this
        # instead of inferring the outcome from the end time.
        self.termination_reason: Optional[str] = None

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
    
    def _update_prediction(self, ped_state: Optional[PedestrianState]):
        """Update and retrieve pedestrian predictions."""
        predicted_traj = None
        predicted_dist = None
        dynamic_obstacles = np.empty((0, 0, 2))
        dynamic_obstacles_dist = None
        t_pred = 0.0

        if ped_state is not None and self.observer.is_ready:
            try:
                # Get observations
                obs_traj, obs_traj_rel, seq_start_end = self.observer.get_observation()

                # Re-anchor predictions from the last observation sample time to
                # the current pedestrian time (observer samples every sgan_dt,
                # so the anchor can be up to sgan_dt - dt stale).
                last_sample_time = self.observer.last_sample_time
                staleness = 0.0
                if last_sample_time is not None:
                    staleness = max(ped_state.timestamp - last_sample_time, 0.0)

                # Predict
                t_start = time.perf_counter()
                predicted_traj, predicted_dist = self.predictor.predict_single_best(
                    obs_traj, obs_traj_rel, seq_start_end, staleness=staleness
                )
                t_pred = time.perf_counter() - t_start
                
                # Preserve time dimension for dynamic collision checks - NO CONVERSION
                dynamic_obstacles = self.coord_converter.pass_through_obstacle(
                    predicted_traj
                )

                # For distribution-aware planning, keep the full sample set (global
                # coordinates, same pass-through as the representative sample).
                if self.distribution_aware_planning and predicted_dist is not None:
                    dynamic_obstacles_dist = np.asarray(predicted_dist)

                logger.debug(
                    f"Predicted {predicted_traj.shape[0]} pedestrian trajectories "
                    f"for {predicted_traj.shape[1]} steps"
                )
                self._consecutive_prediction_failures = 0

            except Exception as e:
                self._consecutive_prediction_failures += 1
                if (self._consecutive_prediction_failures
                        >= self._max_consecutive_prediction_failures):
                    raise RuntimeError(
                        f"Prediction failed {self._consecutive_prediction_failures} times in a row "
                        f"(last error: {e}); a persistent failure (e.g. wrong model for the "
                        f"prediction method) must not silently degrade to the CV fallback"
                    ) from e
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

            # Mirror the current-position prepend for each distribution sample so the
            # time alignment matches the single-sample obstacles.
            if dynamic_obstacles_dist is not None and dynamic_obstacles_dist.size > 0:
                n_samples = dynamic_obstacles_dist.shape[0]
                cur_dist = np.broadcast_to(
                    current_positions[None, ...],
                    (n_samples,) + current_positions.shape
                )
                dynamic_obstacles_dist = np.concatenate(
                    [cur_dist, dynamic_obstacles_dist], axis=2
                )

        return predicted_traj, predicted_dist, dynamic_obstacles, dynamic_obstacles_dist, t_pred

    def _execute_planning_cycle(
        self,
        static_obstacles: np.ndarray,
        dynamic_obstacles: np.ndarray,
        ped_state: Optional[PedestrianState],
        dynamic_obstacles_distribution: Optional[np.ndarray] = None
    ):
        """Execute planning cycle with state machine management and retries."""
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
            constraint_overrides=sm_output.constraint_overrides,
            dynamic_obstacles_distribution=dynamic_obstacles_distribution
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
                ped_radius=self.ped_radius,
                footprint=self.ego_footprint
            )
        else:
            current_metrics = {'min_distance': float('inf'), 'collision': False,
                               'ttc': float('inf'), 'clearance': float('inf')}
        
        # Update SM
        new_sm_output = self.state_machine.update(found_path, current_metrics)

        # Escalate-and-retry loop: while planning fails and the state machine
        # escalates (NORMAL -> CAUTION -> EMERGENCY), re-plan immediately under
        # the relaxed constraints instead of wasting a step. The loop ends once
        # the state stops changing (EMERGENCY reached) or after
        # _max_replan_attempts retries. Retry planning time is part of t_plan
        # so the per-step planning cost is not under-reported on the heaviest
        # steps (M-15).
        while (planned_path is None
               and new_sm_output.state != sm_output.state
               and self._replan_attempts < self._max_replan_attempts):
            logger.warning(
                f"Planning failed in {sm_output.state}. Transitioning to {new_sm_output.state} "
                f"and retrying (attempt {self._replan_attempts + 1}/{self._max_replan_attempts})..."
            )

            # Update local state variable to reflect new state for logging/recording.
            # Copy first: the current object may still be referenced by the
            # previous step's SimulationResult and must not be edited in place.
            self.ego_state = copy.copy(self.ego_state)
            self.ego_state.state = new_sm_output.state
            self._replan_attempts += 1

            target_speed = new_sm_output.target_speed_override
            if target_speed is None:
                target_speed = self.config.ego_target_speed

            t_start = time.perf_counter()
            planned_path = self.planner.plan(
                self.ego_state,
                static_obstacles,
                dynamic_obstacles,
                target_speed=target_speed,
                constraint_overrides=new_sm_output.constraint_overrides,
                dynamic_obstacles_distribution=dynamic_obstacles_distribution
            )
            t_plan += time.perf_counter() - t_start

            if planned_path is not None:
                logger.info(f"Re-planning successful in {new_sm_output.state}")
                break

            logger.error(
                f"Re-planning failed even in {new_sm_output.state} "
                f"(attempt {self._replan_attempts}/{self._max_replan_attempts})"
            )
            sm_output = new_sm_output
            new_sm_output = self.state_machine.update(False, current_metrics)

        if planned_path is None:
            logger.error(
                f"Re-planning exhausted in {new_sm_output.state} "
                f"({self._replan_attempts} retr{'y' if self._replan_attempts == 1 else 'ies'}). "
                f"Proceeding with emergency stop."
            )

        return planned_path, t_plan

    def _update_ego_state(self, planned_path):
        """Update ego vehicle state based on planned path."""
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
        predicted_traj, predicted_dist, dynamic_obstacles, dynamic_obstacles_dist, t_pred = \
            self._update_prediction(ped_state)

        # 3. Plan path with State Machine
        static_obstacles = self.static_obstacle_points.copy()
        planned_path, t_plan = self._execute_planning_cycle(
            static_obstacles, dynamic_obstacles, ped_state, dynamic_obstacles_dist
        )

        # 4. Update ego vehicle state
        self._update_ego_state(planned_path)
        
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
            footprint=self.ego_footprint,
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
        # Replace the state object: the previous one is referenced by the last
        # recorded SimulationResult, and in-place edits would rewrite history.
        self.ego_state = copy.copy(self.ego_state)

        # Use emergency deceleration
        max_dec = self.config.ego_max_accel * 2.0 # Hard braking

        # The vehicle keeps moving while braking: integrate kinematics with the
        # pre-deceleration speed so the braking distance is not silently zero.
        self.ego_state.x += self.ego_state.v * np.cos(self.ego_state.yaw) * self.config.dt
        self.ego_state.y += self.ego_state.v * np.sin(self.ego_state.yaw) * self.config.dt

        self.ego_state.v = max(0.0, self.ego_state.v - max_dec * self.config.dt)
        new_a = -max_dec if self.ego_state.v > 0 else 0.0

        current_jerk = (new_a - old_a) / self.config.dt

        self.ego_state.a = new_a
        self.ego_state.jerk = current_jerk
        self.ego_state.timestamp = self.time + self.config.dt

        # The ego now moves straight, so the previously adopted path's
        # curvature no longer describes it (guarded: some tests build this
        # simulator without a planner).
        planner = getattr(self, 'planner', None)
        if planner is not None:
            planner.reset_ego_curvature()

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
    
    @property
    def goal_reached(self) -> bool:
        """True when the last run() ended by reaching the goal.

        A collision also ends a run early, so the end time alone cannot
        identify goal completion — benchmarks must use this instead.
        """
        return self.termination_reason == 'goal'

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

        # None while running (and after a crash mid-run); 'timeout' is only
        # assigned once the loop genuinely exhausts the step budget.
        self.termination_reason = None
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
                self.termination_reason = 'collision'
                break

            # Check for goal reached
            current_s, _, _, _, _, _ = self.coord_converter.find_nearest_point_on_path(
                self.ego_state.x, self.ego_state.y
            )
            max_s = self.reference_path.s[-1]
            dist_to_goal = max_s - current_s

            if dist_to_goal < 2.0:
                logger.success(f"Goal reached at t={self.time:.1f}s! (Dist to goal: {dist_to_goal:.1f}m)")
                self.termination_reason = 'goal'
                break

        if self.termination_reason is None:
            self.termination_reason = 'timeout'

        logger.info(f"Simulation complete: {len(self.history)} steps "
                    f"(termination: {self.termination_reason})")
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
        ego_yaw = [r.ego_state.yaw for r in self.history]
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
            ego_yaw=np.array(ego_yaw),
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
            metrics = calculate_aggregate_metrics(
                self.history,
                self.config.dt,
                prediction_dt=self.observer.sgan_dt,
                prediction_steps=self.config.pred_len,
            )
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
            "seed": getattr(self.config, 'run_seed', 'not_set'),
            "termination_reason": self.termination_reason,
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
