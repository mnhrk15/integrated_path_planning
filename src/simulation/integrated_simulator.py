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
from ..pedestrian.observer import PedestrianObserver
from ..prediction.trajectory_predictor import TrajectoryPredictor


# Try importing PySocialForce
try:
    import pysocialforce as psf
    PYSOCIALFORCE_AVAILABLE = True
except ImportError:
    PYSOCIALFORCE_AVAILABLE = False
    logger.warning("pysocialforce package not found. Using simplified dynamics.")


class PedestrianSimulator:
    """Pedestrian simulator using Social Force model.
    
    Automatically uses PySocialForce if available, otherwise falls back
    to simplified constant velocity dynamics.
    
    Args:
        initial_states: Initial pedestrian states [n_peds, 6]
                       Format: [x, y, vx, vy, goal_x, goal_y]
        groups: Pedestrian group indices (list of lists)
        obstacles: Static obstacles (list of (x_min, x_max, y_min, y_max))
        dt: Time step [s]
        config_file: Optional path to pysocialforce config file
    """
    
    def __init__(
        self,
        initial_states: np.ndarray,
        groups: Optional[List[List[int]]] = None,
        obstacles: Optional[List] = None,
        dt: float = 0.1,
        config_file: Optional[str] = None
    ):
        self.state = initial_states.copy()  # [n_peds, 6]: [x, y, vx, vy, gx, gy]
        self.groups = groups or [[i] for i in range(len(initial_states))]
        self.obstacles = obstacles
        self.dt = dt
        self.time = 0.0
        self.config_file = config_file
        
        # Initialize simulator
        if PYSOCIALFORCE_AVAILABLE:
            self._init_pysocialforce()
            logger.info(f"PySocialForce simulator initialized with "
                       f"{len(initial_states)} pedestrians, "
                       f"{len(self.groups)} groups")
        else:
            self.simulator = None
            logger.info(f"Simple dynamics simulator initialized with "
                       f"{len(initial_states)} pedestrians (PySocialForce not available)")
    
    def _init_pysocialforce(self):
        """Initialize PySocialForce simulator."""
        try:
            # Create simulator
            self.simulator = psf.Simulator(
                state=self.state,
                groups=self.groups,
                obstacles=self.obstacles,
                config_file=self.config_file
            )
            logger.debug("PySocialForce simulator created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PySocialForce: {e}")
            logger.warning("Falling back to simple dynamics")
            self.simulator = None
    
    def step(self, n: int = 1):
        """Advance simulation by n time steps.
        
        Args:
            n: Number of time steps to simulate
        """
        if self.simulator is not None and PYSOCIALFORCE_AVAILABLE:
            # Use PySocialForce
            self.simulator.step(n)
            
            # Extract state from PySocialForce
            ped_states, _ = self.simulator.get_states()
            current_state = ped_states[-1]  # Get latest state
            
            # Update our state array
            # PySocialForce state: [x, y, vx, vy, dest_x, dest_y, tau]
            self.state[:, 0:6] = current_state[:, 0:6]
            self.time += self.dt * n
        else:
            # Use simple dynamics
            for _ in range(n):
                self._step_simple()
    
    def _step_simple(self):
        """Simple constant velocity dynamics (fallback)."""
        positions = self.state[:, 0:2]
        velocities = self.state[:, 2:4]
        goals = self.state[:, 4:6]
        
        # Calculate desired direction
        to_goal = goals - positions
        dist_to_goal = np.linalg.norm(to_goal, axis=1, keepdims=True)
        desired_direction = np.where(
            dist_to_goal > 0.1,
            to_goal / (dist_to_goal + 1e-6),
            np.zeros_like(to_goal)
        )
        
        # Simple force: move toward goal at 1.0 m/s
        desired_velocity = desired_direction * 1.0
        
        # Update velocity with damping (tau = 0.5s)
        tau = 0.5
        alpha = self.dt / (tau + self.dt)
        new_velocity = alpha * desired_velocity + (1 - alpha) * velocities
        
        # Update position
        new_position = positions + new_velocity * self.dt
        
        # Stop if reached goal
        reached = dist_to_goal.flatten() < 0.5
        new_velocity[reached] = 0.0
        
        # Update state
        self.state[:, 0:2] = new_position
        self.state[:, 2:4] = new_velocity
        
        self.time += self.dt
    
    def get_state(self) -> PedestrianState:
        """Get current pedestrian state.
        
        Returns:
            Current pedestrian state
        """
        return PedestrianState(
            positions=self.state[:, 0:2].copy(),
            velocities=self.state[:, 2:4].copy(),
            goals=self.state[:, 4:6].copy(),
            timestamp=self.time
        )


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
        
        # 2. Initialize pedestrian simulator
        if len(config.ped_initial_states) > 0:
            ped_states = np.array(config.ped_initial_states)
            self.pedestrian_sim = PedestrianSimulator(
                initial_states=ped_states,
                groups=config.ped_groups,
                obstacles=config.static_obstacles,
                dt=config.dt
            )
        else:
            self.pedestrian_sim = None
            logger.warning("No pedestrians in scenario")
        
        # 3. Initialize pedestrian observer
        self.observer = PedestrianObserver(
            obs_len=config.obs_len,
            dt=config.dt
        )
        
        # 4. Initialize trajectory predictor
        self.predictor = TrajectoryPredictor(
            model_path=config.sgan_model_path,
            pred_len=config.pred_len,
            num_samples=1,
            device=config.device
        )
        
        # 5. Initialize path planner
        self.planner = FrenetPlanner(
            reference_path=self.reference_path,
            max_speed=config.ego_max_speed,
            max_accel=config.ego_max_accel,
            max_curvature=config.ego_max_curvature,
            dt=config.dt
        )
        
        # 6. Initialize coordinate converter
        self.coord_converter = CoordinateConverter(self.reference_path)
        
        # 7. Initialize ego vehicle state
        ego_arr = np.array(config.ego_initial_state)
        self.ego_state = EgoVehicleState.from_array(ego_arr, timestamp=0.0)
        
        logger.info("Integrated simulator initialization complete")
    
    def step(self) -> SimulationResult:
        """Execute one simulation step.
        
        Returns:
            Simulation result for this step
        """
        # 1. Advance pedestrian simulation
        ped_state = None
        if self.pedestrian_sim is not None:
            self.pedestrian_sim.step()
            ped_state = self.pedestrian_sim.get_state()
            
            # Update observer
            self.observer.update(ped_state)
        
        # 2. Predict pedestrian trajectories
        predicted_traj = None
        obstacles = np.empty((0, 2))
        
        if ped_state is not None and self.observer.is_ready:
            try:
                # Get observations
                obs_traj, obs_traj_rel, seq_start_end = self.observer.get_observation()
                
                # Predict
                predicted_traj = self.predictor.predict_single_best(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                
                # Convert to obstacle points
                obstacles = self.coord_converter.global_to_frenet_obstacle(
                    predicted_traj
                )
                
                logger.debug(f"Predicted {predicted_traj.shape[0]} pedestrian "
                           f"trajectories, {len(obstacles)} obstacle points")
                
            except Exception as e:
                logger.warning(f"Prediction failed: {e}, using current positions")
                if ped_state is not None:
                    obstacles = ped_state.positions
        elif ped_state is not None:
            # Not enough observations yet, use current positions
            obstacles = ped_state.positions
        
        # 3. Plan path
        planned_path = self.planner.plan(
            self.ego_state,
            obstacles,
            target_speed=self.config.ego_target_speed
        )
        
        # 4. Update ego vehicle state
        if planned_path is not None and len(planned_path) >= 2:
            # Follow the first step of the planned path
            try:
                self.ego_state = planned_path.get_state_at_index(1)
                self.ego_state.timestamp = self.time + self.config.dt
                logger.debug(f"Ego vehicle moved to ({self.ego_state.x:.2f}, "
                           f"{self.ego_state.y:.2f})")
            except IndexError as e:
                logger.warning(f"Path index error: {e}, using index 0 instead")
                if len(planned_path) > 0:
                    self.ego_state = planned_path.get_state_at_index(0)
                    self.ego_state.timestamp = self.time + self.config.dt
                else:
                    logger.warning("No valid path found, emergency stop")
                    self.ego_state.v = max(0.0, self.ego_state.v - self.config.ego_max_accel * self.config.dt)
                    self.ego_state.a = -self.config.ego_max_accel if self.ego_state.v > 0 else 0.0
                    self.ego_state.timestamp = self.time + self.config.dt
        else:
            # No valid path, maintain current state (emergency stop)
            logger.warning("No valid path found, emergency stop")
            self.ego_state.v = max(0.0, self.ego_state.v - self.config.ego_max_accel * self.config.dt)
            self.ego_state.a = -self.config.ego_max_accel if self.ego_state.v > 0 else 0.0
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
            planned_path=planned_path
        )
        
        # Compute metrics
        result.metrics = result.compute_safety_metrics()
        
        # Record history
        self.history.append(result)
        
        # Update time
        self.time += self.config.dt
        self.step_count += 1
        
        return result
    
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
        min_distances = [r.metrics.get('min_distance', float('inf')) for r in self.history]
        
        np.savez(
            trajectory_file,
            times=np.array(times),
            ego_x=np.array(ego_x),
            ego_y=np.array(ego_y),
            ego_v=np.array(ego_v),
            min_distances=np.array(min_distances)
        )
        
        logger.info(f"Results saved to {trajectory_file}")
    
    def visualize(self, output_path: Optional[str] = None):
        """Visualize simulation results.
        
        Args:
            output_path: Output file path for animation
        """
        if not self.config.visualization_enabled:
            logger.info("Visualization disabled")
            return
        
        # This would use matplotlib to create an animation
        # For now, just save a simple plot
        import matplotlib.pyplot as plt
        
        if output_path is None:
            output_path = Path(self.config.output_path) / "simulation.png"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Trajectory
        # Reference path
        s = np.arange(0, self.reference_path.s[-1], 0.1)
        rx, ry = [], []
        for i_s in s:
            ix, iy = self.reference_path.calc_position(i_s)
            if ix is not None and iy is not None:
                rx.append(ix)
                ry.append(iy)
        
        ax1.plot(rx, ry, '--', color='gray', label='Reference Path', linewidth=2)
        
        # Ego trajectory
        ego_x = [r.ego_state.x for r in self.history]
        ego_y = [r.ego_state.y for r in self.history]
        ax1.plot(ego_x, ego_y, 'b-', label='Ego Vehicle', linewidth=2)
        ax1.plot(ego_x[0], ego_y[0], 'go', markersize=10, label='Start')
        ax1.plot(ego_x[-1], ego_y[-1], 'ro', markersize=10, label='End')
        
        # Pedestrian trajectories
        if self.pedestrian_sim is not None:
            for i, result in enumerate(self.history[::10]):  # Sample every 10 steps
                if result.ped_state.n_peds > 0:
                    ax1.plot(result.ped_state.positions[:, 0],
                           result.ped_state.positions[:, 1],
                           'r.', markersize=5, alpha=0.3)
        
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Trajectory')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot 2: Metrics over time
        times = [r.time for r in self.history]
        velocities = [r.ego_state.v for r in self.history]
        min_distances = [r.metrics.get('min_distance', float('inf')) for r in self.history]
        
        ax2_twin = ax2.twinx()
        
        ax2.plot(times, velocities, 'b-', label='Velocity', linewidth=2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True)
        
        ax2_twin.plot(times, min_distances, 'r-', label='Min Distance', linewidth=2)
        ax2_twin.set_ylabel('Min Distance to Pedestrian [m]', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2_twin.axhline(y=1.0, color='orange', linestyle='--', label='Safety Threshold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"Visualization saved to {output_path}")
        plt.close()
