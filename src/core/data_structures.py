"""Core data structures for the integrated path planning system.

This module defines the fundamental data structures used across all components
of the system, ensuring type safety and clear interfaces.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

# Import VehicleState but avoid circular dependency if possible is tricky.
# Instead, we just import it. If circular dependency happens, we might need to move enum here.
# For now, let's redefine the Enum here or import it if appropriate.
# Given it's core, let's look at where state_machine is. src/core/state_machine.py
# data_structures.py is imported BY state_machine.py usually (or used by it).
# So state_machine importing data_structures is fine.
# But data_structures importing state_machine is bad.
# Solution: Move VehicleState Enum to data_structures or a new enums file.
# Let's move VehicleState to here.

from enum import Enum, auto

class VehicleState(Enum):
    """Vehicle operational states."""
    NORMAL = auto()
    CAUTION = auto()
    EMERGENCY = auto()


@dataclass
class EgoVehicleState:
    """State of the ego vehicle.
    
    Attributes:
        x: X coordinate in global frame [m]
        y: Y coordinate in global frame [m]
        yaw: Heading angle [rad]
        v: Velocity [m/s]
        a: Acceleration [m/s²]
        timestamp: Time stamp [s]
    """
    x: float
    y: float
    yaw: float
    v: float
    a: float
    jerk: float = 0.0
    timestamp: float = 0.0
    state: VehicleState = VehicleState.NORMAL
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, yaw, v, a, jerk]."""
        return np.array([self.x, self.y, self.yaw, self.v, self.a, self.jerk])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: float = 0.0) -> 'EgoVehicleState':
        """Create from numpy array [x, y, yaw, v, a, (jerk)]."""
        jerk = arr[5] if len(arr) > 5 else 0.0
        return cls(x=arr[0], y=arr[1], yaw=arr[2], v=arr[3], a=arr[4], 
                   jerk=jerk, timestamp=timestamp)


@dataclass
class PedestrianState:
    """State of pedestrians in the scene.
    
    Attributes:
        positions: Positions of all pedestrians [n_peds, 2] in [x, y] format
        velocities: Velocities of all pedestrians [n_peds, 2] in [vx, vy] format
        goals: Goal positions of all pedestrians [n_peds, 2] in [gx, gy] format
        ids: Optional pedestrian IDs
        timestamp: Time stamp [s]
    """
    positions: np.ndarray  # Shape: (n_peds, 2)
    velocities: np.ndarray  # Shape: (n_peds, 2)
    goals: np.ndarray  # Shape: (n_peds, 2)
    ids: Optional[np.ndarray] = None  # Shape: (n_peds,)
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Validate array shapes."""
        assert self.positions.shape[1] == 2, "Positions must be (n_peds, 2)"
        assert self.velocities.shape[1] == 2, "Velocities must be (n_peds, 2)"
        assert self.goals.shape[1] == 2, "Goals must be (n_peds, 2)"
        assert self.positions.shape[0] == self.velocities.shape[0] == self.goals.shape[0], \
            "All arrays must have same number of pedestrians"
        
        if self.ids is None:
            self.ids = np.arange(self.n_peds)
    
    @property
    def n_peds(self) -> int:
        """Number of pedestrians."""
        return self.positions.shape[0]
    
    @property
    def pedestrians(self):
        """Compatibility alias returning pedestrian positions array."""
        return self.positions
    
    def to_social_force_format(self) -> np.ndarray:
        """Convert to Social Force format [n_peds, 6]: [x, y, vx, vy, gx, gy]."""
        return np.hstack([self.positions, self.velocities, self.goals])
    
    @classmethod
    def from_social_force_format(cls, state: np.ndarray, 
                                  timestamp: float = 0.0) -> 'PedestrianState':
        """Create from Social Force format [n_peds, 6/7]: [x, y, vx, vy, gx, gy, (tau)]."""
        return cls(
            positions=state[:, 0:2],
            velocities=state[:, 2:4],
            goals=state[:, 4:6],
            timestamp=timestamp
        )


@dataclass
class FrenetState:
    """State in Frenet coordinate frame.
    
    Attributes:
        s: Longitudinal position along reference path [m]
        s_d: Longitudinal velocity [m/s]
        s_dd: Longitudinal acceleration [m/s²]
        d: Lateral offset from reference path [m]
        d_d: Lateral velocity [m/s]
        d_dd: Lateral acceleration [m/s²]
    """
    s: float
    s_d: float
    s_dd: float
    d: float
    d_d: float
    d_dd: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [s, s_d, s_dd, d, d_d, d_dd]."""
        return np.array([self.s, self.s_d, self.s_dd, self.d, self.d_d, self.d_dd])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FrenetState':
        """Create from numpy array [s, s_d, s_dd, d, d_d, d_dd]."""
        return cls(s=arr[0], s_d=arr[1], s_dd=arr[2], 
                   d=arr[3], d_d=arr[4], d_dd=arr[5])


@dataclass
class FrenetPath:
    """Trajectory in Frenet coordinate frame.
    
    Attributes:
        t: Time steps
        s, s_d, s_dd, s_ddd: Longitudinal trajectory and derivatives
        d, d_d, d_dd, d_ddd: Lateral trajectory and derivatives
        x, y: Global coordinates
        yaw: Heading angles
        v: Velocities
        a: Accelerations
        c: Curvatures
        cost: Total cost of this path
    """
    t: List[float] = field(default_factory=list)
    s: List[float] = field(default_factory=list)
    s_d: List[float] = field(default_factory=list)
    s_dd: List[float] = field(default_factory=list)
    s_ddd: List[float] = field(default_factory=list)
    d: List[float] = field(default_factory=list)
    d_d: List[float] = field(default_factory=list)
    d_dd: List[float] = field(default_factory=list)
    d_ddd: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    a: List[float] = field(default_factory=list)
    c: List[float] = field(default_factory=list)
    cost: float = float('inf')
    
    def __len__(self) -> int:
        """Length of the path."""
        # Return the minimum length of all required lists to ensure consistency
        # This prevents IndexError when some lists are shorter due to conversion failures
        if len(self.t) == 0:
            return 0
        # Check all required lists and return the minimum length
        lengths = [
            len(self.t),
            len(self.x) if self.x else 0,
            len(self.y) if self.y else 0,
            len(self.yaw) if self.yaw else 0,
            len(self.v) if self.v else 0,
            len(self.a) if self.a else 0,
        ]
        return min(lengths) if lengths else 0
    
    def get_state_at_index(self, idx: int) -> EgoVehicleState:
        """Get ego vehicle state at specific index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for path of length {len(self)}"
            )
        return EgoVehicleState(
            x=self.x[idx],
            y=self.y[idx],
            yaw=self.yaw[idx],
            v=self.v[idx],
            a=self.a[idx],
            timestamp=self.t[idx]
        )


@dataclass
class ObstacleSet:
    """Set of obstacles in the environment.
    
    Attributes:
        static_obstacles: Static line obstacles [(x_min, x_max, y_min, y_max), ...]
        dynamic_obstacles: Dynamic point obstacles [n_obstacles, 2] at current time
        predicted_trajectories: Future trajectories [n_obstacles, time_horizon, 2]
    """
    static_obstacles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    dynamic_obstacles: Optional[np.ndarray] = None
    predicted_trajectories: Optional[np.ndarray] = None
    
    def get_all_obstacle_points(self) -> np.ndarray:
        """Get all obstacle points (static + dynamic + predicted) as array."""
        points = []
        
        # Add dynamic obstacles
        if self.dynamic_obstacles is not None:
            points.append(self.dynamic_obstacles)
        
        # Add predicted trajectories (flattened)
        if self.predicted_trajectories is not None:
            n_obs, time_horizon, _ = self.predicted_trajectories.shape
            points.append(self.predicted_trajectories.reshape(-1, 2))
        
        if len(points) == 0:
            return np.empty((0, 2))
        
        return np.vstack(points)


@dataclass
class SimulationResult:
    """Results from one simulation step.
    
    Attributes:
        time: Current simulation time [s]
        ego_state: Current ego vehicle state
        ped_state: Current pedestrian state
        predicted_trajectories: Predicted pedestrian trajectories [n_peds, pred_len, 2]
        planned_path: Planned ego vehicle path
        metrics: Dictionary of evaluation metrics
        ego_radius: Collision radius for ego vehicle [m]
        ped_radius: Collision radius for pedestrians [m]
        safety_buffer: Additional safety buffer [m]
    """
    time: float
    ego_state: EgoVehicleState
    ped_state: PedestrianState
    predicted_trajectories: Optional[np.ndarray] = None  # [n_peds, n_steps, 2]
    predicted_distribution: Optional[np.ndarray] = None  # [n_samples, n_peds, n_steps, 2]
    planned_path: Optional[FrenetPath] = None
    metrics: dict = field(default_factory=dict)
    ego_radius: float = 1.0
    ped_radius: float = 0.3
    safety_buffer: float = 0.0
    state: VehicleState = VehicleState.NORMAL
    
    def compute_safety_metrics(self) -> dict:
        """Compute safety-related metrics.
        
        Returns:
            Dictionary containing:
                - min_distance: Minimum distance to any pedestrian [m]
                - collision: Whether collision occurred
                - ttc: Time to collision [s] (if applicable)
        """
        # Compute minimum distance to pedestrians
        ego_pos = np.array([self.ego_state.x, self.ego_state.y])
        distances = np.linalg.norm(self.ped_state.positions - ego_pos, axis=1)
        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
        
        combined_radius = self.ego_radius + self.ped_radius + self.safety_buffer
        collision = min_distance < combined_radius
        
        # Time to collision (simplified, along relative approach)
        ttc = float('inf')
        if len(distances) > 0:
            ego_vx = self.ego_state.v * np.cos(self.ego_state.yaw)
            ego_vy = self.ego_state.v * np.sin(self.ego_state.yaw)
            ego_vel = np.array([ego_vx, ego_vy])
            for pos, vel, dist in zip(self.ped_state.positions, self.ped_state.velocities, distances):
                rel_pos = pos - ego_pos
                rel_vel = vel - ego_vel
                rel_speed_along_line = -np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-8)
                if rel_speed_along_line > 1e-5:
                    time_to_collision = (dist - combined_radius) / rel_speed_along_line
                    if time_to_collision >= 0:
                        ttc = min(ttc, time_to_collision)
        
        return {
            'min_distance': min_distance,
            'collision': collision,
            'ttc': ttc
        }
