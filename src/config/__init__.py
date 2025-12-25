"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SimulationConfig:
    """Configuration for integrated simulation.
    
    Attributes:
        # Time parameters
        dt: Time step [s]
        total_time: Total simulation time [s]
        
        # Observation parameters
        obs_len: Number of observation time steps
        pred_len: Number of prediction time steps
        
        # Ego vehicle parameters
        ego_initial_state: Initial state [x, y, yaw, v, a]
        ego_target_speed: Target speed [m/s]
        ego_max_speed: Maximum speed [m/s]
        ego_max_accel: Maximum acceleration [m/s²]
        ego_max_curvature: Maximum curvature [1/m]
        
        # Reference path
        reference_waypoints_x: X coordinates of waypoints
        reference_waypoints_y: Y coordinates of waypoints
        
        # Planner parameters
        d_road_w: Lateral sampling width [m]
        max_road_width: Maximum road width to check [m]
        
        # Pedestrian parameters
        ped_initial_states: List of [x, y, vx, vy, gx, gy]
        ped_groups: List of pedestrian group indices
        static_obstacles: List of (x_min, x_max, y_min, y_max)
        
        # Model paths
        sgan_model_path: Path to Social-GAN model
        prediction_method: Prediction method ('cv', 'lstm', 'sgan')

        # Device
        device: Computation device ('cpu', 'cuda', 'mps')
        
        # Visualization
        visualization_enabled: Enable visualization
        output_path: Output directory for results
        
        # Map Visualization
        map_config: Dictionary defining road map elements
    """
    # Time parameters
    dt: float = 0.1
    total_time: float = 30.0
    
    # Observation parameters
    obs_len: int = 8
    pred_len: int = 8
    num_samples: int = 1
    
    # Ego vehicle
    ego_initial_state: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 5.0, 0.0])
    ego_target_speed: float = 8.33  # 30 km/h
    ego_max_speed: float = 13.89  # 50 km/h
    ego_max_accel: float = 2.0
    ego_max_curvature: float = 1.0
    ego_radius: float = 1.0
    
    # Obstacle / pedestrian safety parameters
    ped_radius: float = 0.2
    obstacle_radius: float = 0.2

    # Planner cost weights (optional override)
    k_j: float = 1.0
    k_t: float = 1.0
    k_d: float = 1.0
    k_s_dot: float = 1.0
    k_lat: float = 1.0
    k_lon: float = 1.0
    
    # Reference path
    reference_waypoints_x: list = field(default_factory=list)
    reference_waypoints_y: list = field(default_factory=list)
    
    # Planner path generation
    d_road_w: float = 0.5
    max_road_width: float = 7.0
    
    # Planner time horizon parameters
    min_t: float = 4.0  # Minimum prediction time [s]
    max_t: float = 5.0  # Maximum prediction time [s]
    d_t_s: float = 5.0 / 3.6  # Target speed sampling width [m/s]
    n_s_sample: int = 1  # Sampling number of target speed
    
    # State machine parameters
    state_machine_safe_distance_caution: float = 0.5  # Safe distance for CAUTION->NORMAL transition [m]
    state_machine_safe_distance_emergency: float = 1.0  # Safe distance for EMERGENCY->CAUTION transition [m]
    state_machine_caution_accel_multiplier: float = 1.5  # Acceleration multiplier in CAUTION state
    state_machine_caution_curvature_multiplier: float = 1.2  # Curvature multiplier in CAUTION state
    state_machine_caution_speed_multiplier: float = 0.8  # Speed multiplier in CAUTION state
    state_machine_emergency_accel_multiplier: float = 3.0  # Acceleration multiplier in EMERGENCY state
    state_machine_emergency_curvature_multiplier: float = 2.0  # Curvature multiplier in EMERGENCY state
    
    # Pedestrians
    ped_initial_states: list = field(default_factory=list)
    ped_groups: list = field(default_factory=list)
    static_obstacles: list = field(default_factory=list)
    
    # Social Force config
    social_force_config: Optional[str] = None
    social_force_params: Dict[str, Any] = field(default_factory=dict)
    
    # Model
    # Model
    sgan_model_path: Optional[str] = None
    prediction_method: str = 'sgan'
    
    # Device
    device: str = 'cpu'
    
    # Visualization
    visualization_enabled: bool = True
    output_path: str = 'output'
    
    # Map Visualization
    map_config: Dict[str, Any] = field(default_factory=dict)
    
    # Internal: loaded from
    config_path: Optional[str] = None


def load_config(config_path: str) -> SimulationConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = SimulationConfig(**config_dict)
    config.config_path = str(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    return config


def save_config(config: SimulationConfig, config_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        config_path: Path to save YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    

    config_dict = {
        'dt': config.dt,
        'total_time': config.total_time,
        'obs_len': config.obs_len,
        'pred_len': config.pred_len,
        'num_samples': config.num_samples,
        'ego_initial_state': config.ego_initial_state,
        'ego_target_speed': config.ego_target_speed,
        'ego_max_speed': config.ego_max_speed,
        'ego_max_accel': config.ego_max_accel,
        'ego_max_curvature': config.ego_max_curvature,
        'reference_waypoints_x': config.reference_waypoints_x,
        'reference_waypoints_y': config.reference_waypoints_y,
        'd_road_w': config.d_road_w,
        'max_road_width': config.max_road_width,
        'min_t': config.min_t,
        'max_t': config.max_t,
        'd_t_s': config.d_t_s,
        'n_s_sample': config.n_s_sample,
        'state_machine_safe_distance_caution': config.state_machine_safe_distance_caution,
        'state_machine_safe_distance_emergency': config.state_machine_safe_distance_emergency,
        'state_machine_caution_accel_multiplier': config.state_machine_caution_accel_multiplier,
        'state_machine_caution_curvature_multiplier': config.state_machine_caution_curvature_multiplier,
        'state_machine_caution_speed_multiplier': config.state_machine_caution_speed_multiplier,
        'state_machine_emergency_accel_multiplier': config.state_machine_emergency_accel_multiplier,
        'state_machine_emergency_curvature_multiplier': config.state_machine_emergency_curvature_multiplier,
        'ped_initial_states': config.ped_initial_states,
        'ped_groups': config.ped_groups,
        'static_obstacles': config.static_obstacles,
        'social_force_config': config.social_force_config,
        'social_force_params': config.social_force_params,
        'sgan_model_path': config.sgan_model_path,
        'prediction_method': config.prediction_method,
        'device': config.device,
        'visualization_enabled': config.visualization_enabled,
        'output_path': config.output_path,
        'map_config': config.map_config,
    }
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
