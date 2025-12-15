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
        
        # Pedestrian parameters
        ped_initial_states: List of [x, y, vx, vy, gx, gy]
        ped_groups: List of pedestrian group indices
        static_obstacles: List of (x_min, x_max, y_min, y_max)
        
        # Model paths
        sgan_model_path: Path to Social-GAN model
        
        # Device
        device: Computation device ('cpu', 'cuda', 'mps')
        
        # Visualization
        visualization_enabled: Enable visualization
        output_path: Output directory for results
    """
    # Time parameters
    dt: float = 0.1
    total_time: float = 30.0
    
    # Observation parameters
    obs_len: int = 8
    pred_len: int = 12
    
    # Ego vehicle
    ego_initial_state: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 5.0, 0.0])
    ego_target_speed: float = 8.33  # 30 km/h
    ego_max_speed: float = 13.89  # 50 km/h
    ego_max_accel: float = 2.0
    ego_max_curvature: float = 1.0
    ego_radius: float = 1.0
    
    # Obstacle / pedestrian safety parameters
    ped_radius: float = 0.3
    obstacle_radius: float = 0.3
    safety_buffer: float = 0.2

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
    
    # Pedestrians
    ped_initial_states: list = field(default_factory=list)
    ped_groups: list = field(default_factory=list)
    static_obstacles: list = field(default_factory=list)
    
    # Social Force config
    social_force_config: Optional[str] = None
    
    # Model
    sgan_model_path: Optional[str] = None
    
    # Device
    device: str = 'cpu'
    
    # Visualization
    visualization_enabled: bool = True
    output_path: str = 'output'


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
    
    # Convert to dict
    config_dict = {
        'dt': config.dt,
        'total_time': config.total_time,
        'obs_len': config.obs_len,
        'pred_len': config.pred_len,
        'ego_initial_state': config.ego_initial_state,
        'ego_target_speed': config.ego_target_speed,
        'ego_max_speed': config.ego_max_speed,
        'ego_max_accel': config.ego_max_accel,
        'ego_max_curvature': config.ego_max_curvature,
        'reference_waypoints_x': config.reference_waypoints_x,
        'reference_waypoints_y': config.reference_waypoints_y,
        'ped_initial_states': config.ped_initial_states,
        'ped_groups': config.ped_groups,
        'static_obstacles': config.static_obstacles,
        'social_force_config': config.social_force_config,
        'sgan_model_path': config.sgan_model_path,
        'device': config.device,
        'visualization_enabled': config.visualization_enabled,
        'output_path': config.output_path,
    }
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
