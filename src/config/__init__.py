"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
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


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: SimulationConfig) -> None:
    """Validate configuration values for consistency and correctness.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If validation fails
    """
    errors: List[str] = []
    
    # Time parameters
    if config.dt <= 0:
        errors.append(f"dt must be positive, got {config.dt}")
    if config.total_time <= 0:
        errors.append(f"total_time must be positive, got {config.total_time}")
    if config.dt > config.total_time:
        errors.append(f"dt ({config.dt}) must be less than total_time ({config.total_time})")
    
    # Observation parameters
    if config.obs_len <= 0:
        errors.append(f"obs_len must be positive, got {config.obs_len}")
    if config.pred_len <= 0:
        errors.append(f"pred_len must be positive, got {config.pred_len}")
    if config.num_samples <= 0:
        errors.append(f"num_samples must be positive, got {config.num_samples}")
    
    # Ego vehicle parameters
    if len(config.ego_initial_state) != 5:
        errors.append(f"ego_initial_state must have 5 elements [x, y, yaw, v, a], got {len(config.ego_initial_state)}")
    if config.ego_target_speed < 0:
        errors.append(f"ego_target_speed must be non-negative, got {config.ego_target_speed}")
    if config.ego_max_speed < 0:
        errors.append(f"ego_max_speed must be non-negative, got {config.ego_max_speed}")
    if config.ego_max_speed < config.ego_target_speed:
        errors.append(f"ego_max_speed ({config.ego_max_speed}) must be >= ego_target_speed ({config.ego_target_speed})")
    if config.ego_max_accel <= 0:
        errors.append(f"ego_max_accel must be positive, got {config.ego_max_accel}")
    if config.ego_max_curvature <= 0:
        errors.append(f"ego_max_curvature must be positive, got {config.ego_max_curvature}")
    if config.ego_radius <= 0:
        errors.append(f"ego_radius must be positive, got {config.ego_radius}")
    
    # Planner parameters
    if config.d_road_w <= 0:
        errors.append(f"d_road_w must be positive, got {config.d_road_w}")
    if config.max_road_width <= 0:
        errors.append(f"max_road_width must be positive, got {config.max_road_width}")
    if config.max_road_width < config.d_road_w:
        errors.append(f"max_road_width ({config.max_road_width}) must be >= d_road_w ({config.d_road_w})")
    
    # Planner time horizon parameters
    if config.min_t <= 0:
        errors.append(f"min_t must be positive, got {config.min_t}")
    if config.max_t <= 0:
        errors.append(f"max_t must be positive, got {config.max_t}")
    if config.min_t >= config.max_t:
        errors.append(f"min_t ({config.min_t}) must be < max_t ({config.max_t})")
    if config.d_t_s <= 0:
        errors.append(f"d_t_s must be positive, got {config.d_t_s}")
    if config.n_s_sample <= 0:
        errors.append(f"n_s_sample must be positive, got {config.n_s_sample}")
    
    # State machine parameters
    if config.state_machine_safe_distance_caution < 0:
        errors.append(f"state_machine_safe_distance_caution must be non-negative, got {config.state_machine_safe_distance_caution}")
    if config.state_machine_safe_distance_emergency < 0:
        errors.append(f"state_machine_safe_distance_emergency must be non-negative, got {config.state_machine_safe_distance_emergency}")
    if config.state_machine_safe_distance_emergency < config.state_machine_safe_distance_caution:
        errors.append(f"state_machine_safe_distance_emergency ({config.state_machine_safe_distance_emergency}) should be >= state_machine_safe_distance_caution ({config.state_machine_safe_distance_caution})")
    if config.state_machine_caution_accel_multiplier <= 0:
        errors.append(f"state_machine_caution_accel_multiplier must be positive, got {config.state_machine_caution_accel_multiplier}")
    if config.state_machine_caution_curvature_multiplier <= 0:
        errors.append(f"state_machine_caution_curvature_multiplier must be positive, got {config.state_machine_caution_curvature_multiplier}")
    if config.state_machine_caution_speed_multiplier <= 0 or config.state_machine_caution_speed_multiplier > 1.0:
        errors.append(f"state_machine_caution_speed_multiplier must be in (0, 1], got {config.state_machine_caution_speed_multiplier}")
    if config.state_machine_emergency_accel_multiplier <= 0:
        errors.append(f"state_machine_emergency_accel_multiplier must be positive, got {config.state_machine_emergency_accel_multiplier}")
    if config.state_machine_emergency_curvature_multiplier <= 0:
        errors.append(f"state_machine_emergency_curvature_multiplier must be positive, got {config.state_machine_emergency_curvature_multiplier}")
    
    # Safety parameters
    if config.ped_radius <= 0:
        errors.append(f"ped_radius must be positive, got {config.ped_radius}")
    if config.obstacle_radius <= 0:
        errors.append(f"obstacle_radius must be positive, got {config.obstacle_radius}")
    
    # Reference path
    if len(config.reference_waypoints_x) < 2:
        errors.append(f"reference_waypoints_x must have at least 2 points, got {len(config.reference_waypoints_x)}")
    if len(config.reference_waypoints_y) < 2:
        errors.append(f"reference_waypoints_y must have at least 2 points, got {len(config.reference_waypoints_y)}")
    if len(config.reference_waypoints_x) != len(config.reference_waypoints_y):
        errors.append(f"reference_waypoints_x ({len(config.reference_waypoints_x)}) and reference_waypoints_y ({len(config.reference_waypoints_y)}) must have the same length")
    
    # Pedestrian parameters
    n_peds = len(config.ped_initial_states)
    if n_peds > 0:
        for i, ped_state in enumerate(config.ped_initial_states):
            if len(ped_state) != 6:
                errors.append(f"ped_initial_states[{i}] must have 6 elements [x, y, vx, vy, gx, gy], got {len(ped_state)}")
        
        # Validate pedestrian groups
        all_group_indices = set()
        for group in config.ped_groups:
            for idx in group:
                if idx < 0 or idx >= n_peds:
                    errors.append(f"Pedestrian group index {idx} is out of range [0, {n_peds-1}]")
                all_group_indices.add(idx)
        
        # Check if all pedestrians are in at least one group (optional warning, not error)
        if len(all_group_indices) < n_peds:
            logger.warning(f"Some pedestrians are not in any group: {set(range(n_peds)) - all_group_indices}")
    
    # Static obstacles
    for i, obs in enumerate(config.static_obstacles):
        if len(obs) != 4:
            errors.append(f"static_obstacles[{i}] must have 4 elements [x_min, x_max, y_min, y_max], got {len(obs)}")
        elif len(obs) == 4:
            x_min, x_max, y_min, y_max = obs
            if x_min >= x_max:
                errors.append(f"static_obstacles[{i}]: x_min ({x_min}) must be < x_max ({x_max})")
            if y_min >= y_max:
                errors.append(f"static_obstacles[{i}]: y_min ({y_min}) must be < y_max ({y_max})")
    
    # Model parameters
    if config.prediction_method not in ['cv', 'lstm', 'sgan']:
        errors.append(f"prediction_method must be one of ['cv', 'lstm', 'sgan'], got '{config.prediction_method}'")
    if config.prediction_method in ['sgan', 'lstm'] and not config.sgan_model_path:
        errors.append(f"sgan_model_path is required when prediction_method is '{config.prediction_method}'")
    if config.sgan_model_path:
        model_path = Path(config.sgan_model_path)
        if not model_path.exists():
            errors.append(f"sgan_model_path does not exist: {config.sgan_model_path}")
    
    # Device
    if config.device not in ['cpu', 'cuda', 'mps']:
        errors.append(f"device must be one of ['cpu', 'cuda', 'mps'], got '{config.device}'")
    
    # Cost weights (should be non-negative, but allow 0 for disabling)
    cost_weights = {
        'k_j': config.k_j,
        'k_t': config.k_t,
        'k_d': config.k_d,
        'k_s_dot': config.k_s_dot,
        'k_lat': config.k_lat,
        'k_lon': config.k_lon,
    }
    for name, value in cost_weights.items():
        if value < 0:
            errors.append(f"{name} must be non-negative, got {value}")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)


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
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {config_path}: {e}") from e
    
    if config_dict is None:
        raise ValueError(f"YAML file {config_path} is empty or contains no valid content")
    
    try:
        config = SimulationConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration structure in {config_path}: {e}") from e
    
    config.config_path = str(config_path)
    
    # Validate configuration
    try:
        validate_config(config)
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed for {config_path}")
        raise
    
    logger.info(f"Configuration loaded and validated from {config_path}")
    
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
