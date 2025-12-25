# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2025-12-25

#### Code Quality Improvements (v3.5)
- **Type Hints Enhancement**: Added comprehensive type hints throughout the codebase
  - `FailSafeStateMachine.__init__` now has proper type hints for `config` parameter
  - `update` method uses `Dict[str, Any]` instead of generic `dict`
  - Used `TYPE_CHECKING` to avoid circular imports while maintaining type safety
  - Improved IDE support and static type checking compatibility
- **Configuration Validation**: Comprehensive validation system for configuration files
  - `ConfigValidationError` custom exception class for clear error reporting
  - `validate_config()` function checks value ranges, consistency, and format
  - Automatic validation on `load_config()` execution
  - Detailed error messages listing all validation failures
  - Validates: time parameters, ego vehicle parameters, planner parameters, state machine parameters, reference paths, pedestrian parameters, static obstacles, model paths, and device settings
- **Code Deduplication**: Refactored safety metrics computation
  - Created `compute_safety_metrics_static()` function for reusable safety metrics calculation
  - Removed duplicate code in `IntegratedSimulator.step()` method
  - `SimulationResult.compute_safety_metrics()` now delegates to the static function
  - Improved maintainability and consistency

### Changed - 2025-12-25
- **State Machine**: Enhanced type hints for better IDE support and type checking
- **Configuration Loading**: Now includes automatic validation with detailed error reporting
- **Safety Metrics**: Centralized computation logic to eliminate code duplication

### Added - 2025-12-16

#### Configuration Improvements (v3.4)
- **Configurable State Machine Parameters**: All hardcoded values in `FailSafeStateMachine` are now configurable via YAML
  - Safe distances for state transitions (`state_machine_safe_distance_caution`, `state_machine_safe_distance_emergency`)
  - Constraint multipliers for CAUTION and EMERGENCY states (acceleration, curvature, speed)
  - Allows fine-tuning of fail-safe behavior per scenario
- **Configurable Planner Time Horizon**: Planner time horizon parameters are now configurable
  - `min_t`, `max_t`: Minimum and maximum prediction time [s]
  - `d_t_s`, `n_s_sample`: Target speed sampling parameters
  - Enables scenario-specific optimization of planning horizon
- Added comprehensive documentation for new configuration parameters in README.md, QUICKSTART.md, and ADDITIONAL_FEATURES.md

### Changed - 2025-12-16
- **State Machine**: Removed hardcoded values (0.5m, 1.0m safe distances, 1.5/1.2/0.8/3.0/2.0 multipliers) and replaced with configurable parameters
- **Frenet Planner**: Replaced module-level constants (`MIN_T`, `MAX_T`, `D_T_S`, `N_S_SAMPLE`) with instance variables that use configuration values
- **Backward Compatibility**: All new parameters have default values matching previous hardcoded values, ensuring existing scenario files work without modification

### Added - 2025-12-15

#### Social-GAN Model Integration
- Added automatic model download script (`scripts/download_sgan_models.py`)
- Added bash alternative for model download (`scripts/download_sgan_models.sh`)
- Integrated official Social-GAN pretrained models
- Added model loading with checkpoint support
- Implemented fallback to constant velocity prediction when model unavailable
- Added support for multiple datasets (ETH, HOTEL, UNIV, ZARA1, ZARA2)
- Added support for both 8-step and 12-step prediction lengths

#### PySocialForce Integration
- Integrated PySocialForce package for realistic pedestrian dynamics
- Implemented full Social Force Model support
- Added pedestrian group behavior simulation
- Added static obstacle handling
- Implemented configuration file support for force parameters
- Added fallback to simple dynamics when PySocialForce unavailable
- Created `PedestrianSimulator` class with full API

#### Animation and Visualization
- Implemented `SimulationAnimator` class using matplotlib.animation
- Added GIF export support (via Pillow)
- Added MP4 export support (via FFmpeg)
- Created 2x2 subplot layout with trajectory, velocity, and distance plots
- Added real-time metrics visualization
- Implemented trajectory trail rendering
- Added predicted trajectory visualization (optional)
- Created `create_simple_animation()` convenience function
- Added `demo_animation.py` example script
- Integrated animation flags into `run_simulation.py`

#### Documentation
- Updated README.md with new features
- Updated QUICKSTART.md with detailed instructions
- Created ADDITIONAL_FEATURES.md comprehensive guide
- Added troubleshooting sections for all new features
- Added usage examples for Python and command-line interfaces

#### Dependencies
- Added `pysocialforce>=1.1.0` (optional)
- Added `pillow>=10.0.0` for GIF generation
- Added `ffmpeg-python>=0.2.0` for MP4 generation
- Added `tqdm>=4.65.0` for progress bars

#### Tests
- Added `test_animator.py` for animation functionality
- Added `test_pedestrian_simulator.py` for pedestrian simulation
- Added `test_trajectory_predictor.py` for prediction module
- Improved test coverage for new features

### Changed
- Updated `TrajectoryPredictor` with proper model loading
- Enhanced `IntegratedSimulator` with animation support
- Improved `PedestrianSimulator` with PySocialForce integration
- Updated command-line arguments in `run_simulation.py`
- Pedestrian observations are downsampled to SGAN's 0.4s cadence; SGAN outputs are resampled to simulation `dt` and extrapolated to the 5s planning horizon.
- Planner collision checks now keep dynamic obstacle time information (no flattening) and evaluate collisions at matching timestamps.

### Fixed
- Improved error handling for missing dependencies
- Added proper warnings when optional packages unavailable
- Fixed device handling for CPU/CUDA/MPS
- Eliminated over-conservative braking caused by treating all future pedestrian positions as immediate obstacles.

## [0.1.0] - Initial Release

### Added
- Basic integrated path planning system
- Frenet coordinate transformation
- CubicSpline path planning
- QuinticPolynomial trajectory generation
- Basic pedestrian simulation
- Simple trajectory prediction
- Static visualization
- Configuration management via YAML
- Example scenarios

[Unreleased]: https://github.com/yourusername/integrated_path_planning/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/integrated_path_planning/releases/tag/v0.1.0
