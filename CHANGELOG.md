# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed
- Improved error handling for missing dependencies
- Added proper warnings when optional packages unavailable
- Fixed device handling for CPU/CUDA/MPS

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
