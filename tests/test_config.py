
import pytest

from src.config import SimulationConfig, ConfigValidationError, validate_config


def _minimal_config(**overrides) -> SimulationConfig:
    """Smallest configuration that passes validation."""
    base = dict(
        reference_waypoints_x=[0.0, 10.0],
        reference_waypoints_y=[0.0, 0.0],
        prediction_method='cv',
    )
    base.update(overrides)
    return SimulationConfig(**base)


def test_minimal_config_is_valid():
    validate_config(_minimal_config())


def test_collision_margin_inflation_default_is_valid():
    config = _minimal_config()
    assert config.collision_margin_inflation == 1.0
    validate_config(config)


def test_collision_margin_inflation_above_one_is_valid():
    validate_config(_minimal_config(collision_margin_inflation=1.2))


def test_collision_margin_inflation_below_one_is_rejected():
    with pytest.raises(ConfigValidationError, match="collision_margin_inflation"):
        validate_config(_minimal_config(collision_margin_inflation=0.9))
