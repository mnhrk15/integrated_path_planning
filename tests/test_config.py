
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


def test_chance_epsilon_range_is_enforced():
    validate_config(_minimal_config(chance_epsilon=0.0))
    validate_config(_minimal_config(chance_epsilon=0.25))
    # eps >= 1.0 would accept paths colliding under every sample
    with pytest.raises(ConfigValidationError, match="chance_epsilon"):
        validate_config(_minimal_config(chance_epsilon=1.0))
    # percent-style typo
    with pytest.raises(ConfigValidationError, match="chance_epsilon"):
        validate_config(_minimal_config(chance_epsilon=5.0))
    with pytest.raises(ConfigValidationError, match="chance_epsilon"):
        validate_config(_minimal_config(chance_epsilon=-0.1))


def test_distribution_aware_requires_multiple_samples():
    with pytest.raises(ConfigValidationError, match="num_samples"):
        validate_config(_minimal_config(distribution_aware_planning=True, num_samples=1))
    validate_config(_minimal_config(distribution_aware_planning=True, num_samples=20))


def test_clearance_keys_validation():
    # Direct clearance keys: positive values pass, ordering enforced.
    validate_config(_minimal_config(
        state_machine_recover_clearance_caution=1.3,
        state_machine_recover_clearance_emergency=1.7))
    with pytest.raises(ConfigValidationError, match="recover_clearance_caution"):
        validate_config(_minimal_config(state_machine_recover_clearance_caution=0.0))
    with pytest.raises(ConfigValidationError, match="recover_clearance_emergency"):
        validate_config(_minimal_config(
            state_machine_recover_clearance_caution=1.7,
            state_machine_recover_clearance_emergency=1.3))
    # Legacy keys are not validated when the clearance keys take over
    # (safe_distance below the combined radius is fine then).
    validate_config(_minimal_config(
        state_machine_safe_distance_caution=0.5,
        state_machine_safe_distance_emergency=1.0,
        state_machine_recover_clearance_caution=1.3,
        state_machine_recover_clearance_emergency=1.7))


def test_trigger_clearance_hysteresis_enforced():
    # Trigger below the recovery gate: OK.
    validate_config(_minimal_config(
        state_machine_trigger_clearance_caution=1.5,
        state_machine_recover_clearance_caution=2.0,
        state_machine_recover_clearance_emergency=2.0))
    # Trigger at/above the recovery gate: the gate would have no effect at
    # the commanded speed (the speed-aware runtime gate then dominates), so
    # validate rejects the configuration as self-inconsistent.
    with pytest.raises(ConfigValidationError, match="hysteresis"):
        validate_config(_minimal_config(
            state_machine_trigger_clearance_caution=2.0,
            state_machine_recover_clearance_caution=2.0,
            state_machine_recover_clearance_emergency=2.0))
    # Also checked against the legacy-derived gate (2.0 - 1.2 = 0.8).
    with pytest.raises(ConfigValidationError, match="hysteresis"):
        validate_config(_minimal_config(state_machine_trigger_clearance_caution=0.9))
    validate_config(_minimal_config(state_machine_trigger_clearance_caution=0.5))


def test_ego_emergency_decel_validation():
    # None (legacy ego_max_accel * 2.0 fallback) and positive values pass.
    validate_config(_minimal_config())
    validate_config(_minimal_config(ego_emergency_decel=4.0))
    with pytest.raises(ConfigValidationError, match="ego_emergency_decel"):
        validate_config(_minimal_config(ego_emergency_decel=0.0))
    with pytest.raises(ConfigValidationError, match="ego_emergency_decel"):
        validate_config(_minimal_config(ego_emergency_decel=-2.0))


def test_ego_emergency_decel_must_cover_max_accel():
    # The adaptive stop clips to [ego_max_accel, ego_emergency_decel]; an
    # inverted range (np.clip lo > hi always returns hi) must be rejected.
    cfg = _minimal_config(ego_emergency_decel=1.0)
    assert cfg.ego_emergency_decel < cfg.ego_max_accel
    with pytest.raises(ConfigValidationError, match="must be >= "):
        validate_config(cfg)
    # Equality is the degenerate-but-consistent boundary: allowed.
    validate_config(_minimal_config(ego_emergency_decel=2.0, ego_max_accel=2.0))


def test_trigger_time_headway_validation():
    with pytest.raises(ConfigValidationError, match="time_headway"):
        validate_config(_minimal_config(state_machine_trigger_time_headway=-0.1))
    # Hysteresis is checked at the CAUTION recovery speed
    # (caution_speed_multiplier 0.8 * ego_target_speed 8.33 = 6.66 m/s):
    # 0.5 + 0.2 * 6.66 = 1.83 < gate 2.0 -> OK.
    validate_config(_minimal_config(
        state_machine_trigger_clearance_caution=0.5,
        state_machine_trigger_time_headway=0.2,
        state_machine_recover_clearance_caution=2.0,
        state_machine_recover_clearance_emergency=2.0))
    # 0.5 + 0.3 * 6.66 = 2.5 >= gate 2.0 -> the gate is ineffective at the
    # commanded speed; rejected as self-inconsistent.
    with pytest.raises(ConfigValidationError, match="hysteresis"):
        validate_config(_minimal_config(
            state_machine_trigger_clearance_caution=0.5,
            state_machine_trigger_time_headway=0.3,
            state_machine_recover_clearance_caution=2.0,
            state_machine_recover_clearance_emergency=2.0))
    # A headway alone (offset 0) activates the trigger and the check.
    with pytest.raises(ConfigValidationError, match="hysteresis"):
        validate_config(_minimal_config(
            state_machine_trigger_time_headway=0.5,
            state_machine_recover_clearance_caution=2.0,
            state_machine_recover_clearance_emergency=2.0))


def test_envelope_keys_validation():
    validate_config(_minimal_config(state_machine_envelope_decel=1.2,
                                    state_machine_envelope_standoff=0.5))
    with pytest.raises(ConfigValidationError, match="envelope_decel"):
        validate_config(_minimal_config(state_machine_envelope_decel=-0.1))
    with pytest.raises(ConfigValidationError, match="envelope_standoff"):
        validate_config(_minimal_config(state_machine_envelope_standoff=-0.1))


def test_state_machine_safe_distances_must_exceed_combined_radius():
    # Defaults (caution 2.0 / emergency 3.0) exceed ego 1.0 + ped 0.2.
    validate_config(_minimal_config())
    # At or below the combined radius the clearance gate is vacuous.
    with pytest.raises(ConfigValidationError, match="safe_distance_caution"):
        validate_config(_minimal_config(state_machine_safe_distance_caution=1.2,
                                        state_machine_safe_distance_emergency=3.0))
    with pytest.raises(ConfigValidationError, match="safe_distance_emergency"):
        validate_config(_minimal_config(state_machine_safe_distance_emergency=1.0))


def test_state_machine_safe_distances_use_footprint_radius():
    # multi_circle (default 4.5 x 2.0, 3 circles) has an effective radius of
    # 1.25 m; 1.4 m clears the circle combined radius (1.2) but not the
    # footprint one (1.45), so it must be rejected in multi_circle mode only.
    validate_config(_minimal_config(state_machine_safe_distance_caution=1.4,
                                    state_machine_safe_distance_emergency=1.5))
    with pytest.raises(ConfigValidationError, match="safe_distance_caution"):
        validate_config(_minimal_config(ego_footprint="multi_circle",
                                        state_machine_safe_distance_caution=1.4,
                                        state_machine_safe_distance_emergency=1.5))
