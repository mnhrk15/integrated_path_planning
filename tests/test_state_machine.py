"""Tests for Fail-Safe State Machine."""

import pytest
from src.core.state_machine import FailSafeStateMachine, VehicleState
from src.config import SimulationConfig

@pytest.fixture
def config():
    c = SimulationConfig()
    c.ego_max_accel = 2.0
    c.ego_max_curvature = 1.0
    c.ego_max_speed = 10.0
    return c

def test_initial_state(config):
    sm = FailSafeStateMachine(config)
    assert sm.current_state == VehicleState.NORMAL

def test_transition_normal_to_caution(config):
    sm = FailSafeStateMachine(config)

    # Planner fails
    output = sm.update(plan_found=False, safety_metrics={'clearance': 8.8})

    assert sm.current_state == VehicleState.CAUTION
    assert output.state == VehicleState.CAUTION
    assert output.constraint_overrides['max_accel'] > config.ego_max_accel

def test_transition_caution_to_emergency(config):
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION

    # Planner fails again
    output = sm.update(plan_found=False, safety_metrics={'clearance': 8.8})

    assert sm.current_state == VehicleState.EMERGENCY
    assert output.state == VehicleState.EMERGENCY
    assert output.target_speed_override == 0.0

def test_recovery_from_caution(config):
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION

    # Planner succeeds and safe enough (clearance above the caution threshold,
    # which is safe_distance_caution 2.0 - combined radius 1.2 = 0.8)
    output = sm.update(plan_found=True, safety_metrics={'clearance': 1.5})

    assert sm.current_state == VehicleState.NORMAL
    assert output.state == VehicleState.NORMAL

def test_emergency_behavior(config):
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.EMERGENCY

    # Planner succeeds but too close (clearance below emergency threshold)
    output = sm.update(plan_found=True, safety_metrics={'clearance': -0.7})

    assert sm.current_state == VehicleState.EMERGENCY

    # Planner succeeds and VERY safe
    output = sm.update(plan_found=True, safety_metrics={'clearance': 3.8})

    assert sm.current_state == VehicleState.CAUTION # Recovers to CAUTION first

def test_recovery_sequence_caution_needs_two_successes(config):
    """After a fresh failure, CAUTION needs two successes to reach NORMAL.

    The first success only clears consecutive_failures; the recovery branch
    requires consecutive_failures == 0 at entry, so NORMAL is reached on the
    second consecutive success (with sufficient clearance).
    """
    sm = FailSafeStateMachine(config)

    sm.update(plan_found=False, safety_metrics={'clearance': 5.0})
    assert sm.current_state == VehicleState.CAUTION

    # First success: failure counter resets, but no recovery yet.
    sm.update(plan_found=True, safety_metrics={'clearance': 5.0})
    assert sm.current_state == VehicleState.CAUTION

    # Second success with ample clearance: recover to NORMAL.
    sm.update(plan_found=True, safety_metrics={'clearance': 5.0})
    assert sm.current_state == VehicleState.NORMAL

def test_recovery_sequence_emergency_to_normal(config):
    """EMERGENCY recovers stepwise: EMERGENCY -> CAUTION -> NORMAL."""
    sm = FailSafeStateMachine(config)

    sm.update(plan_found=False, safety_metrics={'clearance': 5.0})
    sm.update(plan_found=False, safety_metrics={'clearance': 5.0})
    assert sm.current_state == VehicleState.EMERGENCY

    # Success but pedestrian still inside the emergency radius: stay.
    below_emergency = sm.clearance_emergency - 0.1
    sm.update(plan_found=True, safety_metrics={'clearance': below_emergency})
    assert sm.current_state == VehicleState.EMERGENCY

    # Success with clearance above the emergency threshold: drop to CAUTION.
    sm.update(plan_found=True, safety_metrics={'clearance': sm.clearance_emergency + 0.5})
    assert sm.current_state == VehicleState.CAUTION

    # The failure counter from the escalation is still set, so CAUTION needs
    # one success to clear it and a second to recover to NORMAL.
    sm.update(plan_found=True, safety_metrics={'clearance': sm.clearance_caution + 0.5})
    assert sm.current_state == VehicleState.CAUTION
    sm.update(plan_found=True, safety_metrics={'clearance': sm.clearance_caution + 0.5})
    assert sm.current_state == VehicleState.NORMAL

def test_clearance_thresholds_match_legacy_circle_semantics(config):
    # Legacy check compared centre distance against safe_distance; the
    # clearance thresholds must be the same check shifted by the combined
    # collision radius (ego_radius + ped_radius), so single-circle behavior
    # is preserved exactly.
    sm = FailSafeStateMachine(config)
    combined = config.ego_radius + config.ped_radius
    assert sm.clearance_caution == pytest.approx(
        config.state_machine_safe_distance_caution - combined)
    assert sm.clearance_emergency == pytest.approx(
        config.state_machine_safe_distance_emergency - combined)

    # Boundary equivalence: min_distance exactly at the legacy threshold
    # (clearance exactly at the converted threshold) must NOT recover.
    sm.current_state = VehicleState.CAUTION
    legacy_threshold_clearance = config.state_machine_safe_distance_caution - combined
    sm.update(plan_found=True, safety_metrics={'clearance': legacy_threshold_clearance})
    assert sm.current_state == VehicleState.CAUTION

def test_curvature_limit_never_relaxed(config):
    """The curvature limit is kinematic: no state may override it."""
    sm = FailSafeStateMachine(config)

    sm.current_state = VehicleState.CAUTION
    output = sm._get_planner_config()
    assert 'max_curvature' not in output.constraint_overrides

    sm.current_state = VehicleState.EMERGENCY
    output = sm._get_planner_config()
    assert 'max_curvature' not in output.constraint_overrides
    # EMERGENCY relaxes the lateral-acceleration limit towards the friction
    # limit instead (evasive regime).
    assert output.constraint_overrides['max_lat_accel'] == pytest.approx(
        config.ego_max_lat_accel
        * config.state_machine_emergency_lat_accel_multiplier)

def test_preventive_escalation_disabled_by_default(config):
    """trigger 0.0 (default): NORMAL never escalates on proximity alone."""
    sm = FailSafeStateMachine(config)
    sm.update(plan_found=True, safety_metrics={'clearance': 0.01})
    assert sm.current_state == VehicleState.NORMAL

def test_preventive_escalation_and_recovery(config):
    """With a trigger set, NORMAL drops to CAUTION near pedestrians even when
    planning succeeds, and recovers through the ordinary clearance gate
    without oscillating in between."""
    config.state_machine_trigger_clearance_caution = 1.5
    config.state_machine_recover_clearance_caution = 2.0
    config.state_machine_recover_clearance_emergency = 2.0
    sm = FailSafeStateMachine(config)

    # Pedestrian close: preventive CAUTION (not a failure).
    sm.update(plan_found=True, safety_metrics={'clearance': 1.0})
    assert sm.current_state == VehicleState.CAUTION
    assert sm.consecutive_failures == 0

    # In the hysteresis band (trigger 1.5 < clearance < recover 2.0): stay.
    sm.update(plan_found=True, safety_metrics={'clearance': 1.8})
    assert sm.current_state == VehicleState.CAUTION

    # Above the recovery gate: back to NORMAL (counter was never raised).
    sm.update(plan_found=True, safety_metrics={'clearance': 2.5})
    assert sm.current_state == VehicleState.NORMAL

    # And clearly above the trigger it stays NORMAL.
    sm.update(plan_found=True, safety_metrics={'clearance': 2.5})
    assert sm.current_state == VehicleState.NORMAL

def test_speed_dependent_trigger_scales_with_ego_speed(config):
    """RSS-style trigger = clearance_offset + time_headway * v: a fast
    approach escalates at a clearance where a slow one stays NORMAL."""
    config.state_machine_trigger_clearance_caution = 1.0
    config.state_machine_trigger_time_headway = 0.8
    config.state_machine_recover_clearance_caution = 6.0
    config.state_machine_recover_clearance_emergency = 6.0
    sm = FailSafeStateMachine(config)

    # Slow: threshold = 1.0 + 0.8*1.0 = 1.8 < clearance 3.0 -> stay NORMAL.
    sm.update(plan_found=True, safety_metrics={'clearance': 3.0}, ego_speed=1.0)
    assert sm.current_state == VehicleState.NORMAL

    # Fast: threshold = 1.0 + 0.8*6.0 = 5.8 > 3.0 -> preventive CAUTION.
    sm.update(plan_found=True, safety_metrics={'clearance': 3.0}, ego_speed=6.0)
    assert sm.current_state == VehicleState.CAUTION
    assert sm.consecutive_failures == 0

def test_headway_only_trigger_quiet_at_standstill(config):
    """With clearance offset 0 the trigger is purely speed-proportional:
    silent at standstill, active when moving."""
    config.state_machine_trigger_time_headway = 0.8
    sm = FailSafeStateMachine(config)

    sm.update(plan_found=True, safety_metrics={'clearance': 0.5}, ego_speed=0.0)
    assert sm.current_state == VehicleState.NORMAL

    sm.update(plan_found=True, safety_metrics={'clearance': 0.5}, ego_speed=2.0)
    assert sm.current_state == VehicleState.CAUTION

def test_trigger_without_headway_ignores_ego_speed(config):
    """time_headway 0 (legacy): the threshold must not depend on ego_speed."""
    config.state_machine_trigger_clearance_caution = 1.5
    config.state_machine_recover_clearance_caution = 2.0
    config.state_machine_recover_clearance_emergency = 2.0
    sm = FailSafeStateMachine(config)

    sm.update(plan_found=True, safety_metrics={'clearance': 1.6}, ego_speed=10.0)
    assert sm.current_state == VehicleState.NORMAL

    sm.update(plan_found=True, safety_metrics={'clearance': 1.4}, ego_speed=0.0)
    assert sm.current_state == VehicleState.CAUTION

def test_recover_clearance_keys_override_legacy(config):
    """Direct clearance keys replace the centre-distance derivation."""
    config.state_machine_recover_clearance_caution = 1.3
    config.state_machine_recover_clearance_emergency = 1.7
    sm = FailSafeStateMachine(config)
    assert sm.clearance_caution == pytest.approx(1.3)
    assert sm.clearance_emergency == pytest.approx(1.7)

def test_envelope_caps_caution_target_by_clearance(config):
    """Safe-speed envelope: in CAUTION the target speed is capped at
    sqrt(2 * a_env * (clearance - standoff)) — the closer, the slower."""
    config.ego_target_speed = 6.0
    config.state_machine_envelope_decel = 1.2
    config.state_machine_envelope_standoff = 0.5
    # Keep the machine in CAUTION throughout (recovery gates out of reach).
    config.state_machine_recover_clearance_caution = 100.0
    config.state_machine_recover_clearance_emergency = 100.0
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION

    # Clearance 2.0: v_env = sqrt(2*1.2*1.5) = 1.897 < 0.8*6.0 = 4.8.
    sm.update(plan_found=True, safety_metrics={'clearance': 2.0})
    out = sm._get_planner_config()
    assert out.state == VehicleState.CAUTION
    assert out.target_speed_override == pytest.approx((2 * 1.2 * 1.5) ** 0.5)

    # At/below the standoff the target reaches 0 (no restart attempts).
    sm.update(plan_found=True, safety_metrics={'clearance': 0.4})
    out = sm._get_planner_config()
    assert out.target_speed_override == pytest.approx(0.0)

    # With ample clearance the ordinary CAUTION target applies unchanged.
    sm.update(plan_found=True, safety_metrics={'clearance': 50.0})
    out = sm._get_planner_config()
    assert out.target_speed_override == pytest.approx(
        6.0 * config.state_machine_caution_speed_multiplier)

def test_envelope_disabled_by_default(config):
    """envelope_decel 0 (default): legacy fixed CAUTION target even when the
    pedestrian is very close."""
    config.ego_target_speed = 6.0
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION
    sm.update(plan_found=True, safety_metrics={'clearance': 0.1})
    out = sm._get_planner_config()
    assert out.target_speed_override == pytest.approx(
        6.0 * config.state_machine_caution_speed_multiplier)

def test_envelope_handles_missing_clearance(config):
    """No clearance observed yet (inf): the envelope must not constrain."""
    config.ego_target_speed = 6.0
    config.state_machine_envelope_decel = 1.2
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION
    out = sm._get_planner_config()  # before any update()
    assert out.target_speed_override == pytest.approx(
        6.0 * config.state_machine_caution_speed_multiplier)

def test_caution_slows_target_speed(config):
    """CAUTION must lower the planner target speed, not only the speed cap."""
    config.ego_target_speed = 6.0
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION

    output = sm._get_planner_config()
    expected = config.ego_target_speed * config.state_machine_caution_speed_multiplier
    assert output.target_speed_override == pytest.approx(expected)
    # The max-speed cap stays as a second line of defence.
    assert output.constraint_overrides['max_speed'] == pytest.approx(
        config.ego_max_speed * config.state_machine_caution_speed_multiplier)
