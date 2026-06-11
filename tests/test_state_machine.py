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
