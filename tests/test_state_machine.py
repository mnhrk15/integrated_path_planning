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

    # Planner succeeds and safe enough (clearance above caution threshold)
    output = sm.update(plan_found=True, safety_metrics={'clearance': 0.8})

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
