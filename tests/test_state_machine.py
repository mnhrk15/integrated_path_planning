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
    c.safety_buffer = 0.5
    return c

def test_initial_state(config):
    sm = FailSafeStateMachine(config)
    assert sm.current_state == VehicleState.NORMAL

def test_transition_normal_to_caution(config):
    sm = FailSafeStateMachine(config)
    
    # Planner fails
    output = sm.update(plan_found=False, safety_metrics={'min_distance': 10.0})
    
    assert sm.current_state == VehicleState.CAUTION
    assert output.state == VehicleState.CAUTION
    assert output.constraint_overrides['max_accel'] > config.ego_max_accel

def test_transition_caution_to_emergency(config):
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION
    
    # Planner fails again
    output = sm.update(plan_found=False, safety_metrics={'min_distance': 10.0})
    
    assert sm.current_state == VehicleState.EMERGENCY
    assert output.state == VehicleState.EMERGENCY
    assert output.target_speed_override == 0.0

def test_recovery_from_caution(config):
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.CAUTION
    
    # Planner succeeds and safe enough
    output = sm.update(plan_found=True, safety_metrics={'min_distance': 2.0}) # > buffer * 2 (1.0)
    
    assert sm.current_state == VehicleState.NORMAL
    assert output.state == VehicleState.NORMAL

def test_emergency_behavior(config):
    sm = FailSafeStateMachine(config)
    sm.current_state = VehicleState.EMERGENCY
    
    # Planner succeeds but too close
    output = sm.update(plan_found=True, safety_metrics={'min_distance': 0.5})
    
    # Needs buffer * 3 (1.5) to recover
    assert sm.current_state == VehicleState.EMERGENCY 
    
    # Planner succeeds and VERY safe
    output = sm.update(plan_found=True, safety_metrics={'min_distance': 5.0})
    
    assert sm.current_state == VehicleState.CAUTION # Recovers to CAUTION first
