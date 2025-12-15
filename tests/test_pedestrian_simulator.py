"""
Tests for PedestrianSimulator with PySocialForce integration
"""
import pytest
import numpy as np
from src.simulation.integrated_simulator import PedestrianSimulator
from src.core.data_structures import SimulationResult, EgoVehicleState, PedestrianState


@pytest.fixture
def simple_pedestrian_states():
    """Create simple pedestrian initial states"""
    return np.array([
        [0.0, 0.0, 0.5, 0.0, 10.0, 0.0],  # [x, y, vx, vy, gx, gy]
        [0.0, 2.0, 0.5, 0.0, 10.0, 2.0],
    ])


def test_pedestrian_simulator_initialization(simple_pedestrian_states):
    """Test PedestrianSimulator can be initialized"""
    simulator = PedestrianSimulator(
        initial_states=simple_pedestrian_states,
        groups=None,
        obstacles=None,
        dt=0.1
    )
    
    assert simulator is not None
    assert simulator.dt == 0.1


def test_pedestrian_simulator_step(simple_pedestrian_states):
    """Test that simulator can step forward"""
    simulator = PedestrianSimulator(
        initial_states=simple_pedestrian_states,
        groups=None,
        obstacles=None,
        dt=0.1
    )
    
    # Initial state
    state0 = simulator.get_state()
    assert len(state0.pedestrians) == 2
    
    # Step forward
    simulator.step()
    
    # New state
    state1 = simulator.get_state()
    assert len(state1.pedestrians) == 2
    assert state1.timestamp > state0.timestamp


def test_pedestrian_simulator_fallback_without_pysocialforce(simple_pedestrian_states):
    """Test that simulator works even without PySocialForce"""
    simulator = PedestrianSimulator(
        initial_states=simple_pedestrian_states,
        groups=None,
        obstacles=None,
        dt=0.1
    )
    
    # Should work regardless of whether PySocialForce is installed
    for _ in range(5):
        simulator.step()
    
    state = simulator.get_state()
    assert len(state.pedestrians) == 2


def test_static_obstacle_collision_detection():
    """Static obstacles should trigger collision when within radius."""
    # Construct a simple SimulationResult with an obstacle near ego
    ego = EgoVehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0, a=0.0)
    ped = PedestrianState(
        positions=np.array([[0.5, 0.0]]),
        velocities=np.array([[0.0, 0.0]]),
        goals=np.array([[0.0, 0.0]]),
        timestamp=0.0
    )
    res = SimulationResult(
        time=0.0,
        ego_state=ego,
        ped_state=ped,
        predicted_trajectories=None,
        planned_path=None,
        ego_radius=0.6,
        ped_radius=0.3,
        safety_buffer=0.2
    )
    metrics = res.compute_safety_metrics()
    assert bool(metrics['collision']) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
