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


def test_social_force_parameters_are_applied(simple_pedestrian_states):
    """Wrapper parameters should update active PySocialForce and ego-repulsion settings."""
    simulator = PedestrianSimulator(
        initial_states=simple_pedestrian_states,
        groups=None,
        obstacles=None,
        dt=0.1,
        social_force_params={
            "agent_radius": 0.45,
            "social_force.gamma": 0.9,
            "ego_repulsion.sigma": 0.8,
            "ego_repulsion.v0": 4.2,
        },
    )

    active_social_force = next(
        force for force in simulator.sim.forces if type(force).__name__ == "SocialForce"
    )
    assert simulator.sim.peds.agent_radius == pytest.approx(0.45)
    assert simulator.sim.config.sub_config("social_force")("gamma") == pytest.approx(0.9)
    assert active_social_force.config("gamma") == pytest.approx(0.9)
    assert simulator.ego_repulsion_sigma == pytest.approx(0.8)
    assert simulator.ego_repulsion_v0 == pytest.approx(4.2)


def test_ego_repulsive_force_uses_ego_radius():
    """A larger ego radius should increase the explicit repulsive force."""
    states = np.array([[2.0, 0.0, 0.0, 0.0, 10.0, 0.0]])
    ego = EgoVehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0, a=0.0)

    small = PedestrianSimulator(
        initial_states=states,
        dt=0.1,
        ego_radius=0.5,
        social_force_params={
            "agent_radius": 0.2,
            "ego_repulsion.sigma": 0.5,
            "ego_repulsion.v0": 3.0,
        },
    )
    large = PedestrianSimulator(
        initial_states=states,
        dt=0.1,
        ego_radius=1.5,
        social_force_params={
            "agent_radius": 0.2,
            "ego_repulsion.sigma": 0.5,
            "ego_repulsion.v0": 3.0,
        },
    )

    small._overwrite_ego_state(ego)
    large._overwrite_ego_state(ego)
    small_force = small._compute_ego_repulsive_force()
    large_force = large._compute_ego_repulsive_force()

    assert small_force[0, 0] > 0
    assert large_force[0, 0] > small_force[0, 0]
    assert small.sim.peds.size() == len(states)
    assert large.sim.peds.size() == len(states)


def test_static_obstacles_are_converted_to_pysocialforce_segments():
    """Rectangle edges should use PySocialForce's (x1, x2, y1, y2) ordering."""
    simulator = PedestrianSimulator(
        initial_states=np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0]]),
        obstacles=[[-5.0, 5.0, -2.0, -1.0]],
        dt=0.1,
    )

    assert len(simulator.sim.env.obstacles) == 4
    bottom_edge = simulator.sim.env.obstacles[0]
    assert np.allclose(bottom_edge[:, 1], -2.0)
    assert bottom_edge[:, 0].min() == pytest.approx(-5.0)
    assert bottom_edge[:, 0].max() == pytest.approx(5.0)


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
        ped_radius=0.3
    )
    metrics = res.compute_safety_metrics()
    assert bool(metrics['collision']) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_v0_randomization_off_preserves_max_speeds_and_rng(simple_pedestrian_states):
    """Default (off) must neither change desired speeds nor consume the RNG."""
    np.random.seed(42)
    state_before = np.random.get_state()
    np.random.seed(42)
    simulator = PedestrianSimulator(
        initial_states=simple_pedestrian_states.copy(),
        groups=None,
        obstacles=None,
        dt=0.1
    )
    state_after = np.random.get_state()

    expected = 1.3 * np.linalg.norm(simple_pedestrian_states[:, 2:4], axis=1)
    assert np.allclose(simulator.sim.peds.max_speeds, expected)
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])


def test_v0_randomization_is_seeded_and_floored(simple_pedestrian_states):
    """Per-agent noise must be reproducible under the seed and respect the floor."""
    def build(seed, std=0.19, floor=0.3):
        np.random.seed(seed)
        return PedestrianSimulator(
            initial_states=simple_pedestrian_states.copy(),
            groups=None,
            obstacles=None,
            dt=0.1,
            v0_randomization=True,
            v0_std=std,
            v0_min=floor,
        ).sim.peds.max_speeds

    nominal = 1.3 * np.linalg.norm(simple_pedestrian_states[:, 2:4], axis=1)
    a = build(seed=0)
    b = build(seed=0)
    c = build(seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.allclose(a, nominal)

    # A huge negative-noise regime must be floored at v0_min
    floored = build(seed=0, std=100.0, floor=0.3)
    assert np.all(floored >= 0.3)


def test_v0_randomization_survives_simulation_steps(simple_pedestrian_states):
    """pysocialforce recomputes max_speeds per step; the noise must persist."""
    np.random.seed(0)
    simulator = PedestrianSimulator(
        initial_states=simple_pedestrian_states.copy(),
        groups=None,
        obstacles=None,
        dt=0.1,
        v0_randomization=True,
    )
    randomized = simulator.sim.peds.max_speeds.copy()
    simulator.step(n=3)
    assert np.allclose(simulator.sim.peds.max_speeds, randomized)
