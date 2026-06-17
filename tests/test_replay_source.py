import numpy as np
import pytest

from src.core.data_structures import EgoVehicleState, PedestrianState
from src.simulation.replay_source import ReplayPedestrianSource


def _traj():
    # [T=3, N=2, 2]: ped0 moves +1.0 m/step in x, ped1 +1.0 m/step in y.
    return np.array(
        [
            [[0.0, 0.0], [5.0, 5.0]],
            [[1.0, 0.0], [5.0, 6.0]],
            [[2.0, 0.0], [5.0, 7.0]],
        ]
    )


def test_get_state_returns_current_frame():
    src = ReplayPedestrianSource(_traj(), dt=0.4)
    s0 = src.get_state()
    assert isinstance(s0, PedestrianState)
    assert s0.n_peds == 2
    np.testing.assert_allclose(s0.positions, [[0, 0], [5, 5]])
    assert s0.timestamp == pytest.approx(0.0)


def test_step_advances_frame_and_time():
    src = ReplayPedestrianSource(_traj(), dt=0.4)
    src.step()
    s1 = src.get_state()
    np.testing.assert_allclose(s1.positions, [[1, 0], [5, 6]])
    assert s1.timestamp == pytest.approx(0.4)


def test_step_clamps_position_but_time_advances():
    src = ReplayPedestrianSource(_traj(), dt=0.4)
    src.step(n=10)
    s = src.get_state()
    np.testing.assert_allclose(s.positions, [[2, 0], [5, 7]])  # held at last frame
    assert s.timestamp == pytest.approx(10 * 0.4)


def test_velocities_finite_difference():
    src = ReplayPedestrianSource(_traj(), dt=0.4)
    s0 = src.get_state()
    # +1.0 m over 0.4 s = 2.5 m/s along the moving axis.
    np.testing.assert_allclose(s0.velocities[0], [2.5, 0.0])
    np.testing.assert_allclose(s0.velocities[1], [0.0, 2.5])


def test_goals_default_to_final_position():
    src = ReplayPedestrianSource(_traj(), dt=0.4)
    np.testing.assert_allclose(src.get_state().goals, [[2, 0], [5, 7]])


def test_ego_state_is_ignored():
    src = ReplayPedestrianSource(_traj(), dt=0.4)
    ego = EgoVehicleState(x=99.0, y=99.0, yaw=0.0, v=1.0, a=0.0)
    src.step(ego_state=ego)
    np.testing.assert_allclose(src.get_state().positions, [[1, 0], [5, 6]])


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        ReplayPedestrianSource(np.zeros((3, 2)), dt=0.4)  # not [T, N, 2]


def test_supplied_velocities_and_reset():
    traj = _traj()
    vel = np.ones_like(traj)
    src = ReplayPedestrianSource(traj, dt=0.4, velocities=vel)
    np.testing.assert_allclose(src.get_state().velocities, np.ones((2, 2)))
    src.step(n=2)
    src.reset()
    assert src.get_state().timestamp == pytest.approx(0.0)
    np.testing.assert_allclose(src.get_state().positions, [[0, 0], [5, 5]])
