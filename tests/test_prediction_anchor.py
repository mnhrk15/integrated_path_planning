"""Tests for the prediction time-anchor correction (staleness re-anchoring).

The observer samples on a 0.4s grid while predictions are consumed every
sim step (0.1s), so the raw prediction anchor (last observation sample) can
be up to 0.3s stale. The dense output grid must be re-anchored to the
current time: dense index k = current time + (k+1)*sim_dt.

For a constant-velocity pedestrian the re-anchored prediction must match the
true future position exactly, at every observation phase (staleness 0-0.3s).
"""

import numpy as np
import torch

from src.core.data_structures import PedestrianState
from src.pedestrian.observer import PedestrianObserver
from src.prediction.trajectory_predictor import TrajectoryPredictor

SIM_DT = 0.1
SGAN_DT = 0.4
PLAN_HORIZON = 5.0


def make_predictor(num_samples: int = 1) -> TrajectoryPredictor:
    return TrajectoryPredictor(
        model_path=None, pred_len=12, num_samples=num_samples,
        sgan_dt=SGAN_DT, sim_dt=SIM_DT, plan_horizon=PLAN_HORIZON, method="cv",
    )


class TestProcessPredictionAnchor:
    """process_prediction with explicit anchor position and staleness."""

    def _run(self, staleness: float):
        predictor = make_predictor()
        p0 = np.array([2.0, -1.0])      # last observation sample position
        v = np.array([1.2, 0.5])        # true constant velocity
        # Raw SGAN-like prediction: point k is k*sgan_dt after the anchor
        pred_traj = np.stack(
            [p0 + v * (k * SGAN_DT) for k in range(1, 13)], axis=0
        )[:, None, :]                    # (12, 1, 2)
        dense = predictor.process_prediction(
            pred_traj, anchor_pos=p0[None, :], staleness=staleness
        )
        return dense, p0, v

    def test_reanchored_grid_matches_true_future(self):
        """dense[k] must be the position at current time + (k+1)*sim_dt,
        i.e. p0 + v*((k+1)*sim_dt + staleness), for every phase."""
        for j in range(4):
            staleness = j * SIM_DT
            dense, p0, v = self._run(staleness)
            # Within the prediction support (before tail extrapolation)
            support_end = 12 * SGAN_DT - staleness
            for k in range(dense.shape[1]):
                t = (k + 1) * SIM_DT
                if t > support_end:
                    break
                expected = p0 + v * (t + staleness)
                np.testing.assert_allclose(
                    dense[0, k], expected, atol=1e-9,
                    err_msg=f"staleness={staleness}, k={k}"
                )

    def test_no_left_clamp_with_anchor(self):
        """First sim steps interpolate from the anchor, not clamp to the
        first prediction point (the old behaviour put dense[0..2] at the
        +0.4s position when staleness=0)."""
        dense, p0, v = self._run(staleness=0.0)
        np.testing.assert_allclose(dense[0, 0], p0 + v * 0.1, atol=1e-9)
        np.testing.assert_allclose(dense[0, 2], p0 + v * 0.3, atol=1e-9)

    def test_tail_extrapolation_continues_velocity(self):
        """Beyond the (shifted) prediction support the tail extrapolates at
        the clamped tail velocity from the current-time grid."""
        dense, p0, v = self._run(staleness=0.3)
        # support ends at 4.8 - 0.3 = 4.5s; check the last grid point (5.0s)
        k_last = dense.shape[1] - 1
        t_last = (k_last + 1) * SIM_DT
        expected = p0 + v * (t_last + 0.3)
        np.testing.assert_allclose(dense[0, k_last], expected, atol=1e-9)

    def test_zero_staleness_no_anchor_backward_compatible(self):
        """Without anchor/staleness the old grid semantics are preserved."""
        predictor = make_predictor()
        p0 = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        pred_traj = np.stack(
            [p0 + v * (k * SGAN_DT) for k in range(1, 13)], axis=0
        )[:, None, :]
        dense = predictor.process_prediction(pred_traj)
        # Interior grid point on the source grid: t=0.4 -> first pred point
        np.testing.assert_allclose(dense[0, 3], p0 + v * 0.4, atol=1e-9)


class TestPredictCvAnchor:
    def test_cv_origin_shifted_by_staleness(self):
        predictor = make_predictor()
        p_prev = np.array([[0.0, 0.0]])
        p_curr = np.array([[0.48, 0.0]])   # v = 1.2 m/s over one sgan_dt
        obs = torch.from_numpy(
            np.stack([p_prev, p_curr], axis=0)
        ).float()                           # (2, 1, 2)
        for j in range(4):
            staleness = j * SIM_DT
            dense = predictor.predict_cv(obs, staleness=staleness)
            for k in [0, 9, 49]:
                t = (k + 1) * SIM_DT
                expected = p_curr[0] + np.array([1.2, 0.0]) * (t + staleness)
                np.testing.assert_allclose(
                    dense[0, k], expected, atol=1e-9,
                    err_msg=f"staleness={staleness}, k={k}"
                )


class TestEndToEndAnchorAlignment:
    """Observer + predictor driven at sim cadence: for a constant-velocity
    pedestrian the dense prediction must match the true future position at
    EVERY step, regardless of the observation sampling phase. This is the
    regression test for the metrics/planner time-skew (review M-1/M-2)."""

    def test_dense_prediction_matches_truth_at_all_phases(self):
        speed = np.array([1.2, -0.4])
        observer = PedestrianObserver(obs_len=8, dt=SIM_DT, sgan_dt=SGAN_DT)
        predictor = make_predictor()

        def pos(t):
            return np.array([[speed[0] * t, speed[1] * t]])

        # Warm up the observer (32 updates -> 8 samples)
        t = 0.0
        for _ in range(32):
            t = round(t + SIM_DT, 9)
            observer.update(PedestrianState(
                positions=pos(t), velocities=speed[None, :],
                goals=np.array([[100.0, 100.0]]), timestamp=t))

        # Drive 8 more steps covering both sampling phases j=0..3 twice
        for _ in range(8):
            t = round(t + SIM_DT, 9)
            observer.update(PedestrianState(
                positions=pos(t), velocities=speed[None, :],
                goals=np.array([[100.0, 100.0]]), timestamp=t))
            obs_traj, obs_traj_rel, sse = observer.get_observation()
            staleness = t - observer.last_sample_time
            dense = predictor.predict(obs_traj, obs_traj_rel, sse,
                                      staleness=staleness)
            for k in [0, 3, 19, 39]:
                future_t = t + (k + 1) * SIM_DT
                # float32 observation tensors introduce ~1e-6-scale noise;
                # the bug being guarded against is 0.1-0.4 m, so 1e-4 is safe.
                np.testing.assert_allclose(
                    dense[0, k], pos(future_t)[0], atol=1e-4,
                    err_msg=f"t={t:.1f}, staleness={staleness:.1f}, k={k}"
                )
