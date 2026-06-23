"""Determinism / byte-stability guard for closed-loop runs (review R2).

The "byte-stable master_runs/verdicts/folds" reproducibility claim rests on a
fragile RNG-ordering contract: ``set_seed`` (random/numpy/torch) is called once
before constructing+running the simulator, v0 noise is drawn at sim init
(integrated_simulator.py:100), and SGAN draws torch.randn every step
(sgan_vendor/models.py). No test asserts that two same-seed runs are identical,
so a future RNG-reordering refactor would silently break reproducibility with the
suite green. These tests mirror the production contract (set_seed -> build -> run)
and assert bit-identical ego and pedestrian trajectories.
"""
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator
from examples.run_statistical_benchmark import set_seed, resolve_model_path


def _run(method: str, seed: int = 42, total_time: float = 2.0):
    config = load_config("scenarios/scenario_01.yaml")
    config.prediction_method = method
    config.total_time = total_time
    config.visualization_enabled = False
    # Enable v0 randomization so EVEN the CV run draws from the numpy RNG at sim
    # init (integrated_simulator.py:100). Otherwise CV consumes no randomness and
    # the same-seed test would pass even if set_seed were removed -- it must
    # actually exercise the RNG-ordering contract. test_cv_run_differs_across_
    # different_seeds below proves this is non-vacuous.
    config.sfm_v0_randomization = True
    if method != "cv":
        resolve_model_path(config, method)
    set_seed(seed)                      # production contract: seed before build+run
    sim = IntegratedSimulator(config)
    return sim.run()


def _ego_sig(results):
    return np.array([[r.ego_state.x, r.ego_state.y, r.ego_state.yaw,
                      r.ego_state.v, r.ego_state.a] for r in results])


def _ped_sig(results):
    arrs = [np.asarray(r.ped_state.positions).ravel()
            for r in results if r.ped_state is not None]
    return np.concatenate(arrs) if arrs else np.array([])


def _assert_identical(a, b):
    assert len(a) == len(b) and len(a) > 0
    np.testing.assert_array_equal(_ego_sig(a), _ego_sig(b))
    np.testing.assert_array_equal(_ped_sig(a), _ped_sig(b))


def test_cv_run_is_bit_identical_across_same_seed():
    _assert_identical(_run("cv", seed=42), _run("cv", seed=42))


def test_cv_run_differs_across_different_seeds():
    """Proves the same-seed CV test is NOT vacuous: with v0 randomization on, the
    run consumes numpy RNG, so a different seed must change the trajectory. If
    this fails, the determinism guard above is meaningless."""
    a = _run("cv", seed=42)
    b = _run("cv", seed=7)
    assert not np.array_equal(_ped_sig(a), _ped_sig(b))


_SGAN_MODELS_PRESENT = any(Path("models/sgan-p-models").glob("*.pt"))


@pytest.mark.skipif(not _SGAN_MODELS_PRESENT,
                    reason="SGAN (pooling) model weights not downloaded")
def test_sgan_run_is_bit_identical_across_same_seed():
    _assert_identical(_run("sgan"), _run("sgan"))
