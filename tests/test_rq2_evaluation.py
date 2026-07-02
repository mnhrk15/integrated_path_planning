"""Tests for the RQ2 cross-validated evaluation harness (synthetic clips).

Exercises the fold construction (no train/test leakage, correct fold counts) and
the per-fold evaluation contract (full CSV schema, finite metrics, graceful NaN
on an empty test split). Real VCI files are never touched; clips are built with
the same synthetic ``make_clip`` builder the calibration-harness suite uses.
"""
import argparse

import numpy as np

from src.datasets.vci_loader import ClipTracks
from src.datasets.vci_encounter import encounters_from_clips
from tests.test_calibration_harness import make_clip

from examples.run_rq2_evaluation import (
    COLUMNS,
    RAW_KEYS,
    evaluate_fold,
    make_folds,
)


def _stub_clip(stem: str, scenario: str) -> ClipTracks:
    """Minimal ClipTracks: make_folds only reads .clip / .scenario."""
    return ClipTracks(clip=stem, dataset="citr", scenario=scenario,
                      ped=None, veh=None, ped_path=None, veh_path=None, fps=10.0)


def _real_clip(stem: str, scenario: str = "vci_front") -> ClipTracks:
    """A synthetic clip that yields a genuine crossing encounter."""
    clip = make_clip(n_veh=1)
    clip.clip = stem
    clip.scenario = scenario
    return clip


def _args(**kw):
    base = dict(sigma_grid=[0.5, 1.0], v0_grid=[0.0, 2.0],
                no_refine=True, interaction_distance=None)
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# make_folds
# --------------------------------------------------------------------------- #
def test_make_folds_loco_partitions_clips():
    clips = [_stub_clip(f"c{i}", "vci_front") for i in range(4)]
    folds = make_folds(clips, "loco")
    assert len(folds) == 4
    for _name, train, test in folds:
        assert len(test) == 1
        assert len(train) == 3
    # each clip is the held-out test exactly once
    held = sorted(test[0].clip for _n, _tr, test in folds)
    assert held == ["c0", "c1", "c2", "c3"]


def test_make_folds_loso_partitions_scenarios():
    clips = [_stub_clip(f"c{i}", s)
             for s in ("vci_front", "vci_back") for i in range(3)]
    folds = make_folds(clips, "loso")
    assert len(folds) == 2  # one fold per scenario
    for _name, train, test in folds:
        assert len({c.scenario for c in test}) == 1
        assert len(test) == 3
        assert len(train) == 3


def test_make_folds_no_leakage():
    clips = [_stub_clip(f"c{i}", "vci_front") for i in range(5)]
    for _name, train, test in make_folds(clips, "loco"):
        train_ids = {id(c) for c in train}
        test_ids = {id(c) for c in test}
        assert train_ids.isdisjoint(test_ids)


def test_make_folds_loso_needs_two_scenarios():
    import pytest
    clips = [_stub_clip(f"c{i}", "vci_front") for i in range(3)]
    with pytest.raises(SystemExit):
        make_folds(clips, "loso")


# --------------------------------------------------------------------------- #
# evaluate_fold
# --------------------------------------------------------------------------- #
def test_evaluate_fold_returns_documented_keys():
    train = [_real_clip("train0")]
    test = [_real_clip("test0")]
    train_encs = encounters_from_clips(train, 8.0, 5)
    test_encs = encounters_from_clips(test, 8.0, 5)
    assert train_encs and test_encs  # the synthetic geometry must interact
    row, raw = evaluate_fold("test0", "loco", train, test, train_encs, test_encs, _args())
    assert set(row.keys()) == set(COLUMNS)
    assert np.isfinite(row["sigma"]) and np.isfinite(row["v0"])
    assert np.isfinite(row["test_ade"])
    assert np.isfinite(row["train_ade"])
    # The raw-scalar pool (review C1) carries the held-out closest-approach
    # values so main() can pool them across folds into ONE KS.
    assert set(raw.keys()) == set(RAW_KEYS)
    assert len(raw["calibrated_closest"]) == len(raw["real_closest"]) == len(test_encs)


def test_evaluate_fold_empty_test_is_nan():
    train = [_real_clip("train0")]
    train_encs = encounters_from_clips(train, 8.0, 5)
    # calibration still runs (train non-empty), but every test metric is NaN.
    row, raw = evaluate_fold("empty", "loco", train, [], train_encs, [], _args())
    assert np.isfinite(row["sigma"])  # fit succeeded
    assert row["n_test_encs"] == 0
    assert np.isnan(row["test_ade"])
    assert np.isnan(row["test_ks_closest"])
    assert all(raw[k] == [] for k in RAW_KEYS)  # empty test -> empty pools


def test_evaluate_fold_empty_train_skips_calibration():
    test = [_real_clip("test0")]
    test_encs = encounters_from_clips(test, 8.0, 5)
    row, raw = evaluate_fold("notrain", "loco", [], test, [], test_encs, _args())
    assert np.isnan(row["sigma"])  # no calibration possible
    assert row["n_test_encs"] == len(test_encs)
    assert all(raw[k] == [] for k in RAW_KEYS)  # no fit -> no pooled scalars


def _rows_equal(a: dict, b: dict) -> bool:
    """Bit-level row equality with NaN==NaN (rows carry NaN for empty metrics)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if not (va == vb or (np.isnan(va) and np.isnan(vb))):
                return False
        elif va != vb:
            return False
    return True


def _raws_equal(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(np.array_equal(np.asarray(a[k], dtype=float),
                              np.asarray(b[k], dtype=float), equal_nan=True)
               for k in a)


def test_evaluate_fold_custom_objective_and_fidelity_kwargs():
    """The objective_fn / fidelity_kwargs hooks (cap-policy audit) must leave the
    default path bit-identical when omitted or passed explicitly as None, and
    must actually reach the fitter / fidelity calls when provided."""
    from src.simulation.calibration_harness import objective_rollout_ade

    train = [_real_clip("train0")]
    test = [_real_clip("test0")]
    train_encs = encounters_from_clips(train, 8.0, 5)
    test_encs = encounters_from_clips(test, 8.0, 5)

    base_row, base_raw = evaluate_fold(
        "f", "loco", train, test, train_encs, test_encs, _args())
    none_row, none_raw = evaluate_fold(
        "f", "loco", train, test, train_encs, test_encs, _args(),
        objective_fn=None, fidelity_kwargs=None)
    assert _rows_equal(base_row, none_row)
    assert _raws_equal(base_raw, none_raw)

    # A pass-through objective_fn equal to the default must also be identical.
    def default_obj(encs, s, v):
        return objective_rollout_ade(encs, s, v, interaction_distance=None)

    same_row, same_raw = evaluate_fold(
        "f", "loco", train, test, train_encs, test_encs, _args(),
        objective_fn=default_obj)
    assert _rows_equal(base_row, same_row)
    assert _raws_equal(base_raw, same_raw)

    # A cap-policy override must reach BOTH the fitter and every fidelity arm
    # (within-regime controls) -- the uncapped regime changes the held-out
    # closest-approach scalars of the strongly-repelled AVEC-default arm.
    def uncapped_obj(encs, s, v):
        return objective_rollout_ade(encs, s, v, cap_policy="uncapped")

    cap_row, cap_raw = evaluate_fold(
        "f", "loco", train, test, train_encs, test_encs, _args(),
        objective_fn=uncapped_obj, fidelity_kwargs={"cap_policy": "uncapped"})
    assert set(cap_row.keys()) == set(COLUMNS)
    assert np.isfinite(cap_row["test_ade"])
    assert cap_raw["default_closest"] != base_raw["default_closest"]
