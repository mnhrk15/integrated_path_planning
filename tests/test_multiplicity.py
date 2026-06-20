"""Tests for the multiple-comparison ledger (src/core/multiplicity.py).

Covers textbook hand-computed examples, a scipy cross-check, NaN handling (NaNs
are not hypotheses and must not count toward the family size), monotonicity, and
the two-level (within-family + study-wide) ledger assembly.
"""
import numpy as np
import pytest
from scipy.stats import false_discovery_control

from src.core.multiplicity import (adjust, benjamini_hochberg, build_ledger,
                                    holm_bonferroni)


def test_bh_matches_scipy_on_finite_input():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    got = benjamini_hochberg(p)
    expected = false_discovery_control(np.asarray(p), method="bh")
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_holm_textbook_example():
    # Four p-values, m=4. Holm step-down with factors 4,3,2,1 over the sorted
    # order, then running max, then clip to 1.
    #   sorted p     : 0.01, 0.02, 0.03, 0.04
    #   scaled       : 0.04, 0.06, 0.06, 0.04
    #   running max  : 0.04, 0.06, 0.06, 0.06
    p = [0.02, 0.01, 0.04, 0.03]  # deliberately unsorted
    got = holm_bonferroni(p)
    # Map back to input order: 0.02->0.06, 0.01->0.04, 0.04->0.06, 0.03->0.06
    np.testing.assert_allclose(got, [0.06, 0.04, 0.06, 0.06], atol=1e-12)


def test_holm_clips_and_propagates_running_max():
    # m=3, p=[0.3,0.6,0.7]: scaled = 0.9, 1.2, 0.7; running max = 0.9, 1.2, 1.2;
    # clip to 1 -> 0.9, 1.0, 1.0. The largest raw p does NOT "stay itself": the
    # step-down running max carries the earlier 1.2 forward.
    got = holm_bonferroni([0.3, 0.6, 0.7])
    np.testing.assert_allclose(got, [0.9, 1.0, 1.0], atol=1e-12)


def test_bh_is_never_more_conservative_than_holm():
    # A standard property: BH q <= Holm adjusted p for every hypothesis.
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(0, 0.2, size=10))
    bh = benjamini_hochberg(p)
    holm = holm_bonferroni(p)
    assert np.all(bh <= holm + 1e-12)


def test_nan_excluded_from_family_size():
    # Two finite p-values + a NaN: the NaN must not count toward m. With m=2,
    # Holm scales the smaller by 2, the larger by 1.
    got = holm_bonferroni([0.01, np.nan, 0.04])
    assert np.isnan(got[1])
    np.testing.assert_allclose([got[0], got[2]], [0.02, 0.04], atol=1e-12)
    # If the NaN had been treated as a third hypothesis, the smaller would scale
    # by 3 (=0.03); assert it did NOT.
    assert got[0] == pytest.approx(0.02)


def test_all_nan_returns_all_nan():
    assert np.all(np.isnan(benjamini_hochberg([np.nan, np.nan])))
    assert np.all(np.isnan(holm_bonferroni([np.nan, np.nan])))


def test_adjust_reject_flags_and_m():
    out = adjust([0.001, 0.2, np.nan], alpha=0.05)
    assert out["m"] == 2
    assert out["bh_reject"][0] and not out["bh_reject"][1]
    assert not out["bh_reject"][2]  # NaN -> not rejected
    assert out["holm_reject"][0] and not out["holm_reject"][1]


def test_adjust_rejects_on_exact_alpha_boundary():
    # m=1 so the adjusted value equals the raw p; a p exactly on alpha must
    # reject (inclusive <= boundary), not be silently dropped.
    out = adjust([0.05], alpha=0.05)
    assert out["bh_reject"][0]
    assert out["holm_reject"][0]


def test_build_ledger_two_families_and_overall():
    tests = [
        {"test_id": "a", "family": "F1", "p_value": 0.001},
        {"test_id": "b", "family": "F1", "p_value": 0.20},
        {"test_id": "c", "family": "F2", "p_value": 0.04},
    ]
    rows = build_ledger(tests, alpha=0.05)
    by_id = {r["test_id"]: r for r in rows}
    # Within-family sizes: F1 has 2, F2 has 1.
    assert by_id["a"]["family_size"] == 2
    assert by_id["c"]["family_size"] == 1
    # Study-wide family size is 3 for every row.
    assert all(r["overall_size"] == 3 for r in rows)
    # 'a' is strongly significant and should survive both within-family and
    # study-wide BH; 'b' should not.
    assert by_id["a"]["family_bh_reject"] and by_id["a"]["overall_bh_reject"]
    assert not by_id["b"]["family_bh_reject"]
    # Order preserved.
    assert [r["test_id"] for r in rows] == ["a", "b", "c"]
    # Pass-through fields untouched.
    assert by_id["a"]["family"] == "F1"


def test_build_ledger_empty():
    assert build_ledger([]) == []


def test_single_test_unchanged():
    # m=1: BH q == raw p, Holm adjusted == raw p.
    assert benjamini_hochberg([0.03])[0] == pytest.approx(0.03)
    assert holm_bonferroni([0.03])[0] == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# Ledger assembly (examples/make_multiplicity_ledger.py)
# --------------------------------------------------------------------------- #
from examples.make_multiplicity_ledger import (_is_auxiliary,  # noqa: E402
                                               assemble,
                                               rq1b_family_sensitivity)


def _rq1b_test(gt, scenario, p, tier):
    return {"test_id": f"rq1b.rand.fisher.{gt}.{scenario}",
            "family": "rq1b_claim2_fisher", "gt": gt, "scenario": scenario,
            "power_tier": tier, "p_value": p}


def test_is_auxiliary_uses_explicit_field_not_family_suffix():
    # Explicit producer flag / protocol field -- NOT a substring of `family`.
    assert _is_auxiliary({"protocol": "loso"})
    assert _is_auxiliary({"auxiliary": True})
    assert not _is_auxiliary({"protocol": "loco"})
    assert not _is_auxiliary({"family": "rq1b_claim2_fisher"})
    # A family label that merely contains 'loso' must NOT trip the exclusion.
    assert not _is_auxiliary({"family": "rq2_fidelity_ks_loso"})


def test_assemble_excludes_auxiliary_from_overall():
    tests = [
        {"test_id": "rq2.loco", "family": "rq2_fidelity_ks_loco",
         "protocol": "loco", "p_value": 0.007},
        {"test_id": "rq2.loso", "family": "rq2_fidelity_ks_loso",
         "protocol": "loso", "p_value": 0.5},
        {"test_id": "rq1b.a", "family": "rq1b_claim2_fisher", "p_value": 0.02},
    ]
    canonical, auxiliary = assemble(tests, alpha=0.05)
    # The LOSO re-split is split out and NOT counted in the overall family size.
    assert [r["test_id"] for r in auxiliary] == ["rq2.loso"]
    assert all(r["overall_size"] == 2 for r in canonical)
    # Auxiliary tests still get within-family correction (m=1 here).
    assert auxiliary[0]["family_size"] == 1


def test_rq1b_family_sensitivity_three_views():
    # avec: S2 is the strong signal; corners are underpowered noise.
    rows = [
        _rq1b_test("avec", "scenario_01", 0.60, "headline"),
        _rq1b_test("avec", "scenario_02", 0.0078, "headline"),
        _rq1b_test("avec", "scenario_03", 0.2116, "headline"),
        _rq1b_test("calib", "scenario_01", 0.3576, "headline"),
        _rq1b_test("calib", "scenario_02", 1.0, "headline"),
        _rq1b_test("calib", "scenario_03", 0.2199, "headline"),
        _rq1b_test("calib_lo", "scenario_03", 0.0673, "corner"),
        _rq1b_test("calib_hi", "scenario_03", 0.4716, "corner"),
    ]
    sens = rq1b_family_sensitivity(rows, alpha=0.05)
    # avec-only family (m=3): S2 BH q = 0.0078*3/1 = 0.0234 -> survives.
    assert sens["avec_only"]["m"] == 3
    assert sens["avec_only"]["min_bh_q"] == pytest.approx(0.0234, abs=1e-4)
    assert sens["avec_only"]["survives_bh"]
    # headline GTs (avec+calib, m=6): q = 0.0078*6/1 = 0.0468 -> survives.
    assert sens["headline_gts"]["m"] == 6
    assert sens["headline_gts"]["survives_bh"]
    # full scan (m=8 here): q = 0.0078*8/1 = 0.0624 -> does NOT survive.
    assert sens["full_scan"]["m"] == 8
    assert not sens["full_scan"]["survives_bh"]
    # The most-significant test is S2/avec in every view.
    assert all(v["min_test_id"].endswith("avec.scenario_02")
               for v in sens.values())
