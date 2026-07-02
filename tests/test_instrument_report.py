"""Tests for the instrument-audit verdict functions (synthetic frames).

The verdicts are research claims rendered into REPORT.md, so both branches of
every judgement are pinned here (artifact vs structural, improved vs not,
inside vs outside the RQ1b domain), plus the ledger-safety regression: the
audit sidecars must never collide with the canonical rq2.* records and must be
auxiliary-only.
"""
import numpy as np
import pandas as pd

from examples.make_rq2_instrument_report import (
    cap_verdict,
    distmatch_verdict,
    f1_verdict,
    identifiability_summary,
    rq1b_domain_check,
)
from examples.run_rq2_cap_sensitivity import (
    aux_paired_sidecar_tests,
    cap_sidecar_tests,
)


def _cap_df(rows):
    return pd.DataFrame(rows, columns=["policy", "mean_gap", "sign_p",
                                       "n_real_gt_sim", "n_pairs"])


# --------------------------------------------------------------------------- #
# cap_verdict
# --------------------------------------------------------------------------- #
def test_cap_verdict_structural_when_gap_persists():
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("uncapped", 0.76, 8e-7, 25, 26)])
    v = cap_verdict(df)
    assert v["verdict"] == "structural_limit"
    assert v["per_policy"]["uncapped"]["gap_shrunk"] is False
    assert v["per_policy"]["uncapped"]["dominance_broken"] is False


def test_cap_verdict_artifact_when_decoupled_gap_collapses():
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("uncapped", 0.20, 0.4, 15, 26)])  # shrunk 71% + p>0.05
    v = cap_verdict(df)
    assert v["verdict"] == "harness_artifact"
    assert v["artifact_hits"] == ["uncapped"]


def test_cap_verdict_requires_both_criteria():
    # gap shrunk but dominance intact -> NOT an artifact call
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("uncapped", 0.30, 1e-4, 23, 26)])
    assert cap_verdict(df)["verdict"] == "structural_limit"
    # dominance broken but gap not shrunk -> NOT an artifact call
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("uncapped", 0.60, 0.3, 16, 26)])
    assert cap_verdict(df)["verdict"] == "structural_limit"


def test_cap_verdict_closedloop_confound_never_drives_verdict():
    """closedloop meeting the shrink criterion must be flagged, not counted:
    its walking speed is ~1.3x the recorded cruise (documented confound)."""
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("closedloop", 0.20, 0.5, 14, 26),   # would qualify...
                  ("uncapped", 0.76, 8e-7, 25, 26)])   # ...but decoupled says no
    v = cap_verdict(df)
    assert v["verdict"] == "structural_limit"
    assert v["confounded_hits"] == ["closedloop"]
    assert not v["per_policy"]["closedloop"]["verdict_eligible"]


def test_cap_verdict_undetermined_without_median_or_decoupled():
    assert cap_verdict(_cap_df([("uncapped", 0.5, 0.1, 20, 26)]))["verdict"] \
        == "undetermined"
    assert cap_verdict(_cap_df([("median", 0.68, 1e-5, 24, 26),
                                ("closedloop", 0.2, 0.5, 14, 26)]))["verdict"] \
        == "undetermined"


def test_cap_verdict_sign_flip_is_not_an_artifact_claim():
    """A gap that flips side (under- -> over-reproduction) was REPLACED by an
    opposite systematic error, not explained; it must be flagged, never counted
    as shrunk/artifact (review finding: |gap| alone would call -0.40 'shrunk')."""
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("uncapped", -0.40, 0.17, 9, 26)])
    v = cap_verdict(df)
    assert v["verdict"] == "structural_limit"
    d = v["per_policy"]["uncapped"]
    assert d["sign_flipped"] is True
    assert d["gap_shrunk"] is False


def test_cap_verdict_wilcoxon_vetoes_sign_only_dominance_break():
    """sign p > alpha alone (n=26, low power) must not declare 'dominance
    broke' while the Wilcoxon on the same differences is still significant."""
    df = pd.DataFrame(
        [("median", 0.68, 1e-5, 24, 26, 1e-7),
         ("uncapped", 0.40, 0.076, 18, 26, 0.002)],  # closedloop-AVEC profile
        columns=["policy", "mean_gap", "sign_p", "n_real_gt_sim", "n_pairs",
                 "wilcoxon_p"])
    v = cap_verdict(df)
    assert v["per_policy"]["uncapped"]["dominance_broken"] is False
    assert v["verdict"] == "structural_limit"
    # ... and with BOTH tests null, the same numbers do break dominance
    df.loc[df["policy"] == "uncapped", "wilcoxon_p"] = 0.3
    assert cap_verdict(df)["per_policy"]["uncapped"]["dominance_broken"] is True


def test_cap_verdict_partial_explanation_is_surfaced():
    """Gap shrunk but dominance intact: structural verdict, but the middle
    ground must be exported, not hidden by the binary (review finding)."""
    df = _cap_df([("median", 0.68, 1e-5, 24, 26),
                  ("uncapped", 0.30, 5e-4, 22, 26)])
    v = cap_verdict(df)
    assert v["verdict"] == "structural_limit"
    assert v["partially_explained"] == ["uncapped"]


def test_cap_verdict_capfit_nan_multiplier_fails_closed():
    """A capfit row with unknown provenance (no cap_multiplier) must not be
    counted as decoupled evidence (review finding: NaN != 1.0 passed open)."""
    df = pd.DataFrame(
        [("median", 0.68, 1e-5, 24, 26, np.nan),
         ("capfit", 0.20, 0.4, 15, 26, np.nan)],
        columns=["policy", "mean_gap", "sign_p", "n_real_gt_sim", "n_pairs",
                 "cap_multiplier"])
    v = cap_verdict(df)
    assert not v["per_policy"]["capfit"]["verdict_eligible"]
    assert v["verdict"] == "undetermined"


# --------------------------------------------------------------------------- #
# distmatch_verdict / identifiability / f1
# --------------------------------------------------------------------------- #
def _dm_df(rows):
    return pd.DataFrame(rows, columns=["config", "mean_gap", "sign_p",
                                       "n_real_gt_sim", "n_pairs", "test_ade"])


def test_distmatch_verdict_reports_tradeoff_both_ways():
    improved = _dm_df([("w0", 0.68, 1e-5, 24, 26, 0.640),
                       ("w1", 0.30, 0.2, 17, 26, 0.700)])
    v = distmatch_verdict(improved)
    assert v["verdict"] == "standoff_improved"
    assert v["per_config"]["w1"]["gap_shrunk"]
    assert v["per_config"]["w1"]["ade_sacrifice_pct"] > 0

    flat = _dm_df([("w0", 0.68, 1e-5, 24, 26, 0.640),
                   ("w1", 0.65, 1e-5, 24, 26, 0.660)])
    assert distmatch_verdict(flat)["verdict"] == "no_standoff_improvement"
    assert distmatch_verdict(_dm_df([("w1", 0.3, 0.2, 17, 26, 0.7)]))["verdict"] \
        == "undetermined"


def test_distmatch_verdict_w0_alone_is_undetermined_not_negative():
    """Only the (fallback-synthesised) w0 reference present = nothing audited
    yet; 'no_standoff_improvement' would misread a partial run as a completed
    negative result (review finding, reproduced on a cap-only directory)."""
    v = distmatch_verdict(_dm_df([("w0", 0.68, 1e-5, 24, 26, 0.640)]))
    assert v["verdict"] == "undetermined"
    assert "no non-w0" in v["reason"]


def test_distmatch_verdict_sign_flip_is_not_improvement():
    df = _dm_df([("w0", 0.68, 1e-5, 24, 26, 0.640),
                 ("w1", -0.30, 0.2, 9, 26, 0.700)])
    v = distmatch_verdict(df)
    assert v["verdict"] == "no_standoff_improvement"
    assert v["per_config"]["w1"]["sign_flipped"] is True


def test_distmatch_verdict_survives_missing_test_ade_column():
    """A folds CSV lost next to a surviving sidecar must degrade to NaN ADE
    fields, not crash with KeyError (review finding, reproduced)."""
    df = pd.DataFrame([("w0", 0.68, 1e-5, 24, 26),
                       ("w1", 0.30, 0.2, 17, 26)],
                      columns=["config", "mean_gap", "sign_p",
                               "n_real_gt_sim", "n_pairs"])
    v = distmatch_verdict(df)  # must not raise
    assert v["verdict"] == "standoff_improved"
    assert np.isnan(v["per_config"]["w1"]["test_ade"])


def _ident_row(objective, axis, band_width, *, censored=False, band_lo=1.0,
               fitted=1.2, n_nodes=2, edge=False, policy="median"):
    return {"objective": objective, "policy": policy, "axis": axis,
            "band_lo": band_lo, "band_hi": band_lo + band_width,
            "band_width": band_width, "n_nodes_in_band": n_nodes,
            "censored_lo": False, "censored_hi": censored,
            "fitted": fitted, "fitted_on_grid_edge": edge}


def test_identifiability_summary_detects_restoration():
    df = pd.DataFrame([
        _ident_row("ade", "v0", 2.0),
        _ident_row("w1", "v0", 0.4),
        _ident_row("pure", "v0", 1.9),
        # interior sigma-axis fits, so the cross-axis clamp guard stays quiet
        _ident_row("ade", "sigma", 1.4),
        _ident_row("w1", "sigma", 1.0),
        _ident_row("pure", "sigma", 1.0),
    ])
    s = identifiability_summary(df)
    entry = s["per_policy_axis"]["median/v0"]
    assert entry["objectives"]["w1"]["restored"] is True
    assert entry["objectives"]["pure"]["restored"] is False
    assert s["restored_any"] is True


def test_identifiability_summary_degraded_reference_blocks_restoration():
    """A censored ADE reference band makes the <= factor comparison
    meaningless: the whole entry is flagged and cannot grant restoration."""
    df = pd.DataFrame([
        _ident_row("ade", "v0", 2.0, censored=True),
        _ident_row("w1", "v0", 0.4),
        _ident_row("ade", "sigma", 1.4),
        _ident_row("w1", "sigma", 1.0),
    ])
    s = identifiability_summary(df)
    entry = s["per_policy_axis"]["median/v0"]
    assert entry["reference_degraded"] is True
    assert entry["objectives"]["w1"]["restored"] is False
    assert s["restored_any"] is False


def test_identifiability_summary_fitted_outside_band_is_not_restoration():
    """A narrow band that does not contain the refined optimum does not
    describe the optimum (review finding)."""
    df = pd.DataFrame([
        _ident_row("ade", "v0", 2.0),
        _ident_row("w1", "v0", 0.4, fitted=3.0),  # band [1.0, 1.4], fit at 3.0
        _ident_row("ade", "sigma", 1.4),
        _ident_row("w1", "sigma", 1.0),
    ])
    s = identifiability_summary(df)
    w1 = s["per_policy_axis"]["median/v0"]["objectives"]["w1"]
    assert w1["fitted_in_band"] is False
    assert w1["restored"] is False


def test_identifiability_summary_missing_other_axis_fails_closed():
    """Without the other-axis row the slice anchor cannot be verified; the
    guard must fail closed, not silently pass (review finding)."""
    df = pd.DataFrame([
        _ident_row("ade", "v0", 2.0),
        _ident_row("w1", "v0", 0.4),
        # no sigma-axis rows at all
    ])
    s = identifiability_summary(df)
    w1 = s["per_policy_axis"]["median/v0"]["objectives"]["w1"]
    assert w1["other_axis_edge"] is True
    assert w1["restored"] is False


def test_identifiability_summary_censored_band_is_not_restoration():
    """A narrow band hugging the grid edge is a degenerate direction (e.g. the
    EMD term's v0->inf preference), never 'restored' identifiability."""
    df = pd.DataFrame([
        _ident_row("ade", "v0", 1.6),
        _ident_row("pure", "v0", 0.0, censored=True, n_nodes=1),
        _ident_row("ade", "sigma", 1.4),
        _ident_row("pure", "sigma", 1.0),
    ])
    s = identifiability_summary(df)
    p = s["per_policy_axis"]["median/v0"]["objectives"]["pure"]
    assert p["restored"] is False
    assert p["single_node"] is True
    assert s["restored_any"] is False


def test_identifiability_summary_clamped_other_axis_is_not_restoration():
    """A sharp sigma band measured on the v0=grid-edge slice (the fitted v0 ran
    off the grid, e.g. to 874) describes a conditional cross-section far from
    the optimum -- it must not count as restored identifiability."""
    df = pd.DataFrame([
        _ident_row("ade", "sigma", 1.4),
        _ident_row("pure", "sigma", 0.2),          # sharp...
        _ident_row("ade", "v0", 1.6),
        _ident_row("pure", "v0", 0.0, censored=True, fitted=874.0, edge=True),
    ])
    s = identifiability_summary(df)
    p = s["per_policy_axis"]["median/sigma"]["objectives"]["pure"]
    assert p["other_axis_edge"] is True
    assert p["restored"] is False                   # ...but on a clamped slice
    assert s["restored_any"] is False


def test_cap_verdict_capfit_m1_is_median_alias_not_decoupled_evidence():
    """capfit at m=1.0 is the median path by aliasing; even if its numbers met
    the criteria it must not count as decoupled evidence."""
    df = pd.DataFrame(
        [("median", 0.68, 1e-5, 24, 26, np.nan),
         ("capfit", 0.20, 0.4, 15, 26, 1.0)],
        columns=["policy", "mean_gap", "sign_p", "n_real_gt_sim", "n_pairs",
                 "cap_multiplier"])
    v = cap_verdict(df)
    assert v["verdict"] == "undetermined"  # no genuinely decoupled policy ran
    assert not v["per_policy"]["capfit"]["verdict_eligible"]


def test_f1_verdict_requires_joint_improvement():
    stands = f1_verdict([{"label": "cap:median", "ade_calibrated": 0.640,
                          "ade_avec": 0.639, "gap_calibrated": 0.68,
                          "gap_avec": 0.62}])
    assert stands["verdict"] == "f1_stands"
    beats = f1_verdict([{"label": "dm:w1", "ade_calibrated": 0.60,
                         "ade_avec": 0.64, "gap_calibrated": 0.30,
                         "gap_avec": 0.62}])
    assert beats["verdict"] == "calibration_beats_hand_tuning"
    assert beats["beats"] == ["dm:w1"]
    partial = f1_verdict([{"label": "dm:w2", "ade_calibrated": 0.60,
                           "ade_avec": 0.64, "gap_calibrated": 0.60,
                           "gap_avec": 0.62}])
    assert partial["verdict"] == "f1_stands"
    assert partial["partial"] == ["dm:w2"]


def test_f1_verdict_empty_audit_is_undetermined():
    """Zero audited configurations must not read as 'the negative result
    stands' (review finding, reproduced on an empty directory)."""
    assert f1_verdict([])["verdict"] == "undetermined"


def test_f1_verdict_zero_references_cannot_fabricate_a_win():
    """An all-zero degenerate row satisfies both <= comparisons numerically;
    it must not produce the audit's strongest positive claim (review finding)."""
    v = f1_verdict([{"label": "dm:bug", "ade_calibrated": 0.0,
                     "ade_avec": 0.0, "gap_calibrated": 0.0, "gap_avec": 0.0}])
    assert v["verdict"] == "f1_stands"
    assert v["beats"] == []


# --------------------------------------------------------------------------- #
# rq1b_domain_check
# --------------------------------------------------------------------------- #
def test_rq1b_domain_check_box_loso_and_outside():
    assert rq1b_domain_check(1.168, 1.712)["status"] == "inside_box"   # canonical
    assert rq1b_domain_check(1.09, 2.60)["status"] == "near_loso_envelope"
    assert rq1b_domain_check(0.93, 0.94)["status"] == "outside"        # uncapped fit
    assert rq1b_domain_check(float("nan"), 1.7)["status"] == "undefined"


# --------------------------------------------------------------------------- #
# ledger safety: audit sidecars never collide with canonical records
# --------------------------------------------------------------------------- #
def _pools():
    return {"real_closest": [2.0, 2.5, 3.0], "calibrated_closest": [1.5, 2.0, 2.4],
            "default_closest": [1.4, 1.9, 2.6], "norepulsion_closest": [1.0, 1.6, 2.0]}


def test_cap_sidecar_test_ids_do_not_collide_with_canonical():
    tests = cap_sidecar_tests(_pools(), "loco", "uncapped", 1.3)
    assert tests  # 3 arms present
    for t in tests:
        assert t["auxiliary"] is True
        assert t["headline"] is False
        assert t["test_id"].startswith("rq2cap.")
        assert not t["test_id"].startswith("rq2.")
        assert not t["family"].startswith("rq2_fidelity_paired")
        assert t["family"] == "rq2_cap_sensitivity_loco"


def test_aux_sidecar_builder_copies_extra_fields():
    tests = aux_paired_sidecar_tests(
        _pools(), "loco", family="rq2_distmatch_loco",
        prefix="rq2dm.loco.w1", extra={"config": "w1"}, note="n")
    assert all(t["family"] == "rq2_distmatch_loco" for t in tests)
    assert all(t["config"] == "w1" for t in tests)
    assert {t["test_id"] for t in tests} == {
        "rq2dm.loco.w1.closest_sign.calibrated",
        "rq2dm.loco.w1.closest_sign.avec_default",
        "rq2dm.loco.w1.closest_sign.no_repulsion"}
