"""Regression guards for the RQ2 headline fidelity statistic (reviews R4, F5).

The thesis fidelity-gap headline is the PAIRED per-encounter sign test (real vs
sim closest-approach share the encounter, review F5), with the Wilcoxon
signed-rank as the second family member and the former pooled independent-sample
KS demoted to an auxiliary diagnostic. The saturation/de-saturation logic sets
the multiplicity *family size* m that feeds BH-FDR/Holm -- a regression here
could flip whether the fidelity gap survives correction, or silently resurrect
the misspecified KS as a canonical test. These tests pin the paired statistics,
the pairing discipline (joint pair-drop, mismatch refusal), and the family
accounting.
"""
import numpy as np
from scipy.stats import binomtest

from examples.run_rq2_evaluation import (
    _paired_stats,
    _pooled_ks_stat,
    headline_tests,
)


def test_pooled_ks_stat_basic_empty_and_nonfinite():
    real = list(np.linspace(0.0, 1.0, 20))
    sim = list(np.linspace(0.5, 1.5, 25))

    s = _pooled_ks_stat({"sim": sim, "real": real}, "sim", "real")
    assert s is not None
    assert s["n_sim"] == 25 and s["n_real"] == 20
    assert 0.0 <= s["ks"] <= 1.0 and 0.0 <= s["p"] <= 1.0

    # Either side empty (or missing) -> None, never a spurious test record.
    assert _pooled_ks_stat({"sim": [], "real": real}, "sim", "real") is None
    assert _pooled_ks_stat({"real": real}, "sim", "real") is None

    # Non-finite values are dropped from the effective n.
    s2 = _pooled_ks_stat({"sim": sim + [np.nan, np.inf, -np.inf], "real": real},
                         "sim", "real")
    assert s2["n_sim"] == 25


# --------------------------------------------------------------------------- #
# _paired_stats: the F5 paired per-encounter unit
# --------------------------------------------------------------------------- #
def test_paired_stats_direction_and_exact_sign_p():
    real = [1.0, 2.0, 3.0, 4.0]
    sim = [0.5, 1.5, 2.5, 5.0]  # d = real - sim = [+.5, +.5, +.5, -1.0]

    s = _paired_stats({"sim": sim, "real": real}, "sim", "real")
    assert s["n_pairs"] == 4
    assert s["n_real_gt_sim"] == 3 and s["n_real_lt_sim"] == 1
    assert s["sign_p"] == binomtest(3, 4, 0.5).pvalue
    assert np.isfinite(s["wilcoxon_p"]) and 0.0 <= s["wilcoxon_p"] <= 1.0
    assert s["mean_gap"] == np.mean([0.5, 0.5, 0.5, -1.0])


def test_paired_stats_drops_pairs_jointly_and_refuses_mismatch():
    real = [1.0, 2.0, np.nan, 4.0]
    sim = [0.5, np.nan, 2.5, 3.0]

    # A non-finite value on EITHER side drops the whole pair (never one side).
    s = _paired_stats({"sim": sim, "real": real}, "sim", "real")
    assert s["n_pairs"] == 2  # pairs 0 and 3 survive
    assert s["n_real_gt_sim"] == 2

    # Length mismatch = pairing broken upstream -> refuse, don't fake a test.
    assert _paired_stats({"sim": [1.0, 2.0], "real": [1.0]}, "sim", "real") is None
    assert _paired_stats({"sim": [], "real": [1.0]}, "sim", "real") is None
    assert _paired_stats({"real": [1.0]}, "sim", "real") is None


def test_paired_stats_zero_differences_carry_no_direction():
    # One tie: dropped from the sign test's effective n.
    s = _paired_stats({"sim": [1.0, 2.0, 3.0], "real": [1.0, 2.5, 3.5]},
                      "sim", "real")
    assert s["n_real_gt_sim"] == 2 and s["n_real_lt_sim"] == 0
    assert s["sign_p"] == binomtest(2, 2, 0.5).pvalue

    # All-zero differences: no direction at all -> NaN p, never a crash.
    z = _paired_stats({"sim": [1.0, 2.0], "real": [1.0, 2.0]}, "sim", "real")
    assert np.isnan(z["sign_p"]) and np.isnan(z["wilcoxon_p"])


# --------------------------------------------------------------------------- #
# headline_tests: family accounting
# --------------------------------------------------------------------------- #
def _pools(cal, real, default, norep):
    return {
        "calibrated_closest": list(cal),
        "real_closest": list(real),
        "default_closest": list(default),
        "norepulsion_closest": list(norep),
    }


def test_headline_sign_plus_wilcoxon_form_the_canonical_family():
    """Saturated controls -> canonical family = {sign, wilcoxon} (m=2), and the
    demoted KS entries are auxiliary diagnostics, never canonical."""
    real = np.linspace(0.0, 2.0, 30)
    cal = real + 0.6  # d = -0.6 everywhere: sim overshoots real uniformly
    pools = _pools(cal, real, default=cal.copy(), norep=cal.copy())

    out = headline_tests(pools, protocol="loco")

    fam = [t for t in out if t["family"] == "rq2_fidelity_paired_loco"]
    assert [t["test_id"] for t in fam] == [
        "rq2.loco.closest_sign.calibrated",
        "rq2.loco.closest_wilcoxon.calibrated",
    ]
    sign, wilc = fam
    assert sign["headline"] is True and wilc["headline"] is False
    assert not sign.get("auxiliary") and not wilc.get("auxiliary")
    assert set(sign["controls"].keys()) == {"avec_default", "no_repulsion"}

    diags = [t for t in out if t["family"] == "rq2_fidelity_ks_loco_diagnostic"]
    assert {t["test_id"].rsplit(".", 1)[-1] for t in diags} == {
        "calibrated", "avec_default", "no_repulsion"}
    assert all(t["auxiliary"] is True and t["headline"] is False for t in diags)
    assert len(out) == len(fam) + len(diags)  # nothing else leaks in


def test_headline_desaturated_control_becomes_family_member():
    """A control whose paired direction diverges from calibrated is a distinct
    hypothesis counted in the family (family size grows)."""
    real = np.linspace(0.0, 2.0, 30)
    cal = real + 0.6      # n_real_gt_sim = 0
    default = real - 0.6  # n_real_gt_sim = 30 -> de-saturates on the statistic
    pools = _pools(cal, real, default=default, norep=cal.copy())

    out = headline_tests(pools, protocol="loso")

    fam = [t for t in out if t["family"] == "rq2_fidelity_paired_loso"]
    assert len(fam) == 3  # sign + wilcoxon + de-saturated avec_default
    member = next(t for t in fam if t["test_id"].endswith("avec_default"))
    assert member["headline"] is False
    assert member["statistic"] == 30.0
    # The still-saturated no_repulsion stays a control, not a family member.
    head = next(t for t in fam if t["headline"])
    assert set(head["controls"].keys()) == {"no_repulsion"}


def test_reordered_control_de_saturates_the_paired_test():
    """A reordered sim array has the SAME ECDF (identical KS) but DIFFERENT
    per-encounter pairs -- the paired test must treat it as a distinct
    hypothesis while the KS diagnostic still saturates. This pins exactly why
    the paired family replaced the KS headline (F5): the KS cannot see pairing."""
    real = np.linspace(0.0, 2.0, 30)
    cal = real + 0.6
    pools = _pools(cal, real, default=cal[::-1], norep=cal.copy())

    out = headline_tests(pools, protocol="loco")

    fam = [t for t in out if t["family"] == "rq2_fidelity_paired_loco"]
    assert any(t["test_id"].endswith("avec_default") for t in fam)
    # Same ECDF -> the diagnostic KS is numerically identical to calibrated.
    diags = {t["test_id"].rsplit(".", 1)[-1]: t for t in out
             if t["family"] == "rq2_fidelity_ks_loco_diagnostic"}
    assert diags["avec_default"]["statistic"] == diags["calibrated"]["statistic"]
    assert diags["avec_default"]["p_value"] == diags["calibrated"]["p_value"]


def test_headline_empty_when_calibrated_pool_missing():
    real = np.linspace(0.0, 2.0, 30)
    assert headline_tests({"real_closest": list(real)}, protocol="loco") == []
