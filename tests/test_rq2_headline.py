"""Regression guards for the RQ2 headline fidelity statistic (review R4).

The thesis fidelity-gap statistic (pooled closest-approach KS, ~0.46 / p~0.007 /
+0.68 m standoff) and the saturation/de-saturation logic that sets the
multiplicity *family size* m are produced by ``headline_tests`` /
``_pooled_ks_stat`` in examples/run_rq2_evaluation.py, which no test touches. The
1e-12 saturation guard decides whether a control arm counts as a distinct
hypothesis, and m feeds BH-FDR/Holm -- so a regression here could flip whether
the boundary fidelity gap survives correction. These tests pin both branches.
"""
import numpy as np

from examples.run_rq2_evaluation import _pooled_ks_stat, headline_tests


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


def _pools(cal, real, default, norep):
    return {
        "calibrated_closest": list(cal),
        "real_closest": list(real),
        "default_closest": list(default),
        "norepulsion_closest": list(norep),
    }


def test_headline_saturated_controls_excluded_from_family():
    """Controls numerically identical to calibrated saturate -> family size 1."""
    real = np.linspace(0.0, 2.0, 30)
    cal = np.linspace(0.6, 2.6, 30)
    pools = _pools(cal, real, default=cal.copy(), norep=cal.copy())

    out = headline_tests(pools, protocol="loco")

    assert len(out) == 1  # only the calibrated headline; no extra family members
    head = out[0]
    assert head["headline"] is True
    assert head["family"] == "rq2_fidelity_ks_loco"
    assert head["test_id"] == "rq2.loco.closest_ks.calibrated"
    assert set(head["controls"].keys()) == {"avec_default", "no_repulsion"}


def test_headline_desaturated_control_becomes_family_member():
    """A control whose distribution diverges from calibrated becomes a distinct
    hypothesis counted in the family (family size grows)."""
    real = np.linspace(0.0, 2.0, 30)
    cal = np.linspace(0.6, 2.6, 30)
    default = np.linspace(5.0, 9.0, 30)  # clearly different -> de-saturates
    pools = _pools(cal, real, default=default, norep=cal.copy())

    out = headline_tests(pools, protocol="loso")

    assert len(out) == 2  # calibrated headline + de-saturated avec_default
    head = next(t for t in out if t["headline"])
    member = next(t for t in out if not t["headline"])
    assert member["test_id"].endswith("avec_default")
    assert member["family"] == head["family"] == "rq2_fidelity_ks_loso"
    # The still-saturated no_repulsion stays a control, not a family member.
    assert set(head["controls"].keys()) == {"no_repulsion"}


def test_saturation_is_value_equality_not_array_identity():
    """A control that is a DIFFERENT array but yields the identical (ks, p) -- a
    reordering has the same ECDF -- must still saturate. Guards against the
    1e-12 tolerance being mistaken for a byte-identical-array check."""
    real = np.linspace(0.0, 2.0, 30)
    cal = np.linspace(0.6, 2.6, 30)
    pools = _pools(cal, real, default=cal[::-1], norep=cal.copy())  # reversed != identical array
    out = headline_tests(pools, protocol="loco")
    assert len(out) == 1
    assert set(out[0]["controls"].keys()) == {"avec_default", "no_repulsion"}


def test_near_but_distinct_control_de_saturates():
    """A control whose distribution is only slightly perturbed (one point moved)
    diverges from calibrated by more than 1e-12 and must become a family member,
    exercising the tolerance's discriminating edge rather than only wildly
    different inputs."""
    real = np.linspace(0.0, 2.0, 30)
    cal = np.linspace(0.6, 2.6, 30)
    nudged = cal.copy()
    nudged[0] = nudged[0] + 0.5  # small but well beyond the 1e-12 tolerance
    pools = _pools(cal, real, default=nudged, norep=cal.copy())
    out = headline_tests(pools, protocol="loco")
    assert len(out) == 2
    member = next(t for t in out if not t["headline"])
    assert member["test_id"].endswith("avec_default")
    assert set(out[0]["controls"].keys()) == {"no_repulsion"}  # norep still saturates


def test_headline_empty_when_calibrated_pool_missing():
    real = np.linspace(0.0, 2.0, 30)
    assert headline_tests({"real_closest": list(real)}, protocol="loco") == []
