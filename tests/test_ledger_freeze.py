"""Freeze guard for the committed multiplicity-ledger structure.

The thesis (appendix B) transcribes the ledger verbatim and its generator
(MasterThesis/scripts/gen_ledger_tables.py) asserts these same counts, so a
silent change to a committed sidecar -- a family renamed, a test dropped, a
canonical/auxiliary flag flipped -- would break the thesis numbers without
failing any test (full-audit 2026-06-23 Major finding: claim-path guards were
missing). This test re-assembles the ledger from the COMMITTED sidecars and
pins the canonical pool, the study-wide survivors, and the per-family sizes.

Intentional changes (registering a new test) must update EXPECTED_* here AND
the thesis-side generator in the same change set.

Note: the sidecars are read from the WORKING TREE under outputs/ -- the freeze
therefore also catches a partial commit (code committed without its sidecar):
a fresh clone missing a sidecar fails these pins loudly.
"""
from pathlib import Path

from examples.make_multiplicity_ledger import assemble, load_sidecars

REPO = Path(__file__).parent.parent

EXPECTED_CANONICAL_FAMILIES = {
    "rq1b_claim2_fisher": 18,
    "rq2_fidelity_paired_loco": 3,
    "rq3_v1_reactivity": 6,
}
EXPECTED_AUX_FAMILIES = {
    "rq1b_claim2_fisher_aggregate": 12,
    "rq2_ade_contrast_loco": 2,
    "rq2_cap_sensitivity_loco": 9,
    "rq2_distmatch_loco": 12,
    "rq2_dut_fidelity_ks_multivehicle": 3,
    "rq2_dut_fidelity_ks_single": 3,
    "rq2_fidelity_ks_loco_diagnostic": 3,
    "rq2_fidelity_ks_loso_diagnostic": 3,
    "rq2_fidelity_paired_loso": 3,
    "rq3_v1_collision_mcnemar": 5,
    "rq3_v1_collision_mcnemar_ctrl": 14,
    "rq3_v1_reactivity_ctrl": 18,
    "rq3_v2_ranking_gates": 10,
    "rq3_v3_robust_real": 3,
    "rq3_v3_robust_real_ctrl": 12,
    "rq3_v3_robust_wilcoxon": 10,
}
# NaN p rows (degenerate cells, disclosed but not counted in family sizes)
EXPECTED_NULL_P_IDS = {
    "rq3.v3.robust_gain_sign.replay.cv",
    "rq3.v3.robust_gain_sign.calib.cv",
    "rq3.v3.robust_gain_sign.avec.cv",
    "rq3.v3.robust_gain_sign.norep.cv",
    "rq3.v3.robust_gain_sign.calib13x.cv",
}
EXPECTED_SURVIVORS = {
    "rq2.loco.closest_sign.calibrated",
    "rq2.loco.closest_wilcoxon.calibrated",
    "rq2.loco.closest_sign.no_repulsion",
}


def _rows():
    tests, _ = load_sidecars(REPO / "outputs")
    return assemble(tests)


def _family_counts(rows):
    counts = {}
    for r in rows:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    return counts


def test_canonical_pool_frozen():
    canonical, _ = _rows()
    assert _family_counts(canonical) == EXPECTED_CANONICAL_FAMILIES
    assert len(canonical) == 27
    assert all(r["overall_size"] == 27 for r in canonical)


def test_studywide_survivors_frozen():
    canonical, _ = _rows()
    bh = {r["test_id"] for r in canonical if r["overall_bh_reject"]}
    holm = {r["test_id"] for r in canonical if r["overall_holm_reject"]}
    assert bh == EXPECTED_SURVIVORS
    assert holm == EXPECTED_SURVIVORS


def test_auxiliary_families_frozen():
    _, aux = _rows()
    assert _family_counts(aux) == EXPECTED_AUX_FAMILIES
    assert len(aux) == 122


def test_null_p_rows_are_exactly_the_degenerate_cells():
    _, aux = _rows()
    nulls = {r["test_id"] for r in aux
             if r.get("p_value") is None
             or (isinstance(r.get("p_value"), float)
                 and r["p_value"] != r["p_value"])}
    assert nulls == EXPECTED_NULL_P_IDS


def test_auxiliary_never_enters_canonical_pool():
    canonical, aux = _rows()
    canonical_ids = {r["test_id"] for r in canonical}
    assert not canonical_ids & {r["test_id"] for r in aux}
    # loso re-splits are auxiliary even without the explicit flag
    assert all(r.get("protocol") != "loso" for r in canonical)
