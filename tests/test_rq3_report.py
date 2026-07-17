"""Tests for examples/make_rq3_report.py (synthetic-frame pinned verdicts)."""

import json

import numpy as np
import pandas as pd
import pytest

from examples.make_rq3_report import (
    ALPHA,
    STATUS_DEGENERATE,
    STATUS_INVARIANT,
    STATUS_INVARIANT_NS,
    STATUS_NODATA,
    STATUS_REVERSAL,
    STATUS_UNDETERMINED,
    _mcnemar_exact,
    _mcnemar_family_bh,
    encounter_means,
    headline_tests,
    medoid_reference_rows,
    robust_gain_rows,
    tristate,
    v1_rows,
    v2_ranking_preservation,
    v2_robust_gain_preservation,
    write_report,
)
from examples.run_rq3_realloop import PLAN_MODES, planned_cells

ENCS = [f"e{k:02d}" for k in range(6)]


def _runs(arm, pred, plan, min_dists, seeds=(0, 1), collided=None,
          goal=True):
    rows = []
    collided = collided or [0] * len(min_dists)
    for enc, md, coll in zip(ENCS, min_dists, collided):
        for s in seeds:
            rows.append(dict(
                ped_arm=arm, pred=pred, plan=plan, enc_id=enc, seed=s,
                min_dist_m=md + 0.01 * s, min_ttc_s=5.0,
                collision_count=coll, goal_reached=goal, time_s=8.0,
                progress=1.0, censored=not goal,
                ego_dev_mean_m=0.5, ego_dev_max_m=1.0,
            ))
    return rows


def _frame(rows):
    return pd.DataFrame(rows)


class TestEncounterMeans:
    def test_collapses_seeds_to_encounter_mean(self):
        runs = _frame(_runs("replay", "cv", "single", [1.0] * 6))
        em = encounter_means(runs)
        assert len(em) == 6
        assert em["n_seeds"].eq(2).all()
        # mean over seeds 0,1 of md + 0.01*s = md + 0.005
        np.testing.assert_allclose(em["min_dist_m"], 1.005)

    def test_collision_rate_is_seed_fraction(self):
        rows = _runs("replay", "cv", "single", [1.0] * 6,
                     collided=[1, 0, 0, 0, 0, 0])
        # make seed 1 of e00 collision-free -> rate 0.5
        for r in rows:
            if r["enc_id"] == "e00" and r["seed"] == 1:
                r["collision_count"] = 0
        em = encounter_means(_frame(rows))
        e00 = em[em.enc_id == "e00"].iloc[0]
        assert e00["collision_rate"] == pytest.approx(0.5)

    def test_inf_ttc_becomes_nan(self):
        rows = _runs("replay", "cv", "single", [1.0] * 6)
        for r in rows:
            r["min_ttc_s"] = float("inf")
        em = encounter_means(_frame(rows))
        assert em["min_ttc_s"].isna().all()


class TestV1Rows:
    def test_known_positive_delta(self):
        """calib min_dist consistently +0.5 above replay -> 6/6 sign wins."""
        replay = _runs("replay", "cv", "single", [1.0, 1.2, 1.4, 1.1, 1.3, 1.5])
        calib = _runs("calib", "cv", "single", [1.5, 1.7, 1.9, 1.6, 1.8, 2.0])
        em = encounter_means(_frame(replay + calib))
        rows = v1_rows(em, ["calib"], ["cv"], ["single"])
        assert len(rows) == 1
        r = rows[0]
        assert r["n_pairs"] == 6
        assert r["n_arm_gt_replay"] == 6
        assert r["mean_delta_m"] == pytest.approx(0.5)
        assert r["sign_p"] == pytest.approx(2 * 0.5 ** 6)

    def test_pairs_align_on_enc_id_missing_dropped(self):
        replay = _runs("replay", "cv", "single", [1.0] * 6)
        calib = _runs("calib", "cv", "single", [2.0] * 6)
        calib = [r for r in calib if r["enc_id"] != "e03"]
        em = encounter_means(_frame(replay + calib))
        rows = v1_rows(em, ["calib"], ["cv"], ["single"])
        assert rows[0]["n_pairs"] == 5

    def test_collision_discordance_counts(self):
        replay = _runs("replay", "cv", "single", [1.0] * 6,
                       collided=[1, 1, 0, 0, 0, 0])
        calib = _runs("calib", "cv", "single", [2.0] * 6,
                      collided=[0, 0, 0, 0, 0, 0])
        em = encounter_means(_frame(replay + calib))
        r = v1_rows(em, ["calib"], ["cv"], ["single"])[0]
        assert r["replay_coll_encs"] == 2
        assert r["arm_coll_encs"] == 0
        assert r["coll_replay_only"] == 2
        assert r["coll_arm_only"] == 0
        assert r["mcnemar_p"] == pytest.approx(0.5)  # binom(0 of 2, 0.5)


class TestMcNemar:
    def test_no_discordance_is_nan(self):
        assert np.isnan(_mcnemar_exact(0, 0))

    def test_symmetric(self):
        assert _mcnemar_exact(3, 1) == _mcnemar_exact(1, 3)


class TestTristate:
    def test_invariant(self):
        assert tristate([True, True, True], [False, False, False]) \
            == STATUS_INVARIANT

    def test_invariant_all_nonsignificant_is_annotated(self):
        """Agreement counts as invariant, but all-n.s. agreement must not
        read as a fully-powered robustness claim (review M1 companion)."""
        assert tristate([True, True, True], [True, True, True]) \
            == STATUS_INVARIANT_NS

    def test_nonsignificant_disagreement_is_undetermined_not_invariant(self):
        """Review M1: a direction disagreement carried by a non-significant
        arm must NOT claim '方向は全アームで不変' -- it is undetermined."""
        s = tristate([True, True, False], [False, False, True])
        assert s == STATUS_UNDETERMINED
        assert "不変" not in s  # the old wording was structurally false

    def test_significant_disagreement_is_reversal(self):
        assert tristate([True, False], [False, False]) == STATUS_REVERSAL

    def test_one_sided_significance_is_not_reversal(self):
        # Significant + vs non-significant - : undetermined, not reversal.
        assert tristate([True, False], [False, True]) == STATUS_UNDETERMINED

    def test_missing_arm_is_nodata(self):
        assert tristate([True, None], [False, False]) == STATUS_NODATA
        assert tristate([], []) == STATUS_NODATA


class TestRobustGain:
    def _em(self, arm="replay", gain=0.5):
        single = _runs(arm, "cv", "single", [1.0, 1.2, 1.4, 1.1, 1.3, 1.5])
        robust = _runs(arm, "cv", "robust",
                       [1.0 + gain, 1.2 + gain, 1.4 + gain,
                        1.1 + gain, 1.3 + gain, 1.5 + gain])
        return encounter_means(_frame(single + robust))

    def test_positive_gain_detected(self):
        rows = robust_gain_rows(self._em(), ["replay"], ["cv"])
        assert len(rows) == 1
        r = rows[0]
        assert r["n_robust_gt_single"] == 6
        assert r["mean_delta_m"] == pytest.approx(0.5)
        assert r["n_time_pairs"] == 6  # all goal-reached in both plans

    def test_time_pairs_respect_censoring(self):
        single = _runs("replay", "cv", "single", [1.0] * 6)
        robust = _runs("replay", "cv", "robust", [1.5] * 6, goal=False)
        em = encounter_means(_frame(single + robust))
        r = robust_gain_rows(em, ["replay"], ["cv"])[0]
        assert r["n_time_pairs"] == 0
        assert np.isnan(r["mean_time_cost_s"])


class TestV2Preservation:
    def _gain(self, arm, delta, wp):
        return {"ped_arm": arm, "pred": "cv", "mean_delta_m": delta,
                "wilcoxon_p": wp}

    def test_invariant_direction(self):
        rows = v2_robust_gain_preservation(
            [self._gain("replay", 0.5, 0.001), self._gain("calib", 0.4, 0.002)],
            ["replay", "calib"], ["cv"])
        assert rows[0]["value"] == STATUS_INVARIANT

    def test_nonsignificant_flip_is_undetermined(self):
        rows = v2_robust_gain_preservation(
            [self._gain("replay", 0.5, 0.001),
             self._gain("calib", -0.01, 0.8)],
            ["replay", "calib"], ["cv"])
        assert rows[0]["value"] == STATUS_UNDETERMINED

    def test_significant_flip_is_reversal(self):
        rows = v2_robust_gain_preservation(
            [self._gain("replay", 0.5, 0.001),
             self._gain("calib", -0.4, 0.01)],
            ["replay", "calib"], ["cv"])
        assert rows[0]["value"] == STATUS_REVERSAL

    def test_exact_ties_are_degenerate_not_negative(self):
        """CV's robust == single (delta exactly 0) must not be classified as
        a negative direction (review minor: zero-delta misclassification)."""
        rows = v2_robust_gain_preservation(
            [self._gain("replay", 0.0, float("nan")),
             self._gain("calib", 0.0, float("nan"))],
            ["replay", "calib"], ["cv"])
        assert rows[0]["value"] == STATUS_DEGENERATE
        assert "縮退" in rows[0]["detail"]

    def test_tie_plus_consistent_direction_is_invariant(self):
        rows = v2_robust_gain_preservation(
            [self._gain("replay", 0.0, float("nan")),
             self._gain("calib", 0.5, 0.001)],
            ["replay", "calib"], ["cv"])
        assert rows[0]["value"] == STATUS_INVARIANT


class TestSidecar:
    def _v1(self):
        replay = _runs("replay", "cv", "single", [1.0] * 6)
        calib = _runs("calib", "cv", "single", [1.5] * 6)
        avec = _runs("avec", "cv", "single", [1.6] * 6)
        em = encounter_means(_frame(replay + calib + avec))
        return em

    def test_canonical_only_calib_and_namespaced(self):
        em = self._v1()
        v1 = v1_rows(em, ["calib", "avec"], ["cv"], ["single"])
        gains = robust_gain_rows(em, ["replay"], ["cv"])  # empty (no robust)
        tests = headline_tests(v1, gains)
        assert all(t["test_id"].startswith("rq3.") for t in tests)
        canonical = [t for t in tests if t["family"] == "rq3_v1_reactivity"]
        assert len(canonical) == 1
        assert canonical[0]["headline"] is True
        assert "auxiliary" not in canonical[0]
        assert ".calib." in canonical[0]["test_id"]
        ctrl = [t for t in tests if t["family"] == "rq3_v1_reactivity_ctrl"]
        assert all(t["auxiliary"] is True for t in ctrl)
        assert all(".avec." in t["test_id"] for t in ctrl)
        # Guard (review M2a): every NON-auxiliary emission must stay in the
        # user-approved canonical family -- the M2a additions are all
        # auxiliary and must never grow the study-wide pool.
        non_aux = [t for t in tests if not t.get("auxiliary")]
        assert {t["family"] for t in non_aux} == {"rq3_v1_reactivity"}

    def test_no_duplicate_test_ids(self):
        em = self._v1()
        v1 = v1_rows(em, ["calib", "avec"], ["cv"], ["single"])
        tests = headline_tests(v1, [])
        ids = [t["test_id"] for t in tests]
        assert len(ids) == len(set(ids))

    def test_ids_do_not_collide_with_other_rq_namespaces(self):
        em = self._v1()
        tests = headline_tests(v1_rows(em, ["calib"], ["cv"], ["single"]), [])
        for t in tests:
            assert not t["test_id"].startswith(("rq1b.", "rq2.", "rq2cap.",
                                                "rq2dm."))

    def test_cv_robust_mcnemar_duplicate_not_emitted(self):
        """The bit-identical cv/robust cell must not re-enter the McNemar
        family (a duplicated small p can shift BH ranks; review M3)."""
        replay = _runs("replay", "cv", "single", [1.0] * 6,
                       collided=[1, 1, 1, 0, 0, 0])
        replay_r = _runs("replay", "cv", "robust", [1.0] * 6,
                         collided=[1, 1, 1, 0, 0, 0])
        calib = _runs("calib", "cv", "single", [1.5] * 6)
        calib_r = _runs("calib", "cv", "robust", [1.5] * 6)
        em = encounter_means(_frame(replay + replay_r + calib + calib_r))
        v1 = v1_rows(em, ["calib"], ["cv"], ["single", "robust"])
        tests = headline_tests(v1, [])
        mcnemar = [t for t in tests if "collision_mcnemar" in t["test_id"]]
        assert len(mcnemar) == 1  # cv/single only, cv/robust excluded
        assert mcnemar[0]["test_id"].endswith(".cv.single")
        assert "not" in mcnemar[0]["note"]
        # Sign-test family keeps the approved 6-cell structure regardless.
        sign = [t for t in tests if "reactivity_sign" in t["test_id"]]
        assert len(sign) == 2  # single + robust (cv only in this fixture)

    def test_mcnemar_ctrl_arms_use_ctrl_family(self):
        replay = _runs("replay", "cv", "single", [1.0] * 6,
                       collided=[1, 1, 0, 0, 0, 0])
        avec = _runs("avec", "cv", "single", [1.5] * 6)
        em = encounter_means(_frame(replay + avec))
        v1 = v1_rows(em, ["avec"], ["cv"], ["single"])
        tests = headline_tests(v1, [])
        mc = [t for t in tests if "collision_mcnemar" in t["test_id"]]
        assert len(mc) == 1
        assert mc[0]["family"] == "rq3_v1_collision_mcnemar_ctrl"
        assert mc[0]["auxiliary"] is True


class TestSidecarM2aFamilies:
    """Ledger registration of the thesis-quoted V3/V2 p values (review M2a)."""

    def _em_two_arms(self, degenerate_arm=None):
        rows = []
        for arm in ("replay", "calib"):
            single = [1.0, 1.2, 1.4, 1.1, 1.3, 1.5]
            if arm == degenerate_arm:
                robust = single  # exact tie: robust == single bit-for-bit
            else:
                robust = [v + 0.5 for v in single]
            rows += _runs(arm, "cv", "single", single)
            rows += _runs(arm, "cv", "robust", robust)
        return encounter_means(_frame(rows))

    def test_nonreplay_sign_goes_to_ctrl_family(self):
        em = self._em_two_arms()
        gains = robust_gain_rows(em, ["replay", "calib"], ["cv"])
        tests = headline_tests([], gains)
        real = [t for t in tests if t["family"] == "rq3_v3_robust_real"]
        ctrl = [t for t in tests if t["family"] == "rq3_v3_robust_real_ctrl"]
        assert len(real) == 1 and len(ctrl) == 1
        # replay family/test_id unchanged by the M2a addition
        assert real[0]["test_id"] == "rq3.v3.robust_gain_sign.replay.cv"
        assert ctrl[0]["test_id"] == "rq3.v3.robust_gain_sign.calib.cv"
        assert ctrl[0]["auxiliary"] is True

    def test_degenerate_cell_emits_null_p_in_ctrl_family(self):
        em = self._em_two_arms(degenerate_arm="calib")
        gains = robust_gain_rows(em, ["replay", "calib"], ["cv"])
        tests = headline_tests([], gains)
        ctrl = [t for t in tests if t["family"] == "rq3_v3_robust_real_ctrl"]
        assert len(ctrl) == 1
        assert ctrl[0]["p_value"] is None
        assert "degenerate" in ctrl[0]["note"]

    def test_wilcoxon_family_skips_degenerate_cells(self):
        em = self._em_two_arms(degenerate_arm="calib")
        gains = robust_gain_rows(em, ["replay", "calib"], ["cv"])
        tests = headline_tests([], gains)
        wil = [t for t in tests if t["family"] == "rq3_v3_robust_wilcoxon"]
        # only the non-degenerate replay cell has a finite Wilcoxon p
        assert [t["test_id"] for t in wil] \
            == ["rq3.v3.robust_gain_wilcoxon.replay.cv"]
        assert wil[0]["auxiliary"] is True
        assert wil[0]["p_value"] is not None
        assert wil[0]["statistic"] is not None

    def test_ranking_gates_map_one_to_one_with_v2_rows(self):
        rows = []
        for pred, base in (("cv", 1.0), ("lstm", 2.0)):
            rows += _runs("replay", pred, "single",
                          [base, base + 0.2, base + 0.4,
                           base + 0.1, base + 0.3, base + 0.5])
        em = encounter_means(_frame(rows))
        v2 = v2_ranking_preservation(em, ["replay"], ["cv", "lstm"],
                                     ["single"])
        ranking = [r for r in v2 if r["verdict_kind"] == "predictor_ranking"]
        assert len(ranking) == 1
        tests = headline_tests([], [], v2)
        gates = [t for t in tests if t["family"] == "rq3_v2_ranking_gates"]
        assert len(gates) == len(ranking) == 1
        g, r = gates[0], ranking[0]
        assert g["test_id"] == "rq3.v2.ranking_gap_wilcoxon.replay.single"
        assert g["p_value"] == pytest.approx(r["gap_p"])
        assert g["auxiliary"] is True
        # the bottom-two predictors are named in the description
        assert r["most_dangerous"] in g["description"]
        assert r["runner"] in g["description"]

    def test_degenerate_gate_emits_null_p_with_disclosure(self):
        """A gates row with NaN gap_p must disclose the degeneracy."""
        v2 = [{"verdict_kind": "predictor_ranking", "pred_or_plan": "single",
               "ped_arm": "replay", "gap_p": float("nan"),
               "gap_stat": float("nan"), "gap_n_pairs": 6,
               "most_dangerous": "cv", "runner": "lstm"}]
        tests = headline_tests([], [], v2)
        assert len(tests) == 1
        assert tests[0]["p_value"] is None
        assert "degenerate" in tests[0]["note"]

    def test_nonsignificant_gate_still_emitted_as_auxiliary(self):
        v2 = [{"verdict_kind": "predictor_ranking", "pred_or_plan": "single",
               "ped_arm": "replay", "gap_p": 0.135, "gap_stat": 100.0,
               "gap_n_pairs": 26, "most_dangerous": "cv", "runner": "sgan"}]
        tests = headline_tests([], [], v2)
        assert len(tests) == 1  # n.s. gates are disclosed, not dropped
        assert tests[0]["p_value"] == pytest.approx(0.135)
        assert tests[0]["auxiliary"] is True
        assert tests[0]["headline"] is False

    def test_v2_verdict_rows_carry_gap_p(self):
        rows = []
        for pred, base in (("cv", 1.0), ("lstm", 2.0)):
            rows += _runs("replay", pred, "single",
                          [base, base + 0.2, base + 0.4,
                           base + 0.1, base + 0.3, base + 0.5])
        em = encounter_means(_frame(rows))
        v2 = v2_ranking_preservation(em, ["replay"], ["cv", "lstm"],
                                     ["single"])
        r = [x for x in v2 if x["verdict_kind"] == "predictor_ranking"][0]
        assert np.isfinite(r["gap_p"])
        assert np.isfinite(r["gap_stat"])
        assert r["gap_n_pairs"] == 6
        assert r["most_dangerous"] == "cv" and r["runner"] == "lstm"


class TestMcnemarFamilyBH:
    def test_excludes_cv_robust_duplicate(self):
        rows = [
            {"pred": "cv", "plan": "single", "mcnemar_p": 0.0156},
            {"pred": "cv", "plan": "robust", "mcnemar_p": 0.0156},  # dup
            {"pred": "lstm", "plan": "single", "mcnemar_p": 0.0156},
            {"pred": "lstm", "plan": "robust", "mcnemar_p": 0.5},
            {"pred": "sgan", "plan": "single", "mcnemar_p": 0.03125},
            {"pred": "sgan", "plan": "robust", "mcnemar_p": 0.5},
        ]
        bh = _mcnemar_family_bh(rows)
        assert bh["m"] == 5  # duplicate excluded
        assert bh["n_reject"] >= 1

    def test_all_nan_gives_zero_rejections(self):
        rows = [{"pred": "cv", "plan": "single", "mcnemar_p": float("nan")}]
        bh = _mcnemar_family_bh(rows)
        assert bh["n_reject"] == 0
        assert np.isnan(bh["min_q"])


class TestMedoidReference:
    def test_paired_medoid_vs_draw(self):
        draw = _runs("replay", "sgan", "single", [1.0, 1.2, 1.4, 1.1, 1.3, 1.5])
        med = _runs("replay", "sgan", "medoid", [1.2, 1.4, 1.6, 1.3, 1.5, 1.7])
        em = encounter_means(_frame(draw + med))
        rows = medoid_reference_rows(em)
        assert len(rows) == 1
        assert rows[0]["n_pairs"] == 6
        assert rows[0]["mean_medoid_minus_draw_m"] == pytest.approx(0.2)


class TestPlannedCells:
    def test_restricted_plans_do_not_grow_medoid_cells(self):
        cells = planned_cells(["replay"], ["sgan"], ["single"],
                              include_medoid=True)
        assert ("replay", "sgan", "medoid") not in cells

    def test_full_matrix_includes_medoid(self):
        cells = planned_cells(["replay"], ["sgan"], list(PLAN_MODES),
                              include_medoid=True)
        assert ("replay", "sgan", "medoid") in cells

    def test_explicit_medoid_plan_token(self):
        cells = planned_cells(["calib"], ["sgan"], ["medoid"],
                              include_medoid=True)
        assert cells == [("calib", "sgan", "medoid")]

    def test_no_medoid_flag_wins(self):
        cells = planned_cells(["replay"], ["sgan"], list(PLAN_MODES),
                              include_medoid=False)
        assert ("replay", "sgan", "medoid") not in cells


class TestReportDeterminism:
    def test_pipeline_is_reproducible(self):
        rows = (_runs("replay", "cv", "single", [1.0, 1.2, 1.4, 1.1, 1.3, 1.5])
                + _runs("replay", "cv", "robust", [1.3, 1.5, 1.7, 1.4, 1.6, 1.8])
                + _runs("calib", "cv", "single", [1.5, 1.7, 1.9, 1.6, 1.8, 2.0])
                + _runs("calib", "cv", "robust", [1.6, 1.8, 2.0, 1.7, 1.9, 2.1]))
        runs = _frame(rows)
        census = pd.DataFrame({"enc_id": ENCS, "eligible": [True] * 6})

        def build():
            em = encounter_means(runs)
            v1 = v1_rows(em, ["calib"], ["cv"], ["single", "robust"])
            gains = robust_gain_rows(em, ["replay", "calib"], ["cv"])
            v2 = (v2_robust_gain_preservation(gains, ["replay", "calib"],
                                              ["cv"])
                  + v2_ranking_preservation(em, ["replay", "calib"], ["cv"],
                                            ["single", "robust"]))
            v3 = [r for r in gains if r["ped_arm"] == "replay"]
            report = write_report(runs, census, v1, v2, v3, gains, em)
            sidecar = json.dumps({"tests": headline_tests(v1, gains, v2)},
                                 indent=1)
            return report, sidecar

        r1, s1 = build()
        r2, s2 = build()
        assert r1 == r2
        assert s1 == s2
        assert "rq3_v1_reactivity" in s1
        assert "V1" in r1 and "V3" in r1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
