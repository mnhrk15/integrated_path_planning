"""Tests for the RQ1b sensitivity analysis harness.

Covers (a) the GT/cruise override plumbing added to run_da_poc (calibrated SFM
ego-repulsion injection + cruise clamp, without dropping scenario-level keys),
(b) a minimal run_campaign integration round-trip, and (c) the RQ1b verdict
logic (robust-gain / CV-danger and GT-sensitivity flip detection).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.simulation.integrated_simulator import PedestrianSimulator
from examples import run_da_poc
from examples import run_rq1b_sensitivity as rq1b


# --------------------------------------------------------------------------- #
# (a) GT / cruise override plumbing
# --------------------------------------------------------------------------- #
def test_sfm_override_merges_preserving_scenario_keys():
    """Overriding sigma/v0 must keep scenario-level keys (agent_radius)."""
    config = load_config("scenarios/scenario_02.yaml")  # YAML: sigma 0.3, v0 2.1
    assert config.social_force_params["agent_radius"] == pytest.approx(0.3)

    run_da_poc.apply_sfm_and_cruise_overrides(
        config, ego_repulsion_sigma=1.156, ego_repulsion_v0=1.681)

    sfp = config.social_force_params
    assert sfp["ego_repulsion.sigma"] == pytest.approx(1.156)
    assert sfp["ego_repulsion.v0"] == pytest.approx(1.681)
    # The scenario-level key must survive the merge.
    assert sfp["agent_radius"] == pytest.approx(0.3)


def test_sfm_override_partial_keeps_other_repulsion_key():
    """Overriding only sigma must not drop the YAML v0."""
    config = load_config("scenarios/scenario_01.yaml")  # YAML: sigma 0.7, v0 3.5
    run_da_poc.apply_sfm_and_cruise_overrides(config, ego_repulsion_sigma=1.0)
    assert config.social_force_params["ego_repulsion.sigma"] == pytest.approx(1.0)
    assert config.social_force_params["ego_repulsion.v0"] == pytest.approx(3.5)


def test_sfm_override_reaches_pedestrian_simulator():
    """The merged params must actually drive the simulator's ego-repulsion."""
    config = load_config("scenarios/scenario_01.yaml")
    run_da_poc.apply_sfm_and_cruise_overrides(
        config, ego_repulsion_sigma=1.156, ego_repulsion_v0=1.681)
    sim = PedestrianSimulator(
        initial_states=np.array(config.ped_initial_states, dtype=float),
        groups=config.ped_groups,
        obstacles=None,
        dt=config.dt,
        social_force_params=config.social_force_params,
    )
    assert sim.ego_repulsion_sigma == pytest.approx(1.156)
    assert sim.ego_repulsion_v0 == pytest.approx(1.681)


def test_cruise_override_clamps_initial_speed():
    """A cruise override below the YAML initial speed must clamp the latter."""
    config = load_config("scenarios/scenario_01.yaml")  # init v = 5.0
    assert config.ego_initial_state[3] == pytest.approx(5.0)
    run_da_poc.apply_sfm_and_cruise_overrides(config, ego_target_speed=3.0)
    assert config.ego_target_speed == pytest.approx(3.0)
    assert config.ego_initial_state[3] == pytest.approx(3.0)


def test_cruise_override_does_not_raise_initial_speed():
    """A cruise target above the initial speed must not raise the initial v."""
    config = load_config("scenarios/scenario_01.yaml")  # init v = 5.0
    run_da_poc.apply_sfm_and_cruise_overrides(config, ego_target_speed=8.0)
    assert config.ego_target_speed == pytest.approx(8.0)
    assert config.ego_initial_state[3] == pytest.approx(5.0)  # unchanged (min)


# --------------------------------------------------------------------------- #
# (b) run_campaign integration round-trip (CV = no SGAN model needed; short cap)
# --------------------------------------------------------------------------- #
def test_run_campaign_caches_and_records_provenance(tmp_path, monkeypatch):
    campaign_kwargs = dict(
        scenarios=["scenarios/scenario_01.yaml"],
        conditions=[("cv_single", "cv", False, 0.0, 1.00)],
        seeds=[0],
        outdir=tmp_path,
        overrides={
            "ego_repulsion_sigma": 1.156,
            "ego_repulsion_v0": 1.681,
            "ego_target_speed": 3.0,
            "total_time": 2.0,  # keep the closed-loop sim short
        },
    )
    df, failed = run_da_poc.run_campaign(**campaign_kwargs)
    assert failed == 0
    assert len(df) == 1
    row = df.iloc[0]
    # Provenance round-trips through the per-seed JSON cache.
    assert row["ego_repulsion_sigma"] == pytest.approx(1.156)
    assert row["ego_repulsion_v0"] == pytest.approx(1.681)
    assert row["ego_target_speed"] == pytest.approx(3.0)
    assert "rms_jerk" in df.columns and "mean_accel" in df.columns
    # F4: the representative-selection mode is recorded -- scenario_01 sets
    # num_samples: 20, so "single" planning actually uses a medoid-of-20.
    assert row["single_mode"] == "medoid_of_20"
    cache = run_da_poc.cache_path(tmp_path, "scenarios/scenario_01.yaml",
                                  "cv_single", 0)
    assert cache.exists()

    # A second call must resume from cache, NOT re-run the (expensive) sim:
    # patch run_one to blow up so any recompute would fail the test.
    def _no_rerun(*a, **k):
        raise AssertionError("run_one called for an already-cached cell")
    monkeypatch.setattr(run_da_poc, "run_one", _no_rerun)
    df2, failed2 = run_da_poc.run_campaign(**campaign_kwargs)
    assert failed2 == 0
    assert len(df2) == 1
    assert df2.iloc[0]["ego_repulsion_sigma"] == pytest.approx(1.156)


# --------------------------------------------------------------------------- #
# (c) verdict logic
# --------------------------------------------------------------------------- #
def _margin_df(spec, scenarios=("scenario_01", "scenario_02"), seeds=3):
    """spec: dict[condition] -> (min_dist, time)."""
    rows = []
    for sc in scenarios:
        for cond, (md, t) in spec.items():
            for s in range(seeds):
                rows.append(dict(scenario=sc, condition=cond, min_dist_m=md,
                                 time_s=t, collision_count=0, rms_jerk=0.1,
                                 mean_accel=0.5))
    return pd.DataFrame(rows)


def test_margin_verdict_holds_when_robust_dominates():
    # robust: highest MinDist, lowest Time -> no inflation can dominate it.
    spec = {
        "sgan_single_inf1.00": (1.0, 20), "sgan_single_inf1.10": (1.1, 21),
        "sgan_single_inf1.20": (1.2, 22), "sgan_single_inf1.35": (1.3, 24),
        "sgan_single_inf1.50": (1.4, 26), "sgan_robust_eps0.0": (1.8, 19),
    }
    v = rq1b.margin_verdict(_margin_df(spec))
    assert v["robust_gain_holds"] is True
    assert v["dominating_inflations"] == []


def test_margin_verdict_fails_when_an_inflation_dominates_everywhere():
    # inf1.50 has higher MinDist AND lower Time than robust in every scenario.
    spec = {
        "sgan_single_inf1.00": (1.0, 25), "sgan_single_inf1.10": (1.1, 25),
        "sgan_single_inf1.20": (1.2, 24), "sgan_single_inf1.35": (1.4, 22),
        "sgan_single_inf1.50": (2.0, 18), "sgan_robust_eps0.0": (1.5, 20),
    }
    v = rq1b.margin_verdict(_margin_df(spec))
    assert v["robust_gain_holds"] is False
    assert "sgan_single_inf1.50" in v["dominating_inflations"]


def test_margin_verdict_fails_when_robust_itself_collides():
    """A robust planner that collides cannot carry the robust-gain claim, even
    when no inflation dominates it. In a scenario where robust collides on every
    seed, its collision-free time mean is NaN, which vacuously blocks every
    inflation from dominating -> dominating_inflations is empty, yet the verdict
    must be False via the robust-collision safety guard, NOT via domination."""
    # robust wins the MinDist/Time trade-off (highest dist, lowest time) so no
    # inflation dominates -- but robust collides on every seed of scenario_01.
    spec = {
        "sgan_single_inf1.00": (1.0, 20), "sgan_single_inf1.10": (1.1, 21),
        "sgan_single_inf1.20": (1.2, 22), "sgan_single_inf1.35": (1.3, 24),
        "sgan_single_inf1.50": (1.4, 26), "sgan_robust_eps0.0": (1.8, 19),
    }
    rows = []
    for sc in ("scenario_01", "scenario_02"):
        for cond, (md, t) in spec.items():
            for s in range(3):
                coll = int(cond == "sgan_robust_eps0.0" and sc == "scenario_01")
                rows.append(dict(scenario=sc, condition=cond, min_dist_m=md,
                                 time_s=t, collision_count=coll,
                                 rms_jerk=0.1, mean_accel=0.5))
    v = rq1b.margin_verdict(pd.DataFrame(rows))
    assert v["robust_total_collisions"] == 3
    # No inflation dominates robust (robust is Pareto-best on the trade-off)...
    assert v["dominating_inflations"] == []
    # ...so robust_gain_holds is False ONLY because robust itself is unsafe.
    assert v["robust_gain_holds"] is False


def test_rand_verdict_detects_cv_and_lstm_danger():
    # Strong, Fisher-significant danger signal: single collides on every run,
    # the matched robust on none, over 8 seeds -> p well below 0.05.
    rows = []
    coll = {"cv_single": 8, "lstm_single": 7, "lstm_robust_eps0.0": 0,
            "sgan_single_inf1.00": 2, "sgan_robust_eps0.0": 0}
    for cond, c in coll.items():
        for s in range(8):
            rows.append(dict(scenario="scenario_01", condition=cond,
                             min_dist_m=1.0, time_s=20,
                             collision_count=(1 if s < c else 0)))
    v = rq1b.rand_verdict(pd.DataFrame(rows))
    assert v["cv_danger_holds"] is True       # cv 8/8 > sgan robust 0/8, significant
    assert v["cv_fisher_p"] < 0.05
    assert v["cv_danger_undetermined"] is False
    assert v["lstm_danger_holds"] is True      # lstm 7/8 > lstm robust 0/8, significant


def test_rand_verdict_undetermined_when_subthreshold():
    """A directional but non-significant margin (2-vs-0 over 5 seeds) must be
    reported as undetermined, NOT as the danger claim holding (review M8: a
    single-digit flip is indistinguishable from Monte-Carlo noise)."""
    rows = []
    coll = {"cv_single": 2, "lstm_single": 0, "lstm_robust_eps0.0": 0,
            "sgan_single_inf1.00": 0, "sgan_robust_eps0.0": 0}
    for cond, c in coll.items():
        for s in range(5):
            rows.append(dict(scenario="scenario_01", condition=cond,
                             min_dist_m=1.0, time_s=20,
                             collision_count=(1 if s < c else 0)))
    v = rq1b.rand_verdict(pd.DataFrame(rows))
    assert v["cv_danger_direction"] is True    # cv 2 > robust 0 (direction holds)
    assert v["cv_danger_holds"] is False        # ...but not significant
    assert v["cv_danger_undetermined"] is True
    assert v["cv_fisher_p"] > 0.05


def test_rand_verdict_negative_when_robust_not_safer():
    rows = []
    coll = {"cv_single": 0, "lstm_single": 0, "lstm_robust_eps0.0": 2,
            "sgan_single_inf1.00": 0, "sgan_robust_eps0.0": 1}
    for cond, c in coll.items():
        for s in range(5):
            rows.append(dict(scenario="scenario_01", condition=cond,
                             min_dist_m=1.0, time_s=20,
                             collision_count=(1 if s < c else 0)))
    v = rq1b.rand_verdict(pd.DataFrame(rows))
    assert v["cv_danger_holds"] is False       # cv 0 !> robust 1
    assert v["cv_danger_direction"] is False
    assert v["cv_danger_undetermined"] is False
    assert v["lstm_danger_holds"] is False      # lstm single 0 !> lstm robust 2


def test_rand_scenario_rows_classifies_per_scenario():
    """Per-scenario classification must separate genuine single-danger from
    GT-artifact (robust also collides) and no-conflict."""
    def _rand(scenario, coll):
        rows = []
        for cond, c in coll.items():
            for s in range(10):
                rows.append(dict(campaign="rand", gt_label="calib",
                                 scenario=scenario, condition=cond,
                                 collision_count=(1 if s < c else 0)))
        return rows

    rows = []
    # S2: single collides, robust clean -> single-danger
    rows += _rand("scenario_02", {"cv_single": 0, "lstm_single": 3,
                                  "sgan_single_inf1.00": 6,
                                  "lstm_robust_eps0.0": 0,
                                  "sgan_robust_eps0.0": 0})
    # S3: single >> robust > 0 -> mixed (claim-2 direction holds, robust not clean)
    rows += _rand("scenario_03", {"cv_single": 1, "lstm_single": 3,
                                  "sgan_single_inf1.00": 3,
                                  "lstm_robust_eps0.0": 1,
                                  "sgan_robust_eps0.0": 1})
    # S1: nobody collides -> no-conflict
    rows += _rand("scenario_01", {"cv_single": 0, "lstm_single": 0,
                                  "sgan_single_inf1.00": 0,
                                  "lstm_robust_eps0.0": 0,
                                  "sgan_robust_eps0.0": 0})
    # extra: robust >= single > 0 -> GT-artifact (no discrimination)
    rows += _rand("scenario_99", {"cv_single": 1, "lstm_single": 0,
                                  "sgan_single_inf1.00": 1,
                                  "lstm_robust_eps0.0": 1,
                                  "sgan_robust_eps0.0": 2})
    tbl = rq1b.rand_scenario_rows(pd.DataFrame(rows))
    klass = tbl.set_index("scenario")["class"].to_dict()
    assert klass["scenario_02"] == "single-danger"
    assert klass["scenario_03"] == "mixed"
    assert klass["scenario_01"] == "no-conflict"
    assert klass["scenario_99"] == "GT-artifact"


def test_rand_scenario_rows_reports_fisher_significance():
    """Per-scenario single-vs-robust run-level Fisher (M8): an S2-like cell where
    the single planners collide on several runs and both robust planners stay
    clean must be flagged significant (p<0.05) and classed single-danger."""
    rows = []
    coll = {"cv_single": 3, "lstm_single": 3, "sgan_single_inf1.00": 3,
            "lstm_robust_eps0.0": 0, "sgan_robust_eps0.0": 0}
    for cond, c in coll.items():
        for s in range(20):
            rows.append(dict(campaign="rand", gt_label="avec",
                             scenario="scenario_02", condition=cond,
                             collision_count=(1 if s < c else 0)))
    tbl = rq1b.rand_scenario_rows(pd.DataFrame(rows))
    row = tbl.set_index("scenario").loc["scenario_02"]
    assert row["single_collided_runs"] == 9 and row["single_n"] == 60
    assert row["robust_collided_runs"] == 0 and row["robust_n"] == 40
    assert row["fisher_p"] < 0.05           # 9/60 vs 0/40 ~ p=0.008
    assert row["class"] == "single-danger"


def test_scenario_narrative_is_data_driven_over_all_gts():
    """The per-scenario reading must be generated from the table for every GT
    present and must not ship the old hand-written 2-GT ("両 GT") prose that
    could contradict a 4-GT table (review M9)."""
    rows = []
    for gt in ["avec", "calib", "calib_lo", "calib_hi"]:
        for sc in ["scenario_01", "scenario_02"]:
            coll = {"cv_single": 1, "lstm_single": 0, "sgan_single_inf1.00": 0,
                    "lstm_robust_eps0.0": 0, "sgan_robust_eps0.0": 0}
            for cond, c in coll.items():
                for s in range(10):
                    rows.append(dict(campaign="rand", gt_label=gt, scenario=sc,
                                     condition=cond,
                                     collision_count=(1 if s < c else 0)))
    srows = rq1b.rand_scenario_rows(pd.DataFrame(rows))
    text = "\n".join(rq1b._scenario_narrative(srows))
    for gt in ["avec", "calib", "calib_lo", "calib_hi"]:
        assert gt in text                   # every present GT is covered
    assert "両 GT" not in text and "両GT" not in text
    assert "感度分析" in text                # M7 circularity caveat present


def test_scenario_narrative_covers_gt_outside_canonical_order():
    """A GT label absent from _GT_ORDER (e.g. a new GT_CORE entry or a custom
    --report-only CSV) must still appear in the narrative, not just the table:
    otherwise the prose silently drops it while the table keeps it (fillna sort),
    reintroducing the very table/narrative divergence M9 removed."""
    rows = []
    for gt in ["avec", "custom_gt_xyz"]:    # custom_gt_xyz is NOT in _GT_ORDER
        coll = {"cv_single": 1, "lstm_single": 0, "sgan_single_inf1.00": 0,
                "lstm_robust_eps0.0": 0, "sgan_robust_eps0.0": 0}
        for cond, c in coll.items():
            for s in range(10):
                rows.append(dict(campaign="rand", gt_label=gt,
                                 scenario="scenario_01", condition=cond,
                                 collision_count=(1 if s < c else 0)))
    srows = rq1b.rand_scenario_rows(pd.DataFrame(rows))
    assert "custom_gt_xyz" in set(srows.gt_label)       # present in the table
    text = "\n".join(rq1b._scenario_narrative(srows))
    assert "custom_gt_xyz" in text                       # ...and in the prose


def _verdicts_df(rows):
    """Build a minimal verdicts-shaped frame for _sensitivity_status tests."""
    return pd.DataFrame(rows)


def test_sensitivity_status_power_limited_is_not_a_reversal():
    """Review (M8 follow-up): a danger column that is significant for one GT but
    merely *undetermined* (same direction, low-seed corner) for another must NOT
    be reported as a calibration-sensitivity reversal -- that is a detection-power
    artifact, not a direction flip."""
    verdicts = _verdicts_df([
        {"cv_danger_holds": True,  "cv_danger_undetermined": False},
        {"cv_danger_holds": False, "cv_danger_undetermined": True},
    ])
    status = rq1b._sensitivity_status(verdicts, "cv_danger_holds",
                                      "cv_danger_undetermined")
    assert "有意性が落ちる" in status and "反転" not in status.replace("反転ではない", "")


def test_sensitivity_status_genuine_reversal_and_invariant():
    """A real direction flip (one GT holds, another is a true negative) reads as
    a reversal; all-equal reads as robust; a None entry reads as undetermined."""
    reversal = _verdicts_df([
        {"cv_danger_holds": True,  "cv_danger_undetermined": False},
        {"cv_danger_holds": False, "cv_danger_undetermined": False},  # true negative
    ])
    assert rq1b._sensitivity_status(
        reversal, "cv_danger_holds", "cv_danger_undetermined") == "反転あり（較正に感度あり）"

    invariant = _verdicts_df([
        {"cv_danger_holds": True, "cv_danger_undetermined": False},
        {"cv_danger_holds": True, "cv_danger_undetermined": False},
    ])
    assert rq1b._sensitivity_status(
        invariant, "cv_danger_holds", "cv_danger_undetermined") == "全 GT で不変（頑健）"

    uncomputed = _verdicts_df([
        {"robust_gain_holds": True}, {"robust_gain_holds": None}])
    assert "未計算" in rq1b._sensitivity_status(uncomputed, "robust_gain_holds")

    # A 0-row frame must NOT read as "全 GT で不変（頑健）" (robustness over zero
    # GTs); it is undetermined.
    empty = pd.DataFrame({"robust_gain_holds": []})
    status = rq1b._sensitivity_status(empty, "robust_gain_holds")
    assert "判定不能" in status and "頑健" not in status


def test_build_verdicts_detects_gt_flip():
    """A verdict that flips between GT settings is the sensitivity signal."""
    holds = {
        "sgan_single_inf1.00": (1.0, 20), "sgan_single_inf1.10": (1.1, 21),
        "sgan_single_inf1.20": (1.2, 22), "sgan_single_inf1.35": (1.3, 24),
        "sgan_single_inf1.50": (1.4, 26), "sgan_robust_eps0.0": (1.8, 19),
    }
    fails = {
        "sgan_single_inf1.00": (1.0, 25), "sgan_single_inf1.10": (1.1, 25),
        "sgan_single_inf1.20": (1.2, 24), "sgan_single_inf1.35": (1.4, 22),
        "sgan_single_inf1.50": (2.0, 18), "sgan_robust_eps0.0": (1.5, 20),
    }
    a = _margin_df(holds); a["campaign"] = "margin"; a["gt_label"] = "avec"
    b = _margin_df(fails); b["campaign"] = "margin"; b["gt_label"] = "calib"
    master = pd.concat([a, b], ignore_index=True)
    verdicts = rq1b.build_verdicts(master, ["avec", "calib"])

    by_gt = verdicts.set_index("gt_label")["robust_gain_holds"].to_dict()
    assert by_gt["avec"] is True
    assert by_gt["calib"] is False
    # The conclusion is sensitive to the GT reaction model (a flip exists).
    assert len(set(verdicts["robust_gain_holds"])) > 1


# --------------------------------------------------------------------------- #
# (d) DEFAULT_SCENARIOS reproducibility guard (review I1)
# --------------------------------------------------------------------------- #
def test_default_scenarios_point_at_rq1b_variants():
    """Pin DEFAULT_SCENARIOS to the calibrated-domain rq1b/ variants.

    Commit 8524708 documented a silent-wrong-experiment bug where the default
    pointed at the base scenarios/scenario_0X.yaml instead of scenarios/rq1b/.
    A one-line revert would re-introduce it with every other test still green;
    this is the cheap guard that catches it.
    """
    import yaml

    expected = [
        "scenarios/rq1b/scenario_01.yaml",
        "scenarios/rq1b/scenario_02.yaml",
        "scenarios/rq1b/scenario_03.yaml",
    ]
    assert list(rq1b.DEFAULT_SCENARIOS) == expected
    for path in rq1b.DEFAULT_SCENARIOS:
        assert path.startswith("scenarios/rq1b/")
        p = Path(path)
        assert p.exists(), f"{path} missing"
        # Parseable YAML (a renamed/empty variant would corrupt the campaign).
        assert isinstance(yaml.safe_load(p.read_text()), dict)
    # Must NOT silently fall back to the base AVEC scenarios.
    assert not any(s.startswith("scenarios/scenario_") for s in rq1b.DEFAULT_SCENARIOS)


# --------------------------------------------------------------------------- #
# (e) 1.2-3: aggregate Fisher p-values must not bypass the ledger
# --------------------------------------------------------------------------- #
def test_rq1b_aggregate_tests_are_auxiliary_and_skip_unevaluable():
    """The aggregate cv/lstm Fisher cells become AUXILIARY sidecar entries:
    never canonical (auxiliary=True, headline=False), and unevaluable p
    (None/NaN from an empty arm) must not fabricate a hypothesis."""
    verdicts = pd.DataFrame([
        {"gt_label": "avec", "cv_fisher_p": 0.5, "lstm_fisher_p": 0.0287},
        {"gt_label": "calib", "cv_fisher_p": None, "lstm_fisher_p": float("nan")},
    ])
    tests = rq1b.rq1b_aggregate_tests(verdicts)

    assert [t["test_id"] for t in tests] == [
        "rq1b.rand.fisher_aggregate.avec.cv",
        "rq1b.rand.fisher_aggregate.avec.lstm",
    ]
    assert all(t["auxiliary"] is True and t["headline"] is False for t in tests)
    assert all(t["family"] == "rq1b_claim2_fisher_aggregate" for t in tests)
    assert all("noise-grade" in t["caveat"] for t in tests)
    assert tests[1]["p_value"] == pytest.approx(0.0287)

    assert rq1b.rq1b_aggregate_tests(pd.DataFrame()) == []
    assert rq1b.rq1b_aggregate_tests(None) == []


# --------------------------------------------------------------------------- #
# (f) F3: LOSO real-fold envelope arms
# --------------------------------------------------------------------------- #
def test_gt_loso_matches_real_loso_fold_points():
    """GT_LOSO must be the REAL folds_loso.csv points, not synthetic corners.

    F3 (novelty-reinforcement review): the LOSO folds land outside the +/-1SD
    box the campaign sweeps, so the envelope arms must anchor to the actual
    fitted fold points -- a drifting constant here would silently sweep a point
    no calibration ever reached.
    """
    folds = pd.read_csv("outputs/rq2_evaluation/folds_loso.csv").set_index("fold")
    by_label = {g["label"]: g for g in rq1b.GT_LOSO}
    assert set(by_label) == {"calib_loso_vmax", "calib_loso_smin"}

    vmax = by_label["calib_loso_vmax"]
    assert vmax["sigma"] == pytest.approx(folds.loc["vci_back", "sigma"], abs=5e-4)
    assert vmax["v0"] == pytest.approx(folds.loc["vci_back", "v0"], abs=5e-4)

    smin = by_label["calib_loso_smin"]
    assert smin["sigma"] == pytest.approx(folds.loc["vci_lat_bi", "sigma"], abs=5e-4)
    assert smin["v0"] == pytest.approx(folds.loc["vci_lat_bi", "v0"], abs=5e-4)

    # The whole point of F3: both arms lie OUTSIDE the +/-1SD box.
    box_sigma, box_v0 = (1.040, 1.272), (1.542, 1.820)
    assert vmax["v0"] > box_v0[1]
    assert smin["sigma"] < box_sigma[0]


def test_gt_loso_labels_seeds_and_display_order():
    """LOSO arm labels must be distinct cache subdirectories, listed in the
    canonical display order, and budgeted as corner (robustness) arms."""
    core_labels = [g["label"] for g in rq1b.GT_CORE + rq1b.GT_OFFDIAG]
    loso_labels = [g["label"] for g in rq1b.GT_LOSO]
    assert not set(core_labels) & set(loso_labels)
    # New labels = new cache subdirectories; they must also be in the canonical
    # display order so tables/narrative stay deterministic.
    for lbl in loso_labels:
        assert lbl in rq1b._GT_ORDER
    # Envelope arms are robustness checks: corner seed budget, corner tier.
    for lbl in loso_labels:
        assert list(rq1b._seeds_for(lbl, 20, 10)) == list(range(10))


def test_include_loso_defaults_on_and_opt_out_parses(tmp_path, monkeypatch):
    """The committed outputs/rq1b artifacts include the LOSO arms, so the
    DEFAULT invocation must include them (reproducibility contract, same class
    as the DEFAULT_SCENARIOS guard above): a plain --report-only that silently
    dropped LOSO would regress the committed verdicts/REPORT/sidecar. The
    opt-out spelling --no-include-loso must also parse (pre-F3 comparisons).
    """
    import sys as _sys

    captured = {}
    real_build = rq1b.build_verdicts

    def _spy(master, gt_labels):
        captured["gt_labels"] = list(gt_labels)
        return real_build(master, gt_labels)

    monkeypatch.setattr(rq1b, "build_verdicts", _spy)
    # A minimal one-cell cache so main() reaches the verdict stage.
    seed = dict(scenario="scenario_01", condition="sgan_single_inf1.00",
                method="sgan", distribution_aware=False, epsilon=0.0,
                inflation=1.0, seed=0, time_s=10.0, speed_ms=1.0,
                min_dist_m=2.0, min_ttc_s=1.0, collision_count=0,
                ade=0.5, fde=1.0, rms_jerk=0.1, mean_accel=0.5)
    cpath = tmp_path / "margin" / "avec" / "runs" / "scenario_01" / \
        "sgan_single_inf1.00" / "seed_00.json"
    cpath.parent.mkdir(parents=True)
    cpath.write_text(pd.Series(seed).to_json())

    monkeypatch.setattr(_sys, "argv",
                        ["rq1b", "--report-only", "--root", str(tmp_path)])
    rq1b.main()
    loso_labels = {g["label"] for g in rq1b.GT_LOSO}
    assert loso_labels <= set(captured["gt_labels"])  # default INCLUDES loso

    monkeypatch.setattr(_sys, "argv",
                        ["rq1b", "--report-only", "--no-include-loso",
                         "--root", str(tmp_path)])
    rq1b.main()
    assert not loso_labels & set(captured["gt_labels"])  # opt-out works
