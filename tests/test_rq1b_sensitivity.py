"""Tests for the RQ1b sensitivity analysis harness.

Covers (a) the GT/cruise override plumbing added to run_da_poc (calibrated SFM
ego-repulsion injection + cruise clamp, without dropping scenario-level keys),
(b) a minimal run_campaign integration round-trip, and (c) the RQ1b verdict
logic (robust-gain / CV-danger and GT-sensitivity flip detection).
"""
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


def test_rand_verdict_detects_cv_and_lstm_danger():
    rows = []
    coll = {"cv_single": 5, "lstm_single": 3, "lstm_robust_eps0.0": 1,
            "sgan_single_inf1.00": 2, "sgan_robust_eps0.0": 0}
    for cond, c in coll.items():
        for s in range(5):
            rows.append(dict(scenario="scenario_01", condition=cond,
                             min_dist_m=1.0, time_s=20,
                             collision_count=(1 if s < c else 0)))
    v = rq1b.rand_verdict(pd.DataFrame(rows))
    assert v["cv_danger_holds"] is True       # cv 5 > sgan robust 0
    assert v["lstm_danger_holds"] is True      # lstm single 3 > lstm robust 1


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
