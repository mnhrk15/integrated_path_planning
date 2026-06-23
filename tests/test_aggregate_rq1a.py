"""Tests for the RQ1a cross-scene aggregation (review R1).

Pins the aggregation arithmetic on a tiny hand-computable synthetic table so the
RQ1a headline is reproducible and the documented aggregation-sensitivity (eth
unit confound, ordering flip) cannot silently change.
"""
import numpy as np
import pandas as pd
import pytest

from examples.aggregate_rq1a import per_scene_means, cross_scene, aggregate_all


def _df() -> pd.DataFrame:
    """2 scenes x 2 methods. cv is deterministic (1 seed/scene), lstm has 2 seeds.
    eth has 10 trajectories/scene, zara1 has 30 (so weighting matters)."""
    rows = [
        dict(scene="eth", method="cv", seed=0, n_trajectories=10,
             ade=1.0, fde=2.0, ade_per_agent=1.0, fde_per_agent=2.0, nll=np.nan),
        dict(scene="zara1", method="cv", seed=0, n_trajectories=30,
             ade=0.2, fde=0.4, ade_per_agent=0.2, fde_per_agent=0.4, nll=np.nan),
        dict(scene="eth", method="lstm", seed=0, n_trajectories=10,
             ade=0.6, fde=1.0, ade_per_agent=0.5, fde_per_agent=0.9, nll=2.0),
        dict(scene="eth", method="lstm", seed=1, n_trajectories=10,
             ade=0.8, fde=1.2, ade_per_agent=0.7, fde_per_agent=1.1, nll=2.0),
        dict(scene="zara1", method="lstm", seed=0, n_trajectories=30,
             ade=0.4, fde=0.6, ade_per_agent=0.3, fde_per_agent=0.5, nll=1.0),
        dict(scene="zara1", method="lstm", seed=1, n_trajectories=30,
             ade=0.4, fde=0.6, ade_per_agent=0.3, fde_per_agent=0.5, nll=1.0),
    ]
    return pd.DataFrame(rows)


def test_per_scene_means_over_variable_seed_counts():
    ps = per_scene_means(_df(), "ade")
    d = {(r.method, r.scene): r.value for r in ps.itertuples()}
    assert d[("cv", "eth")] == pytest.approx(1.0)
    assert d[("lstm", "eth")] == pytest.approx(0.7)   # mean(0.6, 0.8)
    assert d[("lstm", "zara1")] == pytest.approx(0.4)


def test_cross_scene_unweighted_and_trajectory_weighted():
    df = _df()
    unw = cross_scene(df, "ade", weighted=False)
    assert unw["cv"] == pytest.approx(0.6)     # mean(1.0, 0.2)
    assert unw["lstm"] == pytest.approx(0.55)  # mean(0.7, 0.4)

    wtd = cross_scene(df, "ade", weighted=True)
    assert wtd["cv"] == pytest.approx((1.0 * 10 + 0.2 * 30) / 40)    # 0.40
    assert wtd["lstm"] == pytest.approx((0.7 * 10 + 0.4 * 30) / 40)  # 0.475


def test_dropping_eth_flips_the_cross_scene_ordering():
    df = _df()
    full = cross_scene(df, "ade", drop_eth=False, weighted=False)
    no_eth = cross_scene(df, "ade", drop_eth=True, weighted=False)
    assert full["cv"] > full["lstm"]          # equal-weighted with eth: cv worst
    assert no_eth["cv"] < no_eth["lstm"]      # drop eth: cv best (the R1 flip)
    assert no_eth["cv"] == pytest.approx(0.2)
    assert no_eth["lstm"] == pytest.approx(0.4)


def test_all_nan_metric_aggregates_to_nan_not_error():
    nll = cross_scene(_df(), "nll", weighted=False)
    assert np.isnan(nll["cv"])                 # cv NLL is all-NaN
    assert nll["lstm"] == pytest.approx(1.5)   # mean(eth 2.0, zara1 1.0)


def test_aggregate_all_tidy_schema_and_membership():
    tidy = aggregate_all(_df(), metrics=("ade",))
    assert set(tidy.columns) == {"metric", "aggregation", "method", "value"}
    aggs = set(tidy["aggregation"])
    assert {"unweighted_5scene", "weighted_5scene", "unweighted_no_eth",
            "weighted_no_eth", "scene:eth", "scene:zara1"} <= aggs
