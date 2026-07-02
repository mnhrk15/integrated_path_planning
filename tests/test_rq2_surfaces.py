"""Tests for the identifiability-audit surfaces (profile_band + sigma profile).

profile_band is the single source of the identifiability numbers quoted in the
REPORT (band widths, censoring), so its geometry handling is pinned on small
hand-built surfaces where the correct answer is obvious.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from examples.plot_rq2_loss_surface import (
    load_surface,
    plot_sigma_profile,
    profile_band,
)
from examples.run_rq2_surfaces import make_surface_objective, ylabel_for
from src.simulation.calibration_harness import (
    objective_multi,
    objective_rollout_ade,
)
from tests.test_calibration_harness import make_encounter


def _surf(loss, grid_sigma, grid_v0, sigma, v0):
    return {
        "grid_sigma": np.asarray(grid_sigma, dtype=float),
        "grid_v0": np.asarray(grid_v0, dtype=float),
        "loss": np.ma.masked_invalid(np.asarray(loss, dtype=float)),
        "sigma": float(sigma), "v0": float(v0),
        "grid_best": (float(sigma), float(v0)),
    }


def test_profile_band_width_on_a_clear_valley():
    # v0 profile at sigma=1.0 (middle row): min 0.50 at v0=1; the 2% band
    # (<=0.51) also contains v0=2 (0.505) -> band [1, 2], 2 nodes, uncensored.
    loss = [[2.0, 1.5, 1.6, 1.7, 2.0],
            [1.0, 0.50, 0.505, 0.60, 1.0],
            [2.0, 1.5, 1.6, 1.7, 2.0]]
    surf = _surf(loss, [0.5, 1.0, 1.5], [0, 1, 2, 3, 4], sigma=1.0, v0=1.0)
    band = profile_band(surf, "v0")
    assert band["band_lo"] == 1.0 and band["band_hi"] == 2.0
    assert band["band_width"] == 1.0
    assert band["n_nodes_in_band"] == 2
    assert not band["censored_lo"] and not band["censored_hi"]
    assert not band["fitted_on_grid_edge"]
    assert band["min_loss"] == 0.50
    # sigma profile at v0=1 (second column): [1.5, 0.5, 1.5] -> single node band.
    band_s = profile_band(surf, "sigma")
    assert band_s["band_lo"] == band_s["band_hi"] == 1.0
    assert band_s["band_width"] == 0.0
    assert band_s["n_nodes_in_band"] == 1


def test_profile_band_censoring_at_grid_edge():
    # Monotone profile: the band hugs the low edge -> censored_lo, and a fitted
    # v0 on the edge is flagged.
    loss = [[0.50, 0.505, 0.7, 1.0, 1.5]]
    surf = _surf(loss, [1.0], [0, 1, 2, 3, 4], sigma=1.0, v0=0.0)
    band = profile_band(surf, "v0")
    assert band["censored_lo"] and not band["censored_hi"]
    assert band["band_lo"] == 0.0 and band["band_hi"] == 1.0
    assert band["fitted_on_grid_edge"]  # fitted v0 == grid lower edge


def test_profile_band_degenerate_when_all_masked():
    loss = np.full((2, 3), np.nan)
    surf = _surf(loss, [0.5, 1.0], [0, 1, 2], sigma=0.5, v0=1.0)
    band = profile_band(surf, "v0")
    assert band["degenerate"]
    assert np.isnan(band["band_width"])
    with pytest.raises(ValueError, match="axis"):
        profile_band(surf, "tau")


def test_profile_band_nan_fitted_is_degenerate_not_plausible():
    """A NaN fitted optimum silently selected slice 0 (argmin over all-NaN
    distances) and produced a plausible-looking row, defeating the downstream
    other-axis-edge guard (review finding). It must be degenerate."""
    loss = [[1.0, 0.5, 0.7], [0.9, 0.45, 0.65]]
    surf = _surf(loss, [0.5, 1.0], [0, 2, 4], sigma=float("nan"), v0=2.0)
    for axis in ("v0", "sigma"):
        band = profile_band(surf, axis)
        assert band["degenerate"], axis
        assert np.isnan(band["band_width"])


def test_profile_band_flags_noncontiguous_bimodal_band():
    """Two separated 2%-lobes: the hull [0, 4] is NOT a flat region; the
    contiguity flag must expose it (review finding: the width alone reads as
    'flat over 4 units')."""
    loss = [[0.50, 2.0, 2.0, 2.0, 0.505]]
    surf = _surf(loss, [1.0], [0, 1, 2, 3, 4], sigma=1.0, v0=0.0)
    band = profile_band(surf, "v0")
    assert band["n_nodes_in_band"] == 2
    assert band["band_width"] == 4.0
    assert band["band_contiguous"] is False
    # ... while an unbroken run is contiguous
    loss2 = [[0.50, 0.505, 2.0, 2.0, 2.0]]
    surf2 = _surf(loss2, [1.0], [0, 1, 2, 3, 4], sigma=1.0, v0=0.0)
    assert profile_band(surf2, "v0")["band_contiguous"] is True


def test_merge_identifiability_keeps_other_combos():
    """A partial surfaces re-run must replace only the re-evaluated combos,
    never wipe the other policies' ade reference rows (review finding)."""
    import pandas as pd
    from examples.run_rq2_surfaces import IDENT_COLUMNS, merge_identifiability

    def row(obj, pol, axis, width):
        r = {c: float("nan") for c in IDENT_COLUMNS}
        r.update({"objective": obj, "policy": pol, "axis": axis,
                  "band_width": width})
        return r

    existing = pd.DataFrame([row("ade", "median", "v0", 1.6),
                             row("w1", "median", "v0", 1.5)],
                            columns=IDENT_COLUMNS)
    merged = merge_identifiability(existing, [row("w1", "median", "v0", 0.9),
                                              row("w1", "median", "sigma", 0.2)])
    key = merged.set_index(["objective", "policy", "axis"])["band_width"]
    assert key[("ade", "median", "v0")] == 1.6      # kept
    assert key[("w1", "median", "v0")] == 0.9       # replaced
    assert key[("w1", "median", "sigma")] == 0.2    # added
    assert len(merged) == 3


def test_surface_npz_roundtrips_through_load_surface(tmp_path):
    """The npz written by run_rq2_surfaces must satisfy load_surface's contract
    (same 6 keys as run_rq2_calibration; combo metadata lives in the filename)."""
    p = tmp_path / "loss_surface__ade__median.npz"
    np.savez(p, grid_sigma=np.array([0.5, 1.0]), grid_v0=np.array([0.0, 2.0]),
             grid_loss=np.array([[1.0, np.inf], [0.5, 0.7]]),
             sigma=1.0, v0=0.0, grid_best=np.array([1.0, 0.0]))
    surf = load_surface(p)
    assert surf["sigma"] == 1.0 and surf["v0"] == 0.0
    assert surf["loss"].mask[0, 1]  # inf cell masked
    assert profile_band(surf, "sigma")["n_nodes_total"] == 2


def test_plot_sigma_profile_renders_headless():
    loss = [[1.0, 0.5, 0.7], [0.9, 0.45, 0.65], [1.1, 0.6, 0.8]]
    surf = _surf(loss, [0.5, 1.0, 1.5], [0, 2, 4], sigma=1.0, v0=2.0)
    fig, ax = plt.subplots()
    plot_sigma_profile(ax, surf, ylabel="test loss [m]")
    assert ax.get_ylabel() == "test loss [m]"
    plt.close(fig)


def test_make_surface_objective_ade_alias_is_canonical():
    """'ade' and the distmatch 'w0' config must both be bit-identical to
    objective_rollout_ade (the alias is a labelling convenience, not a fork)."""
    enc = make_encounter(T=12)
    obj_ade = make_surface_objective("ade", [enc], "emd", "median", 1.3)
    obj_w0 = make_surface_objective("w0", [enc], "emd", "median", 1.3)
    ref = objective_rollout_ade([enc], 0.7, 3.5, cap_policy="median")
    assert obj_ade(0.7, 3.5) == ref
    assert obj_w0(0.7, 3.5) == ref
    assert ref == objective_multi([enc], 0.7, 3.5, w_ade=1.0, w_dist=0.0,
                                  cap_policy="median")


def test_ylabel_for_reads_config():
    assert ylabel_for("ade", "emd") == "rollout ADE [m]"
    assert "EMD" in ylabel_for("w1", "emd")
    assert ylabel_for("pure", "emd").startswith("closest-approach EMD")
