#!/usr/bin/env python3
"""Render the RQ2 (sigma, v0) calibration loss surfaces (the thesis ridge figure).

Reads the ``loss_surface_{scenario}.npz`` files written by run_rq2_calibration.py
and draws the rollout-ADE loss over the (sigma, v0) grid as a heatmap with the
identifiability ridge visible, marking the calibrated (refined) optimum and the
AVEC paper's hand-tuned (0.7, 3.5) for contrast. Two outputs:

* ``rq2_loss_surface_all.png`` -- the pooled-CITR surface (paper body).
* ``rq2_loss_surface_grid.png`` -- the four CITR scenarios in a 2x2 grid
  (front / back / lateral-bi / lateral-uni), for the per-geometry stability
  discussion (appendix). Scenarios whose npz is absent are skipped with a warning.

The grid is coarse (6x7), so a pcolormesh (cell-centred) with a thin contour
overlay reads better than a contourf. Non-finite (degenerate / no-sample) cells
are masked so they do not blow up the colour scale.

Usage:
    .venv/bin/python examples/plot_rq2_loss_surface.py --inputs outputs/rq2_calibration
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

logging.getLogger("matplotlib").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figs"

AVEC_DEFAULT = (0.7, 3.5)  # paper's hand-tuned (sigma, v0)

# "Indistinguishable" band: profile values within this relative margin of the
# profile minimum (review C2). Shared by the v0/sigma profile plots and the
# identifiability CSV (profile_band) so figure and table always agree.
BAND_REL = 0.02

# CITR vehicle scenarios for the 2x2 grid, with display titles.
SCENARIO_TITLES = {
    "vci_front": "front",
    "vci_back": "back",
    "vci_lat_bi": "lateral (bi-dir.)",
    "vci_lat_uni": "lateral (uni-dir.)",
}


def load_surface(npz_path: Path) -> Dict[str, object]:
    """Load one loss-surface npz into a dict with the grid loss masked.

    Returns ``grid_sigma`` [S], ``grid_v0`` [V], ``loss`` (a masked [S, V] array
    with non-finite cells masked), the refined optimum ``(sigma, v0)`` and the
    on-grid minimum ``grid_best`` (NaN-filled tuple if the npz predates it).
    """
    data = np.load(npz_path, allow_pickle=False)
    grid_loss = np.asarray(data["grid_loss"], dtype=float)  # [S, V]
    return {
        "grid_sigma": np.asarray(data["grid_sigma"], dtype=float),
        "grid_v0": np.asarray(data["grid_v0"], dtype=float),
        "loss": np.ma.masked_invalid(grid_loss),  # mask inf/NaN cells
        "sigma": float(data["sigma"]),
        "v0": float(data["v0"]),
        "grid_best": (tuple(np.asarray(data["grid_best"], dtype=float))
                      if "grid_best" in data else (float("nan"), float("nan"))),
    }


def plot_single(ax, surf: Dict[str, object], title: str, use_log: bool = False):
    """Draw one (sigma, v0) loss surface onto ``ax``; return the mappable.

    x-axis = sigma, y-axis = v0 (the (sigma, v0) notation order). ``loss`` is
    [S=sigma, V=v0], so it is transposed for pcolormesh, whose C is [rows=y, cols=x].
    """
    gs, gv = surf["grid_sigma"], surf["grid_v0"]
    loss = surf["loss"]  # masked [S, V]
    if loss.count() == 0:  # every cell masked -> nothing to draw
        ax.text(0.5, 0.5, "all cells degenerate", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return None

    norm = None
    if use_log:
        vmin = float(loss.min())
        if vmin > 0:
            norm = LogNorm(vmin=vmin, vmax=float(loss.max()))
    mesh = ax.pcolormesh(gs, gv, loss.T, shading="nearest", cmap="viridis", norm=norm)
    # Thin contour overlay for the ridge shape (skip if too few finite levels).
    if loss.count() >= 4:
        ax.contour(gs, gv, loss.T, levels=8, colors="w", alpha=0.35, linewidths=0.5)

    # calibrated (refined) optimum and the AVEC default for contrast.
    ax.plot(surf["sigma"], surf["v0"], "o", mfc="white", mec="black", mew=1.5,
            ms=9, label=f"calibrated ({surf['sigma']:.2f}, {surf['v0']:.2f})")
    ax.plot(*AVEC_DEFAULT, "*", color="red", ms=13, mec="black", mew=0.6,
            label=f"AVEC default ({AVEC_DEFAULT[0]}, {AVEC_DEFAULT[1]})")

    ax.set_xlabel(r"$\sigma$ [m]")
    ax.set_ylabel(r"$v_0$ [m/s$^2$]")
    ax.set_title(title)
    return mesh


def plot_v0_profile(ax, surf: Dict[str, object],
                    ylabel: str = "rollout ADE [m]") -> None:
    """1-D ADE-vs-v0 profile at the grid sigma nearest the fitted optimum.

    Makes the review-C2 identifiability explicit: along the ridge the rollout ADE
    is nearly flat between the calibrated v0 (~1.7) and the AVEC hand-tuned v0=3.5,
    so the data CANNOT distinguish them (the '<2% ADE' band). Shades the v0 span
    whose ADE is within 2% of the profile minimum. ``ylabel`` overrides the axis
    label when the surface holds a non-ADE objective (the multi-objective sweeps).
    """
    gs, gv = surf["grid_sigma"], surf["grid_v0"]
    loss = surf["loss"]  # masked [S, V]
    si = int(np.argmin(np.abs(gs - surf["sigma"])))
    prof = loss[si]  # masked [V]
    finite = ~np.ma.getmaskarray(prof)
    if finite.sum() < 2:
        ax.text(0.5, 0.5, "profile degenerate", ha="center", va="center",
                transform=ax.transAxes)
        return
    v = gv[finite]
    a = np.asarray(prof[finite])
    amin = float(a.min())
    band = a <= amin * (1.0 + BAND_REL)  # within 2% of min -> indistinguishable
    ax.plot(v, a, "-o", color="#1f77b4", ms=4)
    if band.any():
        ax.axhspan(amin, amin * (1.0 + BAND_REL), color="orange", alpha=0.18,
                   label="within 2% of min (indistinguishable)")
        ax.axvspan(float(v[band].min()), float(v[band].max()),
                   color="orange", alpha=0.10)
    ax.axvline(surf["v0"], color="black", ls="--", lw=1.2,
               label=f"calibrated v0={surf['v0']:.2f}")
    ax.axvline(AVEC_DEFAULT[1], color="red", ls=":", lw=1.5,
               label=f"AVEC v0={AVEC_DEFAULT[1]}")
    ax.set_xlabel(r"$v_0$ [m/s$^2$]")
    ax.set_ylabel(ylabel)
    # Title kept verbatim from the pre-sigma-profile version so re-rendering the
    # committed figs/rq2_v0_profile_all.png stays unchanged ("ADE" reads wrong
    # for a multi-objective surface, but the ylabel carries the objective there).
    ax.set_title(rf"RQ2 ADE vs $v_0$ at $\sigma$={gs[si]:.2f} (identifiability)")
    ax.legend(fontsize=8, framealpha=0.85)


def plot_sigma_profile(ax, surf: Dict[str, object],
                       ylabel: str = "rollout ADE [m]") -> None:
    """1-D loss-vs-sigma profile at the grid v0 nearest the fitted optimum.

    The sigma-axis counterpart of :func:`plot_v0_profile` (review 1.2-2: the S2
    discriminating cell is driven by the repulsion RANGE sigma, so sigma-side
    identifiability must be shown, not only the v0 side). Shades the sigma span
    within ``BAND_REL`` of the profile minimum.
    """
    gs, gv = surf["grid_sigma"], surf["grid_v0"]
    loss = surf["loss"]  # masked [S, V]
    vi = int(np.argmin(np.abs(gv - surf["v0"])))
    prof = loss[:, vi]  # masked [S]
    finite = ~np.ma.getmaskarray(prof)
    if finite.sum() < 2:
        ax.text(0.5, 0.5, "profile degenerate", ha="center", va="center",
                transform=ax.transAxes)
        return
    s = gs[finite]
    a = np.asarray(prof[finite])
    amin = float(a.min())
    band = a <= amin * (1.0 + BAND_REL)
    ax.plot(s, a, "-o", color="#1f77b4", ms=4)
    if band.any():
        ax.axhspan(amin, amin * (1.0 + BAND_REL), color="orange", alpha=0.18,
                   label="within 2% of min (indistinguishable)")
        ax.axvspan(float(s[band].min()), float(s[band].max()),
                   color="orange", alpha=0.10)
    ax.axvline(surf["sigma"], color="black", ls="--", lw=1.2,
               label=f"calibrated sigma={surf['sigma']:.2f}")
    ax.axvline(AVEC_DEFAULT[0], color="red", ls=":", lw=1.5,
               label=f"AVEC sigma={AVEC_DEFAULT[0]}")
    ax.set_xlabel(r"$\sigma$ [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"loss vs $\sigma$ at $v_0$={gv[vi]:.2f} (identifiability)")
    ax.legend(fontsize=8, framealpha=0.85)


def profile_band(surf: Dict[str, object], axis: str,
                 rel: float = BAND_REL) -> Dict[str, float]:
    """Quantify one axis' identifiability band (pure function, CSV-ready).

    Profiles the loss along ``axis`` ("v0" at the grid sigma nearest the fitted
    optimum, or "sigma" at the grid v0 nearest it -- the same slices the profile
    plots draw) and measures the span of grid nodes whose loss is within ``rel``
    of the profile minimum. Returns::

        {axis, fixed_value, band_lo, band_hi, band_width, n_nodes_in_band,
         n_nodes_total, band_contiguous, censored_lo, censored_hi, fitted,
         fitted_on_grid_edge, min_loss, degenerate}

    ``censored_lo/hi``: the band touches the grid edge, so the true flat region
    extends beyond the evaluated grid and ``band_width`` is a LOWER bound.
    ``band_contiguous``: the in-band nodes form one unbroken run; False means a
    MULTI-MODAL profile whose [band_lo, band_hi] hull straddles an out-of-band
    ridge, so ``band_width`` overstates the flat region and must not be quoted
    as one. ``fitted_on_grid_edge``: the refined optimum sits at/beyond the
    grid border (the optimiser may have wanted to leave the box).
    ``degenerate``: fewer than two finite profile nodes, or a non-finite fitted
    optimum (either axis) that would make the slice/edge fields meaningless --
    every band field is NaN and nothing downstream may read the row as a
    measurement.
    """
    if axis not in ("sigma", "v0"):
        raise ValueError(f"axis must be 'sigma' or 'v0', got {axis!r}")
    gs, gv = surf["grid_sigma"], surf["grid_v0"]
    loss = surf["loss"]  # masked [S, V]
    fit_sigma, fit_v0 = float(surf["sigma"]), float(surf["v0"])

    def _degenerate(fixed, fitted):
        return {"axis": axis, "fixed_value": fixed, "fitted": fitted,
                "fitted_on_grid_edge": False, "n_nodes_total": 0,
                "band_lo": float("nan"), "band_hi": float("nan"),
                "band_width": float("nan"), "n_nodes_in_band": 0,
                "band_contiguous": False,
                "censored_lo": False, "censored_hi": False,
                "min_loss": float("nan"), "degenerate": True}

    # A non-finite fitted optimum makes both the slice choice (argmin over
    # |grid - nan| silently picks index 0) and the edge flags meaningless;
    # refuse to emit a plausible-looking row (review: NaN fail-open).
    if not (np.isfinite(fit_sigma) and np.isfinite(fit_v0)):
        return _degenerate(float("nan"), float("nan"))
    if axis == "v0":
        si = int(np.argmin(np.abs(gs - fit_sigma)))
        prof, coords, fixed, fitted = loss[si], gv, float(gs[si]), fit_v0
    else:
        vi = int(np.argmin(np.abs(gv - fit_v0)))
        prof, coords, fixed, fitted = loss[:, vi], gs, float(gv[vi]), fit_sigma
    finite = ~np.ma.getmaskarray(prof)
    grid_lo, grid_hi = float(coords.min()), float(coords.max())
    if finite.sum() < 2:
        out = _degenerate(fixed, fitted)
        out["n_nodes_total"] = int(finite.sum())
        return out
    x = coords[finite]
    a = np.asarray(prof[finite])
    amin = float(a.min())
    band = a <= amin * (1.0 + rel)
    band_idx = np.flatnonzero(band)
    band_lo, band_hi = float(x[band].min()), float(x[band].max())
    return {
        "axis": axis, "fixed_value": fixed, "fitted": fitted,
        "fitted_on_grid_edge": bool(fitted <= grid_lo or fitted >= grid_hi),
        "n_nodes_total": int(finite.sum()),
        "band_lo": band_lo, "band_hi": band_hi,
        "band_width": band_hi - band_lo,
        "n_nodes_in_band": int(band.sum()),
        # One unbroken run of (finite) nodes <=> hull == the actual flat region.
        "band_contiguous": bool(band_idx.size
                                and band_idx[-1] - band_idx[0] + 1 == band_idx.size),
        "censored_lo": bool(band_lo <= float(x.min())),
        "censored_hi": bool(band_hi >= float(x.max())),
        "min_loss": amin, "degenerate": False,
    }


def _find_npz(input_dir: Path, scenario: str) -> Optional[Path]:
    p = input_dir / f"loss_surface_{scenario}.npz"
    return p if p.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", default="outputs/rq2_calibration",
                        help="directory holding loss_surface_*.npz")
    parser.add_argument("--out", default=str(FIGS),
                        help="output directory for the PNGs")
    parser.add_argument("--log", action="store_true",
                        help="log-scale the colour (only if min loss > 0)")
    args = parser.parse_args()

    input_dir = Path(args.inputs)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) pooled-CITR single surface (paper body) ---
    all_npz = _find_npz(input_dir, "all")
    if all_npz is None:
        print(f"WARNING: {input_dir}/loss_surface_all.npz missing; run "
              "run_rq2_calibration.py --scenario all first", file=sys.stderr)
    else:
        surf = load_surface(all_npz)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        mesh = plot_single(ax, surf, "RQ2 loss surface (pooled CITR)", use_log=args.log)
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label="rollout ADE [m]")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        out = out_dir / "rq2_loss_surface_all.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

        # --- (a2) v0 identifiability profile (review C2) ---
        fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
        plot_v0_profile(ax2, surf)
        out2 = out_dir / "rq2_v0_profile_all.png"
        fig2.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"saved {out2}")

    # --- (b) per-scenario 2x2 grid (appendix) ---
    present = [(s, _find_npz(input_dir, s)) for s in SCENARIO_TITLES]
    missing = [s for s, p in present if p is None]
    if missing:
        print(f"WARNING: skipping scenarios with no npz: {missing}", file=sys.stderr)
    available = [(s, p) for s, p in present if p is not None]
    if available:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
        for ax, (scenario, npz_path) in zip(axes.flat, available):
            mesh = plot_single(ax, load_surface(npz_path),
                               SCENARIO_TITLES[scenario], use_log=args.log)
            if mesh is not None:
                fig.colorbar(mesh, ax=ax, label="rollout ADE [m]")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        # blank any unused axes (fewer than 4 scenarios present)
        for ax in axes.flat[len(available):]:
            ax.axis("off")
        fig.suptitle("RQ2 loss surfaces by CITR interaction geometry", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = out_dir / "rq2_loss_surface_grid.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
