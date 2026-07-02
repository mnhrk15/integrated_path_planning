#!/usr/bin/env python3
"""RQ2 identifiability audit: dense (sigma, v0) loss surfaces per objective x policy.

For every requested (objective, cap-policy) combination this evaluates the
pooled-CITR loss on a dense grid (denser than the 6x7 fitting grid), refines
with Nelder-Mead, and writes:

* ``loss_surface__{objective}__{policy}.npz`` -- load_surface-compatible
  (grid_sigma / grid_v0 / grid_loss / sigma / v0 / grid_best; the combo is
  encoded in the FILENAME so the npz stays allow_pickle=False loadable).
* ``identifiability.csv`` -- one row per combo per axis from
  plot_rq2_loss_surface.profile_band: the width of the "within 2% of the
  minimum" band along v0 AND along sigma (review 1.2-2: the S2 discriminating
  cell is driven by sigma, so sigma-side identifiability is mandatory), with
  censoring flags when the band or the fitted optimum touches the grid edge.
* ``identifiability__{objective}__{policy}.png`` -- surface + v0 profile +
  sigma profile, one figure per combo.

Objectives are distmatch config strings (run_rq2_distmatch.parse_config) plus
the alias ``ade`` = the canonical rollout-ADE fitter (bit-identical to ``w0``):
e.g. ``--objectives ade w1 pure``. The question the CSV answers: does adding
the closest-approach EMD term make the v0 (and sigma) valley NARROWER, i.e.
does distribution matching restore identifiability (novelty direction (A)-3)?

Outputs default under outputs/rq2_instrument_audit/; the committed
outputs/rq2_calibration/*.npz are never touched.

Usage:
    .venv/bin/python examples/run_rq2_surfaces.py \\
        --objectives ade w1 pure --policies median uncapped closedloop
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from loguru import logger  # noqa: E402

from examples.plot_rq2_loss_surface import (  # noqa: E402
    load_surface,
    plot_single,
    plot_sigma_profile,
    plot_v0_profile,
    profile_band,
)
from examples.run_rq2_cap_sensitivity import cap_kwargs  # noqa: E402
from examples.run_rq2_distmatch import parse_config  # noqa: E402
from examples.run_rq2_evaluation import VEHICLE_SCENARIOS  # noqa: E402
from src.calibration import calibrate  # noqa: E402
from src.datasets.vci_encounter import encounters_from_clips  # noqa: E402
from src.datasets.vci_loader import load_vci_clips  # noqa: E402
from src.simulation.calibration_harness import (  # noqa: E402
    CAP_POLICIES,
    DIST_METRICS,
    objective_multi,
    objective_rollout_ade,
)

IDENT_COLUMNS = [
    "objective", "policy", "cap_multiplier", "axis", "fixed_value",
    "band_lo", "band_hi", "band_width", "n_nodes_in_band", "n_nodes_total",
    "censored_lo", "censored_hi", "fitted", "fitted_on_grid_edge",
    "min_loss", "degenerate", "fit_sigma", "fit_v0", "fit_loss",
]

# Dense audit grids (vs the 6x7 fitting grid): sigma resolves the 0.3-2.5 m
# range the fits land in; v0 resolves the calibrated ~1.7 / AVEC 3.5 contrast.
DENSE_SIGMA = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
DENSE_V0 = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 5.0, 6.5, 8.0]


def make_surface_objective(spec: str, encs, dist_metric: str, policy: str,
                           cap_multiplier: float):
    """(sigma, v0) -> scalar loss for one objective spec under one cap policy."""
    kw = cap_kwargs(policy, cap_multiplier)
    if spec == "ade":
        def obj(s, v):
            return objective_rollout_ade(encs, s, v, **kw)
    else:
        cfg = parse_config(spec)

        def obj(s, v):
            return objective_multi(
                encs, s, v, w_ade=cfg["w_ade"], w_dist=cfg["w_dist"],
                dist_metric=dist_metric,
                interaction_distance=cfg["interaction_distance"], **kw)
    return obj


def ylabel_for(spec: str, dist_metric: str) -> str:
    unit = "[m^0.5]" if dist_metric == "energy" else "[m]"
    if spec == "ade":
        return "rollout ADE [m]"
    cfg = parse_config(spec)
    if cfg["w_ade"] == 0.0:
        return f"closest-approach {dist_metric.upper()} {unit}"
    return (f"{cfg['w_ade']:g}*ADE + {cfg['w_dist']:g}*"
            f"{dist_metric.upper()}(closest) "
            + ("[mixed units]" if dist_metric == "energy" else "[m]"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objectives", nargs="*", default=["ade", "w1", "pure"],
                   help="'ade' or distmatch config strings (w1, pure, w1_id8, ...)")
    p.add_argument("--policies", nargs="*", default=["median", "uncapped"],
                   choices=list(CAP_POLICIES))
    p.add_argument("--cap-multiplier", type=float, default=1.3,
                   help="capfit headroom when 'capfit' is among --policies")
    p.add_argument("--dist-metric", default="emd", choices=list(DIST_METRICS))
    p.add_argument("--scenario", default="all", choices=VEHICLE_SCENARIOS + ["all"])
    p.add_argument("--root", default="datasets/vci_citr/data")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--min-sep", type=float, default=8.0)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--sigma-grid", type=float, nargs="*", default=DENSE_SIGMA)
    p.add_argument("--v0-grid", type=float, nargs="*", default=DENSE_V0)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--out", default="outputs/rq2_instrument_audit/surfaces")
    p.add_argument("--figs", default="outputs/rq2_instrument_audit/figs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    clips = load_vci_clips(args.root, "citr", fps=args.fps)
    wanted = VEHICLE_SCENARIOS if args.scenario == "all" else [args.scenario]
    clips = [c for c in clips if c.scenario in wanted]
    encs = encounters_from_clips(clips, args.min_sep, args.min_len)
    if not encs:
        raise SystemExit("no encounters extracted (loosen --min-sep/--min-len)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = Path(args.figs)
    figs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nencounters={len(encs)}  grid={len(args.sigma_grid)}x{len(args.v0_grid)}"
          f"  combos={len(args.objectives)}x{len(args.policies)}")

    ident_rows: List[Dict] = []
    for spec in args.objectives:
        for policy in args.policies:
            print(f"=== surface objective={spec} policy={policy} ===")
            obj = make_surface_objective(spec, encs, args.dist_metric, policy,
                                         args.cap_multiplier)
            result = calibrate(obj, args.sigma_grid, args.v0_grid,
                               refine=not args.no_refine)
            npz_path = out_dir / f"loss_surface__{spec}__{policy}.npz"
            np.savez(npz_path, grid_sigma=result.grid_sigma,
                     grid_v0=result.grid_v0, grid_loss=result.grid_loss,
                     sigma=result.sigma, v0=result.v0,
                     grid_best=np.asarray(result.grid_best, dtype=float))
            surf = load_surface(npz_path)

            for axis in ("v0", "sigma"):
                band = profile_band(surf, axis)
                band.update({"objective": spec, "policy": policy,
                             "cap_multiplier": (args.cap_multiplier
                                                if policy == "capfit" else float("nan")),
                             "fit_sigma": result.sigma, "fit_v0": result.v0,
                             "fit_loss": result.loss})
                ident_rows.append(band)
                print(f"  {axis:>5}-band [{band['band_lo']:.3g}, {band['band_hi']:.3g}] "
                      f"width={band['band_width']:.3g} "
                      f"censored={band['censored_lo'] or band['censored_hi']}")

            ylab = ylabel_for(spec, args.dist_metric)
            fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
            mesh = plot_single(axes[0], surf, f"{spec} / {policy}")
            if mesh is not None:
                fig.colorbar(mesh, ax=axes[0], label=ylab)
            axes[0].legend(loc="upper right", fontsize=7, framealpha=0.85)
            plot_v0_profile(axes[1], surf, ylabel=ylab)
            plot_sigma_profile(axes[2], surf, ylabel=ylab)
            fig.suptitle(f"RQ2 identifiability audit: objective={spec}, "
                         f"cap_policy={policy}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig_path = figs_dir / f"identifiability__{spec}__{policy}.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {npz_path.name}, {fig_path.name}")

    df = pd.DataFrame(ident_rows, columns=IDENT_COLUMNS)
    csv_path = out_dir / "identifiability.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nsaved identifiability table to {csv_path}")


if __name__ == "__main__":
    main()
