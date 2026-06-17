#!/usr/bin/env python3
"""Calibrate the SFM ego repulsion (sigma, v0) on real VCI-CITR encounters (RQ2).

This is the thesis spike's RQ2 target: re-identify the AVEC paper's hand-tuned
ego repulsion (sigma=0.7, v0=3.5) from how *real* pedestrians avoided a *real*
vehicle, using the controlled CITR vehicle-crowd clips. The ego is fixed to the
recorded vehicle trajectory and the SFM pedestrians react; (sigma, v0) is fit by
minimising the short-rollout displacement error to the recorded pedestrians.

Pipeline:
    load_vci_clips(CITR) -> encounters_from_clips (align vehicle + extract
      fixed-population interaction spans) -> calibrate(objective_rollout_ade)
      -> fidelity_report at the calibrated vs baseline (AVEC default / no-repulsion).

Outputs the calibrated (sigma, v0), the grid loss surface (saved to .npz for the
thesis ridge figure), the one-step diagnostic, and a control-group fidelity table.

Usage:
    .venv/bin/python examples/run_rq2_calibration.py \
        --scenario vci_front --root datasets/vci_citr/data --fps 29.97
    # held-out generalisation: fit on some clips, report on others
    .venv/bin/python examples/run_rq2_calibration.py --scenario vci_front \
        --holdout front_interaction_04
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from src.calibration import calibrate  # noqa: E402
from src.datasets.vci_loader import load_vci_clips  # noqa: E402
from src.datasets.vci_encounter import encounters_from_clips  # noqa: E402
from src.simulation.calibration_harness import (  # noqa: E402
    fidelity_report,
    objective_one_step,
    objective_rollout_ade,
)

# AVEC paper's hand-tuned values and the no-repulsion null (control group).
AVEC_DEFAULT = (0.7, 3.5)
NO_REPULSION = (1.0, 0.0)  # sigma irrelevant when v0=0

# CITR scenarios that carry a vehicle (the only ones usable for ego calibration).
VEHICLE_SCENARIOS = ["vci_front", "vci_back", "vci_lat_bi", "vci_lat_uni"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="vci_front",
                   choices=VEHICLE_SCENARIOS + ["all"],
                   help="CITR scenario subfolder with vehicles "
                        "(vci_front/back/lat_bi/lat_uni), or 'all' to pool every vehicle scenario")
    p.add_argument("--root", default="datasets/vci_citr/data",
                   help="VCI-CITR data root (the 'data' dir, not legacy/)")
    p.add_argument("--fps", type=float, default=29.97, help="CITR frame rate (NTSC=29.97)")
    p.add_argument("--min-sep", type=float, default=8.0,
                   help="max closest-approach for a span to count as an encounter [m]")
    p.add_argument("--min-len", type=int, default=5, help="min encounter length [frames]")
    p.add_argument("--interaction-distance", type=float, default=None,
                   help="restrict ADE to peds approaching within this distance [m] (default: all)")
    p.add_argument("--holdout", nargs="*", default=None,
                   help="clip stems to EXCLUDE from fitting and report on separately")
    p.add_argument("--sigma-grid", type=float, nargs="*",
                   default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--v0-grid", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    p.add_argument("--no-refine", action="store_true", help="skip Nelder-Mead refinement")
    p.add_argument("--out", default="outputs/rq2_calibration",
                   help="output directory for the loss surface .npz")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _fmt_fidelity(name, params, encs):
    r = fidelity_report(encs, params[0], params[1])
    return (f"  {name:<18} sigma={params[0]:.3f} v0={params[1]:.3f} | "
            f"ADE={r['rollout_ade']:.3f} closest sim/real="
            f"{r['mean_closest_sim']:.2f}/{r['mean_closest_real']:.2f} "
            f"KS_closest={r['ks_closest']:.3f} "
            f"KS_onset={r['ks_onset']:.3f}(n {r['n_onset_sim']}/{r['n_onset_real']})")


def main():
    args = parse_args()
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    clips = load_vci_clips(args.root, "citr", fps=args.fps)
    wanted = VEHICLE_SCENARIOS if args.scenario == "all" else [args.scenario]
    clips = [c for c in clips if c.scenario in wanted]
    if not clips:
        raise SystemExit(f"no clips for scenario {args.scenario!r} under {args.root}")

    holdout = set(args.holdout or [])
    fit_clips = [c for c in clips if c.clip not in holdout]
    held_clips = [c for c in clips if c.clip in holdout]
    # A holdout stem that matches no clip (typo, or it lives under another
    # scenario) would silently fall back to fitting on EVERY clip with an empty
    # held-out block -- the generalisation experiment the flag exists for would
    # quietly not run. Fail loudly instead.
    missing = holdout - {c.clip for c in held_clips}
    if missing:
        raise SystemExit(
            f"--holdout stems not found in scenario {args.scenario!r}: {sorted(missing)} "
            f"(available: {sorted(c.clip for c in clips)})"
        )

    fit_encs = encounters_from_clips(fit_clips, args.min_sep, args.min_len)
    held_encs = encounters_from_clips(held_clips, args.min_sep, args.min_len)
    if not fit_encs:
        raise SystemExit("no encounters extracted for fitting (loosen --min-sep/--min-len)")

    print(f"\nscenario={args.scenario}  fit clips={[c.clip for c in fit_clips]}")
    print(f"fit encounters={len(fit_encs)} (held-out={len(held_encs)})")
    for e in fit_encs:
        print(f"  {e.clip}: T={e.ped_xy.shape[0]} N={e.ped_xy.shape[1]} "
              f"min_sep={e.min_separation:.2f}m ego_v_p50={np.median(e.ego_vel):.2f}m/s")

    def obj(s, v):
        return objective_rollout_ade(fit_encs, s, v,
                                     interaction_distance=args.interaction_distance)

    result = calibrate(obj, args.sigma_grid, args.v0_grid, refine=not args.no_refine)

    print(f"\n=== calibration (rollout ADE fitter) ===")
    print(f"  grid min:  sigma={result.grid_best[0]:.3f} v0={result.grid_best[1]:.3f}")
    print(f"  refined:   sigma={result.sigma:.3f} v0={result.v0:.3f} "
          f"loss(ADE)={result.loss:.4f} (Nelder-Mead {'used' if result.refined else 'no improvement'})")

    # One-step diagnostic at the calibrated point and across v0 (shows v0->0 pull).
    # Restrict to the SAME interaction proximity as the fitter (max_distance =
    # --interaction-distance): otherwise far, non-interacting peds -- whose
    # recorded radial acceleration is pure noise and whose ego force is ~0 --
    # would dominate the residual, so the printed "minimum at v0=0" verdict would
    # describe non-interacting peds rather than the avoidance being calibrated.
    diag = [(v, objective_one_step(fit_encs, result.sigma, v,
                                   max_distance=args.interaction_distance))
            for v in args.v0_grid]
    print(f"\n=== one-step radial-accel DIAGNOSTIC (sigma={result.sigma:.2f}) ===")
    print("  " + "  ".join(f"v0={v:.1f}:{l:.3f}" for v, l in diag))
    print("  (minimum at v0=0 => instantaneous force can't support the paper's repulsion)")

    # calibrated value vs the two null controls (AVEC default, no repulsion).
    baselines = [
        ("calibrated", (result.sigma, result.v0)),
        ("AVEC default", AVEC_DEFAULT),
        ("no repulsion", NO_REPULSION),
    ]
    for title, encs in [("FIT encounters (control group)", fit_encs),
                        ("HELD-OUT encounters", held_encs)]:
        if not encs:
            continue
        print(f"\n=== fidelity report on {title} ===")
        for name, params in baselines:
            print(_fmt_fidelity(name, params, encs))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"loss_surface_{args.scenario}.npz"
    # Save grid_best too: grid_loss carries inf cells (degenerate/no-sample), so a
    # downstream ridge-figure script cannot recover the masked argmin from the raw
    # surface alone.
    np.savez(npz, grid_sigma=result.grid_sigma, grid_v0=result.grid_v0,
             grid_loss=result.grid_loss, sigma=result.sigma, v0=result.v0,
             grid_best=np.asarray(result.grid_best, dtype=float))
    print(f"\nsaved loss surface to {npz}")
    print("\nNote: CITR vehicles are low-speed (p50~1-3 m/s) vs AVEC ego 5-6 m/s; "
          "this calibration is in-domain at low speed (speed term deferred).")


if __name__ == "__main__":
    main()
