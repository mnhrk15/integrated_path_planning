#!/usr/bin/env python3
"""Open-loop prediction evaluation on real ETH/UCY trajectories (RQ1a).

This is the thesis spike's first end-to-end target: feed *recorded* pedestrian
trajectories through the existing observe -> predict pipeline (no ego, no
planner) and measure displacement error / NLL per predictor. It answers whether
the AVEC simulator's CV < LSTM < SGAN ordering is an artefact of fitting the SFM
ground truth, by re-measuring the ordering on in-domain real data.

Pipeline per fixed-population window (eth_ucy_loader.extract_fixed_windows):
    ReplayPedestrianSource (dt=0.4)  ->  PedestrianObserver (8 obs @0.4s)
      ->  TrajectoryPredictor (cv/lstm/sgan, num_samples)
      ->  SimulationResult mini-history  ->  core.metrics (ADE/FDE/NLL).

The predictor runs natively at sgan_dt = sim_dt = 0.4 s with plan_horizon =
pred_len * 0.4, so its dense output grid coincides with the raw SGAN steps
(identity resampling) and matches the metric cadence (prediction_dt = 0.4).

Models are leave-one-out: the checkpoint named after a scene was trained on the
other scenes (see resolve_model). 'lstm' uses the no-pooling models/sgan-models,
'sgan' the pooled models/sgan-p-models.

Usage:
    .venv/bin/python examples/run_openloop_prediction.py --scene zara1 --method sgan
    .venv/bin/python examples/run_openloop_prediction.py --scene zara1 --method cv \
        --max-windows 30   # quick smoke
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from src.core.data_structures import EgoVehicleState, SimulationResult  # noqa: E402
from src.core.metrics import calculate_aggregate_metrics  # noqa: E402
from src.datasets.eth_ucy_loader import (  # noqa: E402
    SCENE_TEST_FILES,
    extract_fixed_windows,
    load_scene,
)
from src.pedestrian.observer import PedestrianObserver  # noqa: E402
from src.prediction.trajectory_predictor import TrajectoryPredictor  # noqa: E402
from src.simulation.replay_source import ReplayPedestrianSource  # noqa: E402

SGAN_DT = 0.4  # native ETH/UCY annotation cadence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_model(scene: str, method: str, pred_len: int):
    """Leave-one-out checkpoint path for a scene/method, or None for CV."""
    if method == "cv":
        return None
    subdir = "sgan-models" if method == "lstm" else "sgan-p-models"
    path = Path("models") / subdir / f"{scene}_{pred_len}_model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint for method='{method}' not found: {path} "
            f"(run scripts/download_sgan_models.py)"
        )
    return str(path)


def evaluate_window(window, predictor, obs_len, dt):
    """Run one [seq_len, N, 2] window through observe->predict.

    Returns a mini-history (list of SimulationResult) with the single prediction
    origin at t = obs_len - 1 carrying the forecast distribution, suitable for
    the standard fixed-horizon ADE/FDE and KDE-NLL metrics.
    """
    source = ReplayPedestrianSource(window, dt=dt)
    observer = PedestrianObserver(obs_len=obs_len, dt=dt, sgan_dt=dt)
    history = []
    n = window.shape[0]
    for t in range(n):
        ped_state = source.get_state()
        observer.update(ped_state)
        pred_single = pred_dist = None
        if t == obs_len - 1 and observer.is_ready:
            obs_traj, obs_traj_rel, seq_start_end = observer.get_observation()
            best, dist = predictor.predict_single_best(
                obs_traj, obs_traj_rel, seq_start_end, staleness=0.0
            )
            pred_single = best
            # Wrap a deterministic forecast as a 1-sample distribution so the
            # ADE/FDE path evaluates it; the NLL path skips <2 samples (CV).
            pred_dist = dist if dist is not None else best[None, ...]
        history.append(
            SimulationResult(
                time=t * dt,
                ego_state=EgoVehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0, a=0.0, timestamp=t * dt),
                ped_state=ped_state,
                predicted_trajectories=pred_single,
                predicted_distribution=pred_dist,
            )
        )
        source.step()
    return history


def evaluate_scene(scenes, predictor, obs_len, pred_len, dt, stride, max_windows):
    seq_len = obs_len + pred_len
    sum_ade = sum_fde = traj_count = 0.0
    sum_ade_pa = sum_fde_pa = 0.0  # per-agent best-of-N (review M1)
    sum_nll = 0.0
    nll_count = 0
    n_windows = 0
    for scene in scenes:
        windows = extract_fixed_windows(scene, seq_len=seq_len, stride=stride)
        if max_windows is not None:
            windows = windows[:max_windows]
        for window in windows:
            history = evaluate_window(window, predictor, obs_len, dt)
            # Reuse the public aggregator (single source of truth for the
            # count-weighting / minADE conventions) instead of the private
            # internals, weighting each window by its evaluation counts.
            m = calculate_aggregate_metrics(history, dt, dt, pred_len)
            ade_count = m["ade_eval_count"]
            if ade_count > 0 and not np.isnan(m["ade"]):
                sum_ade += m["ade"] * ade_count
                sum_fde += m["fde"] * ade_count
                # Per-agent best-of-N reported alongside scene-level so RQ1a can
                # show whether the cv/lstm/sgan ordering is invariant to the
                # (non-canonical) scene-level joint best-of-N selection.
                sum_ade_pa += m["ade_per_agent"] * ade_count
                sum_fde_pa += m["fde_per_agent"] * ade_count
                traj_count += ade_count
            nll_n = m["nll_eval_count"]
            if nll_n > 0 and not np.isnan(m["nll"]):
                sum_nll += m["nll"] * nll_n
                nll_count += nll_n
            n_windows += 1
    return {
        "n_windows": n_windows,
        "n_trajectories": int(traj_count),
        "ade": sum_ade / traj_count if traj_count else float("nan"),
        "fde": sum_fde / traj_count if traj_count else float("nan"),
        "ade_per_agent": sum_ade_pa / traj_count if traj_count else float("nan"),
        "fde_per_agent": sum_fde_pa / traj_count if traj_count else float("nan"),
        "nll": sum_nll / nll_count if nll_count else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="zara1", choices=list(SCENE_TEST_FILES))
    parser.add_argument("--method", default="sgan", choices=["cv", "lstm", "sgan"])
    parser.add_argument("--num-samples", type=int, default=20,
                        help="forecast samples for lstm/sgan (best-of-N); CV is deterministic")
    parser.add_argument("--root", default="datasets", help="dataset root directory")
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1,
                        help="window stride over the frame grid (>1 subsamples)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap windows per file (quick smoke); default all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None,
                        help="append a result row to this CSV (header written if new)")
    parser.add_argument("--quiet", action="store_true", help="suppress info logs")
    args = parser.parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    set_seed(args.seed)
    scenes = load_scene(args.scene, root=args.root)

    num_samples = 1 if args.method == "cv" else args.num_samples
    if args.method in ("lstm", "sgan") and num_samples < 2:
        logger.warning(
            f"--num-samples={num_samples} for method='{args.method}': NLL needs "
            ">=2 samples and will be NaN; ADE/FDE use a single sample."
        )
    predictor = TrajectoryPredictor(
        model_path=resolve_model(args.scene, args.method, args.pred_len),
        pred_len=args.pred_len,
        num_samples=num_samples,
        device=args.device,
        sgan_dt=SGAN_DT,
        sim_dt=SGAN_DT,                 # native 0.4 s evaluation (identity resample)
        plan_horizon=args.pred_len * SGAN_DT,
        method=args.method,
    )

    result = evaluate_scene(
        scenes, predictor, args.obs_len, args.pred_len, SGAN_DT,
        args.stride, args.max_windows,
    )

    print(
        f"\nscene={args.scene} method={args.method} samples={num_samples} "
        f"seed={args.seed}\n"
        f"  windows={result['n_windows']} trajectories={result['n_trajectories']}\n"
        f"  ADE(scene)={result['ade']:.3f} m  FDE(scene)={result['fde']:.3f} m  "
        f"NLL={result['nll']:.3f}\n"
        f"  ADE(per-agent)={result['ade_per_agent']:.3f} m  "
        f"FDE(per-agent)={result['fde_per_agent']:.3f} m  (canonical SGAN minADE/minFDE)"
    )

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = ("scene,method,seed,num_samples,n_windows,n_trajectories,"
                  "ade,fde,ade_per_agent,fde_per_agent,nll\n")
        # Header when the file is absent OR empty: a pre-touched / truncated
        # 0-byte file would otherwise receive a header-less, unparseable append.
        need_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
        with open(csv_path, "a") as f:
            if need_header:
                f.write(header)
            f.write(
                f"{args.scene},{args.method},{args.seed},{num_samples},"
                f"{result['n_windows']},{result['n_trajectories']},"
                f"{result['ade']},{result['fde']},"
                f"{result['ade_per_agent']},{result['fde_per_agent']},{result['nll']}\n"
            )
        print(f"  appended row to {csv_path}")


if __name__ == "__main__":
    main()
