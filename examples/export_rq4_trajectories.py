#!/usr/bin/env python3
"""Export recorded + planner-ego trajectories for the RQ3 real-loop overview
figure (thesis chapter 8).

Re-runs a SMALL, fixed subset of the cached campaign -- four representative
encounters (one per scenario family) x the replay pedestrian arm x the SGAN
predictor x {single, robust} plans x seed 0, about half a minute total --
capturing the planner-ego trajectory that the campaign cache intentionally
discards (``run_rq3_realloop.run_one_rq3`` keeps scalar metrics only). The
recorded ego / pedestrian trajectories come from the same
``enumerate_encounters`` enumeration the campaign used, so the encounter
identity is bit-identical to ``encounters.csv``.

Representative encounters (one per family; chosen so the overview connects
to the V1/V3 narratives -- selection basis in ``TARGETS`` below):

* vci_back__back_interaction_02__e00      rear approach, no collision
* vci_front__front_interaction_01__e00    oncoming; collides under replay
* vci_lat_bi__bidirection_normal_driving_04__e00   bidirectional crossing
* vci_lat_uni__unidirection_normal_driving_03__e00 unidirectional crossing

Outputs (committed so ``plot_rq4_geometry.py`` needs neither the raw dataset
nor a re-run):

* ``outputs/rq3_realloop/rq4_trajectories/<enc_id>.csv`` -- long format
  (kind, t, x, y) with kind in {recorded_ego, recorded_ped_<id>,
  planner_ego_single, planner_ego_robust}
* ``outputs/rq3_realloop/rq4_trajectories/verification.json`` -- per re-run
  min_dist / collision_count compared against the cached campaign JSON
  (``runs/<enc>/replay__sgan__<plan>/seed_00.json``); a mismatch is warned
  and recorded, never silently dropped.

Usage:
    .venv/bin/python examples/export_rq4_trajectories.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from src.core.metrics import calculate_aggregate_metrics  # noqa: E402
from src.simulation.realloop import build_realloop_simulator  # noqa: E402
from examples.inspect_rq3_encounters import enumerate_encounters  # noqa: E402
from examples.run_rq3_realloop import model_path_for  # noqa: E402
from examples.run_statistical_benchmark import set_seed  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# One encounter per scenario family. Selection basis (committed tables):
# back_02   robust gain delta = +1.185 m (v3_encounter_deltas.csv, replay/sgan)
# front_01  single AND robust collide under replay (all_runs.csv) -- the
#           oncoming worst case behind the V1 replay-side discordances
# lat_bi_04 delta = +2.062 m, single collision fraction 0.2 -> robust 0
# lat_uni_03 delta = +3.078 m (largest), single collides -> robust 0
TARGETS = [
    "vci_back__back_interaction_02__e00",
    "vci_front__front_interaction_01__e00",
    "vci_lat_bi__bidirection_normal_driving_04__e00",
    "vci_lat_uni__unidirection_normal_driving_03__e00",
]
PED_ARM = "replay"
PRED = "sgan"
PLANS = ["single", "robust"]
SEED = 0


def planner_rows(plan: str, history, dt: float) -> List[List]:
    # SimulationResult.time is recorded BEFORE the clock increment, so the
    # stored ego state belongs to r.time + dt (recorded_ego_deviation
    # convention).
    return [[f"planner_ego_{plan}", round(float(r.time) + dt, 4),
             round(float(r.ego_state.x), 4), round(float(r.ego_state.y), 4)]
            for r in history]


def recorded_rows(enc) -> List[List]:
    t_rel = np.asarray(enc.times, dtype=float) - float(enc.times[0])
    rows = [["recorded_ego", round(float(t), 4),
             round(float(x), 4), round(float(y), 4)]
            for t, (x, y) in zip(t_rel, enc.ego_xy)]
    for i, pid in enumerate(np.asarray(enc.ped_ids)):
        rows += [[f"recorded_ped_{pid}", round(float(t), 4),
                  round(float(x), 4), round(float(y), 4)]
                 for t, (x, y) in zip(t_rel, enc.ped_xy[:, i, :])]
    return rows


def export(root: str, fps: float, min_sep: float, min_len: int,
           outdir: Path, runs_cache: Path) -> None:
    by_id = {enc_id: (scenario, enc) for enc_id, scenario, enc
             in enumerate_encounters(root, fps, min_sep, min_len)}
    missing = [t for t in TARGETS if t not in by_id]
    if missing:
        raise SystemExit(
            f"target encounters {missing} not found under {root} -- clone "
            "the VCI dataset (see datasets/ README) and re-run")

    outdir.mkdir(parents=True, exist_ok=True)
    verification: List[Dict] = []
    for enc_id in TARGETS:
        _, enc = by_id[enc_id]
        rows = recorded_rows(enc)
        for plan in PLANS:
            set_seed(SEED)
            sim, _ = build_realloop_simulator(
                enc, PED_ARM, PRED, plan,
                sgan_model_path=model_path_for(PRED))
            history = sim.run()
            m = calculate_aggregate_metrics(
                history, sim.config.dt,
                prediction_dt=sim.observer.sgan_dt,
                prediction_steps=sim.config.pred_len,
            )
            rows += planner_rows(plan, history, sim.config.dt)

            rec = {"enc_id": enc_id, "ped_arm": PED_ARM, "pred": PRED,
                   "plan": plan, "seed": SEED,
                   "min_dist_m": float(m["min_dist"]),
                   "collision_count": int(m["collision_count"])}
            cached = runs_cache / enc_id / f"{PED_ARM}__{PRED}__{plan}" \
                / f"seed_{SEED:02d}.json"
            if cached.exists():
                with open(cached) as f:
                    c = json.load(f)
                rec["cached_min_dist_m"] = float(c["min_dist_m"])
                rec["cached_collision_count"] = int(c["collision_count"])
                rec["match"] = (
                    abs(rec["min_dist_m"] - rec["cached_min_dist_m"]) < 1e-6
                    and rec["collision_count"] == rec["cached_collision_count"])
                if not rec["match"]:
                    print(f"WARNING {enc_id}/{plan}: re-run min_dist="
                          f"{rec['min_dist_m']:.6f} coll="
                          f"{rec['collision_count']} differs from cache "
                          f"({rec['cached_min_dist_m']:.6f}, "
                          f"{rec['cached_collision_count']}) -- environment "
                          "drift, disclosed in verification.json")
            else:
                rec["match"] = None
                print(f"NOTE {enc_id}/{plan}: no cached run at {cached}; "
                      "re-run not cross-checked")
            verification.append(rec)
            print(f"{enc_id}/{plan}: min_dist={rec['min_dist_m']:.3f} "
                  f"coll={rec['collision_count']} "
                  f"match={rec.get('match')}")

        out_csv = outdir / f"{enc_id}.csv"
        with out_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "t", "x", "y"])
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")

    with (outdir / "verification.json").open("w") as fh:
        json.dump(verification, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    n_bad = sum(1 for r in verification if r["match"] is False)
    print(f"wrote {outdir / 'verification.json'} "
          f"({len(verification)} runs, {n_bad} cache mismatches)")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="datasets/vci_citr/data")
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--min-sep", type=float, default=8.0)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--outdir", type=Path,
                        default=REPO / "outputs" / "rq3_realloop"
                        / "rq4_trajectories")
    parser.add_argument("--runs-cache", type=Path,
                        default=REPO / "outputs" / "rq3_realloop" / "runs")
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    export(args.root, args.fps, args.min_sep, args.min_len,
           args.outdir, args.runs_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
