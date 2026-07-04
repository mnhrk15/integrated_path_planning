#!/usr/bin/env python3
"""Export per-encounter robust-single min-separation deltas (thesis chapter 8).

Reads the untracked campaign aggregate ``outputs/rq3_realloop/all_runs.csv``
(regenerable from the run cache via ``run_rq3_realloop.py --report-only``),
collapses seeds to encounter means with the exact same reduction as the
committed verdict tables (``make_rq3_report.encounter_means``; the unit of
analysis is the encounter), and writes the small per-encounter delta table

    outputs/rq3_realloop/v3_encounter_deltas.csv
    (ped_arm, pred, enc_id, n_seeds, min_dist_single_m, min_dist_robust_m,
     delta_m)

that the distribution figure ``plot_rq4_v3_gain.py`` reads, so the figure is
reproducible from committed tables alone.

Verification gate: per (ped_arm, pred), the mean of ``delta_m`` rounded to 4
decimals must equal ``mean_delta_m`` in the committed ``v3_robust.csv`` and
the pair count must equal its ``n_pairs``; any mismatch is a SystemExit
(the export never silently disagrees with the committed verdict tables).

Usage:
    .venv/bin/python examples/export_rq4_encounter_deltas.py
"""
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.make_rq3_report import encounter_means  # noqa: E402
from examples.run_rq3_realloop import PED_ARMS, PREDICTORS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_COLUMNS = ["ped_arm", "pred", "enc_id", "n_seeds",
               "min_dist_single_m", "min_dist_robust_m", "delta_m"]


def export(all_runs: Path, v3_csv: Path, out_csv: Path) -> pd.DataFrame:
    if not all_runs.exists():
        raise SystemExit(
            f"{all_runs} missing -- rebuild it from the run cache with "
            "examples/run_rq3_realloop.py --report-only")
    em = encounter_means(pd.read_csv(all_runs))
    v3 = pd.read_csv(v3_csv).set_index(["ped_arm", "pred"])

    rows: List[dict] = []
    for arm in PED_ARMS:
        for pred in PREDICTORS:
            s = em[(em.ped_arm == arm) & (em.pred == pred)
                   & (em.plan == "single")].set_index("enc_id")
            r = em[(em.ped_arm == arm) & (em.pred == pred)
                   & (em.plan == "robust")].set_index("enc_id")
            common = s.index.intersection(r.index).sort_values()
            deltas = (r.loc[common, "min_dist_m"]
                      - s.loc[common, "min_dist_m"])

            ref = v3.loc[(arm, pred)]
            if len(common) != int(ref["n_pairs"]):
                raise SystemExit(
                    f"{arm}/{pred}: {len(common)} encounter pairs != "
                    f"n_pairs={int(ref['n_pairs'])} in {v3_csv}")
            if round(float(deltas.mean()), 4) != float(ref["mean_delta_m"]):
                raise SystemExit(
                    f"{arm}/{pred}: mean delta {deltas.mean():.6f} does not "
                    f"reproduce mean_delta_m={ref['mean_delta_m']} in "
                    f"{v3_csv} -- all_runs.csv and the committed verdict "
                    "tables disagree; regenerate both from the same cache")

            for enc_id in common:
                rows.append({
                    "ped_arm": arm, "pred": pred, "enc_id": enc_id,
                    "n_seeds": int(s.loc[enc_id, "n_seeds"]),
                    "min_dist_single_m":
                        round(float(s.loc[enc_id, "min_dist_m"]), 6),
                    "min_dist_robust_m":
                        round(float(r.loc[enc_id, "min_dist_m"]), 6),
                    "delta_m": round(float(deltas.loc[enc_id]), 6),
                })
    df = pd.DataFrame(rows, columns=OUT_COLUMNS)
    df.to_csv(out_csv, index=False)
    n_cells = df.groupby(["ped_arm", "pred"]).ngroups
    print(f"wrote {out_csv} ({len(df)} rows, {n_cells} arm x pred cells; "
          "gate vs v3_robust.csv passed)")
    return df


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = REPO / "outputs" / "rq3_realloop"
    parser.add_argument("--all-runs", type=Path,
                        default=root / "all_runs.csv")
    parser.add_argument("--v3", type=Path, default=root / "v3_robust.csv")
    parser.add_argument("--out", type=Path,
                        default=root / "v3_encounter_deltas.csv")
    args = parser.parse_args(argv)
    export(args.all_runs, args.v3, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
