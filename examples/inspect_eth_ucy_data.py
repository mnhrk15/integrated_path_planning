#!/usr/bin/env python3
"""Validate the ETH/UCY pre-processing assumptions RQ1a depends on.

The ADE/FDE ordering in RQ1a (open-loop prediction) is only meaningful if the
parsed ETH/UCY trajectories satisfy a few assumptions the loader makes
implicitly. The code-review completeness critic flagged four that were never
checked; this tool quantifies each from the real files (read-only; it changes no
results). It is the ETH/UCY analogue of ``inspect_vci_data.py``.

Gaps diagnosed (one report block each):
  1. Annotation cadence -- the loader treats every scene as world-frame metres
     (verified correct: the UCY scenes share one spatial span) but the wall-clock
     seconds one annotation step represents differs by scene. ``eth`` was recorded
     from an accelerated video (Trajectron++ issue #67), so its whole speed
     distribution is ~2x too fast under the uniform 0.4 s assumption and normalises
     at ~0.8 s; ``univ`` is a genuine 0.4 s scene whose LOW MEDIAN is a loitering
     crowd, not a cadence error. This tool therefore evaluates each scene at its
     per-scene cadence (:func:`scene_dt`) and flags on the MOVING speed (p90, robust
     to loiterers), NOT the median -- so a slow-but-correct crowd is not mislabelled.
     The 0.4 s SGAN evaluation protocol (and ADE/FDE, which is dt-independent) is
     unaffected; see ``docs/GAPB_eth_ucy_cadence_resolution.md``.
  2. ``univ`` two-file concatenation -- ``univ`` ships students001/003 as two
     recordings with OVERLAPPING frame ids and ped ids. The loader returns them as
     separate SceneTrajectories (no cross-file window), but reused frame/ped ids
     mean the same (frame, ped) key denotes DIFFERENT physical agents in the two
     files; this quantifies the overlap and the coordinate divergence so the
     "separate handling" assumption is verified rather than assumed.
  3. Missing-frame straddle -- windows index the sorted frame grid and treat a
     larger gap (a frame where nobody is annotated) as a single 0.4 s step. A
     fixed-population window that spans such a hole has a physical horizon longer
     than seq_len*0.4 s. Reports the gap distribution and the fraction of EMITTED
     windows that straddle a hole.
  4. observer->GT time anchor -- RQ1a runs the observer at sim_dt == sgan_dt
     (0.4 s), so there is no 0.1->0.4 downsampling phase (that only exists in the
     closed-loop sim, out of RQ1a scope). Reports the resulting prediction index
     grid and flags only if sim_dt != sgan_dt (which WOULD introduce a phase).

Usage:
    .venv/bin/python examples/inspect_eth_ucy_data.py --root datasets
    .venv/bin/python examples/inspect_eth_ucy_data.py --root datasets --scene eth
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.metrics import _steps_for_interval  # noqa: E402
from src.datasets.eth_ucy_loader import (  # noqa: E402
    SCENE_TEST_FILES,
    SGAN_PROTOCOL_DT,
    SceneTrajectories,
    extract_fixed_windows,
    load_scene,
    load_scene_file,
    scene_dt,
    walking_speed_stats,
)

# Plausible band for the MOVING walking speed (p90 of per-step speeds) [m/s].
# p90 isolates the actively-walking pedestrians, so it is robust to a loitering
# population that only drags the MEDIAN down (e.g. the univ students plaza). A
# scene whose p90 leaves this band -- evaluated at its per-scene cadence -- has a
# cadence/scale problem, not merely a slow crowd.
MOVING_SPEED_BAND = (0.8, 2.0)
# A median below this floor WITH a plausible p90 just means a slow/loitering
# population; reported as an informational note, never a cadence flag.
LOITER_MEDIAN_FLOOR = 0.8


def scene_cadence_report(scene: SceneTrajectories, dt: float = 0.4) -> Dict[str, float]:
    """Gap 1: walking-speed stats + a cadence-plausibility flag for one file.

    Flags on the MOVING speed (p90), not the median, so a genuine loitering
    population (low median but normal walkers -- the univ students plaza) is NOT
    mislabelled as a coordinate/cadence error. A flag means even the active
    walkers are implausible under ``dt``: evidence the annotation cadence (or, in
    principle, the coordinate scale) does not match the 0.4 s SGAN assumption.
    ``dt`` should be the scene's physical cadence (:func:`scene_dt`).
    """
    speeds = walking_speed_stats(scene, dt=dt)
    if speeds.size == 0:
        # No adjacent-step pairs (e.g. a <2-frame file or one whose only gaps
        # are holes) -> cannot assess cadence. This is a missing-DATA condition,
        # NOT a cadence/scale mismatch, so it must not raise the flag (n_speeds==0
        # is already visible in the printout).
        return {"n_speeds": 0, "median": float("nan"), "p90": float("nan"),
                "p95": float("nan"), "frame_step": scene.frame_step,
                "flagged": False, "low_median": False}
    median = float(np.median(speeds))
    p90 = float(np.percentile(speeds, 90))
    flagged = not (MOVING_SPEED_BAND[0] <= p90 <= MOVING_SPEED_BAND[1])
    return {
        "n_speeds": int(speeds.size),
        "median": median,
        "p90": p90,
        "p95": float(np.percentile(speeds, 95)),
        "frame_step": scene.frame_step,
        "flagged": flagged,
        # Genuinely slow crowd (not a cadence error): low median, plausible p90.
        "low_median": (not flagged) and (median < LOITER_MEDIAN_FLOOR),
    }


def frame_gap_report(scene: SceneTrajectories) -> Dict[str, object]:
    """Gap 3a: distribution of consecutive-frame gaps and hole count."""
    if scene.n_frames < 2:
        return {"frame_step": scene.frame_step, "gaps": {}, "n_holes": 0}
    diffs = np.diff(scene.frames)
    step = scene.frame_step
    values, counts = np.unique(diffs, return_counts=True)
    n_holes = int((diffs > step * 1.5).sum()) if step > 0 else 0
    return {
        "frame_step": step,
        "gaps": {float(v): int(c) for v, c in zip(values, counts)},
        "n_holes": n_holes,
    }


def window_straddle_report(
    scene: SceneTrajectories, seq_len: int, stride: int = 1, min_peds: int = 1
) -> Dict[str, float]:
    """Gap 3b: fraction of EMITTED fixed windows whose frames straddle a hole.

    A straddling window has a non-uniform frame gap, so its physical horizon
    exceeds seq_len*0.4 s -- the ADE/FDE for that window is measured over a longer
    time than the predictor was asked for. Only windows that actually pass the
    population filter (and would be evaluated) are counted.
    """
    step = scene.frame_step
    total = 0
    straddling = 0
    for start in range(0, scene.n_frames - seq_len + 1, stride):
        frame_dicts = scene.by_frame[start: start + seq_len]
        present = set(frame_dicts[0].keys())
        for fd in frame_dicts[1:]:
            present &= set(fd.keys())
        if len(present) < min_peds:
            continue
        total += 1
        fr = scene.frames[start: start + seq_len]
        if step > 0 and np.any(~np.isclose(np.diff(fr), step)):
            straddling += 1
    return {
        "emitted_windows": total,
        "straddling_windows": straddling,
        "straddle_frac": (straddling / total) if total else 0.0,
    }


def univ_overlap_report(scenes: List[SceneTrajectories]) -> Dict[str, object]:
    """Gap 2: frame/ped-id reuse and coordinate divergence across univ's 2 files.

    Returns the overlap counts and, for keys (frame, ped) present in BOTH files,
    the fraction whose recorded (x, y) differ by more than 1 m -- evidence that a
    reused key denotes a different physical agent, confirming the files MUST be
    handled separately (which the loader does).
    """
    if len(scenes) < 2:
        return {"applicable": False}
    a, b = scenes[0], scenes[1]
    fa, fb = set(a.frames.tolist()), set(b.frames.tolist())
    pa, pb = set(a.ped_ids.tolist()), set(b.ped_ids.tolist())
    fa_idx = {f: i for i, f in enumerate(a.frames.tolist())}
    fb_idx = {f: i for i, f in enumerate(b.frames.tolist())}
    frame_overlap = fa & fb
    diverging = 0
    compared = 0
    for f in frame_overlap:
        da = a.by_frame[fa_idx[f]]
        db = b.by_frame[fb_idx[f]]
        for pid in set(da.keys()) & set(db.keys()):
            compared += 1
            if float(np.linalg.norm(da[pid] - db[pid])) > 1.0:
                diverging += 1
    return {
        "applicable": True,
        "n_frame_overlap": len(frame_overlap),
        "n_ped_overlap": len(pa & pb),
        "n_keys_in_both": compared,
        "n_diverging_gt1m": diverging,
        "diverge_frac": (diverging / compared) if compared else 0.0,
    }


def anchor_report(obs_len: int, pred_len: int, sim_dt: float,
                  sgan_dt: float) -> Dict[str, object]:
    """Gap 4: prediction index grid for the open-loop config + phase flag.

    RQ1a uses sim_dt == sgan_dt, so the observer samples every frame and the GT
    indexing has no sub-grid phase. Reports the indices and flags a mismatch.
    """
    # A non-integer dt ratio is itself a phase mismatch (no clean step grid);
    # _steps_for_interval raises in that case, so catch it and report the
    # mismatch instead of crashing -- the function's contract is to FLAG a
    # mismatch, not to require a clean ratio.
    try:
        stride = _steps_for_interval(sgan_dt, sim_dt)
        pred_indices = (stride * np.arange(1, pred_len + 1) - 1).tolist()
        future_offsets = (stride * np.arange(1, pred_len + 1)).tolist()
    except ValueError:
        stride = None
        pred_indices = []
        future_offsets = []
    return {
        "sim_dt": sim_dt,
        "sgan_dt": sgan_dt,
        "stride": stride,
        "phase_mismatch": not np.isclose(sim_dt, sgan_dt),
        "pred_indices": pred_indices,
        "future_offsets": future_offsets,
        "obs_len": obs_len,
    }


def _fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets", help="ETH/UCY data root")
    ap.add_argument("--scene", default="all",
                    help="scene name (eth/hotel/univ/zara1/zara2) or 'all'")
    ap.add_argument("--seq-len", type=int, default=20, help="obs_len+pred_len")
    ap.add_argument("--obs-len", type=int, default=8)
    ap.add_argument("--pred-len", type=int, default=12)
    ap.add_argument("--dt", type=float, default=None,
                    help="override the physical cadence [s] uniformly for all "
                         "scenes (diagnostic); default uses the per-scene "
                         "scene_dt (eth=0.8, others=0.4)")
    args = ap.parse_args()

    scenes = (list(SCENE_TEST_FILES) if args.scene == "all" else [args.scene])
    root = Path(args.root)

    mode = (f"uniform dt={args.dt}s (override)" if args.dt is not None
            else "per-scene cadence (scene_dt)")
    print("=" * 78)
    print(f"ETH/UCY pre-processing validation  root={root}  seq_len={args.seq_len}")
    print(f"cadence mode: {mode};  ADE/FDE protocol dt={SGAN_PROTOCOL_DT}s "
          "(uniform, dt-independent -- unaffected)")
    print("=" * 78)

    any_flag = False
    for name in scenes:
        try:
            objs = load_scene(name, root=root)
        except (KeyError, FileNotFoundError, OSError) as e:
            print(f"\n[{name}] SKIP: {e}")
            continue

        dt_used = args.dt if args.dt is not None else scene_dt(name)
        print(f"\n### scene={name}  ({len(objs)} file(s))  "
              f"cadence dt={dt_used}s (x{dt_used / SGAN_PROTOCOL_DT:.2g} of "
              f"{SGAN_PROTOCOL_DT}s protocol step)")
        for i, sc in enumerate(objs):
            sr = scene_cadence_report(sc, dt=dt_used)
            gr = frame_gap_report(sc)
            wr = window_straddle_report(sc, args.seq_len)
            any_flag = any_flag or sr["flagged"]
            if sr["flagged"]:
                tag = ("  <<< CADENCE-FLAG (p90 moving speed outside "
                       "%.1f-%.1f m/s)" % MOVING_SPEED_BAND)
            elif sr["low_median"]:
                tag = "  (note: low median -- loitering/slow crowd; walkers normal)"
            else:
                tag = ""
            print(f"  file[{i}] {Path(sc.source).name}: "
                  f"frames={sc.n_frames} peds={len(sc.ped_ids)} "
                  f"frame_step={_fmt(sr['frame_step'],1)}")
            print(f"    [1 cadence] walk median={_fmt(sr['median'])} "
                  f"p90={_fmt(sr['p90'])} p95={_fmt(sr['p95'])} m/s "
                  f"(n={sr['n_speeds']}){tag}")
            print(f"    [3 holes ] gaps={gr['gaps']} n_holes={gr['n_holes']}; "
                  f"emitted_windows={wr['emitted_windows']} "
                  f"straddling={wr['straddling_windows']} "
                  f"({_fmt(100*wr['straddle_frac'],1)}%)")

        if len(objs) >= 2:
            ur = univ_overlap_report(objs)
            print(f"  [2 concat] frame_overlap={ur['n_frame_overlap']} "
                  f"ped_overlap={ur['n_ped_overlap']} "
                  f"keys_in_both={ur['n_keys_in_both']} "
                  f"diverging>1m={ur['n_diverging_gt1m']} "
                  f"({_fmt(100*ur['diverge_frac'],1)}%)")

    # The anchor check is about the EVALUATION PROTOCOL dt (sim_dt vs sgan_dt),
    # not the physical cadence above, so it always uses SGAN_PROTOCOL_DT.
    ar = anchor_report(args.obs_len, args.pred_len, SGAN_PROTOCOL_DT, SGAN_PROTOCOL_DT)
    print(f"\n[4 anchor] sim_dt={ar['sim_dt']} sgan_dt={ar['sgan_dt']} "
          f"stride={ar['stride']} phase_mismatch={ar['phase_mismatch']}")
    print(f"    pred_indices={ar['pred_indices']}")
    print(f"    future_offsets={ar['future_offsets']}")
    print("    (RQ1a uses sim_dt==sgan_dt => no sub-grid phase; the 0.1->0.4 "
          "downsampling exists only in the closed-loop sim.)")

    print("\n" + "=" * 78)
    print("SUMMARY: " + ("CADENCE FLAG raised -- a scene's MOVING speed (p90) is "
                         "outside the plausible band at its per-scene cadence; its "
                         "annotation cadence (or, in principle, coordinate scale) "
                         "may not match the 0.4 s SGAN step. NOTE: ADE/FDE is "
                         "dt-independent and unaffected." if any_flag
                         else "no cadence flags; every scene's moving speed (p90) "
                         "is plausible at its per-scene cadence. A 'low median' "
                         "note marks a genuine loitering crowd, not an error."))


if __name__ == "__main__":
    main()
