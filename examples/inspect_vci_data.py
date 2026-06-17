#!/usr/bin/env python3
"""Inspect downloaded VCI (DUT/CITR) data against the loader's assumptions.

Run this once after manually downloading the datasets (see
``scripts/download_vci_data.sh``) and extracting into ``datasets/vci_dut`` /
``datasets/vci_citr``. It validates -- on the *real* files -- the structural
assumptions ``src/datasets/vci_loader.py`` only checks against synthetic CSVs:

  * STRUCTURE      clip count, per-scenario breakdown, ped/veh track counts
  * COLUMNS        real CSV headers vs the expected column sets
  * PHYSICAL SANITY metre units via per-clip coord span / walking ~1.3 m/s /
                   vehicle ~1-3 m/s from positions / raw psi_est is radians
  * FPS ESTIMATE   (CITR only) frame_span/clip_seconds vs a walking-speed
                   self-consistency estimate -- CITR's fps is unstated upstream
  * SPEED MISMATCH vehicle speed quantiles vs the AVEC ego (~5-6 m/s), the
                   input to the RQ2 velocity-extrapolation decision
  * MISSINGNESS    NaN (absent-span) rate, empty/unpaired clips

Output is stdout only (no files written). The report functions are pure (take
ClipTracks, return lines) so they are unit-testable on synthetic trees.

Usage:
    .venv/bin/python examples/inspect_vci_data.py --dataset dut
    .venv/bin/python examples/inspect_vci_data.py --dataset citr --fps 29.97
    .venv/bin/python examples/inspect_vci_data.py --dataset citr   # estimates fps
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.vci_loader import (  # noqa: E402
    PED_SUFFIX,
    VEH_SUFFIX,
    AgentTracks,
    ClipTracks,
    _discover_clip_files,
    agent_speed_samples,
    load_vci_clips,
    vehicle_speed_samples,
)

# A tuple entry is an alias group satisfied by any one of its names: the real
# filtered CSVs spell pedestrian velocity vx_est/vy_est, the README xv_est/yv_est
# (mirrors PED_V*_ALIASES in the loader, so neither spelling reads as a mismatch).
PED_EXPECTED = ["id", "frame", "label", "x_est", "y_est",
                ("vx_est", "xv_est"), ("vy_est", "yv_est")]
VEH_EXPECTED = ["id", "frame", "label", "x_est", "y_est", "psi_est", "vel_est"]

DEFAULT_ROOTS = {"dut": "datasets/vci_dut", "citr": "datasets/vci_citr"}

# Plausible ranges for the metre-unit sanity check; outside -> flag for review.
WALK_MEDIAN_RANGE = (0.5, 2.5)  # normal walking ~1.3 m/s
VEH_MEDIAN_RANGE = (0.3, 6.0)  # DUT/CITR vehicles ~1-3 m/s
COORD_SPAN_PIXELS = 200.0  # a metre scene spans tens of m; px would be ~1000s
# A single-frame clip has frame span 1.0 (max == min): no elapsed time, so it
# carries no rate information and would drag the fps-estimate median toward ~0.
# Both fps estimates (estimate_citr_fps, report_fps_estimate) exclude it.
MIN_FRAME_SPAN = 1.0


# --- small helpers -----------------------------------------------------------


def _all_agent_speeds(tracks_list: List[Optional[AgentTracks]]) -> np.ndarray:
    """Position-differenced speeds [m/s] pooled over ped *or* veh tracks.

    dt defaults to each track's own grid step (via agent_speed_samples), so this
    is a unit check on the *positions* -- the quantity the loader actually
    consumes for calibration -- independent of any recorded speed column.
    """
    # agent_speed_samples already returns [] for T<2, so only filter None here.
    parts = [agent_speed_samples(t) for t in tracks_list if t is not None]
    return np.concatenate(parts) if parts else np.array([])


def _recorded_veh_speeds(clips: List[ClipTracks]) -> np.ndarray:
    """All recorded vehicle 'vel' samples [m/s] (the channel, not positions)."""
    parts = [vehicle_speed_samples(c.veh) for c in clips if c.veh is not None]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.array([])


def _max_axis_span(clips: List[ClipTracks]) -> float:
    """Largest per-clip, per-axis coordinate extent [m] (origin-insensitive).

    Pooling a single global min/max over all clips would union their bounding
    boxes, so clips with different origins inflate the span and falsely trip the
    pixel flag. Taking the max over (clip, axis) extents keeps it a true
    per-scene size: tens of m for metres, ~1000s for pixels.
    """
    spans = []
    for c in clips:
        for tr in (c.ped, c.veh):
            if tr is None or tr.positions.size == 0:
                continue
            for axis in (0, 1):
                col = tr.positions[..., axis]
                finite = col[np.isfinite(col)]
                if finite.size:
                    spans.append(float(finite.max() - finite.min()))
    return max(spans) if spans else float("nan")


def _read_single_column(path: Path, col: str) -> np.ndarray:
    """Finite float values of one column from a raw CSV; empty array if unusable.

    Header whitespace is stripped to match the loader's ``_read_agent_csv``, so a
    space-padded header (e.g. ``"id, frame, ..."``, which the loader tolerates and
    ingests) resolves the same column rather than silently missing it -- otherwise
    the inspector's raw checks would be vacuously skipped on exactly the files the
    loader still consumes. A missing column, an unparseable/empty file, or a
    non-numeric cell all yield an empty array rather than a traceback -- the
    loader's own validation surfaces the real diagnostic instead. The float
    coercion is inside the guard so a non-numeric cell (ValueError) is demoted too,
    and NaN cells are dropped rather than propagated into a NaN result. Shared by
    the frame-span and psi-range checks.
    """
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        v = df[col].to_numpy(dtype=float)
    except (ValueError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return np.array([])
    return v[np.isfinite(v)]


def _raw_frame_span(path: Path) -> float:
    """Frame count span (max-min+1) from a raw CSV's 'frame' column.

    Returns 0.0 if the file is unparseable, has no 'frame' column, or the column
    is non-numeric/empty, so the loader's own validation surfaces the real
    diagnostic rather than a raw pandas traceback from the inspector. Assumes a
    single contiguous recording -- a frame discontinuity (concatenated
    sub-sequences) would inflate the span and bias the fps estimate high; the
    duration-independent walking self-consistency estimate is the safeguard.
    """
    f = _read_single_column(path, "frame")
    return (float(f.max()) - float(f.min()) + 1.0) if f.size else 0.0


def _first_positive_frame_span(paths: List[Optional[Path]]) -> Optional[float]:
    """First usable raw frame span from candidate CSVs."""
    for path in paths:
        if path is None:
            continue
        span = _raw_frame_span(path)
        if span > MIN_FRAME_SPAN:  # skip single-frame clips (no rate info)
            return span
    return None


def _raw_psi_range(clips: List[ClipTracks]) -> Optional[Tuple[float, float]]:
    """Min/max of the *raw* psi_est column across vehicle CSVs (radian check).

    Reads the raw column, not ``ClipTracks.veh.extra['psi']`` -- the loader
    unwraps and wraps that channel into (-pi, pi], which makes a radian check
    vacuously pass even for degree-valued input. Reading the file the loader
    consumed is what lets degrees (|.|>pi) or other units actually show up.
    """
    los, his = [], []
    for c in clips:
        if c.veh_path is None:
            continue
        v = _read_single_column(c.veh_path, "psi_est")
        if v.size:
            los.append(float(v.min()))
            his.append(float(v.max()))
    return (min(los), max(his)) if los else None


def _quantile_line(arr: np.ndarray, qs=(5, 25, 50, 75, 95)) -> str:
    if arr.size == 0:
        return "no samples"
    vals = np.percentile(arr, qs)
    return "  ".join(f"p{q}={v:.2f}" for q, v in zip(qs, vals))


def estimate_citr_fps(root, clip_seconds: float) -> Optional[float]:
    """Coarse fps from median raw frame_span / clip_seconds (CITR pre-load).

    Returns None when no clip yields a positive frame span; the caller must then
    ask the user for --fps rather than fabricate a rate (it must NOT fall back to
    DUT's drone fps for CITR, which would silently resample at the wrong rate).
    """
    files = _discover_clip_files(root, "citr")
    spans = []
    for paths in files.values():
        span = _first_positive_frame_span([paths.get("ped"), paths.get("veh")])
        if span is not None:
            spans.append(span)
    if not spans:
        return None
    return float(np.median(spans)) / clip_seconds


# --- report sections (pure: ClipTracks -> lines) -----------------------------


def report_structure(clips: List[ClipTracks]) -> List[str]:
    lines = [f"clips: {len(clips)}"]
    by_scn = {}
    for c in clips:
        by_scn.setdefault(c.scenario, 0)
        by_scn[c.scenario] += 1
    for scn in sorted(by_scn, key=lambda s: (s or "")):
        lines.append(f"  scenario {scn!r}: {by_scn[scn]} clips")
    for c in clips:
        n_ped = c.ped.ids.size if c.ped is not None else "-"
        n_veh = c.veh.ids.size if c.veh is not None else "-"
        # Grid length from whichever side has the longer grid (a ped-empty clip
        # with a populated veh must not report grid=0).
        n_t = max(
            (tr.times.size for tr in (c.ped, c.veh) if tr is not None),
            default=0,
        )
        tag = "" if (c.ped is not None and c.veh is not None) else "  [UNPAIRED]"
        lines.append(
            f"  {c.scenario or '.'}/{c.clip}: ped={n_ped} veh={n_veh} grid={n_t}{tag}"
        )
    return lines


def report_columns(clips: List[ClipTracks]) -> List[str]:
    """Compare each real CSV header to the expected columns (raw, not loaded)."""
    lines = []
    checks = [("ped", PED_EXPECTED), ("veh", VEH_EXPECTED)]
    mismatches = 0
    for c in clips:
        for kind, expected in checks:
            path = c.ped_path if kind == "ped" else c.veh_path
            if path is None:
                continue
            try:
                cols = [str(x).strip() for x in pd.read_csv(path, nrows=0).columns]
            except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                # The CITR/strict=False path keeps unreadable files; report them
                # here (this section's purpose) instead of crashing the inspector.
                # OSError covers a file removed/unreadable between load and report
                # (the loader's own _load already demotes OSError the same way).
                mismatches += 1
                lines.append(f"  {c.scenario or '.'}/{c.clip} [{kind}]: UNREADABLE ({type(e).__name__})")
                continue
            # An alias group (tuple) is satisfied by any one of its names; a bare
            # string must be present. Report a missing group as "a|b".
            missing = []
            for e in expected:
                group = (e,) if isinstance(e, str) else e
                if not any(name in cols for name in group):
                    missing.append("|".join(group))
            allowed = {name for e in expected for name in ((e,) if isinstance(e, str) else e)}
            extra = [a for a in cols if a not in allowed]
            if missing or extra:
                mismatches += 1
                lines.append(
                    f"  {c.scenario or '.'}/{c.clip} [{kind}]: "
                    f"missing={missing} extra={extra}"
                )
    lines.insert(0, f"header mismatches: {mismatches}" + ("" if mismatches else "  (all OK)"))
    return lines


def report_physical_sanity(clips: List[ClipTracks]) -> List[str]:
    lines = []

    span = _max_axis_span(clips)
    flag = "  <- PIXELS? (a metre scene spans tens of m)" if (
        np.isfinite(span) and span > COORD_SPAN_PIXELS
    ) else ""
    lines.append(f"max per-clip axis span: {span:.1f} m{flag}")

    ped = _all_agent_speeds([c.ped for c in clips])
    if ped.size:
        med = float(np.median(ped))
        flag = "" if WALK_MEDIAN_RANGE[0] <= med <= WALK_MEDIAN_RANGE[1] else "  <- off ~1.3 m/s"
        lines.append(f"walking speed median: {med:.2f} m/s  (n={ped.size}){flag}")
    else:
        lines.append("walking speed median: no pedestrian samples")

    # The metre-unit check for vehicles differences positions (the quantity
    # whose unit is in question), same as pedestrians. The recorded 'vel'
    # channel is shown alongside as an independent cross-check but may carry its
    # own units, so it does not drive the flag.
    veh = _all_agent_speeds([c.veh for c in clips])
    if veh.size:
        med = float(np.median(veh))
        flag = "" if VEH_MEDIAN_RANGE[0] <= med <= VEH_MEDIAN_RANGE[1] else "  <- off ~1-3 m/s"
        lines.append(f"vehicle speed median (from positions): {med:.2f} m/s  (n={veh.size}){flag}")
    else:
        lines.append("vehicle speed median (from positions): no vehicle samples")
    veh_rec = _recorded_veh_speeds(clips)
    if veh_rec.size:
        lines.append(f"vehicle speed median (recorded 'vel'): {float(np.median(veh_rec)):.2f} m/s")

    psi = _raw_psi_range(clips)
    if psi is not None:
        lo, hi = psi
        # Raw psi_est should already be radians in [-pi, pi]; degree-valued
        # (|.|>pi) input trips this. Reading the raw column (not the loader's
        # unwrapped channel) is what makes it detectable. One-sided check: a
        # degree clip that barely turns (range << pi) is indistinguishable from
        # radians here, so absence of the flag is not proof of radians.
        within = (-np.pi - 1e-2) <= lo and hi <= (np.pi + 1e-2)
        flag = "" if within else "  <- not radians? (expected [-pi, pi])"
        lines.append(f"raw heading range: [{lo:.2f}, {hi:.2f}] rad{flag}")

    return lines


def report_fps_estimate(
    clips: List[ClipTracks], clip_seconds: float, dt: float
) -> List[str]:
    """Two independent CITR fps estimates; agreement = confidence (not a verdict)."""
    cur = clips[0].fps if clips else float("nan")
    lines = [f"loaded with fps={cur:.2f}; clip_seconds assumption={clip_seconds:.1f} s"]
    dur_ests, walk_ests = [], []
    for c in clips:
        span = _first_positive_frame_span([c.ped_path, c.veh_path])
        if span is not None:
            dur_ests.append(span / clip_seconds)
        if c.ped is not None and c.ped.positions.shape[0] >= 2:
            # Grid speed scales ~linearly with fps (the grid's real-time step is
            # target_dt but frames map to t via fps), so nudge cur toward 1.3 m/s.
            sp = agent_speed_samples(c.ped, dt=dt)
            med = float(np.median(sp)) if sp.size else 0.0
            if med > 0:
                walk_ests.append(cur * 1.3 / med)
    if dur_ests:
        lines.append(
            f"  from frame_span/{clip_seconds:.0f}s: median {np.median(dur_ests):.2f} "
            f"(range {min(dur_ests):.2f}-{max(dur_ests):.2f})"
        )
    if walk_ests:
        lines.append(
            f"  from walking=1.3 m/s self-consistency: median {np.median(walk_ests):.2f} "
            f"(range {min(walk_ests):.2f}-{max(walk_ests):.2f})"
        )
    lines.append(
        "  (the frame_span estimate assumes every clip is exactly clip_seconds long, so "
        "its spread mixes duration variance with fps; the walking self-consistency "
        "estimate is duration-independent and the more reliable of the two)"
    )
    return lines


def report_speed_mismatch(clips: List[ClipTracks], ego_speed: float) -> List[str]:
    veh = _recorded_veh_speeds(clips)
    lines = [f"vehicle speed quantiles [m/s]: {_quantile_line(veh)}"]
    if veh.size:
        med = float(np.median(veh))
        p95 = float(np.percentile(veh, 95))
        lines.append(
            f"  vs ego {ego_speed:.1f} m/s: median ratio {med / ego_speed:.2f}, "
            f"p95={p95:.2f} {'reaches' if p95 >= ego_speed - 1.0 else 'below'} ego domain"
        )
        lines.append("  -> informs RQ2 velocity-extrapolation (speed term / stratify / cap)")
    return lines


def report_missingness(clips: List[ClipTracks]) -> List[str]:
    lines = []
    nan_rates = []
    empty = unpaired = 0
    for c in clips:
        if c.ped is None or c.veh is None:
            unpaired += 1
        for tr in (c.ped, c.veh):
            if tr is None or tr.positions.size == 0:
                if tr is not None:
                    empty += 1
                continue
            nan_rates.append(float(np.isnan(tr.positions).mean()))
    if nan_rates:
        lines.append(
            f"NaN (absent-span) rate: mean {np.mean(nan_rates):.2%} "
            f"max {np.max(nan_rates):.2%}"
        )
    lines.append(f"empty tracks: {empty}   unpaired clips: {unpaired}")
    return lines


def build_report(clips, dataset, ego_speed, dt, clip_seconds) -> List[tuple]:
    sections = [
        ("STRUCTURE", report_structure(clips)),
        ("COLUMNS", report_columns(clips)),
        ("PHYSICAL SANITY", report_physical_sanity(clips)),
    ]
    if dataset == "citr":
        sections.append(("FPS ESTIMATE", report_fps_estimate(clips, clip_seconds, dt)))
    sections.append(("SPEED MISMATCH vs EGO", report_speed_mismatch(clips, ego_speed)))
    sections.append(("MISSINGNESS", report_missingness(clips)))
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["dut", "citr"])
    parser.add_argument("--root", default=None, help="dataset root (default datasets/vci_<ds>)")
    parser.add_argument("--fps", type=float, default=None,
                        help="frame rate; DUT defaults to 23.98, CITR is estimated if omitted")
    parser.add_argument("--target-dt", type=float, default=0.4)
    parser.add_argument("--max-clips", type=int, default=None, help="cap clips (smoke)")
    parser.add_argument("--ego-speed", type=float, default=5.5, help="AVEC ego speed [m/s]")
    parser.add_argument("--clip-seconds", type=float, default=20.0,
                        help="assumed CITR clip duration for fps estimation")
    args = parser.parse_args()
    # `not (x > 0)` (rather than `x <= 0`) also rejects NaN, since `nan > 0` is
    # False: `--fps nan` would otherwise slip through and silently poison every
    # track instead of failing fast like the bounds below.
    if not (args.ego_speed > 0):
        parser.error("--ego-speed must be a positive number")
    if not (args.clip_seconds > 0):
        parser.error("--clip-seconds must be a positive number")
    # fps <= 0 makes frame/fps non-increasing, which np.interp does not validate
    # (it returns silently wrong positions); target_dt <= 0 divides by zero when
    # building the resample grid -- guard both the way the two estimates above are.
    if not (args.target_dt > 0):
        parser.error("--target-dt must be a positive number")
    if args.fps is not None and not (args.fps > 0):
        parser.error("--fps must be a positive number")
    # Applied as clips[:max_clips]; a negative value would silently drop trailing
    # clips and 0 would look like 'no clips found', so require a positive cap.
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("--max-clips must be positive")

    root = args.root or DEFAULT_ROOTS[args.dataset]
    if not Path(root).exists():
        print(f"Dataset root {root!r} not found. Download per scripts/download_vci_data.sh "
              f"and extract into {root}/.")
        return

    fps = args.fps
    if fps is None and args.dataset == "citr":
        fps = estimate_citr_fps(root, args.clip_seconds)
        if fps is None:
            print("[fps] could not estimate CITR fps (no readable frame spans under "
                  f"{root!r}). Pass --fps explicitly.")
            return
        print(f"[fps] CITR fps not given; using estimate ~{fps:.2f} "
              f"(frame_span/{args.clip_seconds:.0f}s). Pass --fps to override.")

    # strict=False so one malformed CSV does not abort the whole scan before the
    # COLUMNS section (which exists to diagnose exactly that) can report it.
    clips = load_vci_clips(root, args.dataset, fps=fps, target_dt=args.target_dt,
                           strict=False)
    if args.max_clips is not None:
        clips = clips[: args.max_clips]
    if not clips:
        print(f"No VCI clips found under {root!r} (looked for *{PED_SUFFIX} / *{VEH_SUFFIX}).")
        return

    for title, lines in build_report(clips, args.dataset, args.ego_speed,
                                     args.target_dt, args.clip_seconds):
        print(f"\n=== {title} ===")
        for ln in lines:
            print(ln)


if __name__ == "__main__":
    main()
