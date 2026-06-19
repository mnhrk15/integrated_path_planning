"""Tests for the ETH/UCY pre-processing inspection tool (synthetic scenes).

The diagnostics are pure functions over ``SceneTrajectories`` built in-process, so
the suite never touches the (gitignored) real ETH/UCY files. They pin the four
gap checks the tool reports: annotation-cadence flag (on the moving-speed p90, not
the median), missing-frame straddle, the univ two-file key reuse / coordinate
divergence, and the open-loop anchor grid.
"""
import numpy as np

from src.datasets.eth_ucy_loader import SceneTrajectories
from examples.inspect_eth_ucy_data import (
    MOVING_SPEED_BAND,
    anchor_report,
    frame_gap_report,
    scene_cadence_report,
    univ_overlap_report,
    window_straddle_report,
)


def _scene(frames, peds_per_frame, source="syn"):
    """frames: list of frame ids; peds_per_frame: list[dict[pid -> (x, y)]]."""
    frames = np.asarray(frames, dtype=float)
    by_frame = [{int(pid): np.asarray(xy, dtype=float) for pid, xy in d.items()}
                for d in peds_per_frame]
    pids = sorted({pid for d in peds_per_frame for pid in d})
    return SceneTrajectories(frames=frames, ped_ids=np.asarray(pids, dtype=int),
                             by_frame=by_frame, source=source)


def _const_velocity_scene(speed_ms, n=10, frame_step=10, dt=0.4):
    """One ped walking in +x at ``speed_ms`` on a uniform grid."""
    disp = speed_ms * dt  # per-step displacement [m]
    frames = [i * frame_step for i in range(n)]
    peds = [{1: (i * disp, 0.0)} for i in range(n)]
    return _scene(frames, peds)


def _loiter_scene(walk_speed=1.3, n_walk=3, n_stand=7, n=12, frame_step=10, dt=0.4):
    """A loitering crowd: ``n_walk`` peds walk at ``walk_speed``, ``n_stand`` peds
    stand still. Low MEDIAN speed (mostly standing) but a normal MOVING p90 -- the
    univ students plaza signature that must NOT raise a cadence flag.
    """
    disp = walk_speed * dt
    frames = [i * frame_step for i in range(n)]
    peds = []
    for i in range(n):
        d = {}
        for w in range(n_walk):                      # walkers move in +x
            d[w] = (i * disp, float(w))
        for s in range(n_stand):                     # standers fixed in place
            d[100 + s] = (0.0, 10.0 + float(s))
        peds.append(d)
    return _scene(frames, peds)


def test_cadence_report_flags_on_moving_speed_not_median():
    ok = scene_cadence_report(_const_velocity_scene(1.3), dt=0.4)
    assert np.isclose(ok["median"], 1.3, atol=1e-6)
    assert ok["flagged"] is False and ok["low_median"] is False

    # eth-like: the WHOLE distribution is shifted ~2x -> p90 implausibly fast.
    fast = scene_cadence_report(_const_velocity_scene(3.0), dt=0.4)
    assert fast["p90"] > MOVING_SPEED_BAND[1]
    assert fast["flagged"] is True

    # Correcting the cadence (here, halving the implied speed via dt=0.8) brings a
    # uniformly-fast scene back into the band -- the eth=0.8 s fix in miniature.
    corrected = scene_cadence_report(_const_velocity_scene(3.0), dt=0.8)
    assert corrected["flagged"] is False

    # Uniformly slow (no normal walkers at all) -> p90 too low -> still a flag.
    slow = scene_cadence_report(_const_velocity_scene(0.5), dt=0.4)
    assert slow["p90"] < MOVING_SPEED_BAND[0]
    assert slow["flagged"] is True


def test_cadence_report_does_not_flag_genuine_loitering_crowd():
    """univ signature: low median (standing crowd) but normal moving p90. Must be
    reported as an informational low_median note, NEVER a cadence flag."""
    rep = scene_cadence_report(_loiter_scene(walk_speed=1.3), dt=0.4)
    assert rep["median"] < MOVING_SPEED_BAND[0]      # dragged down by standers
    assert MOVING_SPEED_BAND[0] <= rep["p90"] <= MOVING_SPEED_BAND[1]  # walkers OK
    assert rep["flagged"] is False
    assert rep["low_median"] is True


def test_frame_gap_report_counts_holes():
    # uniform step 10 with one big gap (a frame hole) at the end
    scene = _scene([0, 10, 20, 60], [{1: (0, 0)} for _ in range(4)])
    rep = frame_gap_report(scene)
    assert rep["frame_step"] == 10.0
    assert rep["n_holes"] == 1            # the 40-frame jump (20 -> 60)
    assert rep["gaps"][10.0] == 2 and rep["gaps"][40.0] == 1


def test_window_straddle_flags_window_spanning_a_hole():
    # ped present in all three frames; frames 0,10,30 -> the window straddles a
    # hole (gap 20 != grid step 10), so its physical horizon is stretched.
    scene = _scene([0, 10, 30], [{1: (0, 0)}, {1: (0.5, 0)}, {1: (1.5, 0)}])
    rep = window_straddle_report(scene, seq_len=3)
    assert rep["emitted_windows"] == 1
    assert rep["straddling_windows"] == 1
    assert rep["straddle_frac"] == 1.0

    # a clean uniform scene straddles nothing
    clean = window_straddle_report(_const_velocity_scene(1.3, n=10), seq_len=4)
    assert clean["straddling_windows"] == 0


def test_univ_overlap_detects_reused_keys_with_diverging_coords():
    # two "files" reusing frame 0 / ped 1 but at far-apart coordinates
    a = _scene([0, 10], [{1: (0.0, 0.0)}, {1: (0.5, 0.0)}])
    b = _scene([0, 10], [{1: (20.0, 20.0)}, {1: (20.5, 20.0)}])
    rep = univ_overlap_report([a, b])
    assert rep["applicable"] is True
    assert rep["n_frame_overlap"] == 2 and rep["n_ped_overlap"] == 1
    assert rep["n_keys_in_both"] == 2
    assert rep["n_diverging_gt1m"] == 2      # same key, >1 m apart -> different agent
    assert rep["diverge_frac"] == 1.0


def test_univ_overlap_not_applicable_for_single_file():
    assert univ_overlap_report([_const_velocity_scene(1.3)])["applicable"] is False


def test_cadence_report_does_not_flag_empty_speed_scene():
    """Zero adjacent-step pairs (a <2-frame file, or one whose only gaps are
    holes) is a missing-DATA condition, not a cadence/scale mismatch, so it must
    NOT raise the flag (nor the low_median note) -- that would misattribute
    missing data."""
    one_frame = _scene([0], [{1: (0.0, 0.0)}])
    rep = scene_cadence_report(one_frame, dt=0.4)
    assert rep["n_speeds"] == 0
    assert rep["flagged"] is False
    assert rep["low_median"] is False


def test_anchor_report_no_phase_when_dts_equal_and_phase_when_not():
    eq = anchor_report(obs_len=8, pred_len=12, sim_dt=0.4, sgan_dt=0.4)
    assert eq["stride"] == 1 and eq["phase_mismatch"] is False
    assert eq["pred_indices"] == list(range(12))
    assert eq["future_offsets"] == list(range(1, 13))

    mism = anchor_report(obs_len=8, pred_len=12, sim_dt=0.1, sgan_dt=0.4)
    assert mism["stride"] == 4 and mism["phase_mismatch"] is True


def test_anchor_report_flags_non_multiple_dts_without_crashing():
    """A non-integer dt ratio is itself a phase mismatch; anchor_report must
    FLAG it (its documented contract) rather than crash in _steps_for_interval."""
    rep = anchor_report(obs_len=8, pred_len=12, sim_dt=0.3, sgan_dt=0.4)
    assert rep["phase_mismatch"] is True
    assert rep["stride"] is None
    assert rep["pred_indices"] == [] and rep["future_offsets"] == []
