"""Tests for the ETH/UCY pre-processing inspection tool (synthetic scenes).

The diagnostics are pure functions over ``SceneTrajectories`` built in-process, so
the suite never touches the (gitignored) real ETH/UCY files. They pin the four
gap checks the tool reports: coordinate scale flag, missing-frame straddle, the
univ two-file key reuse / coordinate divergence, and the open-loop anchor grid.
"""
import numpy as np

from src.datasets.eth_ucy_loader import SceneTrajectories
from examples.inspect_eth_ucy_data import (
    anchor_report,
    frame_gap_report,
    scene_scale_report,
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


def test_scale_report_flags_only_implausible_median():
    ok = scene_scale_report(_const_velocity_scene(1.3), dt=0.4)
    assert np.isclose(ok["median"], 1.3, atol=1e-6)
    assert ok["flagged"] is False

    fast = scene_scale_report(_const_velocity_scene(2.5), dt=0.4)  # eth-like
    assert np.isclose(fast["median"], 2.5, atol=1e-6)
    assert fast["flagged"] is True

    slow = scene_scale_report(_const_velocity_scene(0.5), dt=0.4)  # univ-like
    assert slow["flagged"] is True


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


def test_scale_report_does_not_scale_flag_empty_speed_scene():
    """Zero adjacent-step pairs (a <2-frame file, or one whose only gaps are
    holes) is a missing-DATA condition, not a coordinate-scale mismatch, so it
    must NOT raise the SCALE flag (which would misattribute missing data)."""
    one_frame = _scene([0], [{1: (0.0, 0.0)}])
    rep = scene_scale_report(one_frame, dt=0.4)
    assert rep["n_speeds"] == 0
    assert rep["flagged"] is False


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
