"""Synthetic E2E for examples/inspect_vci_data.py report functions.

The script is loaded by path (examples/ is not a package); its report functions
are pure (ClipTracks -> lines), so they are exercised on synthetic clip trees
without any real download. The key assertion is that the metre-unit sanity check
flags pixel-scale coordinates.
"""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "inspect_vci_data",
    Path(__file__).resolve().parent.parent / "examples" / "inspect_vci_data.py",
)
inspect_vci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inspect_vci)

PED_SUFFIX = inspect_vci.PED_SUFFIX
VEH_SUFFIX = inspect_vci.VEH_SUFFIX

# Metre-scale clip: coords span a few m, walking 1 m/s at fps=10/target_dt=0.5.
METRE_PED = (
    "id,frame,label,x_est,y_est,xv_est,yv_est\n"
    "1,0,ped,0.0,0.0,1.0,0.0\n"
    "1,10,ped,1.0,0.0,1.0,0.0\n"
    "1,20,ped,2.0,0.0,1.0,0.0\n"
)
METRE_VEH = (
    "id,frame,label,x_est,y_est,psi_est,vel_est\n"
    "1,0,veh,0.0,0.0,0.0,2.0\n"
    "1,10,veh,2.0,0.0,0.0,2.0\n"
)
# Pixel-scale clip: same motion but ~1000x coords -> must trip the PIXELS flag.
PIXEL_PED = (
    "id,frame,label,x_est,y_est,xv_est,yv_est\n"
    "1,0,ped,100.0,100.0,1.0,0.0\n"
    "1,10,ped,600.0,100.0,1.0,0.0\n"
    "1,20,ped,1100.0,100.0,1.0,0.0\n"
)


def _build(tmp_path, ped_text, veh_text=METRE_VEH):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    (d / f"intersection_01{PED_SUFFIX}").write_text(ped_text)
    (d / f"intersection_01{VEH_SUFFIX}").write_text(veh_text)
    return inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)


def test_physical_sanity_flags_pixel_coords(tmp_path):
    clips = _build(tmp_path, PIXEL_PED)
    lines = inspect_vci.report_physical_sanity(clips)
    assert any("PIXELS" in ln for ln in lines)


def test_physical_sanity_passes_metre_coords(tmp_path):
    clips = _build(tmp_path, METRE_PED)
    lines = inspect_vci.report_physical_sanity(clips)
    assert not any("PIXELS" in ln for ln in lines)


def test_build_report_sections_for_dut(tmp_path):
    clips = _build(tmp_path, METRE_PED)
    sections = inspect_vci.build_report(
        clips, "dut", ego_speed=5.5, dt=0.5, clip_seconds=20.0
    )
    titles = [t for t, _ in sections]
    assert "STRUCTURE" in titles and "COLUMNS" in titles
    assert "FPS ESTIMATE" not in titles  # DUT fps is fixed (23.98)
    # COLUMNS should report no mismatch for well-formed headers.
    cols = dict(sections)["COLUMNS"]
    assert any("all OK" in ln for ln in cols)


def test_build_report_includes_fps_for_citr(tmp_path):
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    (d / f"front_01{PED_SUFFIX}").write_text(METRE_PED)
    (d / f"front_01{VEH_SUFFIX}").write_text(METRE_VEH)
    clips = inspect_vci.load_vci_clips(tmp_path, "citr", fps=10.0, target_dt=0.5)
    sections = inspect_vci.build_report(
        clips, "citr", ego_speed=5.5, dt=0.5, clip_seconds=20.0
    )
    assert "FPS ESTIMATE" in [t for t, _ in sections]


# --- CITR fps estimation -----------------------------------------------------


def test_raw_frame_span(tmp_path):
    csv = "id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,0,0,1,0\n1,599,ped,1,0,1,0\n"
    p = tmp_path / "x.csv"
    p.write_text(csv)
    assert inspect_vci._raw_frame_span(p) == 600.0  # max - min + 1


def test_raw_frame_span_drops_nan_and_nonnumeric(tmp_path):
    # A blank 'frame' cell (NaN) must be dropped, not propagated into a NaN span;
    # a wholly non-numeric column must be demoted to 0.0, not crash the inspector.
    nan_frame = tmp_path / "nan.csv"
    nan_frame.write_text(
        "id,frame,label,x_est,y_est,xv_est,yv_est\n"
        "1,0,ped,0,0,1,0\n1,,ped,1,0,1,0\n1,10,ped,2,0,1,0\n"
    )
    assert inspect_vci._raw_frame_span(nan_frame) == 11.0  # blank dropped: 10-0+1

    bad = tmp_path / "bad.csv"
    bad.write_text("id,frame,label,x_est,y_est,xv_est,yv_est\n1,oops,ped,0,0,1,0\n")
    assert inspect_vci._raw_frame_span(bad) == 0.0  # non-numeric -> [] -> 0.0


def test_raw_readers_tolerate_spaced_headers(tmp_path):
    # The loader (_read_agent_csv) strips header whitespace and ingests such
    # files, so the inspector's raw readers must strip too -- otherwise they miss
    # the column the loader uses and silently skip the frame-span / psi checks.
    p = tmp_path / "spaced.csv"
    p.write_text(
        "id, frame, label, x_est, y_est, psi_est, vel_est\n"
        "1,0,veh,0,0,0.1,2.0\n1,10,veh,1,0,0.2,2.0\n"
    )
    assert inspect_vci._raw_frame_span(p) == 11.0  # 'frame' resolved despite spaces
    assert inspect_vci._read_single_column(p, "psi_est").size == 2


def test_estimate_citr_fps_from_frame_span(tmp_path):
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    # frame span 600 over an assumed 20 s clip -> 30 fps
    ped = "id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,0,0,1,0\n1,599,ped,1,0,1,0\n"
    (d / f"c1{PED_SUFFIX}").write_text(ped)
    fps = inspect_vci.estimate_citr_fps(tmp_path, clip_seconds=20.0)
    assert abs(fps - 30.0) < 1e-9


def test_estimate_citr_fps_falls_back_to_vehicle_span(tmp_path):
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    ped = "id,frame,label,x_est,y_est,xv_est,yv_est\n"
    veh = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0,0,0.0,2.0\n1,599,veh,1,0,0.0,2.0\n"
    (d / f"c1{PED_SUFFIX}").write_text(ped)
    (d / f"c1{VEH_SUFFIX}").write_text(veh)
    fps = inspect_vci.estimate_citr_fps(tmp_path, clip_seconds=20.0)
    assert abs(fps - 30.0) < 1e-9


def test_estimate_citr_fps_uses_median_over_clips(tmp_path):
    # spans 200, 200, 800 -> median 200 (mean would be 400). Asymmetric on
    # purpose: the assertion fails if the implementation averaged instead of
    # taking the median. /20 s -> 10 fps.
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    for i, last in enumerate((199, 199, 799)):
        ped = f"id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,0,0,1,0\n1,{last},ped,1,0,1,0\n"
        (d / f"c{i}{PED_SUFFIX}").write_text(ped)
    assert abs(inspect_vci.estimate_citr_fps(tmp_path, 20.0) - 10.0) < 1e-9


def test_estimate_citr_fps_returns_none_when_no_clips(tmp_path):
    # Must NOT fabricate a rate (esp. not DUT's drone fps) for CITR.
    (tmp_path / "trajectories_filtered").mkdir()
    assert inspect_vci.estimate_citr_fps(tmp_path, 20.0) is None


def test_report_fps_estimate_reports_both_methods(tmp_path):
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    ped = "id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,0,0,1,0\n1,200,ped,1,0,1,0\n"
    (d / f"c1{PED_SUFFIX}").write_text(ped)
    clips = inspect_vci.load_vci_clips(tmp_path, "citr", fps=10.0, target_dt=0.5)
    lines = inspect_vci.report_fps_estimate(clips, clip_seconds=20.0, dt=0.5)
    # frame span 201 (0..200) / 20 s -> 10.05 fps; assert the actual value, not
    # just that a line exists, so a broken duration estimate is caught.
    assert any("frame_span" in ln and "10.05" in ln for ln in lines)
    assert any("self-consistency" in ln for ln in lines)  # walking-speed method present


def test_report_fps_estimate_falls_back_to_vehicle_span(tmp_path):
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    ped = "id,frame,label,x_est,y_est,xv_est,yv_est\n"
    veh = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0,0,0.0,2.0\n1,599,veh,1,0,0.0,2.0\n"
    (d / f"c1{PED_SUFFIX}").write_text(ped)
    (d / f"c1{VEH_SUFFIX}").write_text(veh)
    clips = inspect_vci.load_vci_clips(
        tmp_path, "citr", fps=30.0, target_dt=0.5, strict=False
    )
    lines = inspect_vci.report_fps_estimate(clips, clip_seconds=20.0, dt=0.5)
    assert any("frame_span" in ln and "30.00" in ln for ln in lines)


# --- physical-sanity flags and other report sections -------------------------


def test_physical_sanity_flags_fast_walking(tmp_path):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    # 1 m per 0.1 s step -> 10 m/s (off ~1.3) while coords stay small (no PIXELS)
    ped = "id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,0,0,1,0\n1,1,ped,1,0,1,0\n"
    (d / f"c1{PED_SUFFIX}").write_text(ped)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.1)
    lines = inspect_vci.report_physical_sanity(clips)
    assert any("off ~1.3" in ln for ln in lines)
    assert not any("PIXELS" in ln for ln in lines)


def test_physical_sanity_flags_fast_vehicle_from_positions(tmp_path):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    # vehicle moves 10 m in one 0.1 s grid step -> 100 m/s from POSITIONS (off
    # ~1-3); the recorded vel=2.0 is deliberately in-range to prove the flag is
    # driven by position differencing, not the vel channel.
    veh = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0,0,0.0,2.0\n1,1,veh,10,0,0.0,2.0\n"
    (d / f"c1{VEH_SUFFIX}").write_text(veh)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.1)
    lines = inspect_vci.report_physical_sanity(clips)
    assert any("from positions" in ln and "off ~1-3" in ln for ln in lines)


def test_report_columns_detects_missing_and_extra(tmp_path):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    # required cols present (loader accepts) but xv_est/yv_est missing, 'foo' extra
    ped = "id,frame,label,x_est,y_est,foo\n1,0,ped,0,0,9\n1,10,ped,1,0,9\n"
    (d / f"c1{PED_SUFFIX}").write_text(ped)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0)
    lines = inspect_vci.report_columns(clips)
    assert lines[0] == "header mismatches: 1"
    assert any("xv_est" in ln for ln in lines[1:])  # missing reported
    assert any("foo" in ln for ln in lines[1:])  # extra reported


def test_report_speed_mismatch_has_quantiles_and_ego_ratio(tmp_path):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    veh = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0,0,0.0,2.0\n1,10,veh,1,0,0.0,2.0\n"
    (d / f"c1{VEH_SUFFIX}").write_text(veh)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    lines = inspect_vci.report_speed_mismatch(clips, ego_speed=5.5)
    # all vehicle speed samples are 2.0 m/s -> every quantile is 2.00, and the
    # ego ratio is 2.0/5.5 = 0.36. Assert the values (the "quantiles" header is
    # printed even with no samples, so a header-only check would be vacuous).
    assert any("p50=2.00" in ln for ln in lines)
    assert any("ratio 0.36" in ln for ln in lines)


def test_report_missingness_counts_unpaired(tmp_path):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    (d / f"c1{PED_SUFFIX}").write_text(METRE_PED)  # ped only -> unpaired
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    lines = inspect_vci.report_missingness(clips)
    assert any("unpaired clips: 1" in ln for ln in lines)


def test_physical_sanity_flags_degree_psi(tmp_path):
    # Raw psi_est in degrees (0..360) must be flagged, even though the loader
    # silently unwraps/wraps the channel into (-pi, pi]. Reading the raw column
    # is what catches it; reading veh.extra['psi'] would pass vacuously.
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    veh = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0,0,10.0,2.0\n1,10,veh,1,0,350.0,2.0\n"
    (d / f"c1{VEH_SUFFIX}").write_text(veh)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    lines = inspect_vci.report_physical_sanity(clips)
    assert any("not radians" in ln for ln in lines)


def test_physical_sanity_radian_psi_not_flagged(tmp_path):
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    veh = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0,0,-1.5,2.0\n1,10,veh,1,0,1.5,2.0\n"
    (d / f"c1{VEH_SUFFIX}").write_text(veh)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    lines = inspect_vci.report_physical_sanity(clips)
    assert any("raw heading range" in ln and "not radians" not in ln for ln in lines)


def test_physical_sanity_origin_offset_clips_no_false_pixels(tmp_path):
    # Two metre clips ~30 m each but at different origins (0 and 270). A global
    # bounding box would span ~300 m and falsely flag PIXELS; per-clip per-axis
    # spans stay ~30 m and must not flag.
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    a = "id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,0,0,1,0\n1,10,ped,30,0,1,0\n"
    b = "id,frame,label,x_est,y_est,xv_est,yv_est\n1,0,ped,270,0,1,0\n1,10,ped,300,0,1,0\n"
    (d / f"c1{PED_SUFFIX}").write_text(a)
    (d / f"c2{PED_SUFFIX}").write_text(b)
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    lines = inspect_vci.report_physical_sanity(clips)
    assert not any("PIXELS" in ln for ln in lines)


def test_estimate_citr_fps_returns_none_when_all_spans_zero(tmp_path):
    # clips exist but every frame_span is 0 (header-only) -> None via the span>0
    # filter, not the empty-discovery path the sibling test exercises.
    d = tmp_path / "trajectories_filtered" / "vci_front"
    d.mkdir(parents=True)
    (d / f"c1{PED_SUFFIX}").write_text("id,frame,label,x_est,y_est,xv_est,yv_est\n")
    assert inspect_vci.estimate_citr_fps(tmp_path, 20.0) is None


def test_report_columns_flags_unreadable_csv(tmp_path):
    # A 0-byte CSV is kept by load_vci_clips(strict=False) with its path; the
    # COLUMNS section must report it as UNREADABLE rather than crash the inspector.
    d = tmp_path / "trajectories_filtered"
    d.mkdir(parents=True)
    (d / f"bad{PED_SUFFIX}").write_text("")
    clips = inspect_vci.load_vci_clips(tmp_path, "dut", fps=10.0, strict=False)
    lines = inspect_vci.report_columns(clips)  # must not raise
    assert any("UNREADABLE" in ln for ln in lines)
