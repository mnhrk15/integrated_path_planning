import numpy as np
import pytest

from src.datasets.vci_loader import (
    DUT_FPS,
    PED_SUFFIX,
    VEH_SUFFIX,
    agent_speed_samples,
    extract_fixed_windows,
    load_vci_clips,
    load_vci_pedestrians,
    load_vci_vehicles,
    vehicle_speed_samples,
)

PED_CSV = """id,frame,label,x_est,y_est,xv_est,yv_est
1,0,ped,0.0,0.0,1.0,0.0
1,10,ped,1.0,0.0,1.0,0.0
1,20,ped,2.0,0.0,1.0,0.0
2,0,ped,5.0,5.0,1.0,0.0
2,10,ped,6.0,5.0,1.0,0.0
"""

VEH_CSV = """id,frame,label,x_est,y_est,psi_est,vel_est
1,0,veh,0.0,0.0,0.0,2.0
1,10,veh,2.0,0.0,0.0,2.0
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_pedestrian_resample_to_grid(tmp_path):
    # fps=10 -> frames 0/10/20 are t=0/1/2 s; target_dt=0.5 -> grid 0,0.5,1,1.5,2.
    tracks = load_vci_pedestrians(_write(tmp_path, "ped.csv", PED_CSV), fps=10.0, target_dt=0.5)
    np.testing.assert_allclose(tracks.times, [0.0, 0.5, 1.0, 1.5, 2.0])
    assert list(tracks.ids) == [1, 2]
    # ped1 spans the whole grid, x interpolates linearly 0..2.
    np.testing.assert_allclose(tracks.positions[:, 0, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
    # ped2 only spans t in [0,1]; x interpolates 5->6, beyond span it is NaN.
    np.testing.assert_allclose(tracks.positions[:3, 1, 0], [5.0, 5.5, 6.0])
    assert np.all(np.isnan(tracks.positions[3:, 1, 0]))


def test_extract_fixed_windows_drops_absent_agents(tmp_path):
    tracks = load_vci_pedestrians(_write(tmp_path, "ped.csv", PED_CSV), fps=10.0, target_dt=0.5)
    windows = extract_fixed_windows(tracks, seq_len=3, stride=1)
    assert len(windows) == 3
    # start=0 (grid 0,0.5,1): both peds present -> N=2.
    assert windows[0].shape == (3, 2, 2)
    # later windows include grid >=1.5 where ped2 is NaN -> N=1.
    assert windows[1].shape == (3, 1, 2)
    assert windows[2].shape == (3, 1, 2)


def test_vehicle_loads_speed_and_heading_channels(tmp_path):
    tracks = load_vci_vehicles(_write(tmp_path, "veh.csv", VEH_CSV), fps=10.0, target_dt=0.5)
    assert "vel" in tracks.extra and "psi" in tracks.extra
    speeds = vehicle_speed_samples(tracks)
    assert speeds.size > 0
    np.testing.assert_allclose(speeds, 2.0)


def test_missing_columns_raise(tmp_path):
    bad = "id,frame,x\n1,0,0.0\n"
    with pytest.raises(ValueError):
        load_vci_pedestrians(_write(tmp_path, "bad.csv", bad), fps=10.0)


# --- pedestrian velocity channels (vx_est/vy_est real vs xv_est/yv_est README) -

PED_CSV_REAL_VEL = """id,frame,label,x_est,y_est,vx_est,vy_est
1,0,ped,0.0,0.0,1.5,0.0
1,10,ped,1.5,0.0,1.5,0.0
1,20,ped,3.0,0.0,1.5,0.0
"""


def test_pedestrian_reads_velocity_real_spelling(tmp_path):
    # Real filtered CSVs spell ped velocity vx_est/vy_est -> must fill 'vx'/'vy'.
    tracks = load_vci_pedestrians(
        _write(tmp_path, "ped.csv", PED_CSV_REAL_VEL), fps=10.0, target_dt=0.5
    )
    assert "vx" in tracks.extra and "vy" in tracks.extra
    vx = tracks.extra["vx"][:, 0]
    np.testing.assert_allclose(vx[np.isfinite(vx)], 1.5)  # constant across the span


def test_pedestrian_reads_velocity_readme_spelling(tmp_path):
    # The documented xv_est/yv_est spelling is equally accepted (PED_CSV uses it).
    tracks = load_vci_pedestrians(_write(tmp_path, "ped.csv", PED_CSV), fps=10.0, target_dt=0.5)
    assert "vx" in tracks.extra and "vy" in tracks.extra


def test_pedestrian_without_velocity_omits_channels(tmp_path):
    # Velocity is optional: positions still load, no vx/vy channels appear.
    csv = "id,frame,label,x_est,y_est\n1,0,ped,0.0,0.0\n1,10,ped,1.0,0.0\n"
    tracks = load_vci_pedestrians(_write(tmp_path, "ped.csv", csv), fps=10.0, target_dt=0.5)
    assert "vx" not in tracks.extra and "vy" not in tracks.extra
    assert tracks.positions.shape[1] == 1  # ped still loaded


def test_vehicle_heading_interpolates_across_pi_wrap(tmp_path):
    # psi goes 3.0 -> -3.0 (a ~0.28 rad turn across +/-pi). Linear interp would
    # wrongly pass through 0; angular unwrap must keep the midpoint near +/-pi.
    wrap = "id,frame,label,x_est,y_est,psi_est,vel_est\n1,0,veh,0.0,0.0,3.0,2.0\n1,10,veh,1.0,0.0,-3.0,2.0\n"
    tracks = load_vci_vehicles(_write(tmp_path, "veh_wrap.csv", wrap), fps=10.0, target_dt=0.5)
    psi_mid = tracks.extra["psi"][1, 0]  # grid t=0.5
    assert abs(abs(psi_mid) - np.pi) < 0.05  # near +/-pi, NOT 0


def test_empty_csv_returns_empty_tracks(tmp_path):
    empty = "id,frame,label,x_est,y_est,xv_est,yv_est\n"
    tracks = load_vci_pedestrians(_write(tmp_path, "empty.csv", empty), fps=10.0)
    assert tracks.times.size == 0
    assert tracks.ids.size == 0
    assert extract_fixed_windows(tracks, seq_len=3) == []


# --- agent_speed_samples (ped/veh position-difference speeds) ----------------


def test_agent_speed_samples_only_finite_steps(tmp_path):
    # fps=10, target_dt=0.5: ped1 spans the whole 5-point grid (4 steps), ped2
    # spans t=0..1 (2 finite steps, then NaN). All steps are 0.5 m / 0.5 s.
    tracks = load_vci_pedestrians(_write(tmp_path, "ped.csv", PED_CSV), fps=10.0, target_dt=0.5)
    speeds = agent_speed_samples(tracks, dt=0.5)
    assert speeds.size == 6  # 4 (ped1) + 2 (ped2); NaN steps excluded
    np.testing.assert_allclose(speeds, 1.0)


# --- multi-clip discovery / load_vci_clips -----------------------------------

EMPTY_PED = "id,frame,label,x_est,y_est,xv_est,yv_est\n"
EMPTY_VEH = "id,frame,label,x_est,y_est,psi_est,vel_est\n"


def _build_clip(dirpath, stem, ped=True, veh=True, ped_text=PED_CSV, veh_text=VEH_CSV):
    """Write a clip's ped/veh CSVs under dirpath using the real name suffixes."""
    dirpath.mkdir(parents=True, exist_ok=True)
    if ped:
        (dirpath / f"{stem}{PED_SUFFIX}").write_text(ped_text)
    if veh:
        (dirpath / f"{stem}{VEH_SUFFIX}").write_text(veh_text)
    return dirpath


def test_discover_dut_flat_pairs(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01")
    _build_clip(d, "roundabout_03")
    clips = load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    assert [c.clip for c in clips] == ["intersection_01", "roundabout_03"]
    assert all(c.scenario is None for c in clips)
    assert all(c.ped is not None and c.veh is not None for c in clips)


def test_discover_citr_nested_scenario(tmp_path):
    d = tmp_path / "trajectories_filtered" / "vci_front"
    _build_clip(d, "front_interaction_01")
    clips = load_vci_clips(tmp_path, "citr", fps=10.0)
    assert len(clips) == 1
    assert clips[0].scenario == "vci_front"
    assert clips[0].clip == "front_interaction_01"


def test_scan_ignores_txt_and_pdf(tmp_path):
    # CITR co-locates ratio .txt and plot .pdf with the CSVs; suffix matching
    # must not pick them up (only *_traj_{ped,veh}_filtered.csv).
    d = tmp_path / "trajectories_filtered" / "vci_front"
    _build_clip(d, "front_interaction_01")
    (d / "front_interaction_01_ratio_pixel2meter.txt").write_text("1.0\n")
    (d / "front_interaction_01_traj_plot.pdf").write_text("%PDF-1.4\n")
    # A raw (non-filtered) CSV must also be ignored.
    (d / "front_interaction_01_traj_ped.csv").write_text(PED_CSV)
    clips = load_vci_clips(tmp_path, "citr", fps=10.0)
    assert len(clips) == 1
    assert clips[0].ped is not None and clips[0].veh is not None


def test_ped_only_clip(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01", veh=False)
    clips = load_vci_clips(tmp_path, "dut", fps=10.0)
    assert len(clips) == 1
    assert clips[0].ped is not None and clips[0].veh is None
    assert clips[0].veh_path is None


def test_veh_only_clip(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01", ped=False)
    clips = load_vci_clips(tmp_path, "dut", fps=10.0)
    assert len(clips) == 1
    assert clips[0].veh is not None and clips[0].ped is None
    assert clips[0].ped_path is None


def test_require_both_skips_unpaired(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01")  # paired
    _build_clip(d, "intersection_02", veh=False)  # ped only
    clips = load_vci_clips(tmp_path, "dut", fps=10.0, require_both=True)
    assert [c.clip for c in clips] == ["intersection_01"]


def test_empty_clip_csv_does_not_crash(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01", ped_text=EMPTY_PED, veh_text=EMPTY_VEH)
    clips = load_vci_clips(tmp_path, "dut", fps=10.0)
    assert len(clips) == 1
    assert clips[0].ped.ids.size == 0


def test_clips_sorted_deterministic(tmp_path):
    d = tmp_path / "trajectories_filtered"
    for stem in ("roundabout_02", "intersection_10", "intersection_01"):
        _build_clip(d, stem)
    clips = load_vci_clips(tmp_path, "dut", fps=10.0)
    assert [c.clip for c in clips] == ["intersection_01", "intersection_10", "roundabout_02"]


def test_scenario_keys_disambiguate_same_stem(tmp_path):
    tf = tmp_path / "trajectories_filtered"
    _build_clip(tf / "vci_front", "interaction_01")
    _build_clip(tf / "vci_back", "interaction_01")
    clips = load_vci_clips(tmp_path, "citr", fps=10.0)
    assert len(clips) == 2
    assert {(c.scenario, c.clip) for c in clips} == {
        ("vci_back", "interaction_01"),
        ("vci_front", "interaction_01"),
    }


def test_citr_requires_explicit_fps(tmp_path):
    with pytest.raises(ValueError):
        load_vci_clips(tmp_path, "citr")  # fps omitted -> error


def test_dut_defaults_to_dut_fps(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01")
    clips = load_vci_clips(tmp_path, "dut")  # fps omitted -> DUT_FPS
    assert clips[0].fps == DUT_FPS


def test_invalid_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        load_vci_clips(tmp_path, "inD", fps=10.0)


def test_load_clips_reuses_single_file_api(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "intersection_01")
    clips = load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)
    direct = load_vci_pedestrians(d / f"intersection_01{PED_SUFFIX}", fps=10.0, target_dt=0.5)
    np.testing.assert_array_equal(clips[0].ped.positions, direct.positions)
    np.testing.assert_array_equal(clips[0].ped.ids, direct.ids)


# --- loader edge cases: nesting, missing root, cross-scenario, dup frames -----


def test_discover_handles_extra_wrapper_dirs(tmp_path):
    # A zip extraction typically adds wrapper dirs above trajectories_filtered;
    # rglob is depth-agnostic so the clip is still found from the dataset root.
    d = tmp_path / "vci-dataset-dut-master" / "data" / "trajectories_filtered"
    _build_clip(d, "intersection_01")
    clips = load_vci_clips(tmp_path, "dut", fps=10.0)
    assert [c.clip for c in clips] == ["intersection_01"]
    assert clips[0].scenario is None


def test_dut_same_stem_in_different_dirs_raises(tmp_path):
    # DUT keys on (None, stem) (directory is not a scenario), so the same stem
    # under two wrapper dirs would collapse to one key and silently drop a clip.
    # Discovery must fail loudly instead of dropping nondeterministically.
    _build_clip(tmp_path / "dirA" / "trajectories_filtered", "intersection_01")
    _build_clip(tmp_path / "dirB" / "trajectories_filtered", "intersection_01")
    with pytest.raises(ValueError):
        load_vci_clips(tmp_path, "dut", fps=10.0)


def test_dut_split_pair_across_different_dirs_raises(tmp_path):
    # Same stem but different parent dirs must not synthesize a ped/veh pair from
    # two unrelated wrapper dirs when each side appears only once.
    _build_clip(
        tmp_path / "dirA" / "trajectories_filtered",
        "intersection_01",
        veh=False,
    )
    _build_clip(
        tmp_path / "dirB" / "trajectories_filtered",
        "intersection_01",
        ped=False,
    )
    with pytest.raises(ValueError):
        load_vci_clips(tmp_path, "dut", fps=10.0)


def test_discover_citr_scenario_under_deep_nesting(tmp_path):
    d = tmp_path / "vci-dataset-citr-master" / "data" / "trajectories_filtered" / "vci_front"
    _build_clip(d, "front_01")
    clips = load_vci_clips(tmp_path, "citr", fps=10.0)
    assert clips[0].scenario == "vci_front"  # parent dir name, regardless of depth
    assert clips[0].clip == "front_01"


def test_load_clips_missing_root_returns_empty(tmp_path):
    # rglob over a non-existent path yields nothing (no FileNotFoundError).
    assert load_vci_clips(tmp_path / "nope", "dut", fps=10.0) == []


def test_load_clips_empty_tree_returns_empty(tmp_path):
    (tmp_path / "trajectories_filtered").mkdir()
    assert load_vci_clips(tmp_path, "dut", fps=10.0) == []


def test_ped_veh_in_different_scenarios_do_not_pair(tmp_path):
    tf = tmp_path / "trajectories_filtered"
    _build_clip(tf / "vci_front", "c1", veh=False)  # ped only in one scenario
    _build_clip(tf / "vci_back", "c1", ped=False)  # veh only in another
    clips = load_vci_clips(tmp_path, "citr", fps=10.0)
    assert len(clips) == 2  # scenario key prevents cross-scenario pairing
    front = next(c for c in clips if c.scenario == "vci_front")
    back = next(c for c in clips if c.scenario == "vci_back")
    assert front.ped is not None and front.veh is None
    assert back.ped is None and back.veh is not None


def test_load_clips_handles_unsorted_and_duplicate_frames(tmp_path):
    # Rows out of order with a duplicate (id, frame); the reused single-file
    # loader sorts and keeps the first dup, exercised via the multi-clip path.
    text = (
        "id,frame,label,x_est,y_est,xv_est,yv_est\n"
        "1,20,ped,2.0,0.0,1.0,0.0\n"
        "1,0,ped,0.0,0.0,1.0,0.0\n"
        "1,10,ped,1.0,0.0,1.0,0.0\n"
        "1,10,ped,9.9,9.9,0.0,0.0\n"  # duplicate frame 10 -> keep first (x=1.0)
    )
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "c1", ped_text=text, veh=False)
    ped = load_vci_clips(tmp_path, "dut", fps=10.0, target_dt=0.5)[0].ped
    np.testing.assert_allclose(ped.positions[:, 0, 0], [0.0, 0.5, 1.0, 1.5, 2.0])


def test_load_clips_strict_true_raises_on_bad_csv(tmp_path):
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "good")
    (d / f"bad{PED_SUFFIX}").write_text("id,frame,label\n1,0,ped\n")  # missing x_est/y_est
    with pytest.raises(ValueError):
        load_vci_clips(tmp_path, "dut", fps=10.0)  # strict=True default


def test_load_clips_strict_false_keeps_path_and_other_clips(tmp_path):
    # One malformed CSV must not abort the scan: the bad side becomes None (path
    # retained for diagnosis), the good clip survives.
    d = tmp_path / "trajectories_filtered"
    _build_clip(d, "good")
    (d / f"bad{PED_SUFFIX}").write_text("id,frame,label\n1,0,ped\n")  # missing x_est/y_est
    clips = load_vci_clips(tmp_path, "dut", fps=10.0, strict=False)
    assert [c.clip for c in clips] == ["bad", "good"]
    bad = next(c for c in clips if c.clip == "bad")
    assert bad.ped is None and bad.ped_path is not None
    good = next(c for c in clips if c.clip == "good")
    assert good.ped is not None and good.veh is not None


def test_agent_speed_samples_dt_defaults_to_grid_step(tmp_path):
    # dt omitted -> derived from the grid step (times[1]-times[0]), matching the
    # target_dt the tracks were resampled with -- not the old hardcoded 0.4.
    tracks = load_vci_pedestrians(_write(tmp_path, "ped.csv", PED_CSV), fps=10.0, target_dt=0.5)
    np.testing.assert_allclose(
        agent_speed_samples(tracks), agent_speed_samples(tracks, dt=0.5)
    )
