import numpy as np
import pytest

from src.datasets.vci_loader import (
    extract_fixed_windows,
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
