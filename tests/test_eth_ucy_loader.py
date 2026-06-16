import numpy as np
import pytest

from src.datasets.eth_ucy_loader import (
    extract_fixed_windows,
    load_scene_file,
    walking_speed_stats,
)

# Synthetic scene rows: (frame, ped_id, x, y).
# ped1 present at frames 0/10/20/30 (moving +0.4 m/step in x).
# ped2 present at frames 0/10/20 only (moving +0.4 m/step in y), absent at 30.
SYNTH = [
    (0, 1, 0.0, 0.0), (0, 2, 10.0, 0.0),
    (10, 1, 0.4, 0.0), (10, 2, 10.0, 0.4),
    (20, 1, 0.8, 0.0), (20, 2, 10.0, 0.8),
    (30, 1, 1.2, 0.0),
]


def _write(tmp_path, rows):
    p = tmp_path / "scene.txt"
    p.write_text("\n".join("\t".join(str(v) for v in r) for r in rows) + "\n")
    return p


def test_load_scene_file_parses_frames_and_peds(tmp_path):
    scene = load_scene_file(_write(tmp_path, SYNTH))
    assert list(scene.frames) == [0, 10, 20, 30]
    assert list(scene.ped_ids) == [1, 2]
    assert scene.frame_step == 10.0
    np.testing.assert_allclose(scene.by_frame[0][1], [0.0, 0.0])
    np.testing.assert_allclose(scene.by_frame[1][2], [10.0, 0.4])


def test_extract_fixed_windows_drops_transient_peds(tmp_path):
    scene = load_scene_file(_write(tmp_path, SYNTH))
    windows = extract_fixed_windows(scene, seq_len=3, stride=1)
    assert len(windows) == 2
    # Window 0 spans frames 0/10/20: both pedestrians present -> N=2.
    assert windows[0].shape == (3, 2, 2)
    np.testing.assert_allclose(windows[0][:, 0, 0], [0.0, 0.4, 0.8])  # ped1 x
    # Window 1 spans frames 10/20/30: only ped1 present (ped2 left) -> N=1.
    assert windows[1].shape == (3, 1, 2)


def test_extract_fixed_windows_min_peds_filter(tmp_path):
    scene = load_scene_file(_write(tmp_path, SYNTH))
    windows = extract_fixed_windows(scene, seq_len=3, stride=1, min_peds=2)
    assert len(windows) == 1  # only window 0 has >=2 spanning pedestrians


def test_walking_speed_stats(tmp_path):
    scene = load_scene_file(_write(tmp_path, SYNTH))
    speeds = walking_speed_stats(scene, dt=0.4)
    # Every adjacent-frame move is 0.4 m over 0.4 s = 1.0 m/s.
    np.testing.assert_allclose(np.median(speeds), 1.0)


def test_load_scene_file_rejects_too_few_columns(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 1 5.0\n")  # only 3 columns
    with pytest.raises(ValueError):
        load_scene_file(p)
