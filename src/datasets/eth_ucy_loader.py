"""ETH/UCY pedestrian trajectory loader for open-loop prediction (RQ1a).

The ETH/UCY datasets (SGAN distribution, fetched by ``scripts/download_data.sh``
into ``datasets/``) are tab/space-separated ``frame ped_id x y`` text files in
world-frame metres, annotated on a fixed grid (the smallest frame gap == one
0.4 s SGAN step; pedestrians enter/leave so larger gaps appear where a frame
holds no one).

This module parses a scene file and extracts *fixed-population windows*:
contiguous ``seq_len`` frames in which a set of pedestrians is present for the
entire window. This matches SGAN's leave-one-out evaluation protocol and the
fixed-N assumption of :class:`PedestrianObserver` and ``core.metrics`` (whose
ADE/FDE requires ``gt.shape == (N, T, 2)``). Windows index the *sorted frame
grid*, treating adjacent frames as one 0.4 s step (SGAN convention; the physical
gap at a missing frame is ignored, exactly as in the original SGAN dataloader).

Full handling of pedestrians entering/leaving mid-window is deferred (the spike
uses fixed windows). Raw data is gitignored and not redistributed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

# Scene -> test split file(s). ``univ`` ships two recordings.
SCENE_TEST_FILES: Dict[str, List[str]] = {
    "eth": ["eth/test/biwi_eth.txt"],
    "hotel": ["hotel/test/biwi_hotel.txt"],
    "univ": ["univ/test/students001.txt", "univ/test/students003.txt"],
    "zara1": ["zara1/test/crowds_zara01.txt"],
    "zara2": ["zara2/test/crowds_zara02.txt"],
}

@dataclass
class SceneTrajectories:
    """Parsed trajectories for one scene file, on the native frame grid."""

    frames: np.ndarray  # [F] sorted unique frame ids
    ped_ids: np.ndarray  # [P] sorted unique pedestrian ids
    by_frame: List[Dict[int, np.ndarray]]  # by_frame[f_idx][ped_id] = (x, y)
    source: str

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def frame_step(self) -> float:
        """Most common gap between consecutive annotated frames (the 0.4 s grid).

        Uses the mode rather than the min so a single off-grid/anomalous close
        pair cannot mislabel the grid step (which would otherwise make
        walking_speed_stats reject every normal pair).
        """
        if len(self.frames) < 2:
            return 0.0
        values, counts = np.unique(np.diff(self.frames), return_counts=True)
        return float(values[np.argmax(counts)])


def load_scene_file(path: Union[str, Path]) -> SceneTrajectories:
    """Parse one ETH/UCY ``frame ped_id x y`` file into a SceneTrajectories."""
    data = np.loadtxt(str(path))
    if data.size == 0:
        raise ValueError(f"{path}: empty trajectory file")
    data = np.atleast_2d(data)  # handle single-row and single-scalar files
    if data.shape[1] < 4:
        raise ValueError(
            f"{path}: expected >=4 columns (frame ped_id x y), got {data.shape[1]}"
        )

    frames = np.unique(data[:, 0])
    frame_index = {f: i for i, f in enumerate(frames)}
    by_frame: List[Dict[int, np.ndarray]] = [dict() for _ in frames]
    for row in data:
        frame, pid, x, y = row[0], row[1], row[2], row[3]
        by_frame[frame_index[frame]][int(pid)] = np.array([x, y], dtype=float)

    ped_ids = np.unique(data[:, 1].astype(int))
    return SceneTrajectories(
        frames=frames, ped_ids=ped_ids, by_frame=by_frame, source=str(path)
    )


def load_scene(
    scene: str, root: Union[str, Path] = "datasets"
) -> List[SceneTrajectories]:
    """Load all test-split files for a named scene (e.g. 'zara1')."""
    if scene not in SCENE_TEST_FILES:
        raise KeyError(
            f"unknown scene '{scene}', expected one of {list(SCENE_TEST_FILES)}"
        )
    root = Path(root)
    return [load_scene_file(root / rel) for rel in SCENE_TEST_FILES[scene]]


def extract_fixed_windows(
    scene: SceneTrajectories,
    seq_len: int,
    stride: int = 1,
    min_peds: int = 1,
) -> List[np.ndarray]:
    """Return ``[seq_len, N, 2]`` windows of pedestrians present throughout.

    Population N is fixed within each window but varies between windows. Windows
    slide over the sorted frame grid with the given ``stride``; windows with
    fewer than ``min_peds`` pedestrians spanning the whole window are skipped.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    windows: List[np.ndarray] = []
    for start in range(0, scene.n_frames - seq_len + 1, stride):
        frame_dicts = scene.by_frame[start : start + seq_len]
        present = set(frame_dicts[0].keys())
        for fd in frame_dicts[1:]:
            present &= set(fd.keys())
        if len(present) < min_peds:
            continue
        ids = sorted(present)
        arr = np.empty((seq_len, len(ids), 2), dtype=float)
        for t, fd in enumerate(frame_dicts):
            for j, pid in enumerate(ids):
                arr[t, j] = fd[pid]
        windows.append(arr)
    return windows


def walking_speed_stats(scene: SceneTrajectories, dt: float = 0.4) -> np.ndarray:
    """Per-step speeds [m/s] for pedestrians present in consecutive frames.

    Only adjacent-frame pairs are used (so transient gaps do not inflate the
    apparent speed), giving a sanity-check distribution that should peak near
    ~1.3 m/s for normal walking.
    """
    speeds: List[float] = []
    step = scene.frame_step
    for i in range(scene.n_frames - 1):
        # Only count pairs exactly one grid step apart; frame holes (larger
        # gaps) would otherwise be divided by a too-small dt and inflate speed.
        if step > 0 and not np.isclose(scene.frames[i + 1] - scene.frames[i], step):
            continue
        a = scene.by_frame[i]
        b = scene.by_frame[i + 1]
        for pid in set(a.keys()) & set(b.keys()):
            speeds.append(float(np.linalg.norm(b[pid] - a[pid]) / dt))
    return np.array(speeds)
