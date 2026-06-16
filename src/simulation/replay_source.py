"""Replay recorded pedestrian trajectories as ground truth.

:class:`ReplayPedestrianSource` is a drop-in replacement for
``PedestrianSimulator`` (``integrated_simulator.py``) that supplies *recorded*
pedestrian trajectories instead of simulating them with the Social Force model.
It implements the minimal contract the integrated simulator / its warmup loop
rely on:

* ``step(ego_state=None, n=1)`` -- advance the playback head by ``n`` frames.
  ``ego_state`` is accepted for interface compatibility but **ignored**: replayed
  pedestrians follow the recorded ground truth and do not react to the ego
  vehicle. This is intentional for open-loop prediction evaluation (RQ1a), where
  the ego must not perturb the ground truth. Closed-loop calibration that needs
  pedestrians to react to the ego must keep using the SFM simulator.
* ``get_state() -> PedestrianState`` -- the pedestrian state at the current frame.

The input is a fixed-population window ``[T, N, 2]`` (metres, world frame), e.g.
one window from :func:`eth_ucy_loader.extract_fixed_windows`. Velocities are
finite-differenced from the positions when not supplied; goals default to each
pedestrian's final recorded position.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.data_structures import EgoVehicleState, PedestrianState


class ReplayPedestrianSource:
    """Frame-by-frame replay of recorded pedestrian trajectories."""

    def __init__(
        self,
        trajectories: np.ndarray,
        dt: float = 0.4,
        velocities: Optional[np.ndarray] = None,
        goals: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
    ):
        traj = np.asarray(trajectories, dtype=float)
        if traj.ndim != 3 or traj.shape[2] != 2:
            raise ValueError(
                f"trajectories must be [T, N, 2], got shape {traj.shape}"
            )
        self.trajectories = traj
        self.n_frames, self.n_peds, _ = traj.shape
        self.dt = float(dt)
        self.time = 0.0
        self._idx = 0

        if velocities is not None:
            vel = np.asarray(velocities, dtype=float)
            if vel.shape != traj.shape:
                raise ValueError(
                    f"velocities shape {vel.shape} != trajectories {traj.shape}"
                )
            self.velocities = vel
        else:
            self.velocities = self._finite_difference(traj, self.dt)

        if goals is not None:
            goals = np.asarray(goals, dtype=float)
            if goals.shape != (self.n_peds, 2):
                raise ValueError(
                    f"goals shape {goals.shape} != ({self.n_peds}, 2)"
                )
            self.goals = goals
        else:
            self.goals = traj[-1].copy()  # final recorded position per pedestrian

        self.ids = (
            np.asarray(ids) if ids is not None else np.arange(self.n_peds)
        )

    @staticmethod
    def _finite_difference(traj: np.ndarray, dt: float) -> np.ndarray:
        """Forward difference; last step duplicates the previous velocity."""
        vel = np.zeros_like(traj)
        if traj.shape[0] >= 2:
            vel[:-1] = (traj[1:] - traj[:-1]) / dt
            vel[-1] = vel[-2]
        return vel

    def step(self, ego_state: Optional[EgoVehicleState] = None, n: int = 1) -> None:
        """Advance the playback head by ``n`` frames (clamped at the last frame).

        ``ego_state`` is ignored (see module docstring). Simulation time keeps
        advancing past the final frame so the source stays compatible with the
        simulator's warmup loop, which calls ``step`` a fixed number of times.
        """
        for _ in range(n):
            if self._idx < self.n_frames - 1:
                self._idx += 1
            self.time += self.dt

    def get_state(self) -> PedestrianState:
        """Pedestrian state at the current frame."""
        return PedestrianState(
            positions=self.trajectories[self._idx].copy(),
            velocities=self.velocities[self._idx].copy(),
            goals=self.goals.copy(),
            ids=self.ids.copy(),
            timestamp=self.time,
        )

    def reset(self) -> None:
        """Rewind to the first frame."""
        self._idx = 0
        self.time = 0.0
