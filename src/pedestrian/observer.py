"""Pedestrian observation module.

This module accumulates pedestrian observations over time and converts them
to the format required by Social-GAN.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from collections import deque
from loguru import logger

from ..core.data_structures import PedestrianState


class PedestrianObserver:
    """Observer for accumulating pedestrian observations.

    This class maintains a sliding window of pedestrian observations
    and converts them to the format required by the trajectory predictor.

    Args:
        obs_len: Number of observation time steps (default: 8)
        dt: Simulation time step between observations [s] (default: 0.4)
        sgan_dt: Desired sampling interval for SGAN input [s] (default: 0.4)
    """

    def __init__(self, obs_len: int = 8, dt: float = 0.4, sgan_dt: float = 0.4):
        self.obs_len = obs_len
        self.dt = dt
        self.sgan_dt = sgan_dt
        self.history: deque = deque(maxlen=obs_len)
        self.timestamps: deque = deque(maxlen=obs_len)
        self.n_peds = 0
        self.accumulated_time: float = 0.0

        logger.info(
            f"Pedestrian observer initialized with obs_len={obs_len}, "
            f"dt={dt}s, sgan_dt={sgan_dt}s"
        )
    
    def reset(self):
        """Reset the observation history."""
        self.history.clear()
        self.timestamps.clear()
        self.n_peds = 0
        self.accumulated_time = 0.0
        logger.debug("Observer history reset")
    
    def update(self, ped_state: PedestrianState):
        """Add a new pedestrian observation.

        Args:
            ped_state: Current pedestrian state
        """
        # Accumulate simulation time and only sample when reaching SGAN interval
        if len(self.timestamps) > 0:
            delta_t = max(
                ped_state.timestamp - self.timestamps[-1],
                0.0
            )
        else:
            # Fall back to configured simulation dt for the first step
            delta_t = self.dt

        self.accumulated_time += delta_t

        if self.accumulated_time + 1e-9 >= self.sgan_dt:
            self.history.append(ped_state.positions.copy())
            self.timestamps.append(ped_state.timestamp)
            self.n_peds = ped_state.n_peds
            # Keep leftover time so we stay aligned to sgan_dt
            self.accumulated_time %= self.sgan_dt

            logger.debug(
                f"Observer sampled at t={ped_state.timestamp:.2f}s, "
                f"history length={len(self.history)}/{self.obs_len}"
            )
    
    @property
    def is_ready(self) -> bool:
        """Check if enough observations have been accumulated.
        
        Returns:
            True if we have at least obs_len observations
        """
        return len(self.history) >= self.obs_len
    
    def get_observation(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get observations in Social-GAN format.
        
        Returns:
            obs_traj: Absolute positions [obs_len, n_peds, 2]
            obs_traj_rel: Relative positions [obs_len, n_peds, 2]
            seq_start_end: Sequence delimiters [1, 2]
            
        Raises:
            ValueError: If not enough observations have been accumulated
        """
        if not self.is_ready:
            raise ValueError(
                f"Not enough observations: {len(self.history)}/{self.obs_len}"
            )
        
        # Stack observations into array
        obs_traj = np.stack(list(self.history), axis=0)  # (obs_len, n_peds, 2)
        
        # Calculate relative positions (displacement from previous frame)
        obs_traj_rel = np.zeros_like(obs_traj)
        obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
        obs_traj_rel[0] = 0.0  # First frame has no previous frame
        
        # Sequence start/end indices
        seq_start_end = torch.LongTensor([[0, self.n_peds]])
        
        # Convert to tensors
        obs_traj_tensor = torch.from_numpy(obs_traj).float()
        obs_traj_rel_tensor = torch.from_numpy(obs_traj_rel).float()
        
        logger.debug(f"Generated observation: obs_traj shape={obs_traj_tensor.shape}, "
                    f"n_peds={self.n_peds}")
        
        return obs_traj_tensor, obs_traj_rel_tensor, seq_start_end
    
    def get_latest_positions(self) -> Optional[np.ndarray]:
        """Get the most recent pedestrian positions.
        
        Returns:
            Latest positions [n_peds, 2], or None if no observations
        """
        if len(self.history) == 0:
            return None
        return self.history[-1].copy()
