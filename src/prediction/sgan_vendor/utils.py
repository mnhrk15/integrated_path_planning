"""
Minimal utilities from the official SGAN repo.
Source: https://github.com/agrimgupta92/sgan (MIT License)
"""

import torch


def relative_to_abs(rel_traj: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    """
    Convert relative trajectories to absolute coordinates.

    Args:
        rel_traj: Tensor (seq_len, batch, 2)
        start_pos: Tensor (batch, 2)
    Returns:
        abs_traj: Tensor (seq_len, batch, 2)
    """
    rel_traj = rel_traj.permute(1, 0, 2)  # batch, seq_len, 2
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)  # seq_len, batch, 2
