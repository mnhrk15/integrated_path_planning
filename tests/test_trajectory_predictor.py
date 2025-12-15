"""
Tests for TrajectoryPredictor with Social-GAN integration
"""
import pytest
import numpy as np
import torch
from src.prediction.trajectory_predictor import TrajectoryPredictor


def test_trajectory_predictor_initialization():
    """Test TrajectoryPredictor can be initialized"""
    predictor = TrajectoryPredictor(
        model_path=None,
        pred_len=12,
        num_samples=1,
        device='cpu'
    )
    
    assert predictor is not None
    assert predictor.pred_len == 12


def test_trajectory_predictor_without_model():
    """Predict should raise if model is not loaded."""
    obs_traj = torch.randn(8, 2, 2)
    obs_traj_rel = torch.diff(obs_traj, dim=0, prepend=obs_traj[:1])
    seq_start_end = torch.tensor([[0, 2]])

    predictor = TrajectoryPredictor(
        model_path=None,
        pred_len=12,
        num_samples=1,
        device='cpu'
    )

    with pytest.raises(RuntimeError):
        predictor.predict(obs_traj, obs_traj_rel, seq_start_end)


def test_trajectory_predictor_with_dummy_model(monkeypatch):
    """Test predictor path when a model exists (using dummy generator)."""
    obs_traj = torch.randn(8, 1, 2)
    obs_traj_rel = torch.diff(obs_traj, dim=0, prepend=obs_traj[:1])
    seq_start_end = torch.tensor([[0, 1]])

    class DummyGen(torch.nn.Module):
        def forward(self, obs_traj, obs_traj_rel, seq_start_end):
            # Return zeros with same pred_len and batch size
            pred_len = 12
            batch = obs_traj.shape[1]
            return torch.zeros(pred_len, batch, 2, device=obs_traj.device)

    predictor = TrajectoryPredictor(model_path=None, pred_len=12, num_samples=1, device='cpu')
    predictor.generator = DummyGen()

    pred = predictor.predict(obs_traj, obs_traj_rel, seq_start_end)

    assert pred.shape == (12, 1, 2)
    # Since dummy returns zeros in relative coords, absolute should stay at last obs position
    last_obs = obs_traj[-1].cpu().numpy()
    expected = np.repeat(last_obs[None, :, :], 12, axis=0)
    assert np.allclose(pred, expected)


def test_trajectory_predictor_with_vendor_sgan(monkeypatch):
    """Ensure vendorized SGAN generator path produces correct shapes."""
    from src.prediction.sgan_vendor.models import TrajectoryGenerator
    from src.prediction.sgan_vendor.utils import relative_to_abs

    class DummyGen(TrajectoryGenerator):
        def __init__(self):
            super().__init__(
                obs_len=8, pred_len=12, pooling_type=None, noise_dim=(0,), device=torch.device("cpu")
            )

        def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
            # Return zeros in relative coords
            pred_len = self.pred_len
            batch = obs_traj.shape[1]
            return torch.zeros(pred_len, batch, 2)

    obs_traj = torch.randn(8, 1, 2)
    obs_traj_rel = torch.diff(obs_traj, dim=0, prepend=obs_traj[:1])
    seq_start_end = torch.tensor([[0, 1]])

    predictor = TrajectoryPredictor(model_path=None, pred_len=12, num_samples=1, device='cpu')
    predictor.generator = DummyGen()

    pred = predictor.predict(obs_traj, obs_traj_rel, seq_start_end)

    assert pred.shape == (12, 1, 2)
    last_obs = obs_traj[-1].cpu().numpy()
    expected = np.repeat(last_obs[None, :, :], 12, axis=0)
    assert np.allclose(pred, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
