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
    """Test predictor works without trained model"""
    obs_traj = torch.randn(8, 2, 2)
    obs_traj_rel = torch.diff(obs_traj, dim=0, prepend=obs_traj[:1])
    seq_start_end = torch.tensor([[0, 2]])
    
    predictor = TrajectoryPredictor(
        model_path=None,
        pred_len=12,
        num_samples=1,
        device='cpu'
    )
    
    pred_traj_rel = predictor.predict(obs_traj, obs_traj_rel, seq_start_end)
    
    assert pred_traj_rel is not None
    assert pred_traj_rel.shape[0] == 12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
