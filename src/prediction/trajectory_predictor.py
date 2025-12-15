"""Trajectory prediction using Social-GAN.

This module wraps the Social-GAN model for predicting pedestrian trajectories.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from loguru import logger

# Try importing from sgan package (if installed from original repo)
# Otherwise, use standalone implementation
try:
    from sgan.models import TrajectoryGenerator as SGANGenerator
    from sgan.utils import relative_to_abs as sgan_relative_to_abs
    SGAN_AVAILABLE = True
except ImportError:
    SGAN_AVAILABLE = False
    logger.warning("sgan package not found. Trying standalone implementation...")

# Standalone implementation based on official Social-GAN
# This allows the system to work without installing the full sgan package

# Standalone implementation based on official Social-GAN
# This allows the system to work without installing the full sgan package

# Note: For production use, install the official sgan package:
#   git clone https://github.com/agrimgupta92/sgan.git
#   cd sgan && pip install -e .
# The standalone implementation below is simplified and may not achieve
# the same performance as the official implementation.


class StandaloneTrajectoryGenerator(nn.Module):
    """Simplified standalone Social-GAN generator.
    
    This is a minimal implementation for cases where the full sgan package
    is not installed. For best results, use the official implementation.
    
    Note: This implementation requires pre-trained weights from the official
    Social-GAN repository. It cannot be trained from scratch without the
    full discriminator and training infrastructure.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        logger.warning(
            "Using standalone TrajectoryGenerator. For best results, "
            "install the official sgan package from "
            "https://github.com/agrimgupta92/sgan"
        )
        
        # Store parameters
        self.obs_len = kwargs.get('obs_len', 8)
        self.pred_len = kwargs.get('pred_len', 12)
        
        # This is a minimal placeholder that can load pre-trained weights
        # but requires the official implementation for full functionality
        # We'll raise an informative error if trying to use without weights
        self._loaded = False
    
    def load_state_dict(self, state_dict, strict=True):
        """Load pre-trained weights."""
        # This would need to match the official architecture
        # For now, we just store the state dict
        self._state_dict = state_dict
        self._loaded = True
        logger.warning(
            "Weights loaded into standalone generator. "
            "This may not work correctly without the official implementation."
        )
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end):
        """Forward pass."""
        if not self._loaded:
            raise RuntimeError(
                "Standalone generator cannot be used without pre-trained weights. "
                "Please install the official sgan package."
            )
        
        # This would need the full encoder-decoder architecture
        # For now, raise an error directing users to the official implementation
        raise NotImplementedError(
            "Standalone generator requires the official sgan package. "
            "Please install it:\n"
            "  git clone https://github.com/agrimgupta92/sgan.git\n"
            "  cd sgan && pip install -e .\n"
            "Then the TrajectoryPredictor will automatically use the official implementation."
        )


def relative_to_abs(rel_traj: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    """Convert relative trajectory to absolute coordinates.
    
    Args:
        rel_traj: Relative trajectory [seq_len, batch, 2]
        start_pos: Starting positions [batch, 2]
        
    Returns:
        Absolute trajectory [seq_len, batch, 2]
    """
    rel_traj = rel_traj.permute(1, 0, 2)  # [batch, seq_len, 2]
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)  # [seq_len, batch, 2]


class TrajectoryPredictor:
    """Pedestrian trajectory predictor using Social-GAN.
    
    This class loads a pre-trained Social-GAN model and provides
    a simple interface for trajectory prediction.
    
    Args:
        model_path: Path to pre-trained model checkpoint (.pt file)
        pred_len: Prediction horizon (number of time steps)
        num_samples: Number of trajectory samples to generate
        device: Device to run inference on ('cpu', 'cuda', 'mps')
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        pred_len: int = 12,
        num_samples: int = 1,
        device: str = 'cpu'
    ):
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.device = torch.device(device)
        self.generator = None
        
        if model_path is not None:
            self.load_model(model_path)
        
        logger.info(f"Trajectory predictor initialized with pred_len={pred_len}, "
                   f"num_samples={num_samples}, device={device}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained Social-GAN model.
        
        Args:
            model_path: Path to model checkpoint
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            logger.info(f"Loading Social-GAN model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model arguments
            if 'args' in checkpoint:
                args = checkpoint['args']
                if isinstance(args, dict):
                    # Already a dict
                    pass
                else:
                    # Convert AttrDict or Namespace to dict
                    args = vars(args) if hasattr(args, '__dict__') else dict(args)
            else:
                logger.warning("No args found in checkpoint, using defaults")
                args = {}
            
            # Log model configuration
            logger.info(f"Model configuration:")
            logger.info(f"  obs_len: {args.get('obs_len', 8)}")
            logger.info(f"  pred_len: {args.get('pred_len', 12)}")
            logger.info(f"  pooling_type: {args.get('pooling_type', 'pool_net')}")
            
            # Create generator
            if SGAN_AVAILABLE:
                # Use official implementation
                self.generator = SGANGenerator(
                    obs_len=args.get('obs_len', 8),
                    pred_len=args.get('pred_len', 12),
                    embedding_dim=args.get('embedding_dim', 64),
                    encoder_h_dim=args.get('encoder_h_dim_g', 64),
                    decoder_h_dim=args.get('decoder_h_dim_g', 128),
                    mlp_dim=args.get('mlp_dim', 1024),
                    num_layers=args.get('num_layers', 1),
                    noise_dim=args.get('noise_dim', (8,)),
                    noise_type=args.get('noise_type', 'gaussian'),
                    noise_mix_type=args.get('noise_mix_type', 'ped'),
                    pooling_type=args.get('pooling_type', 'pool_net'),
                    pool_every_timestep=args.get('pool_every_timestep', True),
                    dropout=args.get('dropout', 0.0),
                    bottleneck_dim=args.get('bottleneck_dim', 1024),
                    neighborhood_size=args.get('neighborhood_size', 2.0),
                    grid_size=args.get('grid_size', 8),
                    batch_norm=args.get('batch_norm', False)
                )
                logger.info("Using official sgan.models.TrajectoryGenerator")
            else:
                # Use standalone implementation (defined below)
                self.generator = StandaloneTrajectoryGenerator(
                    obs_len=args.get('obs_len', 8),
                    pred_len=args.get('pred_len', 12),
                    embedding_dim=args.get('embedding_dim', 64),
                    encoder_h_dim=args.get('encoder_h_dim_g', 64),
                    decoder_h_dim=args.get('decoder_h_dim_g', 128),
                    mlp_dim=args.get('mlp_dim', 1024),
                    num_layers=args.get('num_layers', 1),
                    noise_dim=args.get('noise_dim', (8,)),
                    noise_type=args.get('noise_type', 'gaussian'),
                    noise_mix_type=args.get('noise_mix_type', 'ped'),
                    pooling_type=args.get('pooling_type', 'pool_net'),
                    pool_every_timestep=args.get('pool_every_timestep', True),
                    dropout=args.get('dropout', 0.0),
                    bottleneck_dim=args.get('bottleneck_dim', 1024),
                    neighborhood_size=args.get('neighborhood_size', 2.0),
                    grid_size=args.get('grid_size', 8),
                    batch_norm=args.get('batch_norm', False)
                )
                logger.info("Using standalone TrajectoryGenerator implementation")
            
            # Load state dict
            if 'g_state' in checkpoint:
                self.generator.load_state_dict(checkpoint['g_state'])
                logger.info("Loaded generator state dict")
            elif 'g_best_state' in checkpoint:
                self.generator.load_state_dict(checkpoint['g_best_state'])
                logger.info("Loaded best generator state dict")
            else:
                logger.warning("No generator state found in checkpoint!")
            
            self.generator.to(self.device)
            self.generator.eval()
            
            logger.info(f"✓ Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        obs_traj: torch.Tensor,
        obs_traj_rel: torch.Tensor,
        seq_start_end: torch.Tensor
    ) -> np.ndarray:
        """Predict future trajectories.
        
        Args:
            obs_traj: Observed trajectories [obs_len, n_peds, 2]
            obs_traj_rel: Observed relative trajectories [obs_len, n_peds, 2]
            seq_start_end: Sequence boundaries [n_seq, 2]
            
        Returns:
            Predicted trajectories [n_peds, pred_len, 2]
        """
        if self.generator is None:
            # Fallback: constant velocity prediction
            logger.warning("No model loaded, using constant velocity prediction")
            return self._constant_velocity_prediction(obs_traj, obs_traj_rel)
        
        with torch.no_grad():
            obs_traj = obs_traj.to(self.device)
            obs_traj_rel = obs_traj_rel.to(self.device)
            seq_start_end = seq_start_end.to(self.device)
            
            # Generate prediction
            pred_traj_rel = self.generator(obs_traj, obs_traj_rel, seq_start_end)
            
            # Convert to absolute coordinates: [pred_len, n_peds, 2]
            pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1]).cpu().numpy()
        
        logger.debug(f"Predicted {pred_traj.shape[0]} pedestrian trajectories "
                    f"for {pred_traj.shape[1]} time steps")
        
        return pred_traj
    
    def _constant_velocity_prediction(
        self,
        obs_traj: torch.Tensor,
        obs_traj_rel: torch.Tensor
    ) -> np.ndarray:
        """Fallback: constant velocity prediction.
        
        Args:
            obs_traj: Observed trajectories [obs_len, n_peds, 2]
            obs_traj_rel: Observed relative trajectories [obs_len, n_peds, 2]
            
        Returns:
            Predicted trajectories [n_peds, pred_len, 2]
        """
        # Calculate average velocity from recent observations
        recent_vel = obs_traj_rel[-3:].mean(dim=0)  # [n_peds, 2]
        
        # Last observed position
        last_pos = obs_traj[-1]  # [n_peds, 2]
        
        # Predict by extrapolating with constant velocity
        n_peds = last_pos.shape[0]
        pred_traj = np.zeros((self.pred_len, n_peds, 2))
        
        current_pos = last_pos.cpu().numpy()
        velocity = recent_vel.cpu().numpy()
        
        for t in range(self.pred_len):
            current_pos = current_pos + velocity
            pred_traj[t, :, :] = current_pos
        
        logger.debug("Using constant velocity prediction as fallback")
        
        return pred_traj
    
    def predict_single_best(
        self,
        obs_traj: torch.Tensor,
        obs_traj_rel: torch.Tensor,
        seq_start_end: torch.Tensor
    ) -> np.ndarray:
        """Predict a single trajectory (best sample).
        
        If num_samples > 1, samples multiple trajectories and returns
        the one closest to the average.
        
        Args:
            obs_traj: Observed trajectories [obs_len, n_peds, 2]
            obs_traj_rel: Observed relative trajectories [obs_len, n_peds, 2]
            seq_start_end: Sequence boundaries [n_seq, 2]
            
        Returns:
            Best predicted trajectory [n_peds, pred_len, 2]
        """
        if self.num_samples == 1:
            return self.predict(obs_traj, obs_traj_rel, seq_start_end)
        
        # Generate multiple samples
        samples = []
        for _ in range(self.num_samples):
            pred = self.predict(obs_traj, obs_traj_rel, seq_start_end)
            samples.append(pred)
        
        samples = np.stack(samples, axis=0)  # [num_samples, n_peds, pred_len, 2]
        
        # Select best sample (closest to mean)
        mean_traj = samples.mean(axis=0)
        distances = np.linalg.norm(samples - mean_traj[None, ...], axis=-1).sum(axis=(1, 2))
        best_idx = np.argmin(distances)
        
        return samples[best_idx]
