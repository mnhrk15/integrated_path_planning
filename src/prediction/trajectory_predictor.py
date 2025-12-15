"""Trajectory prediction using Social-GAN (vendor implementation only)."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

# Always use vendorized SGAN implementation
from .sgan_vendor.models import TrajectoryGenerator
from .sgan_vendor.utils import relative_to_abs


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
        """Load a pre-trained vendor Social-GAN model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading Social-GAN model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model arguments
        if "args" in checkpoint:
            args = checkpoint["args"]
            if not isinstance(args, dict):
                args = vars(args) if hasattr(args, "__dict__") else dict(args)
        else:
            logger.warning("No args found in checkpoint, using defaults")
            args = {}

        logger.info("Model configuration:")
        logger.info(f"  obs_len: {args.get('obs_len', 8)}")
        logger.info(f"  pred_len: {args.get('pred_len', 12)}")
        logger.info(f"  pooling_type: {args.get('pooling_type', 'pool_net')}")

        # Build vendor TrajectoryGenerator
        self.generator = TrajectoryGenerator(
            obs_len=args.get("obs_len", 8),
            pred_len=args.get("pred_len", 12),
            embedding_dim=args.get("embedding_dim", 64),
            encoder_h_dim=args.get("encoder_h_dim_g", 64),
            decoder_h_dim=args.get("decoder_h_dim_g", 128),
            mlp_dim=args.get("mlp_dim", 1024),
            num_layers=args.get("num_layers", 1),
            noise_dim=args.get("noise_dim", (8,)),
            noise_type=args.get("noise_type", "gaussian"),
            noise_mix_type=args.get("noise_mix_type", "ped"),
            pooling_type=args.get("pooling_type", "pool_net"),
            pool_every_timestep=args.get("pool_every_timestep", True),
            dropout=args.get("dropout", 0.0),
            bottleneck_dim=args.get("bottleneck_dim", 1024),
            neighborhood_size=args.get("neighborhood_size", 2.0),
            grid_size=args.get("grid_size", 8),
            batch_norm=args.get("batch_norm", False),
            device=self.device,
        )

        # Load state dict (required)
        if "g_state" in checkpoint:
            self.generator.load_state_dict(checkpoint["g_state"])
            logger.info("Loaded generator state dict")
        elif "g_best_state" in checkpoint:
            self.generator.load_state_dict(checkpoint["g_best_state"])
            logger.info("Loaded best generator state dict")
        else:
            raise ValueError("No generator state found in checkpoint.")

        self.generator.to(self.device)
        self.generator.eval()

        logger.info(f"✓ Model loaded successfully from {model_path}")
    
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
            raise RuntimeError("Generator not loaded. Call load_model before predict().")
        
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
