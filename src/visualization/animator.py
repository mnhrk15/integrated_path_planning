"""Advanced visualization with animation support.

This module provides animated visualization of simulation results using
matplotlib.animation.FuncAnimation.
"""

import os
import numpy as np
import matplotlib

# 強制的に非GUIバックエンドを使う（保存のみでも安定するように）
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyArrow, Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

from ..core.data_structures import SimulationResult


class SimulationAnimator:
    """Create animations from simulation results.
    
    Features:
    - Animated ego vehicle trajectory
    - Animated pedestrian trajectories
    - Predicted trajectories visualization
    - Reference path display
    - Real-time metrics display
    - Export to GIF or MP4
    
    Args:
        results: List of simulation results
        figsize: Figure size (width, height) in inches
        dpi: Dots per inch for rendered output
        interval: Frame interval in milliseconds
    """
    
    def __init__(
        self,
        results: List[SimulationResult],
        figsize: Tuple[float, float] = (14, 8),
        dpi: int = 100,
        interval: int = 100  # ms between frames
    ):
        if len(results) == 0:
            raise ValueError("SimulationAnimator requires at least one result")
        self.results = results
        self.figsize = figsize
        self.dpi = dpi
        self.interval = interval
        self.n_frames = len(results)
        
        # Setup figure
        self.fig = None
        self.ax_main = None
        self.ax_velocity = None
        self.ax_distance = None
        self.anim = None
        
        logger.info(f"Animator initialized with {self.n_frames} frames")
    
    def create_animation(
        self,
        show_predictions: bool = True,
        show_metrics: bool = True,
        trail_length: int = 50,
        ego_color: str = 'blue',
        ped_color: str = 'red',
        pred_color: str = 'orange',
        save_path: Optional[Path] = None,
        writer: str = 'pillow',  # 'pillow' for GIF, 'ffmpeg' for MP4
        fps: int = 10
    ) -> animation.FuncAnimation:
        """Create animation from simulation results.
        
        Args:
            show_predictions: Show predicted pedestrian trajectories
            show_metrics: Show metrics subplots
            trail_length: Number of past positions to show as trail
            ego_color: Color for ego vehicle
            ped_color: Color for pedestrians
            pred_color: Color for predicted trajectories
            save_path: Path to save animation (None = don't save)
            writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)
            fps: Frames per second for saved animation
            
        Returns:
            FuncAnimation object
        """
        logger.info("Creating animation...")
        
        # Setup figure with subplots
        if show_metrics:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            self.ax_main = self.fig.add_subplot(gs[:, 0])
            self.ax_velocity = self.fig.add_subplot(gs[0, 1])
            self.ax_distance = self.fig.add_subplot(gs[1, 1])
        else:
            self.fig, self.ax_main = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Setup main plot
        self._setup_main_plot()
        
        # Setup metric plots
        if show_metrics:
            self._setup_metric_plots()
        
        # Create artists for animation
        artists = self._create_artists(
            show_predictions, trail_length,
            ego_color, ped_color, pred_color
        )
        
        # Create animation
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            init_func=lambda: self._init_animation(artists),
            fargs=(artists, show_predictions, show_metrics, trail_length),
            frames=self.n_frames,
            interval=self.interval,
            blit=False,  # Set to False for better compatibility
            repeat=True
        )
        
        logger.info(f"Animation created with {self.n_frames} frames")
        
        # Save if requested
        if save_path is not None:
            self._save_animation(save_path, writer, fps)
        
        return self.anim
    
    def _setup_main_plot(self):
        """Setup main trajectory plot."""
        self.ax_main.set_xlabel('X [m]', fontsize=12)
        self.ax_main.set_ylabel('Y [m]', fontsize=12)
        self.ax_main.set_title('Autonomous Vehicle Path Planning Simulation',
                               fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.set_aspect('equal', adjustable='box')
        
        # Set limits based on trajectory bounds
        all_x = []
        all_y = []
        for result in self.results:
            all_x.append(result.ego_state.x)
            all_y.append(result.ego_state.y)
            if result.ped_state is not None:
                all_x.extend(result.ped_state.positions[:, 0])
                all_y.extend(result.ped_state.positions[:, 1])
        
        margin = 5.0
        if all_x and all_y:
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
            self.ax_main.set_xlim(x_min, x_max)
            self.ax_main.set_ylim(y_min, y_max)
    
    def _setup_metric_plots(self):
        """Setup metric subplots."""
        # Velocity plot
        self.ax_velocity.set_xlabel('Time [s]', fontsize=10)
        self.ax_velocity.set_ylabel('Velocity [m/s]', fontsize=10)
        self.ax_velocity.set_title('Ego Vehicle Velocity', fontsize=11, fontweight='bold')
        self.ax_velocity.grid(True, alpha=0.3)
        
        # Distance plot
        self.ax_distance.set_xlabel('Time [s]', fontsize=10)
        self.ax_distance.set_ylabel('Distance [m]', fontsize=10)
        self.ax_distance.set_title('Min Distance to Pedestrians', fontsize=11, fontweight='bold')
        self.ax_distance.grid(True, alpha=0.3)
        self.ax_distance.axhline(y=1.0, color='r', linestyle='--', 
                                linewidth=1, alpha=0.7, label='Safety threshold')
        self.ax_distance.legend(fontsize=9)
    
    def _create_artists(self, show_predictions, trail_length,
                       ego_color, ped_color, pred_color):
        """Create matplotlib artists for animation."""
        artists = {}
        
        # Ego vehicle (triangle/arrow)
        artists['ego'] = FancyArrow(
            0, 0, 1, 0,
            width=1.5, head_width=2.5, head_length=2.0,
            fc=ego_color, ec='black', alpha=0.8,
            transform=self.ax_main.transData
        )
        self.ax_main.add_patch(artists['ego'])
        
        # Ego trail
        artists['ego_trail'], = self.ax_main.plot(
            [], [], '-', color=ego_color, alpha=0.3, linewidth=1,
            label='Ego trajectory'
        )
        
        # Pedestrians (circles)
        artists['pedestrians'] = []
        artists['ped_trails'] = []
        
        # Predicted trajectories
        if show_predictions:
            artists['predictions'] = []
        
        # Metric lines
        if hasattr(self, 'ax_velocity'):
            artists['velocity_line'], = self.ax_velocity.plot(
                [], [], 'b-', linewidth=2
            )
            artists['velocity_point'], = self.ax_velocity.plot(
                [], [], 'bo', markersize=8
            )
        
        if hasattr(self, 'ax_distance'):
            artists['distance_line'], = self.ax_distance.plot(
                [], [], 'g-', linewidth=2
            )
            artists['distance_point'], = self.ax_distance.plot(
                [], [], 'go', markersize=8
            )
        
        # Time text
        artists['time_text'] = self.ax_main.text(
            0.02, 0.98, '', transform=self.ax_main.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        return artists
    
    def _init_animation(self, artists):
        """Initialize animation."""
        return list(artists.values())
    
    def _update_frame(self, frame, artists, show_predictions, show_metrics, trail_length):
        """Update frame for animation."""
        result = self.results[frame]
        
        # Update ego vehicle
        self._update_ego(artists, result, frame, trail_length)
        
        # Update pedestrians
        self._update_pedestrians(artists, result, show_predictions)
        
        # Update metrics
        if show_metrics:
            self._update_metrics(artists, frame)
        
        # Update time text
        artists['time_text'].set_text(f'Time: {result.time:.1f} s')
        
        return list(artists.values())
    
    def _update_ego(self, artists, result, frame, trail_length):
        """Update ego vehicle visualization."""
        ego = result.ego_state
        
        # Update arrow position and rotation
        dx = 2.0 * np.cos(ego.yaw)
        dy = 2.0 * np.sin(ego.yaw)
        artists['ego'].set_data(
            x=ego.x, y=ego.y, dx=dx, dy=dy
        )
        
        # Update trail
        start_idx = max(0, frame - trail_length)
        trail_x = [self.results[i].ego_state.x for i in range(start_idx, frame + 1)]
        trail_y = [self.results[i].ego_state.y for i in range(start_idx, frame + 1)]
        artists['ego_trail'].set_data(trail_x, trail_y)
    
    def _update_pedestrians(self, artists, result, show_predictions):
        """Update pedestrian visualization."""
        # Remove old pedestrian artists
        for ped_circle in artists.get('pedestrians', []):
            ped_circle.remove()
        artists['pedestrians'] = []
        
        for ped_trail in artists.get('ped_trails', []):
            ped_trail.remove()
        artists['ped_trails'] = []
        
        if show_predictions:
            for pred_line in artists.get('predictions', []):
                pred_line.remove()
            artists['predictions'] = []
        
        # Add new pedestrian artists
        if result.ped_state is not None:
            for i, pos in enumerate(result.ped_state.positions):
                # Pedestrian circle
                circle = Circle(
                    pos, radius=0.3,
                    facecolor='red', edgecolor='darkred',
                    alpha=0.7, zorder=5
                )
                self.ax_main.add_patch(circle)
                artists['pedestrians'].append(circle)
            
            # Predicted trajectories
            if show_predictions:
                pred_traj = getattr(result, "predicted_trajectories", None)
                if pred_traj is None:
                    pred_traj = getattr(result, "predicted_traj", None)
                if pred_traj is not None:
                    # Normalize shape to [n_peds, pred_len, 2]
                    if pred_traj.ndim == 3:
                        # If time-first (pred_len, n_peds, 2), swap axes
                        if result.ped_state is not None and pred_traj.shape[1] == result.ped_state.n_peds:
                            pred_traj = np.transpose(pred_traj, (1, 0, 2))
                    for i in range(pred_traj.shape[0]):
                        traj = pred_traj[i]
                        line, = self.ax_main.plot(
                            traj[:, 0], traj[:, 1],
                            'o-', color='orange', alpha=0.4,
                            markersize=3, linewidth=1, zorder=3
                        )
                        artists['predictions'].append(line)
    
    def _update_metrics(self, artists, frame):
        """Update metric plots."""
        times = [r.time for r in self.results[:frame+1]]
        
        # Velocity
        if 'velocity_line' in artists:
            velocities = [r.ego_state.v for r in self.results[:frame+1]]
            artists['velocity_line'].set_data(times, velocities)
            artists['velocity_point'].set_data([times[-1]], [velocities[-1]])
            
            self.ax_velocity.relim()
            self.ax_velocity.autoscale_view()
        
        # Distance
        if 'distance_line' in artists:
            distances = [r.metrics.get('min_distance', np.inf) 
                        for r in self.results[:frame+1]]
            artists['distance_line'].set_data(times, distances)
            artists['distance_point'].set_data([times[-1]], [distances[-1]])
            
            self.ax_distance.relim()
            self.ax_distance.autoscale_view()
    
    def _save_animation(self, save_path: Path, writer: str, fps: int):
        """Save animation to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving animation to {save_path} (writer={writer}, fps={fps})...")
        
        try:
            if writer == 'pillow':
                # Save as GIF
                if not str(save_path).endswith('.gif'):
                    save_path = save_path.with_suffix('.gif')
                self.anim.save(
                    str(save_path),
                    writer='pillow',
                    fps=fps,
                    dpi=self.dpi
                )
            elif writer == 'ffmpeg':
                # Save as MP4
                if not str(save_path).endswith('.mp4'):
                    save_path = save_path.with_suffix('.mp4')
                self.anim.save(
                    str(save_path),
                    writer='ffmpeg',
                    fps=fps,
                    dpi=self.dpi,
                    extra_args=['-vcodec', 'libx264']
                )
            else:
                raise ValueError(f"Unsupported writer: {writer}")
            
            size_mb = save_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Animation saved successfully ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            logger.info("Make sure you have pillow (for GIF) or ffmpeg (for MP4) installed")
            raise
    
    def show(self):
        """Display the animation."""
        if self.anim is None:
            raise RuntimeError("Animation not created. Call create_animation() first.")
        plt.show()
    
    def close(self):
        """Close the animation and release resources."""
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.anim = None


def create_simple_animation(
    results: List[SimulationResult],
    output_path: Optional[Path] = None,
    show: bool = True,
    show_predictions: bool = True,
    show_metrics: bool = True,
    trail_length: int = 50,
    fps: int = 10,
    **kwargs,
) -> SimulationAnimator:
    """Convenience function to create and display/save animation.
    
    Args:
        results: Simulation results
        output_path: Path to save animation (None = don't save)
        show: Whether to display the animation
        **kwargs: Additional arguments passed to create_animation()
        
    Returns:
        SimulationAnimator instance
    """
    animator = SimulationAnimator(results)
    
    # Determine writer from file extension
    writer = 'pillow'  # Default to GIF
    if output_path is not None:
        ext = Path(output_path).suffix.lower()
        if ext == '.mp4':
            writer = 'ffmpeg'
        elif ext == '.gif':
            writer = 'pillow'
    
    animator.create_animation(
        show_predictions=show_predictions,
        show_metrics=show_metrics,
        trail_length=trail_length,
        save_path=output_path,
        writer=writer,
        fps=fps,
        **kwargs
    )
    
    if show:
        animator.show()
    
    return animator
