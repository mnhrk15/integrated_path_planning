
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger
from matplotlib.patches import Rectangle

from ..core.data_structures import SimulationResult

class DashboardGenerator:
    """Generates a static dashboard report of the simulation."""

    def __init__(self, history: List[SimulationResult]):
        self.history = history
        if not history:
            raise ValueError("History is empty")

    def generate(self, output_path: str, summary_metrics: dict = None, map_config: Optional[Dict] = None):
        """Create and save the dashboard image.
        
        Args:
            output_path: Path to save the image
            summary_metrics: Optional dictionary of aggregate metrics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Data prep
        times = [r.time for r in self.history]
        velocities = [r.ego_state.v for r in self.history]
        accelerations = [r.ego_state.a for r in self.history]
        jerks = [r.ego_state.jerk for r in self.history]
        min_dists = [r.metrics.get('min_distance', np.inf) for r in self.history]
        ttcs = [r.metrics.get('ttc', np.inf) for r in self.history]
        ttcs_finite = [t for t in ttcs if t < 10.0 and t > 0] # Filter for histogram
        
        # Setup Figure
        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        gs = gridspec.GridSpec(3, 3, figure=fig) # Changed to 3 columns
        
        # 1. Trajectory Map (Top-Left, spans 2 cols)
        ax_map = fig.add_subplot(gs[0:2, 0:2])
        self._plot_map(ax_map, map_config)
        
        # 2. Dynamics (Top-Right)
        ax_dyn = fig.add_subplot(gs[0, 2])
        ax_dyn.plot(times, velocities, label='Velocity (m/s)', color='blue')
        ax_dyn.set_ylabel('Velocity', color='blue')
        ax_dyn2 = ax_dyn.twinx()
        ax_dyn2.plot(times, accelerations, label='Accel (m/s²)', color='green', linestyle='--')
        ax_dyn2.set_ylabel('Acceleration', color='green')
        ax_dyn.set_title("Vehicle Dynamics")
        ax_dyn.grid(True, alpha=0.3)
        
        # 3. Comfort / Jerk (Middle-Right)
        ax_jerk = fig.add_subplot(gs[1, 2])
        ax_jerk.plot(times, jerks, color='purple')
        ax_jerk.set_title("Comfort (Jerk)")
        ax_jerk.set_ylabel("Jerk [m/s³]")
        ax_jerk.grid(True, alpha=0.3)
        ax_jerk.axhline(2.0, color='red', linestyle=':', alpha=0.5)
        ax_jerk.axhline(-2.0, color='red', linestyle=':', alpha=0.5)
        
        # 4. Safety (Bottom-Left)
        ax_safe = fig.add_subplot(gs[2, 0])
        ax_safe.plot(times, min_dists, color='orange')
        ax_safe.axhline(1.0, color='red', linestyle='--', label='Critical Threshold')
        ax_safe.set_title("Minimum Distance")
        ax_safe.set_xlabel("Time [s]")
        ax_safe.set_ylabel("Distance [m]")
        ax_safe.grid(True)
        
        # 5. Risk Analysis / TTC (Bottom-Middle)
        ax_ttc = fig.add_subplot(gs[2, 1])
        if ttcs_finite:
            ax_ttc.hist(ttcs_finite, bins=20, color='red', alpha=0.7)
            ax_ttc.set_xlabel("TTC [s]")
            ax_ttc.set_ylabel("Frequency")
        else:
            ax_ttc.text(0.5, 0.5, "No global risk detected", ha='center')
        ax_ttc.set_title("TTC Distribution")
        
        # 6. Summary Table (Bottom-Right)
        ax_table = fig.add_subplot(gs[2, 2])
        ax_table.axis('off')
        if summary_metrics:
            self._plot_summary_table(ax_table, summary_metrics)
        else:
            ax_table.text(0.5, 0.5, "No metrics available", ha='center')
        
        # Save
        fig.suptitle("Simulation Evaluation Report", fontsize=16)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Dashboard saved to {output_path}")

    def generate_trajectory_map(self, output_path: str, map_config: Optional[Dict] = None):
        """Generate just the trajectory map (simulation.png)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        self._plot_map(ax, map_config)
        
        # Ensure aspect ratio is preserved
        ax.set_aspect('equal')
        
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        logger.info(f"Simulation plot saved to {output_path}")

    def _plot_summary_table(self, ax, metrics: dict):
        """Render summary table."""
        # Format data
        data = [
            ["Metric", "Value", "Unit"],
            ["Min Distance", f"{metrics.get('min_dist', 0):.2f}", "m"],
            ["Min TTC", f"{metrics.get('min_ttc', np.inf):.2f}", "s"],
            ["Collisions", f"{int(metrics.get('collision_count', 0))}", "#"],
            ["Max Jerk", f"{metrics.get('max_jerk', 0):.2f}", "m/s³"],
            ["ADE (Pred)", f"{metrics.get('ade', 0):.3f}", "m"],
            ["FDE (Pred)", f"{metrics.get('fde', 0):.3f}", "m"],
        ]
        
        table = ax.table(
            cellText=data,
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.3, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("Performance Summary")

    def _plot_map(self, ax, map_config: Optional[Dict] = None):
        if map_config:
            # Draw road borders (Solid black lines)
            for border in map_config.get('road_borders', []):
                if len(border) == 4:
                    x1, y1, x2, y2 = border
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2.0, alpha=0.8, zorder=1)
                    
            # Draw lanes (Dashed lighter lines)
            for lane in map_config.get('lanes', []):
                if len(lane) == 4:
                    x1, y1, x2, y2 = lane
                    ax.plot([x1, x2], [y1, y2], 'k--', linewidth=1.0, alpha=0.4, zorder=1)
                    
            # Draw crosswalks (Rectangles)
            for cw in map_config.get('crosswalks', []):
                if len(cw) >= 4:
                    x, y, w, h = cw[:4]
                    angle = cw[4] if len(cw) > 4 else 0.0
                    rect = Rectangle(
                        (x, y), w, h, angle=angle,
                        facecolor='white', edgecolor='gray',
                        hatch='///', alpha=0.3, zorder=0
                    )
                    ax.add_patch(rect)

        # Plot Trajectories
        ego_x = [r.ego_state.x for r in self.history]
        ego_y = [r.ego_state.y for r in self.history]
        ax.plot(ego_x, ego_y, 'b-', linewidth=2, label='Ego', zorder=2)
        ax.plot(ego_x[0], ego_y[0], 'go', label='Start', zorder=2)
        ax.plot(ego_x[-1], ego_y[-1], 'ro', label='End', zorder=2)
        
        # Plot Pedestrians (Sampled)
        for r in self.history[::5]: # Sample every 5 steps
            if r.ped_state and r.ped_state.n_peds > 0:
                ax.plot(r.ped_state.positions[:, 0], r.ped_state.positions[:, 1], 'r.', markersize=2, alpha=0.2, zorder=2)



        ax.set_title("Trajectory Map")
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
            fontsize=9
        )

def create_dashboard(history: List[SimulationResult], output_path: str, metrics: dict = None, map_config: Optional[Dict] = None):
    """Convenience function."""
    gen = DashboardGenerator(history)
    gen.generate(output_path, metrics, map_config)

def create_simulation_plot(history: List[SimulationResult], output_path: str, map_config: Optional[Dict] = None):
    """Convenience function for trajectory map."""
    gen = DashboardGenerator(history)
    gen.generate_trajectory_map(output_path, map_config)
