
import sys
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.integrated_simulator import IntegratedSimulator
from src.config import SimulationConfig

def test_headless():
    output_dir = Path("output/test_headless")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    # Create config with visualization disabled
    config = SimulationConfig()
    config.visualization_enabled = False
    config.dt = 0.1
    config.total_time = 1.0 # Short run
    config.output_path = str(output_dir)
    config.reference_waypoints_x = [0.0, 10.0, 20.0, 30.0, 40.0]
    config.reference_waypoints_y = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initialize simulator
    sim = IntegratedSimulator(config)
    
    # Run a few steps
    sim.run(n_steps=5)
    
    # Save results
    sim.save_results()
    
    # Check if dashboard.png exists
    dashboard_path = output_dir / "dashboard.png"
    if dashboard_path.exists():
        print("FAIL: dashboard.png was generated despite visualization_enabled=False")
        sys.exit(1)
    else:
        print("PASS: dashboard.png was NOT generated")
        
    # Check if trajectory.npz exists (should exist)
    traj_path = output_dir / "trajectory.npz"
    if not traj_path.exists():
        print("FAIL: trajectory.npz was NOT generated")
        sys.exit(1)
    else:
        print("PASS: trajectory.npz was generated")

if __name__ == "__main__":
    test_headless()
