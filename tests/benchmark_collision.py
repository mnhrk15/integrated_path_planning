
import time
import numpy as np
from src.planning.frenet_planner import FrenetPlanner, FrenetPath
from src.planning.cubic_spline import CubicSpline2D

def benchmark_collision():
    # Setup
    wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    csp = CubicSpline2D(wx, wy)
    planner = FrenetPlanner(csp)
    
    # Create a dummy path with many points (e.g. 5 seconds at 0.1s dt = 50 points)
    fp = FrenetPath()
    fp.t = list(np.arange(0, 5.0, 0.1))
    fp.x = list(np.linspace(0, 50, len(fp.t)))
    fp.y = list(np.zeros(len(fp.t)))
    
    # CASE 1: Many Static Obstacles
    n_static = 100
    static_obs = np.random.rand(n_static, 2) * 50  # Random layout
    
    # CASE 2: Many Dynamic Obstacles
    n_dynamic = 50
    n_steps = 50
    # [n_obs, n_steps, 2]
    dynamic_obs = np.random.rand(n_dynamic, n_steps, 2) * 50
    
    # Benchmark
    n_iter = 1000
    
    start_time = time.time()
    for _ in range(n_iter):
        planner._check_collision(fp, static_obs, dynamic_obs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_iter * 1000  # ms
    print(f"Average collision check time: {avg_time:.4f} ms per call")
    print(f"Load: {len(fp.x)} path points, {n_static} static obs, {n_dynamic} dynamic obs (x{n_steps} steps)")
    
    # Verification of correctness (sanity check)
    # Create an obstacle strictly ON the path
    static_hit = np.array([[0.0, 0.0]])
    fp_hit = FrenetPath()
    fp_hit.x = [0.0]
    fp_hit.y = [0.0]
    fp_hit.t = [0.0]
    
    assert not planner._check_collision(fp_hit, static_hit, None), "Should detect collision"
    assert planner._check_collision(fp_hit, np.array([[100.0, 100.0]]), None), "Should not detect collision"
    print("Correctness sanity check passed.")

if __name__ == "__main__":
    benchmark_collision()
