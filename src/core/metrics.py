
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
from .data_structures import SimulationResult

def calculate_ade_fde(history: List[SimulationResult], dt: float) -> Tuple[float, float, int]:
    """Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        history: List of simulation results
        dt: Simulation time step
        
    Returns:
        tuple: (ADE, FDE, number of samples)
    """
    total_ade = 0.0
    total_fde = 0.0
    count = 0
    
    # Iterate through history to find predictions
    for i, result in enumerate(history):
        if result.predicted_trajectories is None or result.predicted_trajectories.size == 0:
            continue
            
        # Prediction: [n_peds, n_steps, 2]
        # These predictions start from time t + dt
        preds = result.predicted_trajectories
        n_peds, n_steps, _ = preds.shape
        
        # We need to compare with ground truth at future steps i+1, i+2, ...
        # Available history length
        remaining_steps = len(history) - (i + 1)
        
        # We can only evaluate up to the available future or prediction length
        eval_steps = min(n_steps, remaining_steps)
        
        if eval_steps == 0:
            continue
            
        for p in range(n_peds):
            # Ground truth trajectory for this pedestrian
            # We assume pedestrian index p stays consistent (simplification)
            gt_traj = []
            valid_gt = True
            
            for k in range(eval_steps):
                d_idx = i + 1 + k
                # Check if pedestrian p exists in ground truth
                # Assuming ped_state.positions has shape [n_current_peds, 2]
                # and p < n_current_peds.
                # WARNING: This assumes constant ID/ordering.
                gt_state = history[d_idx].ped_state
                if p < gt_state.n_peds:
                    gt_traj.append(gt_state.positions[p])
                else:
                    valid_gt = False
                    break
            
            if not valid_gt or len(gt_traj) < 1:
                continue
                
            gt_traj = np.array(gt_traj) # [eval_steps, 2]
            pred_traj = preds[p, :len(gt_traj), :] # [eval_steps, 2]
            
            # Displacement error for this pedestrian at this start time
            displacement = np.linalg.norm(pred_traj - gt_traj, axis=1)
            
            # Accumulate
            total_ade += np.mean(displacement)
            total_fde += displacement[-1]
            count += 1
            
    if count == 0:
        return 0.0, 0.0, 0
        
    return total_ade / count, total_fde / count, count

def calculate_aggregate_metrics(history: List[SimulationResult], dt: float) -> Dict[str, float]:
    """Calculate aggregate metrics for the entire simulation."""
    
    # Safety
    min_distances = [r.metrics.get('min_distance', float('inf')) for r in history]
    ttc_list = [r.metrics.get('ttc', float('inf')) for r in history]
    ttc_valid = [t for t in ttc_list if t > 0 and t != float('inf')]
    
    # Comfort
    jerks = [abs(r.ego_state.jerk) for r in history]
    accels = [abs(r.ego_state.a) for r in history]
    
    # Prediction Accuracy
    ade, fde, n_samples = calculate_ade_fde(history, dt)
    
    metrics = {
        "min_dist": min(min_distances) if min_distances else 0.0,
        "collision_count": sum(1 for r in history if r.metrics.get('collision', False)),
        "min_ttc": min(ttc_valid) if ttc_valid else float('inf'),
        "max_jerk": max(jerks) if jerks else 0.0,
        "mean_jerk": np.mean(jerks) if jerks else 0.0,
        "max_accel": max(accels) if accels else 0.0,
        "ade": ade,
        "fde": fde,
        "pred_samples": n_samples
    }
    
    return metrics
