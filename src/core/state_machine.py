"""Fail-Safe State Machine for autonomous vehicle simulation.

This module defines the states and transitions for the vehicle's fail-safe behavior.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from .data_structures import VehicleState

@dataclass
class StateMachineOutput:
    """Output from the state machine to control the planner."""
    state: VehicleState
    target_speed_override: Optional[float] = None
    constraint_overrides: Optional[Dict[str, float]] = None

class FailSafeStateMachine:
    """Manages vehicle state transitions based on planning results and safety metrics."""
    
    def __init__(self, config):
        self.config = config
        self.current_state = VehicleState.NORMAL
        self.consecutive_failures = 0
        
    def update(self, plan_found: bool, safety_metrics: dict) -> StateMachineOutput:
        """Update state based on current iteration results.
        
        Args:
            plan_found: Whether the planner found a valid path
            safety_metrics: Dictionary of safety metrics (e.g., 'min_distance', 'ttc')
            
        Returns:
            StateMachineOutput containing the new state and planner constraints
        """
        # Default transitions
        if self.current_state == VehicleState.NORMAL:
            if not plan_found:
                self.current_state = VehicleState.CAUTION
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0
                
        elif self.current_state == VehicleState.CAUTION:
            if plan_found and self.consecutive_failures == 0:
                # If we recovered and planned successfully, try to go back to NORMAL
                # But maybe we should stay in CAUTION for a bit? 
                # For simplicity, if we found a path, check safety.
                min_dist = safety_metrics.get('min_distance', float('inf'))
                if min_dist > self.config.safety_buffer * 2:
                    self.current_state = VehicleState.NORMAL
            elif not plan_found:
                # If CAUTION extraction failed, escalate
                self.current_state = VehicleState.EMERGENCY
                self.consecutive_failures += 1
            else:
                # Plan found in CAUTION mode, stay there until safe
                self.consecutive_failures = 0

        elif self.current_state == VehicleState.EMERGENCY:
            # In EMERGENCY, we usually stay until stopped or explicitly reset
            # But if a very safe path appears, we could recover
            if plan_found:
                 min_dist = safety_metrics.get('min_distance', float('inf'))
                 if min_dist > self.config.safety_buffer * 3:
                     self.current_state = VehicleState.CAUTION
            else:
                # Keep trying to stop
                pass

        return self._get_planner_config()
    
    def _get_planner_config(self) -> StateMachineOutput:
        """Generate planner configuration for the current state."""
        if self.current_state == VehicleState.NORMAL:
            return StateMachineOutput(
                state=VehicleState.NORMAL,
                target_speed_override=None, # Use default config
                constraint_overrides=None
            )
            
        elif self.current_state == VehicleState.CAUTION:
            # Relax constraints to find a path
            return StateMachineOutput(
                state=VehicleState.CAUTION,
                target_speed_override=None, # Keep trying to move, maybe slower?
                # For now, let's say CAUTION means "move carefully" but allows higher jerk/accel if needed
                constraint_overrides={
                    "max_accel": self.config.ego_max_accel * 1.5,
                    "max_curvature": self.config.ego_max_curvature * 1.2,
                    "max_speed": self.config.ego_max_speed * 0.8 # Slow down
                }
            )
            
        elif self.current_state == VehicleState.EMERGENCY:
            # STOP immediately
            return StateMachineOutput(
                state=VehicleState.EMERGENCY,
                target_speed_override=0.0,
                # Allow extreme maneuvers to stop
                constraint_overrides={
                    "max_accel": self.config.ego_max_accel * 3.0,
                    "max_curvature": self.config.ego_max_curvature * 2.0
                }
            )
            
        return StateMachineOutput(VehicleState.NORMAL)
