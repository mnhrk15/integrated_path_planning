"""Fail-Safe State Machine for autonomous vehicle simulation.

This module defines the states and transitions for the vehicle's fail-safe behavior.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SimulationConfig

from .data_structures import VehicleState

@dataclass
class StateMachineOutput:
    """Output from the state machine to control the planner."""
    state: VehicleState
    target_speed_override: Optional[float] = None
    constraint_overrides: Optional[Dict[str, float]] = None

class FailSafeStateMachine:
    """Manages vehicle state transitions based on planning results and safety metrics."""
    
    def __init__(self, config: 'SimulationConfig') -> None:
        """Initialize the fail-safe state machine.
        
        Args:
            config: Simulation configuration containing state machine parameters
        """
        self.config = config
        self.current_state = VehicleState.NORMAL
        self.consecutive_failures = 0
        
    def update(self, plan_found: bool, safety_metrics: Dict[str, Any]) -> StateMachineOutput:
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
                # Check safety distance from config
                min_dist = safety_metrics.get('min_distance', float('inf'))
                safe_distance = getattr(self.config, 'state_machine_safe_distance_caution', 0.5)
                if min_dist > safe_distance:
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
                 safe_distance = getattr(self.config, 'state_machine_safe_distance_emergency', 1.0)
                 if min_dist > safe_distance:
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
            accel_mult = getattr(self.config, 'state_machine_caution_accel_multiplier', 1.5)
            curvature_mult = getattr(self.config, 'state_machine_caution_curvature_multiplier', 1.2)
            speed_mult = getattr(self.config, 'state_machine_caution_speed_multiplier', 0.8)
            return StateMachineOutput(
                state=VehicleState.CAUTION,
                target_speed_override=None, # Keep trying to move, maybe slower?
                constraint_overrides={
                    "max_accel": self.config.ego_max_accel * accel_mult,
                    "max_curvature": self.config.ego_max_curvature * curvature_mult,
                    "max_speed": self.config.ego_max_speed * speed_mult
                }
            )
            
        elif self.current_state == VehicleState.EMERGENCY:
            # STOP immediately
            accel_mult = getattr(self.config, 'state_machine_emergency_accel_multiplier', 3.0)
            curvature_mult = getattr(self.config, 'state_machine_emergency_curvature_multiplier', 2.0)
            return StateMachineOutput(
                state=VehicleState.EMERGENCY,
                target_speed_override=0.0,
                # Allow extreme maneuvers to stop
                constraint_overrides={
                    "max_accel": self.config.ego_max_accel * accel_mult,
                    "max_curvature": self.config.ego_max_curvature * curvature_mult
                }
            )
            
        return StateMachineOutput(VehicleState.NORMAL)
