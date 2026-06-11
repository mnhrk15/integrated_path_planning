"""Fail-Safe State Machine for autonomous vehicle simulation.

This module defines the states and transitions for the vehicle's fail-safe behavior.
"""

import math
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
    # When set, the planner must come to a stop within this travel distance
    # [m]: candidates that stop later (or not at all) are rejected. Issued
    # when the safe-speed envelope demands a full stop, to defeat the
    # receding-horizon procrastination of jerk-optimal stop profiles (each
    # replan otherwise re-stretches the stop over the whole horizon and the
    # actual braking never starts until the fail-safe slams).
    max_stop_distance: Optional[float] = None

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
        # Recovery thresholds operate on clearance (= min_distance minus the
        # combined collision radius) so their meaning does not depend on the
        # footprint mode. They can be specified directly via
        # state_machine_recover_clearance_{caution,emergency}; when those are
        # None the legacy single-circle keys (centre-to-pedestrian distance)
        # are converted instead:
        # min_distance > safe_distance  <=>  clearance > safe_distance - combined
        # In multi_circle mode the effective ego radius is the footprint circle
        # radius (validate_config enforces safe_distance > combined either way,
        # so the clearance thresholds are strictly positive).
        if getattr(config, 'ego_footprint', None) is not None:
            from .footprint import effective_ego_radius
            ego_radius = effective_ego_radius(config)
        else:
            ego_radius = getattr(config, 'ego_radius', 1.0)
        combined_radius = ego_radius + getattr(config, 'ped_radius', 0.2)
        recover_caution = getattr(config, 'state_machine_recover_clearance_caution', None)
        recover_emergency = getattr(config, 'state_machine_recover_clearance_emergency', None)
        self.clearance_caution = (
            recover_caution if recover_caution is not None
            else getattr(config, 'state_machine_safe_distance_caution', 2.0) - combined_radius
        )
        self.clearance_emergency = (
            recover_emergency if recover_emergency is not None
            else getattr(config, 'state_machine_safe_distance_emergency', 3.0) - combined_radius
        )
        # Preventive escalation: clearance below which NORMAL drops to CAUTION
        # even though planning succeeds (defensive slowdown near pedestrians).
        # RSS-style speed dependence: threshold = trigger_clearance +
        # time_headway * ego_speed, so fast approaches escalate earlier while
        # the low-speed regime stays quiet. Both 0.0 disables the trigger
        # (legacy behavior). validate_config enforces that the threshold at
        # the CAUTION recovery speed stays below clearance_caution so
        # NORMAL<->CAUTION cannot oscillate.
        self.trigger_clearance_caution = getattr(
            config, 'state_machine_trigger_clearance_caution', 0.0
        )
        self.trigger_time_headway = getattr(
            config, 'state_machine_trigger_time_headway', 0.0
        )
        # Safe-speed envelope in CAUTION: cap the planner target at the speed
        # from which a constant envelope_decel braking stops envelope_standoff
        # short of the nearest pedestrian ("the closer, the slower"). 0.0
        # disables the envelope (legacy fixed CAUTION target). The clearance
        # is taken from the most recent update() call.
        self.envelope_decel = getattr(config, 'state_machine_envelope_decel', 0.0)
        self.envelope_standoff = getattr(config, 'state_machine_envelope_standoff', 0.5)
        # Omnidirectional clearance (trigger / recovery gates) and the
        # forward-restricted one (envelope / stop directive: braking only
        # helps against frontal conflicts, and a pedestrian beside or behind
        # a stopped vehicle must not pin its speed target at zero).
        self._last_clearance = float('inf')
        self._last_clearance_ahead = float('inf')

    def update(self, plan_found: bool, safety_metrics: Dict[str, Any],
               ego_speed: float = 0.0) -> StateMachineOutput:
        """Update state based on current iteration results.

        Args:
            plan_found: Whether the planner found a valid path
            safety_metrics: Dictionary of safety metrics (e.g., 'min_distance', 'ttc')
            ego_speed: Current ego speed [m/s] for the speed-dependent
                preventive trigger (0.0 reproduces the fixed-clearance trigger)

        Returns:
            StateMachineOutput containing the new state and planner constraints
        """
        self._last_clearance = safety_metrics.get('clearance', float('inf'))
        self._last_clearance_ahead = safety_metrics.get(
            'clearance_ahead', self._last_clearance)
        trigger_threshold = (self.trigger_clearance_caution
                             + self.trigger_time_headway * max(ego_speed, 0.0))
        # Default transitions
        if self.current_state == VehicleState.NORMAL:
            if not plan_found:
                self.current_state = VehicleState.CAUTION
                self.consecutive_failures += 1
            elif (trigger_threshold > 0.0
                  and safety_metrics.get('clearance', float('inf'))
                  < trigger_threshold):
                # Preventive escalation: planning succeeded but a pedestrian
                # is close — slow down defensively. Not a failure, so the
                # counter stays at 0 and the ordinary recovery branch can
                # return to NORMAL as soon as the clearance gate reopens.
                self.current_state = VehicleState.CAUTION
                self.consecutive_failures = 0
            else:
                self.consecutive_failures = 0
                
        elif self.current_state == VehicleState.CAUTION:
            if plan_found and self.consecutive_failures == 0:
                # If we recovered and planned successfully, try to go back to NORMAL
                # Check safety clearance from config
                clearance = safety_metrics.get('clearance', float('inf'))
                if clearance > self.clearance_caution:
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
                 clearance = safety_metrics.get('clearance', float('inf'))
                 if clearance > self.clearance_emergency:
                     self.current_state = VehicleState.CAUTION
            else:
                # Keep trying to stop
                pass

        return self._get_planner_config()
    
    def _get_planner_config(self) -> StateMachineOutput:
        """Generate planner configuration for the current state."""
        if self.current_state == VehicleState.NORMAL:
            # The safe-speed envelope is a state-independent cap ("never
            # faster than what a comfortable stop can handle"): applying it
            # already in NORMAL starts the pre-slowdown several metres before
            # the preventive trigger fires. Against a converging pedestrian
            # the clearance can collapse at ~3 m/s, so braking that only
            # starts at the CAUTION transition is physically too late for a
            # gentle stop — the cap must bind while the gap is still wide.
            target_override = None
            v_env = self._envelope_speed()
            if v_env is not None and v_env < self.config.ego_target_speed:
                target_override = v_env
            return StateMachineOutput(
                state=VehicleState.NORMAL,
                target_speed_override=target_override,
                constraint_overrides=None
            )
            
        elif self.current_state == VehicleState.CAUTION:
            # Relax the acceleration limit to find a path, and slow down
            # preventively. The curvature limit is NOT relaxed: it is a
            # kinematic property of the vehicle (minimum turning radius) and
            # cannot be traded against risk.
            accel_mult = getattr(self.config, 'state_machine_caution_accel_multiplier', 1.5)
            speed_mult = getattr(self.config, 'state_machine_caution_speed_multiplier', 0.8)
            target_speed = self.config.ego_target_speed * speed_mult
            max_stop_distance = None
            v_env = self._envelope_speed()
            if v_env is not None:
                target_speed = min(target_speed, v_env)
                if v_env <= 0.0:
                    # Inside the standoff the envelope demands a full stop:
                    # also bound WHERE the stop happens, or the jerk-optimal
                    # selection keeps re-stretching the stop over the whole
                    # horizon and never actually brakes.
                    max_stop_distance = self._stop_room_to_pedestrian()
            return StateMachineOutput(
                state=VehicleState.CAUTION,
                target_speed_override=target_speed,
                constraint_overrides={
                    "max_accel": self.config.ego_max_accel * accel_mult,
                    "max_speed": self.config.ego_max_speed * speed_mult
                },
                max_stop_distance=max_stop_distance
            )

        elif self.current_state == VehicleState.EMERGENCY:
            # STOP immediately. Braking effort and lateral acceleration are
            # relaxed towards the friction limit (evasive regime), but the
            # curvature limit stays (the vehicle cannot out-steer its own
            # geometry).
            accel_mult = getattr(self.config, 'state_machine_emergency_accel_multiplier', 3.0)
            lat_mult = getattr(self.config, 'state_machine_emergency_lat_accel_multiplier', 2.0)
            return StateMachineOutput(
                state=VehicleState.EMERGENCY,
                target_speed_override=0.0,
                constraint_overrides={
                    "max_accel": self.config.ego_max_accel * accel_mult,
                    "max_lat_accel": getattr(self.config, 'ego_max_lat_accel', 3.0) * lat_mult
                },
                # Commit the planned stop to the available room too (same
                # anti-procrastination as CAUTION; only with the envelope
                # enabled so legacy scenarios keep their exact behaviour).
                max_stop_distance=(self._stop_room_to_pedestrian()
                                   if self.envelope_decel > 0.0 else None)
            )
            
        return StateMachineOutput(VehicleState.NORMAL)

    def _envelope_speed(self) -> Optional[float]:
        """Safe-speed envelope value for the last observed clearance.

        v_env = sqrt(2 * envelope_decel * (clearance - standoff)): the speed
        from which a constant envelope_decel braking stops envelope_standoff
        short of the nearest pedestrian ("the closer, the slower"). The
        target ramps down continuously while approaching a conflict, reaches
        0 at clearance <= standoff (which also suppresses restart attempts
        while pedestrians pass), and rises smoothly as the clearance reopens.
        Returns None when the envelope is disabled or nothing was observed.
        """
        if self.envelope_decel <= 0.0 or not math.isfinite(self._last_clearance_ahead):
            return None
        stop_room = max(self._last_clearance_ahead - self.envelope_standoff, 0.0)
        return math.sqrt(2.0 * self.envelope_decel * stop_room)

    def _stop_room_to_pedestrian(self) -> Optional[float]:
        """Travel distance within which a commanded stop must complete.

        0.2 m last-resort margin to the nearest pedestrian (smaller than the
        envelope standoff on purpose: when already inside the standoff the
        vehicle may legitimately come to rest closer than the planner would
        normally aim for). None when no pedestrian has been observed.
        """
        if not math.isfinite(self._last_clearance_ahead):
            return None
        return max(self._last_clearance_ahead - 0.2, 0.05)
