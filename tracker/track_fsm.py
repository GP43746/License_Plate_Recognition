"""
Track Finite State Machine
===========================
States per persistent_plate_id:
  NEW         → just appeared, no OCR yet
  STABILIZING → has OCR hits but not yet stable
  STABLE      → vote_threshold met
  LOST        → SORT dropped it; shown in panel until buffer expires
"""

from enum import Enum


class TrackState(Enum):
    NEW         = "NEW"
    STABILIZING = "STAB"
    STABLE      = "STABLE"
    LOST        = "LOST"


# Color map for state badges (BGR)
STATE_COLOR = {
    TrackState.NEW:         (180, 180,   0),   # yellow
    TrackState.STABILIZING: (200, 130,   0),   # orange
    TrackState.STABLE:      (  0, 210,   0),   # green
    TrackState.LOST:        ( 80,  80,  80),   # grey
}

STATE_LABEL = {
    TrackState.NEW:         "[NEW]",
    TrackState.STABILIZING: "[STAB]",
    TrackState.STABLE:      "[STABLE]",
    TrackState.LOST:        "[LOST]",
}


class TrackFSM:
    def __init__(self):
        self._states: dict[int, TrackState] = {}

    def get(self, pid: int) -> TrackState:
        return self._states.get(pid, TrackState.NEW)

    def transition(self, pid: int, stable_plate: str,
                   stability_ratio: float, is_active: bool):
        """Drive state transitions based on pipeline observations."""
        if not is_active:
            self._states[pid] = TrackState.LOST
            return

        current = self._states.get(pid, TrackState.NEW)

        if stable_plate:
            # Once STABLE, stay STABLE (no regression)
            self._states[pid] = TrackState.STABLE
        elif stability_ratio > 0:
            if current != TrackState.STABLE:
                self._states[pid] = TrackState.STABILIZING
        else:
            if current not in (TrackState.STABILIZING, TrackState.STABLE):
                self._states[pid] = TrackState.NEW

    def remove(self, pid: int):
        self._states.pop(pid, None)
