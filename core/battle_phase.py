"""Temporal battle-phase evidence, separate from visual HUD presence."""
from enum import Enum

from core.ui import ScreenState


class BattlePhase(str, Enum):
    LOADING = "loading"
    PREPARING = "preparing"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class BattlePhaseTracker:
    def __init__(self, duration_seconds=1200):
        self.duration_seconds = duration_seconds
        self.previous = None
        self.active = False

    def observe(self, screen, clock, now, *, active_evidence=False):
        if screen != ScreenState.BATTLE:
            if screen in {ScreenState.LOADING, ScreenState.PORT, ScreenState.RESULTS}:
                self.previous = None
                self.active = False
            return BattlePhase.LOADING if screen == ScreenState.LOADING else BattlePhase.UNKNOWN
        if clock is None or not 0 <= clock <= self.duration_seconds:
            self.previous = None
            # A single unreadable clock frame must not demote a match that was
            # already confirmed by wall-clock tracking. The phase is cleared by
            # a real scene change above (loading/port/results), not by an OCR
            # miss. Demoting here used to stop every helm command until two
            # fresh reads arrived -- about 12 s each time -- and it happened
            # repeatedly during a battle, so the ship kept coasting.
            return BattlePhase.ACTIVE if self.active else BattlePhase.UNKNOWN
        if clock == self.duration_seconds:
            self.previous = (clock, now)
            self.active = False
            return BattlePhase.PREPARING
        if self.active:
            self.previous = (clock, now)
            return BattlePhase.ACTIVE
        previous = self.previous
        self.previous = (clock, now)
        # Short preparation countdowns are not the match timer. Once active,
        # however, the final minute is still a live battle.
        if clock <= 60 and not active_evidence:
            return BattlePhase.PREPARING
        if previous is not None:
            value, observed_at = previous
            elapsed = now - observed_at
            dropped = value - clock
            # The match clock must track wall time, not merely decrease: a
            # frozen capture repeats the same value, and a stale sample moves
            # the clock far less than the elapsed time. Requiring both bounds
            # is what makes a slow sampling rate safe.
            #
            # The window used to be a flat 10 s, but the scene-recovery loop
            # that re-enters battle control samples about once every 13 s (the
            # recovery probe plus one control pass). A live match could then
            # never satisfy ``elapsed <= 10``, so the phase stayed UNKNOWN and
            # the bot never sent a single helm command after entering battle.
            if (
                0 < elapsed <= 60
                and 0 < dropped <= elapsed + 2
                and dropped >= elapsed - 3
            ):
                self.active = True
                return BattlePhase.ACTIVE
        return BattlePhase.UNKNOWN
