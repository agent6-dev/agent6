# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Track the dispatch loop's repeat and error streaks.

The repeat streak (the same tool and args back to back) powers the identical-result stub and
the repeat warning; the error streak (the same tool failing the same way) climbs the
nudge, escalate, stop ladder. A successful dispatch clears the whole error spiral in
`note_success`, the one reset site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent6.harness._guards import Ladder, Rung
from agent6.harness._nudges import (
    TOOL_ERROR_ESCALATE_AFTER,
    TOOL_ERROR_NUDGE_AFTER,
    TOOL_ERROR_STOP_AFTER,
)


def tool_error_ladder() -> Ladder:
    """Return the ladder a streak of tool errors sharing one signature climbs."""
    return Ladder(TOOL_ERROR_NUDGE_AFTER, TOOL_ERROR_ESCALATE_AFTER, TOOL_ERROR_STOP_AFTER)


@dataclass(slots=True)
class SpiralGuard:
    """Hold the repeat and error streaks of one execution.

    Never snapshotted: a resumed execution starts unspiralled.

    Attributes:
        last_call_sig: The last dispatched (tool, args) signature.
        call_streak: How many times that signature ran back to back.
        last_served_content: The bytes most recently served to the model, success or error.
        warned_at_iteration: When the repeat warning last fired; it re-arms after a quiet turn.
        error_sig: The signature of the current error streak.
        error_streak: How many times that error repeated.
        error_ladder: The tool-error ladder the streak climbs.
        last_error_was_denial: Whether the last error was an operator's denial.
    """

    last_call_sig: str | None = None
    call_streak: int = 0
    last_served_content: str | None = None
    warned_at_iteration: int = 0
    error_sig: str | None = None
    error_streak: int = 0
    error_ladder: Ladder = field(default_factory=tool_error_ladder)
    last_error_was_denial: bool = False

    def note_call(self, sig: str, *, polling: bool = False) -> None:
        """Extend the repeat streak on the same signature, restart it on any other.

        A poll is never a repeat: `read_background` is meant to be called again with the same
        id until the job ends.

        Args:
            sig: The (tool, args) signature of the call.
            polling: Whether the call polls a background job.
        """
        if sig == self.last_call_sig and not polling:
            self.call_streak += 1
        else:
            self.last_call_sig = sig
            self.call_streak = 1

    def stub_repeat(self, content: str, *, min_chars: int) -> bool:
        """Return whether a repeat's unchanged result is served as a short stub.

        Args:
            content: The result bytes the call produced.
            min_chars: The size below which the stub saves nothing.

        Returns:
            True for a back-to-back repeat whose result is unchanged and longer than `min_chars`.
        """
        return (
            self.call_streak >= 2
            and content == self.last_served_content
            and len(content) > min_chars
        )

    def note_success(self, content: str) -> None:
        """Record a successful dispatch and clear the whole error spiral.

        Args:
            content: The result bytes served to the model.
        """
        self.last_served_content = content
        self.error_sig = None
        self.error_streak = 0
        self.error_ladder.rearm()
        self.last_error_was_denial = False

    def note_error(self, sig: str, *, denial: bool, content: str) -> None:
        """Extend the error streak on the same signature, restart it and re-arm on a new one.

        Args:
            sig: The error's signature.
            denial: Whether the error was an operator's denial.
            content: The error bytes served to the model.
        """
        self.last_served_content = content
        self.last_error_was_denial = denial
        if sig == self.error_sig:
            self.error_streak += 1
        else:
            self.error_sig = sig
            self.error_streak = 1
            self.error_ladder.rearm()

    def climb_error(self) -> Rung | None:
        """Return the rung the error streak reaches on the tool-error ladder."""
        return self.error_ladder.climb(self.error_streak)
