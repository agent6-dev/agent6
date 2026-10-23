# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Track the transitions, notifications and log lines a machine watcher has surfaced."""

from __future__ import annotations

import dataclasses
import pathlib

from agent6.viewmodel import _machine_logs, machine_state


def notification_key(n: machine_state.NotificationView) -> tuple[str, str, str]:
    """Return a notification's identity for dedup across the sliding window.

    Args:
        n: A notification from `machine_state.MachineState.notifications`.

    Returns:
        Its `(ts, state, message)`, mirroring the web client's `ts|state|message`.
    """
    return (n.ts, n.state, n.message)


@dataclasses.dataclass
class MachineWatchCursor:
    """What a live machine watcher has already surfaced.

    The three dedup rules every front-end agrees on: transitions by count,
    notifications by identity (the tuple is a sliding window, so a count would miss
    every notify past its cap), and the newest state log by path and byte offset.

    Attributes:
        seen_steps: How many transitions were surfaced.
        seen_notifications: The keys of the notifications surfaced; None before seeding.
        log_path: The state log being followed.
        log_offset: Where the last complete line of that log ended.
    """

    seen_steps: int = 0
    seen_notifications: set[tuple[str, str, str]] | None = None
    log_path: pathlib.Path | None = None
    log_offset: int = 0

    def seed_notifications(self, ms: machine_state.MachineState) -> None:
        """Mark every recorded notification as seen, so opening a watch does not replay them."""
        self.seen_notifications = {notification_key(n) for n in ms.notifications}

    def new_transitions(self, ms: machine_state.MachineState) -> list[machine_state.TransitionView]:
        """Return the transitions not yet surfaced and mark them seen."""
        out = list(ms.transitions[self.seen_steps :])
        self.seen_steps = len(ms.transitions)
        return out

    def new_notifications(
        self, ms: machine_state.MachineState
    ) -> list[machine_state.NotificationView]:
        """Return the notifications not yet surfaced and mark them seen."""
        if self.seen_notifications is None:
            self.seen_notifications = set()
        out: list[machine_state.NotificationView] = []
        for n in ms.notifications:
            key = notification_key(n)
            if key not in self.seen_notifications:
                self.seen_notifications.add(key)
                out.append(n)
        return out

    def advance_log(self, root: pathlib.Path) -> tuple[pathlib.Path | None, bool]:
        """Follow the newest per-state log under the instance dir.

        Args:
            root: The instance's dir.

        Returns:
            The current log and whether it changed; on a change the caller resets its
            render state and announces the new agent state.
        """
        newest = _machine_logs.newest_state_log(root)
        if newest != self.log_path:
            self.log_path, self.log_offset = newest, 0
            return newest, True
        return newest, False

    def read_log_lines(self) -> list[str]:
        """Return the complete lines the current state log gained since the last poll."""
        if self.log_path is None:
            return []
        lines, self.log_offset = _machine_logs.read_complete_lines(self.log_path, self.log_offset)
        return lines
