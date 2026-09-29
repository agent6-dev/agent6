# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Define the steer seam front-ends hand the lifecycle, and its file-bridge form.

The bridge serves surfaces with no controlling terminal; the CLI layers its
SIGINT pause menu on the same shape, `app.frontend.SteerHooks`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent6.sessions.ipc import (
    clear_steer_answer,
    clear_steer_request,
    read_steer_answer,
    steer_answer_is_abort,
    steer_interrupt_pending,
    steer_request_pending,
)


@dataclass
class SteerState:
    """The steer hooks a front-end hands the lifecycle.

    Attributes:
        requested: A steer is pending at the next boundary.
        clear: Drop the pending request and its answer.
        prompt: Read the steer text, or None when the front-end abandoned the prompt.
        restore: Put the terminal back.
        abort_pending: Polled during a streaming call, so a Stop interrupts a long turn.
        interrupt: Polled during a streaming call; True aborts it so the prompt runs now.
        reset_stage: Called at each execution entry, so a stage armed in a finished
            execution cannot open a phantom pause; the marker files stay for `--steer`.
        armed: A Ctrl-C pause is armed, which the approval prompt reads as a boundary.
        prompt_now: Run the pause menu at once and seed its action as the next
            boundary's answer; a no-op off the terminal.
    """

    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]
    abort_pending: Callable[[], bool]
    interrupt: Callable[[], bool]
    reset_stage: Callable[[], None]
    armed: Callable[[], bool] = field(default=lambda: False)
    prompt_now: Callable[[], None] = field(default=lambda: None)


def file_bridge_steer(session_dir: Path) -> SteerState:
    """Return the steer hooks for a run with no controlling terminal.

    No SIGINT handler; requests and answers travel over the front-end file bridge.
    """

    def prompt() -> str | None:
        answer = read_steer_answer(session_dir)
        # An abandoned prompt clears its request, or the next boundary would block again.
        if answer is None:
            clear_steer_request(session_dir)
        return answer

    def clear() -> None:
        clear_steer_answer(session_dir)
        clear_steer_request(session_dir)

    return SteerState(
        requested=lambda: steer_request_pending(session_dir),
        clear=clear,
        prompt=prompt,
        restore=lambda: None,
        abort_pending=lambda: steer_answer_is_abort(session_dir),
        interrupt=lambda: steer_interrupt_pending(session_dir),
        reset_stage=lambda: None,
    )
