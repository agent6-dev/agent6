# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The operator's side of a run: the callables a front-end injects for
steering, stopping and compacting the loop, the steer verbs, and the pin
invariant."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agent6.types import AutoCommitDirective
from agent6.workflows.subrun import GroupLaneSpawner

# `/pin` instructions are re-injected verbatim after every tier-2 restart, so
# their total is capped. Over the cap a pin lands as an ordinary steer (the
# instruction still reaches the model once); only its survives-compaction
# durability is refused.
PINS_MAX_CHARS = 4_000

# The steer texts that end or hand over the run, as the operator types them:
# the verb the loop acts on, the event that records it, the log line.
STEER_VERBS: dict[str, tuple[str, str, str]] = {
    "abort": ("abort", "loop.steer.aborted", "abort - halting the run"),
    "exit": ("exit", "loop.steer.exited", "exit - halting the run and leaving the terminal"),
    "/undo": ("undo", "loop.steer.undo", "/undo - forking back before the last message"),
    "detach": ("detach", "loop.steer.detached", "detach - stopping to resume in the background"),
}


@dataclass(frozen=True, slots=True)
class OperatorBridge:
    """What the operator can do to a running loop, as the front-end injects
    it. The defaults do nothing: a loop with no operator runs unattended."""

    # Polled between iterations; on a positive the loop asks `steer_prompt`
    # for the instruction (or "abort") and `steer_clear` consumes the request.
    steer_requested: Callable[[], bool] = field(default=lambda: False)
    steer_clear: Callable[[], None] = field(default=lambda: None)
    steer_prompt: Callable[[], str | None] = field(default=lambda: None)
    # Called at each leg entry (run/resume): disarms a SIGINT stage the prior
    # leg never consumed, without touching the steer marker files.
    steer_reset: Callable[[], None] = field(default=lambda: None)
    # "Compact now" from a front-end: polled at the same pre-call boundary as
    # the tiered thresholds; a positive forces the tier-2 summarise-and-restart.
    # The marker travels the same file bridge as steer.
    compact_requested: Callable[[], str | None] = field(default=lambda: None)
    compact_clear: Callable[[], None] = field(default=lambda: None)
    # "Stop after this step": polled at each completed-iteration boundary
    # (tool results and auto-commit landed), ending the run cleanly there. The
    # mid-turn immediate stop is the steer "abort" answer.
    stop_requested: Callable[[], bool] = field(default=lambda: False)
    stop_clear: Callable[[], None] = field(default=lambda: None)
    # Polled DURING a streaming model call: True once the operator asked to
    # stop, so a long reasoning turn aborts promptly.
    should_abort: Callable[[], bool] = field(default=lambda: False)
    # Polled DURING a streaming call: True once the operator asked to steer
    # (Ctrl-C, the TUI's `s`), so the watchdog ends the turn and the loop
    # reaches its steer boundary at once instead of waiting the turn out.
    should_interrupt: Callable[[], bool] = field(default=lambda: False)
    # Called once per landed auto-commit. "stop" ends the loop cleanly as
    # interactive_stop; "undo" takes the steer's /undo path; "continue" (the
    # default) runs the next iteration. `agent6 run -i` installs its REPL
    # prompt here.
    after_auto_commit: Callable[[int, str], AutoCommitDirective] = field(
        default=lambda _i, _sha: "continue"
    )
    # `/undo`: commits the tree as it stands onto the session's ref, forks the
    # session at the state before its last operator message and puts the
    # checkout back to that tree (app.undo.undo_fork, injected: workflows never
    # import app); returns (new_session_id, undone_text), or None with the
    # reason printed.
    undo_forker: Callable[[], tuple[str, str] | None] | None = None
    # `/parallel` steer dispatch: the ui-side group spawner that runs a sibling
    # group of subordinate lanes to completion and imports their branches into
    # this run's repo. None (the default, every headless path, and inside a
    # lane: depth 1) makes a `/parallel` directive answer with feedback and
    # continue.
    lane_spawner: GroupLaneSpawner | None = None


def try_pin(pins: list[str], instruction: str) -> bool:
    """Append *instruction* to *pins* when it is non-empty and fits the
    PINS_MAX_CHARS total; whether it was pinned. The one owner of the pin
    invariants: `/pin` and the pre-run --pin seeding both go through it."""
    instruction = instruction.strip()
    if not instruction:
        return False
    if sum(len(p) for p in pins) + len(instruction) > PINS_MAX_CHARS:
        return False
    pins.append(instruction)
    return True
