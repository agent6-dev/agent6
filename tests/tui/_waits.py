# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Waits for a TUI test, so the deadline rule has one owner.

`wait_for` is the general one: a condition, a deadline, and a name to fail by.
An iteration-capped pause loop spins through in milliseconds under load while
the awaited work lags behind, then falls through silently to fail at some later
assert; a wall-clock deadline fails at the wait that actually missed.

The rest wait on an approval row.

A row is queryable a frame before the labels it composes, so a test that reads
its presence as readiness races that mount: `focus_answers` finds no labels and
silently leaves the focus in the composer, where the answer keys are the letters
they are, and a render of the row raises NoMatches. The product defers its own
call through `call_after_refresh`, which is what these waits stand in for.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable
from typing import Any

from agent6.ui.tui import composer

TIMEOUT_S = 10.0


async def wait_for(
    pilot: Any,
    cond: Callable[[], bool],
    what: str,
    *,
    pump: Callable[[], None] | None = None,
    timeout: float = TIMEOUT_S,
) -> None:
    """Wait until the condition holds, failing by name at the deadline.

    `pump` drives whatever the condition waits on (a host tick, a screen poll) each pass.
    """
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        if pump is not None:
            pump()
        await pilot.pause(0.05)


def answerable(view: Any) -> bool:
    """Whether *view*'s approval row can take an answer: mounted, labels and all.

    `query(ApprovalRow)` alone is true a frame earlier.
    """
    rows = view.query(composer.ApprovalRow)
    return bool(rows) and bool(rows.first().query(".answer-yes"))


async def focus_answers(view: Any, pilot: Any, timeout: float = TIMEOUT_S) -> None:
    """Put the focus on the row's first answer and wait until it holds there.

    A focus that lands on nothing leaves the composer focused, where the answer keys are text.
    """

    def holds() -> bool:
        rows = view.query(composer.ApprovalRow)
        return bool(rows) and bool(rows.first().holds_focus())

    def nudge() -> None:
        rows = view.query(composer.ApprovalRow)
        if rows:
            rows.first().focus_answers()

    await wait_for(pilot, holds, "the answers to take the focus", pump=nudge, timeout=timeout)


async def row_gone(
    view: Any, pilot: Any, pump: Callable[[], None] | None = None, timeout: float = TIMEOUT_S
) -> bool:
    """Return whether the row has unmounted, which follows an answer a tick later.

    `pump` feeds the fold each pass, for a withdrawal that waits on an unread event.
    """
    deadline = time.monotonic() + timeout
    while view.query(composer.ApprovalRow):
        if time.monotonic() >= deadline:
            return False
        if pump is not None:
            pump()
        await pilot.pause(0.05)
    return True


async def answer_written(
    run: pathlib.Path, pilot: Any, name: str = "ap1", timeout: float = TIMEOUT_S
) -> str:
    """The answer file's text once the click's or key's answer has landed through the host.

    Three paths answer nothing and leave no file: a key a text field kept or
    that reached no row, a screen holding no open approval, and a host that
    reads dead. The wait ends in which of them it was, since the file's absence
    alone names none.
    """
    path = run / "approvals" / f"{name}.answer"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            app = pilot.app
            screen = app.screen
            raise AssertionError(
                f"no answer for {name} in {timeout:.0f}s:"
                f" focus={type(app.focused).__name__}"
                f" rows={len(screen.query(composer.ApprovalRow))}"
                f" open={getattr(screen, '_approval', None)}"
                f" controllable={app.session_controllable()} status={app.dir_status}"
                f" files={sorted(p.name for p in (run / 'approvals').iterdir())}"
            )
        await pilot.pause(0.05)
    return path.read_text(encoding="utf-8")
