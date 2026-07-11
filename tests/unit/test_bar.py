# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""BarController's steer holder -- the thread-safe seam the loop polls in
[cli] input = "bar" mode -- tested without a terminal."""

from __future__ import annotations

from agent6.ui.cli._bar import BarController


def test_steer_holder_maps_input_to_actions() -> None:
    bar = BarController()
    ss = bar.steer_state()
    assert ss.requested() is False

    bar.submit("focus on the parser")  # a typed instruction
    assert ss.requested() is True
    assert ss.prompt() == "focus on the parser"
    assert ss.requested() is False  # consumed by prompt()

    bar.submit("/stop")  # slash form -> abort (matches the toolbar)
    assert ss.abort_pending() is True
    assert ss.prompt() == "abort"
    assert ss.abort_pending() is False

    bar.submit("")  # empty -> continue: a no-op, nothing queued
    assert ss.requested() is False


def test_prompt_runs_directly_before_the_loop_starts() -> None:
    # bar.prompt runs fn directly when the bar loop is not up yet (no event loop
    # to route to), so callers never block on a not-yet-running bar.
    assert BarController().prompt(lambda: "answer") == "answer"
