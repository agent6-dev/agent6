# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""operator_inputs: the journal's task and steers, in order, repeats collapsed."""

from __future__ import annotations

from agent6.viewmodel import operator_inputs


def test_task_then_steers_in_order_across_legs() -> None:
    events = [
        {"type": "session.start", "user_task": "polish the TUI", "mode": "run"},
        {"type": "role.result", "text": "done"},
        {"type": "loop.steer.injected", "chars": 14, "text": "focus on tests"},
        {"type": "session.end", "reason": "steer_abort"},
        # A resume appends to the same journal; its steer joins the list.
        {"type": "loop.resume.start"},
        {"type": "loop.steer.injected", "chars": 7, "text": "ship it"},
    ]
    assert operator_inputs(events) == ["polish the TUI", "focus on tests", "ship it"]


def test_blank_countonly_and_consecutive_repeats_drop() -> None:
    events = [
        {"type": "session.start", "user_task": "t"},
        {"type": "loop.steer.injected", "text": "again"},
        {"type": "loop.steer.injected", "text": "again"},
        {"type": "loop.steer.injected", "text": "   "},
        {"type": "loop.steer.injected", "chars": 5},  # an older log: count only
        {"type": "loop.steer.injected", "text": "again"},
    ]
    # The blank and count-only entries render nothing, so the final "again"
    # still sits next to the first and collapses with it.
    assert operator_inputs(events) == ["t", "again"]
