# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What an editor is told a run did.

Projected from the SAME fold the CLI, TUI and web render, so a fourth surface
cannot disagree with the other three about what happened.
"""

from __future__ import annotations

import json
from typing import Any

from agent6.ui.acp.updates import updates_for, updates_for_events
from agent6.viewmodel.transcript import TranscriptItem


def _kinds(updates: list[dict[str, Any]]) -> list[str]:
    return [u["params"]["update"]["sessionUpdate"] for u in updates]


def test_reasoning_and_answer_are_different_channels() -> None:
    """An editor renders thinking collapsed; conflating them would present the
    model's scratch work as its answer."""
    thinking = updates_for(TranscriptItem("thinking", body="let me look"), session_id="s")
    text = updates_for(TranscriptItem("text", body="the answer"), session_id="s")
    assert _kinds(thinking) == ["agent_thought_chunk"]
    assert _kinds(text) == ["agent_message_chunk"]


def test_the_operators_own_words_echo_back_as_theirs() -> None:
    """A steer is the human speaking. Attributing it to the agent would make
    the transcript lie about who said what."""
    updates = updates_for(TranscriptItem("operator", body="also add a flag"), session_id="s")
    assert _kinds(updates) == ["user_message_chunk"]


def test_a_tool_is_a_call_and_then_an_outcome() -> None:
    """ACP models a tool call as a thing with a lifecycle. An editor that only
    ever saw the finished one could not show work in progress -- which for a
    long verify is the whole point."""
    updates = updates_for(
        TranscriptItem("tool", name="run_verify_command", arg="pytest", ok=True, detail="12s"),
        session_id="s",
    )
    assert _kinds(updates) == ["tool_call", "tool_call_update"]
    call, done = (u["params"]["update"] for u in updates)
    assert call["toolCallId"] == done["toolCallId"], "the update must pair with its call"
    assert call["status"] == "pending"
    assert done["status"] == "completed"


def test_a_failed_tool_says_so() -> None:
    updates = updates_for(
        TranscriptItem("tool", name="run_command", arg="ls", ok=False), session_id="s"
    )
    assert updates[1]["params"]["update"]["status"] == "failed"


def test_a_tool_still_running_is_not_reported_failed() -> None:
    """`ok=None` is "no outcome yet", which is not the same as a failure."""
    updates = updates_for(TranscriptItem("tool", name="grep", arg="x"), session_id="s")
    assert updates[1]["params"]["update"]["status"] == "completed"


def test_an_empty_body_produces_nothing() -> None:
    """A blank chunk renders as an empty bubble in the editor."""
    assert updates_for(TranscriptItem("text", body="   "), session_id="s") == []


def test_every_notification_is_addressed_and_well_formed() -> None:
    updates = updates_for(TranscriptItem("text", body="hi"), session_id="sess-1")
    (one,) = updates
    assert one["jsonrpc"] == "2.0"
    assert one["method"] == "session/update"
    assert one["params"]["sessionId"] == "sess-1"
    assert "id" not in one, "a notification expects no reply"
    json.dumps(one)  # it has to survive the wire


def test_deltas_are_folded_once_across_the_whole_run() -> None:
    """The fold is stateful: deltas accumulate and flush at a turn boundary.
    A fresh fold per event would emit every partial message as if it were
    whole."""
    events: list[dict[str, Any]] = [
        {"type": "role.text_delta", "text": "the "},
        {"type": "role.text_delta", "text": "answer"},
        {"type": "role.result", "role": "worker", "ok": True},
    ]
    updates = updates_for_events(events, session_id="s")
    assert _kinds(updates) == ["agent_message_chunk"]
    assert updates[0]["params"]["update"]["content"]["text"] == "the answer"


def test_a_headless_run_still_has_something_to_show() -> None:
    """No streaming means no deltas; the settled text on role.result is what
    the fold falls back to, and it must reach the editor too."""
    events: list[dict[str, Any]] = [
        {"type": "role.result", "role": "worker", "ok": True, "text": "done it"}
    ]
    updates = updates_for_events(events, session_id="s")
    assert updates[0]["params"]["update"]["content"]["text"] == "done it"
