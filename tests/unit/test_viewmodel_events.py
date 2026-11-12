# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The journal's tolerant reader: parse_event maps raw rows to the records.

Pins the parse boundary itself (the fold's end-to-end behaviour is pinned by
test_viewmodel_state and the golden compat test). The contract: every family the
SessionState fold consumes parses to its record with the fold's historical
coercion, and everything else, telemetry, unknown and future types, a typeless line,
degrades to RawEvent so the fold drops it instead of crashing.
"""

from __future__ import annotations

from agent6.sessions import journal
from agent6.viewmodel import events as ev


def test_known_families_parse_to_their_records() -> None:
    assert ev.parse_event({"type": "session.start", "user_task": "t"}) == journal.SessionStart(
        user_task="t", mode="", execution_id=""
    )
    assert ev.parse_event(
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    ) == journal.SessionStop(
        execution_id="",
        reason="finish_session",
        turns=0,
        verification="passed",
        summary="",
        scoped=False,
    )
    scoped = ev.parse_event(
        {
            "type": "session.stop",
            "reason": "finish_session",
            "verification": "passed",
            "scoped": True,
        }
    )
    assert isinstance(scoped, journal.SessionStop) and scoped.scoped is True
    assert ev.parse_event({"type": "session.resume"}) == journal.SessionResume(
        session_id="", user_task="", steer="", mode="", turn=0, messages=0, execution_id=""
    )
    assert ev.parse_event({"type": "session.steer_requested"}) == journal.SessionSteerRequested(
        source=""
    )


def test_unknown_and_telemetry_and_typeless_become_rawevent() -> None:
    # A telemetry type the fold never consumes, an unknown/future type, and a line
    # with no `type` all fall to RawEvent (the fold's old `case _`).
    samples = ({"type": "loop.auto_commit.failed", "sha": "x"}, {"type": "totally.new"}, {"foo": 1})
    for raw in samples:
        parsed = ev.parse_event(raw)
        assert isinstance(parsed, ev.RawEvent)
        assert parsed.raw is raw  # carries the dict for the log-line renderer


def test_coercion_matches_the_folds_historical_defaults() -> None:
    # Missing fields default, not raise (the fold never validated).
    assert ev.parse_event({"type": "role.call"}) == journal.RoleCall(
        role="", model="", provider="", model_call_id="", seat_index=None
    )
    # The pairing fields read with defaults, so a journal written without them folds as before;
    # a seat index is an int, never a bool.
    paired = ev.parse_event(
        {"type": "role.call", "role": "review:x", "model_call_id": "abc", "seat_index": 1}
    )
    assert paired == journal.RoleCall(
        role="review:x", model="", provider="", model_call_id="abc", seat_index=1
    )
    result = ev.parse_event(
        {"type": "role.result", "ok": False, "model_call_id": "abc", "seat_index": True}
    )
    assert isinstance(result, journal.RoleResult)
    assert result.ok is False and result.seat_index is None and result.tokens_in == 0
    # An unknown verification word reads as none; a garbled numeric degrades the row.
    stop = ev.parse_event({"type": "session.stop", "verification": "maybe"})
    assert isinstance(stop, journal.SessionStop) and ev.verification_word(stop.verification) is None
    assert isinstance(ev.parse_event({"type": "verify.end", "exit_code": "x"}), ev.RawEvent)
    # A tool call without an id reads None, so the fold matches its result by name.
    call = ev.parse_event({"type": "tool.call", "name": "read_file", "args_preview": {}})
    assert isinstance(call, journal.ToolCall) and call.call_id is None
    # A journal written before the rename keys the preview `args`.
    old = ev.parse_event({"type": "tool.call", "name": "read_file", "args": {"path": "a"}})
    assert isinstance(old, journal.ToolCall) and old.args_preview == {"path": "a"}
    # A command's outcome rides along; a row without one leaves the fields unset.
    done = ev.parse_event(
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "exit=1",
            "exit_code": 1,
        }
    )
    assert isinstance(done, journal.ToolResult) and done.exit_code == 1 and done.duration_s is None


def test_tool_rows_and_deltas_read_their_call_identity() -> None:
    """The execution, the model call, the seat and the provider's block id ride into the records."""
    call = ev.parse_event(
        {
            "type": "tool.call",
            "name": "read_file",
            "args_preview": {},
            "call_id": 3,
            "execution_id": "e1",
            "model_call_id": "c1",
            "panel_id": "p",
            "seat_index": 1,
            "provider_tool_call_id": "toolu_1",
        }
    )
    assert isinstance(call, journal.ToolCall)
    assert (call.execution_id, call.model_call_id, call.panel_id, call.seat_index) == (
        "e1",
        "c1",
        "p",
        1,
    )
    assert call.provider_tool_call_id == "toolu_1"
    delta = ev.parse_event(
        {"type": "role.text_delta", "role": "worker", "text": "hi", "model_call_id": "c1"}
    )
    assert isinstance(delta, journal.RoleTextDelta) and delta.model_call_id == "c1"
    assert delta.execution_id is None  # a row without the field leaves it unset
