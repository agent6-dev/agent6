from __future__ import annotations

import dataclasses

from agent6.viewmodel import state_types, state_view


def test_every_state_field_has_an_explicit_wire_classification() -> None:
    """A new state field requires an explicit public or private decision."""
    public = {
        "last_verify",
        "last_event_ep",
        "budget",
        "session_id",
        "cursor_task_id",
        "tool_count",
        "finish_summary",
        "log_count",
        "approvals",
        "tool_calls",
        "commits",
        "recent_diffs",
        "started",
        "undone_to",
        "review",
        "compact_gists_live",
        "compact_elided",
        "latest_diff",
        "log_tail",
        "undone_text",
        "finished",
        "last_role",
        "pins",
        "questions",
        "stop_reason",
        "end_summary",
        "unattended_questions",
        "tasks",
        "mode",
        "steer_requests",
        "verification",
        "verify_scoped",
        "stranded",
    }
    private = {"targets", "task_label"}
    assert set(state_view.PUBLIC_STATE_FIELDS) == public
    assert not public & private
    assert {f.name for f in dataclasses.fields(state_types.SessionState)} == public | private
    payload = state_view.session_state_as_dict(state_types.SessionState())
    assert public <= payload.keys()
    assert not private & payload.keys()
