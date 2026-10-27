from __future__ import annotations

import dataclasses
import io
from typing import Any

import pytest

from agent6 import review_records
from agent6.ui.acp import updates
from agent6.ui.cli import _console_view
from agent6.ui.tui import conversation
from agent6.viewmodel import log_line, review, state, transcript


def start_event(**changes: Any) -> dict[str, Any]:
    return {
        "type": "loop.review.start",
        "panel_id": "panel",
        "execution_id": "exec",
        "target": dataclasses.asdict(
            review_records.ReviewTarget("b", "bt", "h", "ht", "h", (), "revisions")
        ),
        "seats": [],
        "rule_sources": [{"path": "REVIEW.md", "blob_id": "abc123", "supplied_bytes": 9}],
        "rules_complete": True,
        "rules_reason": None,
        **changes,
    }


def test_rule_metadata_survives_start_completion_and_wire_round_trip() -> None:
    rows = [
        start_event(),
        {"type": "loop.review.freeform", "panel_id": "panel", "execution_id": "exec"},
    ]
    view = state.fold_session(rows).review
    assert view.completion == "complete"
    assert view.rules_complete is True
    assert view.rule_sources == (review_records.RuleSource("REVIEW.md", "abc123", 9),)
    wire = review.as_dict(view)
    assert wire["rule_sources"] == wire["panels"][0]["rule_sources"] == rows[0]["rule_sources"]
    assert wire["rules_complete"] is True
    assert wire["panels"][0]["rules_reason"] is None


@pytest.mark.parametrize(
    "complete,reason", [(True, None), (False, "REVIEW.md: unreadable blob"), (None, None)]
)
def test_absent_incomplete_and_legacy_rules_remain_distinct(
    complete: bool | None, reason: str | None
) -> None:
    view = state.fold_session(
        [start_event(rule_sources=[], rules_complete=complete, rules_reason=reason)]
    ).review
    assert view.rules_complete is complete
    assert view.panels[0].rules_reason == reason
    assert not view.read_failure


def test_cli_tui_and_acp_render_the_same_recorded_sources() -> None:
    event = start_event()
    items = transcript.fold_transcript([event])
    assert len(items) == 1
    item = items[0]
    expected = "REVIEW.md: blob abc123, 9 bytes"
    assert expected in item.body and "rules complete" in item.body
    assert "base tree bt" in item.body
    output = io.StringIO()
    console = _console_view.ConsoleView(out=output, color=False)
    console.feed(event)
    console.close()
    assert expected in output.getvalue()
    rendered = conversation._item_renderables(item, detail="collapsed")
    assert expected in "\n".join(line.plain for line in rendered)
    notifications = updates.updates_for(item, acp_session_id="session")
    assert expected in str(notifications)
    assert expected in log_line.format_log_line(event)


def test_incomplete_and_absent_notices_name_the_difference() -> None:
    absent = transcript.fold_transcript([start_event(rule_sources=[])])
    failed = transcript.fold_transcript(
        [start_event(rule_sources=[], rules_complete=False, rules_reason="REVIEW.md: hidden")]
    )
    assert absent and "No base rules exist." in absent[0].body
    assert failed and "REVIEW.md: hidden" in failed[0].body
    assert "rules incomplete" in failed[0].body


@pytest.mark.parametrize("field", ["source", "reason"])
def test_rule_notices_escape_controls_without_losing_the_name(field: str) -> None:
    """Rule paths and refusal reasons must not become terminal commands on any text surface."""
    name = "dir\x1b]52;c;c2VjcmV0\x07/REVIEW.md"
    event = start_event(
        rule_sources=[{"path": name, "blob_id": "abc123", "supplied_bytes": 9}]
        if field == "source"
        else [],
        rules_complete=field == "source",
        rules_reason=f"{name}: unreadable" if field == "reason" else None,
    )
    (item,) = transcript.fold_transcript([event])
    output = io.StringIO()
    console = _console_view.ConsoleView(out=output, color=False)
    console.feed(event)
    console.close()
    tui = "\n".join(line.plain for line in conversation._item_renderables(item, detail="collapsed"))
    (notification,) = updates.updates_for(item, acp_session_id="session")
    acp = notification["params"]["update"]["content"]["text"]
    panel = state.fold_session([event]).review.panels[0]
    for rendered in (
        review.rules_notice(event),
        review.rules_notice(panel),
        item.body,
        output.getvalue(),
        tui,
        acp,
        log_line.format_log_line(event),
    ):
        assert "\x1b" not in rendered and "\x07" not in rendered, repr(rendered)
        assert r"dir\u001b]52;c;c2VjcmV0\u0007/REVIEW.md" in rendered
