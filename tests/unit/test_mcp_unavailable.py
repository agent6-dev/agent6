# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A server that never started says so where the operator is looking.

`[mcp] failed to start 'x'` on stderr is visible from a terminal and nowhere
else: under `agent6 acp` it lands in the editor's log pane, not the
conversation. The journal is what every surface folds, so the failure goes
there.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import events as agent6_events
from agent6.app import _setup
from agent6.config import Config
from agent6.viewmodel import transcript


def _cfg(command: list[str]) -> Config:
    return Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"notes": {"command": command}}}}
    )


def test_a_server_that_cannot_spawn_is_recorded_not_just_logged(tmp_path: pathlib.Path) -> None:
    """A server that cannot spawn is recorded, not just logged.

    The manager knows which servers are missing; a logger that may go nowhere is not a record.
    """
    mgr = _setup.start_mcp_manager_if_enabled(_cfg(["/nonexistent/mcp-server"]), tmp_path, "none")
    assert mgr is not None
    try:
        assert [f.name for f in mgr.failures] == ["notes"]
        assert mgr.failures[0].error
    finally:
        mgr.close()


def test_the_failure_reaches_the_journal(tmp_path: pathlib.Path) -> None:
    logs = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(logs)
    mgr = _setup.start_mcp_manager_if_enabled(
        _cfg(["/nonexistent/mcp-server"]), tmp_path, "none", events=events
    )
    assert mgr is not None
    mgr.close()

    written = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    unavailable = [e for e in written if e["type"] == "mcp.server_unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0]["server"] == "notes"
    assert unavailable[0]["error"]


def test_a_server_that_starts_emits_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the failure is news.

    A per-server "started" event would be noise in every conversation.
    """
    from agent6.tools import mcp_client

    def _ok(_self: object) -> None:
        return None

    monkeypatch.setattr(mcp_client._MCPServer, "start", _ok)  # pyright: ignore[reportPrivateUsage]
    logs = tmp_path / "logs.jsonl"
    mgr = _setup.start_mcp_manager_if_enabled(
        _cfg(["true"]), tmp_path, "none", events=agent6_events.EventSink(logs)
    )
    assert mgr is not None
    mgr.close()
    assert not logs.exists() or "mcp.server_unavailable" not in logs.read_text(encoding="utf-8")


def test_the_conversation_shows_it_on_every_surface() -> None:
    """The conversation shows an unavailable server on every surface.

    A marker in the shared transcript fold, so the CLI, TUI, web and ACP all render it without each
    learning the event.
    """
    fold = transcript.TranscriptFold()
    items = list(
        fold.feed(
            {
                "type": "mcp.server_unavailable",
                "server": "notes",
                "error": "could not spawn MCP server 'notes': No such file",
            }
        )
    )
    assert [i.kind for i in items] == ["marker"]
    assert "notes" in items[0].body
    assert "No such file" in items[0].body
    assert "tools are missing" in items[0].body
    # The error already names the server; the marker must not say it twice.
    assert items[0].body.count("notes") == 1


def test_the_editor_is_told_too() -> None:
    """The whole reason it is an event.

    ACP projects the same fold, so the editor gets it in the conversation instead of a log pane it
    may not show.
    """
    from agent6.ui.acp import updates as acp_updates

    event = {
        "type": "mcp.server_unavailable",
        "server": "notes",
        "error": "could not spawn MCP server 'notes': boom",
    }
    updates = [
        u
        for item in transcript.TranscriptFold().feed(event)
        for u in acp_updates.updates_for(item, acp_session_id="s")
    ]
    assert updates, "the editor was told nothing"
    text = updates[0]["params"]["update"]["content"]["text"]
    assert "notes" in text and "tools are missing" in text


def test_check_names_the_spawn_error_not_a_symptom(capsys: pytest.CaptureFixture[str]) -> None:
    """`agent6 check` names the spawn error, not a symptom.

    "started but exposed no tools" for a server that never started is a symptom and a false claim.
    """
    from agent6.ui.cli import check_cmds

    _doctor_check_mcp = check_cmds._doctor_check_mcp  # pyright: ignore[reportPrivateUsage]

    checks = _doctor_check_mcp(_cfg(["/nonexistent/mcp-server"]))
    assert [c.status for c in checks] == ["FAIL"]
    assert "No such file" in checks[0].detail
    assert "exposed no tools" not in checks[0].detail
