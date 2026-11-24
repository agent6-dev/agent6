# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A hosted web search's record: the journal event, the budget count and each surface's line."""

from __future__ import annotations

import io
import json
import pathlib
from typing import Any

import httpx2
import pytest

from agent6 import budget as agent6_budget
from agent6.app import role_providers
from agent6.providers import anthropic
from agent6.sessions import event_log
from agent6.ui.cli import console_render, sessions_show
from agent6.ui.tui import conversation as tui_conversation
from agent6.ui.web import model as web_model
from agent6.viewmodel import log_line, transcript
from agent6.viewmodel.session import log as session_log
from agent6.viewmodel.session import view as session_view
from tests.acp import v1
from tests.unit import test_web_search_messages as reply

_LINE = "Search: httpx2 release notes · https://example.com/a https://example.com/b"


def _journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, *, streamed: bool
) -> list[dict[str, Any]]:
    """Make one instrumented worker call against the recorded reply and return its journal."""

    def fake_post(url: str, **_kwargs: Any) -> Any:
        body = {"content": reply._CONTENT, "stop_reason": "tool_use", "usage": reply._USAGE}
        return reply._FakeResponse(body)

    def fake_stream(method: str, url: str, **_kwargs: Any) -> Any:
        return reply._FakeStream(reply._stream_of(reply._CONTENT))

    monkeypatch.setattr("agent6.providers.transport.http_post", fake_post)
    monkeypatch.setattr(httpx2, "stream", fake_stream)
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    logs = tmp_path / "logs.jsonl"
    provider = role_providers.InstrumentedProvider(
        inner=anthropic.AnthropicProvider(api_key="k", model="m", provider_name="p", budget=budget),
        actor="worker",
        model="m",
        provider_name="p",
        events=event_log.EventSink(logs),
        budget=budget,
        stream_text=streamed,
    )
    provider.call(system="s", messages=[{"role": "user", "content": "q"}])
    return [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("streamed", [False, True])
def test_a_search_is_journaled_with_its_call_and_counted_in_the_budget_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, streamed: bool
) -> None:
    """One `model.web_search` per search, before its call's result; the count reaches the row."""
    events = _journal(monkeypatch, tmp_path, streamed=streamed)
    kinds = [e["type"] for e in events if not e["type"].endswith("_delta")]
    assert kinds == [
        "model.call",
        "provider.request",
        "model.web_search",
        "model.attempt",
        "model.result",
        "budget.update",
    ]
    search = next(e for e in events if e["type"] == "model.web_search")
    call = events[0]
    assert {k: search[k] for k in ("actor", "provider", "model", "model_call_id")} == {
        k: call[k] for k in ("actor", "provider", "model", "model_call_id")
    }
    assert (search["search_id"], search["query"], search["status"]) == (
        "srvtoolu_01",
        "httpx2 release notes",
        "completed",
    )
    assert search["urls"] == ["https://example.com/a", "https://example.com/b"]
    assert events[-1]["web_searches"] == 1


def test_a_reply_with_no_search_journals_what_it_did_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A reply with no search emits no search event and leaves the budget row's keys alone."""
    monkeypatch.setattr(reply, "_CONTENT", reply._CONTENT[:1])
    monkeypatch.setitem(reply._USAGE, "server_tool_use", {"web_search_requests": 0})
    events = _journal(monkeypatch, tmp_path, streamed=False)
    assert "model.web_search" not in {e["type"] for e in events}
    assert "web_searches" not in events[-1]


@pytest.fixture
def search_event() -> dict[str, Any]:
    return {
        "ts": "2026-11-22T12:00:00.000000+00:00",
        "type": "model.web_search",
        "actor": "worker",
        "provider": "p",
        "model": "m",
        "model_call_id": "c1",
        "search_id": "srvtoolu_01",
        "query": "httpx2 release notes",
        "status": "completed",
        "urls": ["https://example.com/a", "https://example.com/b"],
    }


def test_every_surface_shows_the_search_line(search_event: dict[str, Any]) -> None:
    """The shared fold makes the line once; the CLI, the TUI, the web and ACP all print it."""
    items = transcript.TranscriptFold().feed(search_event)
    assert [(item.kind, item.body) for item in items] == [("marker", _LINE)]

    assert log_line.format_log_line(search_event).endswith(f"model.web_search   {_LINE}")

    out = io.StringIO()
    console = console_render.ConsoleView(out, color=False)
    console.feed(search_event)
    console.close()
    assert f"── {_LINE} ──" in out.getvalue()

    drawn = tui_conversation._item_renderables(items[0], detail="collapsed")
    assert drawn[0].plain == f"── {_LINE} ──"

    web = web_model.conversation_items([search_event])
    assert web == [{"kind": "marker", "lines": [[(f"── {_LINE} ──", "marker")]]}]

    told = v1.updates_for(items[0], acp_session_id="s")
    assert told[0]["params"]["update"]["content"]["text"] == f"[agent6] {_LINE}\n"


def test_a_failed_search_names_its_outcome_and_provider_text_stays_inert() -> None:
    """An outcome other than completed is said; a control sequence in a query cannot paint."""
    fold = transcript.TranscriptFold()
    failed: dict[str, Any] = {
        "type": "model.web_search",
        "query": "q",
        "status": "max_uses_exceeded",
        "urls": [],
    }
    assert fold.feed(failed)[0].body == "Search: q (max_uses_exceeded)"
    hostile = {
        "type": "model.web_search",
        "query": "a\x1b[2J\nb",
        "urls": ["https://x/\x1b]0;t\x07"],
    }
    body = fold.feed(hostile)[0].body
    assert "\x1b" not in body and "\n" not in body and "\x07" not in body
    assert fold.feed({"type": "model.web_search", "query": ""})[0].body == "Search"


def test_sessions_show_carries_the_search_count(tmp_path: pathlib.Path) -> None:
    """The budget row's count reaches `sessions show`: its usage line and its JSON."""
    rows = [
        {"type": "session.start", "session_id": "s", "mode": "run", "user_task": "t"},
        {
            "type": "budget.update",
            "usd_total": 0.01,
            "input_total": 5,
            "output_total": 7,
            "web_searches": 2,
        },
    ]
    logs = tmp_path / "logs.jsonl"
    logs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    scan = session_log.scan_session_log(logs)
    assert sessions_show._usage_line(scan) == "in=5 out=7 searches=2  usage $0.01"
    assert session_view.budget_dict(scan.budget)["execution"]["web_searches"] == 2
