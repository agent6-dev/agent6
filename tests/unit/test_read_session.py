# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A session can read this project's other sessions."""

from __future__ import annotations

import json
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.runs.layout import session_layout
from agent6.tools.sessions import ROSTER_MAX, conversation, roster, session_briefs


def _session(state: Path, bucket: str, sid: str, mode: str, task: str, turns: list[str]) -> Path:
    d = state / bucket / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "mode": mode,
                "run_id": sid,
                "user_task": task,
                "start_ts": "2026-07-31T01:00:00",
            }
        ),
        encoding="utf-8",
    )
    lines = [json.dumps({"type": "run.start", "user_task": task})]
    lines += [json.dumps({"type": "role.result", "text": t}) for t in turns]
    (d / "logs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_the_roster_spans_every_bucket(tmp_path: Path) -> None:
    """A run, a plan and an ask are all sessions; a roster that showed only
    runs would hide exactly the quick ask you wanted to pick up."""
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask", "how do I convert h264", ["use ffmpeg"])
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "add a flag", ["done"])
    modes = {b.id: b.mode for b in session_briefs(tmp_path)}
    assert modes == {"quiet-fox-AAAAAA": "ask", "brave-elk-BBBBBB": "run"}


def test_the_conversation_reads_oldest_first(tmp_path: Path) -> None:
    d = _session(
        tmp_path,
        "asks",
        "quiet-fox-AAAAAA",
        "ask",
        "how do I convert h264",
        ["use ffmpeg", "with libx265"],
    )
    layout = session_layout(tmp_path, "quiet-fox-AAAAAA")
    assert layout is not None and layout.run_dir == d
    text = conversation(layout, max_chars=10_000)
    assert text.index("how do I convert") < text.index("use ffmpeg") < text.index("libx265")
    assert text.startswith("user: how do I convert")


def test_truncation_keeps_the_tail(tmp_path: Path) -> None:
    """A later session usually wants what the earlier one CONCLUDED; the head
    is the task the roster already carries."""
    _session(tmp_path, "runs", "long-BBBBBB", "run", "t", ["x" * 400, "THE ANSWER"])
    layout = session_layout(tmp_path, "long-BBBBBB")
    assert layout is not None
    text = conversation(layout, max_chars=200)
    assert "THE ANSWER" in text
    assert "earlier characters elided" in text


def test_a_query_finds_a_session_by_its_content(tmp_path: Path) -> None:
    """An id is useless to a model that does not know it; content and recency
    are how a session is actually found."""
    _session(
        tmp_path, "asks", "quiet-fox-AAAAAA", "ask", "video question", ["use ffmpeg -c:v libx265"]
    )
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "add a flag", ["unrelated"])
    assert [b.id for b in roster(tmp_path, "libx265").briefs] == ["quiet-fox-AAAAAA"]
    assert [b.id for b in roster(tmp_path, "add a flag").briefs] == ["brave-elk-BBBBBB"]
    assert roster(tmp_path, "nothing here").briefs == ()


def test_a_torn_journal_line_does_not_break_the_read(tmp_path: Path) -> None:
    """A live session's last line can be half-written."""
    d = _session(tmp_path, "runs", "live-BBBBBB", "run", "t", ["first"])
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "role.res')
    layout = session_layout(tmp_path, "live-BBBBBB")
    assert layout is not None
    assert "first" in conversation(layout, max_chars=10_000)


def test_a_session_with_no_conversation_says_so(tmp_path: Path) -> None:
    d = tmp_path / "runs" / "empty-BBBBBB"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "run"}), encoding="utf-8")
    layout = session_layout(tmp_path, "empty-BBBBBB")
    assert layout is not None
    assert "no readable journal" in conversation(layout, max_chars=100)


@pytest.mark.parametrize("escape", ["../../etc", "..", "/etc/passwd", "a/../../b"])
def test_no_path_from_the_model_reaches_the_filesystem(tmp_path: Path, escape: str) -> None:
    """The model names a session by id, never a path: resolution matches real
    directory names in the project's buckets, so traversal cannot resolve."""
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "t", ["x"])
    assert session_layout(tmp_path, escape) is None


def test_the_tool_returns_the_roster_and_refuses_an_unknown_id(tmp_path: Path) -> None:
    """Dispatch-level: the roster rides on every answer (like read_background),
    and an id the project does not have is an error, not an empty read."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher, ToolError

    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask", "video question", ["use ffmpeg"])
    d = ToolDispatcher(root=tmp_path, config=Config(), state_dir=tmp_path)
    out = d.dispatch("read_session", {}).to_wire()
    assert out["sessions"] == ["[quiet-fox-AAAAAA] ask · 2026-07-31T01:00: video question"]
    assert "conversation" not in out
    got = d.dispatch("read_session", {"id": "quiet-fox-AAAAAA"}).to_wire()
    assert "use ffmpeg" in got["conversation"]
    with pytest.raises(ToolError, match="no session"):
        d.dispatch("read_session", {"id": "nope"})


def test_the_tool_is_unwired_without_a_project_state_dir(tmp_path: Path) -> None:
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher, ToolError

    d = ToolDispatcher(root=tmp_path, config=Config())
    with pytest.raises(ToolError, match="state dir"):
        d.dispatch("read_session", {})


def test_a_project_with_many_sessions_does_not_flood_the_context(tmp_path: Path) -> None:
    """Every read_session call pays for the roster. At 2000 sessions the
    uncapped list rendered ~70k tokens, so one lookup cost more than the answer."""
    for i in range(ROSTER_MAX + 25):
        _session(tmp_path, "runs", f"s{i:04d}-AAAAAA", "run", "t", [])
    got = roster(tmp_path, "")
    assert len(got.briefs) == ROSTER_MAX
    assert got.more
    assert "narrow with `query`" in got.lines()[-1], "a silent cut reads as the whole project"


def test_a_query_matching_everything_is_capped_too(tmp_path: Path) -> None:
    for i in range(ROSTER_MAX + 25):
        _session(tmp_path, "runs", f"s{i:04d}-AAAAAA", "run", "refactor the parser", [])
    got = roster(tmp_path, "parser")
    assert len(got.briefs) == ROSTER_MAX and got.more


def test_a_query_reads_journals_without_holding_them_in_memory(tmp_path: Path) -> None:
    """Journals reach megabytes; slurping each one to answer a yes/no was ~1 GB
    per call. The needle is planted across a chunk boundary."""
    big = _session(tmp_path, "runs", "big-AAAAAA", "run", "t", []) / "logs.jsonl"
    with big.open("w") as fh:
        fh.write("x" * ((1 << 16) - 4))  # the needle straddles a chunk boundary
        fh.write("nEeDlE")
        fh.write("y" * (4 << 20))
    size = big.stat().st_size
    peak = _peak_bytes_reading(lambda: roster(tmp_path, "needle"))
    assert [b.id for b in roster(tmp_path, "needle").briefs] == ["big-AAAAAA"]
    assert peak < size // 4, f"held {peak} bytes of a {size}-byte journal"


def _peak_bytes_reading(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_reader_sees_what_the_assistant_SAID_in_a_real_journal(tmp_path: Path) -> None:
    """Written by the real emitter, not by hand. The prose reached the journal
    only as `role.text_delta`, which is emitted only when streaming is on -- so
    a headless run (CI, a redirected stdout, every spawned ask) recorded no
    assistant text, and this tool returned the task and a list of tool names.
    Every fixture that hand-wrote `{"type": "role.result", "text": ...}` passed
    against a shape the engine never emitted."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent6.app.providers import InstrumentedProvider
    from agent6.budget import BudgetTracker
    from agent6.events import EventSink
    from agent6.runs.layout import session_layout

    d = tmp_path / "asks" / "quiet-fox-AAAAAA"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "ask", "user_task": "how do I convert h264"}),
        encoding="utf-8",
    )
    events = EventSink(d / "logs.jsonl")
    events.emit("run.start", user_task="how do I convert h264")
    inner = MagicMock()
    inner.call.return_value = SimpleNamespace(
        text="use ffmpeg -c:v libx265",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={},
    )
    InstrumentedProvider(
        inner=inner,
        role="worker",
        model="m",
        provider_name="p",
        events=events,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
    ).call(system="s", messages=[{"role": "user", "content": "q"}], tools=[], max_tokens=64)

    layout = session_layout(tmp_path, "quiet-fox-AAAAAA")
    assert layout is not None
    text = conversation(layout, max_chars=10_000)
    assert "use ffmpeg -c:v libx265" in text, "the answer the other session reached is missing"
    assert "how do I convert h264" in text
