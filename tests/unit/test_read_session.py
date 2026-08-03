# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A session can read this project's other sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.runs.layout import session_layout
from agent6.tools.sessions import conversation, matching_sessions, session_briefs


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
    assert [b.id for b in matching_sessions(tmp_path, "libx265")] == ["quiet-fox-AAAAAA"]
    assert [b.id for b in matching_sessions(tmp_path, "add a flag")] == ["brave-elk-BBBBBB"]
    assert matching_sessions(tmp_path, "nothing here") == []


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
