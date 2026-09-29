# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Anything a listing shows is reachable by every command that takes an id.

A site that rebuilds "id -> layout" with `runs/` hardcoded sees one bucket and cannot open a plan or
an ask.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.paths import state_dir


def _seed(state: Path, bucket: str, session_id: str, *, mode: str, marker: str = "") -> Path:
    d = state / "sessions" / bucket / session_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    (d / "manifest.json").write_text(
        json.dumps({"version": 1, "session_id": session_id, "mode": mode, "user_task": "t"}),
        encoding="utf-8",
    )
    if marker:
        (d / marker).mkdir()
    return d


def test_history_graph_without_an_id_finds_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """History graph without an id finds a plan."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    state = state_dir(tmp_path)
    _seed(state, "plans", "brave-oak-AAAAAA", mode="plan", marker="graph")

    main(["sessions", "graph"])
    err = capsys.readouterr().err
    # It resolved the plan: an empty graph is a separate, honest complaint.
    assert "brave-oak-AAAAAA" in err
    assert "no sessions with a graph" not in err


def test_history_transcript_without_an_id_finds_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    state = state_dir(tmp_path)
    d = _seed(state, "plans", "brave-oak-AAAAAA", mode="plan", marker="transcripts")
    (d / "transcripts" / "0001.json").write_text(
        json.dumps({"seq": 1, "request": {}, "response": {}}), encoding="utf-8"
    )

    assert main(["sessions", "transcript"]) == 0
    assert "brave-oak-AAAAAA" in capsys.readouterr().err


def test_sessions_diff_names_the_real_problem_for_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sessions diff on a plan says it has no branch, not that no session matches the id."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    state = state_dir(tmp_path)
    _seed(state, "plans", "brave-oak-AAAAAA", mode="plan")
    # A populated runs/, so the resolver takes its real path instead of failing on a missing bucket.
    _seed(state, "runs", "quiet-fox-BBBBBB", mode="run")

    main(["sessions", "diff", "brave-oak-AAAAAA"])
    err = capsys.readouterr().err
    assert "no session matches" not in err, err


def test_the_repl_watch_reads_its_own_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The REPL watch reads its own log under asks/."""
    from agent6.ui.cli._repl import repl_show_recent_events  # pyright: ignore[reportPrivateUsage]

    state = state_dir(tmp_path)
    _seed(state, "asks", "brave-oak-AAAAAA", mode="ask")

    repl_show_recent_events(tmp_path, "brave-oak-AAAAAA", n=5)
    out = capsys.readouterr()
    assert "no logs.jsonl" not in out.err + out.out


def test_the_mcp_tools_see_every_bucket(tmp_path: Path) -> None:
    """`list_sessions` lists every bucket, so a plan or an ask is visible over MCP."""
    import io

    from agent6.config import Config
    from agent6.ui.mcp_server import MCPServer

    state = state_dir(tmp_path)
    _seed(state, "plans", "brave-oak-AAAAAA", mode="plan")
    _seed(state, "asks", "quiet-fox-BBBBBB", mode="ask")

    server = MCPServer(root=tmp_path, config=Config(), stdin=io.BytesIO(), stdout=io.BytesIO())
    listed = {s["session_id"] for s in server._h_list_sessions({})["sessions"]}  # pyright: ignore[reportPrivateUsage]
    assert listed == {"brave-oak-AAAAAA", "quiet-fox-BBBBBB"}
