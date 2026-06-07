# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 history graph` DFS tree rendering."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent6.cli import main
from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, write_node


def _node(
    nid: str,
    *,
    parent: str | None,
    title: str,
    children: tuple[str, ...] = (),
    status: str = "pending",
    commit_sha: str = "",
) -> TaskNode:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return TaskNode.model_validate(
        {
            "id": nid,
            "parent_id": parent,
            "title": title,
            "rationale": "",
            "acceptance": "",
            "relevant_paths": (),
            "depends_on": (),
            "children": children,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "created_by": "planner",
            "commit_sha": commit_sha,
        }
    )


def _seed_tree(tmp_path: Path, run_id: str) -> None:
    """Build a small tree:
    root
      step1 (passed, commit aaaaaaa...)
        sub1a
        sub1b
      step2 (failed)
    """
    layout = RunLayout(state_dir=tmp_path / ".agent6", run_id=run_id)
    layout.ensure()
    root_id = "0" * 25 + "R"
    s1_id = "0" * 25 + "1"
    s2_id = "0" * 25 + "2"
    s1a_id = "0" * 25 + "A"
    s1b_id = "0" * 25 + "B"
    root = _node(root_id, parent=None, title="root task", children=(s1_id, s2_id))
    s1 = _node(
        s1_id,
        parent=root_id,
        title="step 1",
        children=(s1a_id, s1b_id),
        status="passed",
        commit_sha="abcdef1234567890",
    )
    s2 = _node(s2_id, parent=root_id, title="step 2", status="failed")
    s1a = _node(s1a_id, parent=s1_id, title="sub 1a")
    s1b = _node(s1b_id, parent=s1_id, title="sub 1b")
    nodes = {n.id: n for n in (root, s1, s2, s1a, s1b)}
    for n in nodes.values():
        write_node(layout, nodes, n)


def test_history_graph_renders_dfs_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_tree(tmp_path, "test-run-AAAA11")
    rc = main(["history", "graph", "test-run-AAAA11"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in out.splitlines() if line and not line.startswith("Run id:")]
    # Strict DFS: root, then step1, then deep-left sub1a, then sub1b, then step2.
    assert lines == [
        "[pending] root task",
        "  [passed] step 1  (commit: abcdef1)",
        "    [pending] sub 1a",
        "    [pending] sub 1b",
        "  [failed] step 2",
    ]


def test_history_graph_uses_most_recent_when_no_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_tree(tmp_path, "older-run-AAAA11")
    _seed_tree(tmp_path, "newer-run-BBBB22")
    # Touch the newer run to make sure mtime ordering picks it.
    (tmp_path / ".agent6" / "runs" / "newer-run-BBBB22").touch()
    rc = main(["history", "graph"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[pending] root task" in out


def test_history_graph_missing_run_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["history", "graph", "nonexistent"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no runs directory" in err or "no run matches" in err


def test_history_graph_empty_graph_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    layout = RunLayout(state_dir=tmp_path / ".agent6", run_id="empty-run-CCCC33")
    layout.ensure()
    rc = main(["history", "graph", "empty-run-CCCC33"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no persisted graph nodes" in err
