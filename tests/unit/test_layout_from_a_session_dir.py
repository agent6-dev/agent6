# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rebuilding a layout must not lose the bucket it came from.

`SessionLayout(state_dir=..., session_id=...)` defaults to `runs`, so a rebuild from an id retargets
a plan or an ask at a directory that does not exist.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent6.graph.models import TaskNode
from agent6.graph.storage import write_node
from agent6.sessions.layout import SessionLayout, bucket_dir, layout_of

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def test_a_layout_round_trips_through_its_own_directory() -> None:
    for bucket in ("runs", "plans", "asks", "machines"):
        original = SessionLayout(state_dir=Path("/s"), session_id="brave-oak-AAAAAA", subdir=bucket)
        assert layout_of(original.session_dir) == original


def test_the_end_of_run_task_tree_renders_for_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The task tree reads a plan's graph from its own bucket, and a failure is reported."""
    from agent6.ui.cli.sessions_show import _print_task_tree  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    session = bucket_dir(tmp_path / "state", "plans") / "brave-oak-AAAAAA"
    layout = SessionLayout(
        state_dir=tmp_path / "state", session_id="brave-oak-AAAAAA", subdir="plans"
    )
    layout.ensure()
    # Written by the real writer, so the test reads the graph's format, not the fixture's.
    nodes: dict[str, TaskNode] = {}
    for node_id, title, parent in (
        ("01AAAAAAAAAAAAAAAAAAAAAAAA", "root task", None),
        ("01BBBBBBBBBBBBBBBBBBBBBBBB", "step one", "01AAAAAAAAAAAAAAAAAAAAAAAA"),
    ):
        node = TaskNode(
            id=node_id,
            title=title,
            parent_id=parent,
            created_at=_TS,
            updated_at=_TS,
            created_by="planner",
        )
        nodes[node_id] = node
        write_node(layout, nodes, node)

    _print_task_tree(session)
    # The subject is that the plan's graph is read at all; the layout pointed at runs/ before.
    out = capsys.readouterr().out
    assert "plan:" in out and "root task" in out


def test_no_new_site_builds_a_layout_without_naming_its_bucket() -> None:
    """Every rebuilt layout names its bucket; `subdir`'s runs/ default is a silent assumption."""
    src = Path(__file__).resolve().parents[2] / "src" / "agent6"
    # Where defaulting is the point, with the reason.
    allowed = {
        # `sessions diff|merge|commits` with no id means the most recent run.
        "ui/cli/sessions_cmds.py",
        "ui/cli/sessions_merge.py",
        # The owner of this directory's layout passes the bucket it read off the path.
        "sessions/layout.py",
    }
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        for call in re.findall(r"SessionLayout\((?:[^()]|\([^()]*\))*\)", path.read_text("utf-8")):
            if "subdir=" not in call and str(path.relative_to(src)) not in allowed:
                offenders.append(f"{path.relative_to(src)}: {' '.join(call.split())[:70]}")
    assert not offenders, (
        "these assume the runs/ bucket; pass subdir=session_bucket(<mode>),"
        f" or add the file to `allowed` with the reason: {offenders}"
    )


def test_a_task_tree_render_failure_is_not_an_empty_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while rendering the tree reaches the CLI's reporter, never a silent exit 0."""
    from agent6.ui.cli.sessions_show import _print_task_tree  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    session = bucket_dir(tmp_path / "state", "plans") / "brave-oak-BBBBBB"
    layout = SessionLayout(
        state_dir=tmp_path / "state", session_id="brave-oak-BBBBBB", subdir="plans"
    )
    layout.ensure()
    nodes: dict[str, TaskNode] = {}
    for node_id, title, parent in (
        ("01AAAAAAAAAAAAAAAAAAAAAAAA", "root task", None),
        ("01BBBBBBBBBBBBBBBBBBBBBBBB", "step one", "01AAAAAAAAAAAAAAAAAAAAAAAA"),
    ):
        node = TaskNode(
            id=node_id,
            title=title,
            parent_id=parent,
            created_at=_TS,
            updated_at=_TS,
            created_by="planner",
        )
        nodes[node_id] = node
        write_node(layout, nodes, node)

    def broken(*_a: object, **_k: object) -> list[str]:
        raise RuntimeError("a render defect")

    monkeypatch.setattr("agent6.ui.cli._task_tree.task_tree_lines", broken)
    with pytest.raises(RuntimeError, match="a render defect"):
        _print_task_tree(session)
