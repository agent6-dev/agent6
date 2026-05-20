# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Integration tests for the alignment guard wired into ImplementWorkflow.

We do not exercise the full state machine here (that path needs git +
sandbox). Instead we drive the alignment helpers directly with a stubbed
guard sub-agent and confirm verdicts translate to the right StepResult
or graph mutation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.graph.client import GraphClient, spawn_curator
from agent6.graph.storage import RunLayout
from agent6.models import AlignmentVerdict, Step
from agent6.workflows import implement as impl_module
from agent6.workflows.implement import ImplementWorkflow


def _silent(_msg: str) -> None:
    return None


def _step(title: str) -> Step:
    return Step(title=title, rationale="r", acceptance="a", relevant_paths=())


def _make_workflow(tmp_path: Path, *, graph_client: GraphClient | None = None) -> ImplementWorkflow:
    return ImplementWorkflow(
        root=tmp_path,
        config=MagicMock(),
        planner=MagicMock(),
        worker=MagicMock(),
        reviewer=MagicMock(),
        critic=MagicMock(),
        dispatcher=MagicMock(),
        graph_client=graph_client,
        alignment_guard=MagicMock(),  # presence enables the guard path
        alignment_period=2,
        logger=_silent,
    )


@pytest.fixture
def curator_client(tmp_path: Path) -> Iterator[GraphClient]:
    layout = RunLayout(root=tmp_path, run_id="r1")
    layout.ensure()
    sock = layout.run_dir / "curator.sock"
    proc = spawn_curator(tmp_path, layout.run_id, sock)
    try:
        with GraphClient(sock) as client:
            yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
            proc.wait()


def _stub_guard(monkeypatch: pytest.MonkeyPatch, verdicts: list[AlignmentVerdict]) -> list[Any]:
    it = iter(verdicts)
    calls: list[Any] = []

    def fake(provider: Any, **kwargs: Any) -> AlignmentVerdict:
        calls.append(kwargs)
        return next(it)

    monkeypatch.setattr(impl_module, "alignment_check", fake)
    return calls


def test_pre_execute_proceed_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_guard(monkeypatch, [AlignmentVerdict(verdict="proceed", reasoning="ok")])
    wf = _make_workflow(tmp_path)
    result = wf._alignment_pre_execute(  # pyright: ignore[reportPrivateUsage]
        node_id=None,
        step=_step("do work"),
        root_node_id=None,
        original_task="ship",
    )
    assert result is None


def test_pre_execute_reject_returns_obsolete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_guard(
        monkeypatch,
        [AlignmentVerdict(verdict="reject", reasoning="off-topic subtask")],
    )
    wf = _make_workflow(tmp_path)
    result = wf._alignment_pre_execute(  # pyright: ignore[reportPrivateUsage]
        node_id=None,
        step=_step("do unrelated thing"),
        root_node_id=None,
        original_task="ship",
    )
    assert result is not None
    assert result.status == "obsolete"
    assert "off-topic subtask" in result.notes


def test_pre_execute_reorder_logs_and_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_guard(
        monkeypatch,
        [
            AlignmentVerdict(
                verdict="reorder",
                reasoning="do sibling first",
                suggested_reorder=("01" * 13, "02" * 13),
            )
        ],
    )
    wf = _make_workflow(tmp_path)
    result = wf._alignment_pre_execute(  # pyright: ignore[reportPrivateUsage]
        node_id=None,
        step=_step("step out of order"),
        root_node_id=None,
        original_task="ship",
    )
    # v1.1 semantics: log + proceed
    assert result is None


def test_pre_execute_escalation_returns_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_guard(
        monkeypatch,
        [AlignmentVerdict(verdict="re-plan-subtree", reasoning="bad subtree")],
    )
    wf = _make_workflow(tmp_path)
    result = wf._alignment_pre_execute(  # pyright: ignore[reportPrivateUsage]
        node_id=None,
        step=_step("step"),
        root_node_id=None,
        original_task="ship",
    )
    assert result is not None
    assert result.status == "failed"
    assert "re-plan-subtree" in result.notes


def test_periodic_only_fires_on_multiples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub_guard(
        monkeypatch,
        [
            AlignmentVerdict(verdict="proceed", reasoning="ok"),
            AlignmentVerdict(verdict="proceed", reasoning="ok"),
        ],
    )
    wf = _make_workflow(tmp_path)
    # alignment_period=2; passed_count=1 → no call
    assert (
        wf._alignment_periodic(  # pyright: ignore[reportPrivateUsage]
            passed_count=1,
            node_id=None,
            step=_step("s"),
            root_node_id=None,
            original_task="ship",
        )
        is None
    )
    assert calls == []
    # passed_count=2 → call, proceed → None
    assert (
        wf._alignment_periodic(  # pyright: ignore[reportPrivateUsage]
            passed_count=2,
            node_id=None,
            step=_step("s"),
            root_node_id=None,
            original_task="ship",
        )
        is None
    )
    assert len(calls) == 1


def test_pre_execute_reject_marks_graph_node_obsolete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft

    root = curator_client.add_subtask(
        AddSubtaskIntent(
            parent_id=None,
            draft=TaskNodeDraft(title="root", created_by="planner"),
        )
    )
    child = curator_client.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(title="child", created_by="planner"),
        )
    )

    _stub_guard(
        monkeypatch,
        [AlignmentVerdict(verdict="reject", reasoning="scope creep")],
    )
    wf = _make_workflow(tmp_path, graph_client=curator_client)
    result = wf._alignment_pre_execute(  # pyright: ignore[reportPrivateUsage]
        node_id=child.id,
        step=_step("child"),
        root_node_id=root.id,
        original_task="ship",
    )
    assert result is not None
    assert result.status == "obsolete"
    state = curator_client.get_state()
    nodes = state["nodes"]
    assert nodes[child.id]["status"] == "obsolete"
