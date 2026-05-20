# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for mid-run steering helpers on `ImplementWorkflow`.

We do not run the full state machine here — we exercise the
`_maybe_steer` boundary directly so the steering UX can be tested without
git / sandbox / curator setup. The curator branch (graph mutations) is
covered separately in the alignment integration tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.models import Plan, Step
from agent6.types import RepoSummary
from agent6.workflows import implement as impl_module
from agent6.workflows.implement import ImplementWorkflow


def _silent(_msg: str) -> None:
    return None


def _plan(*titles: str) -> Plan:
    return Plan(
        summary="sum",
        steps=tuple(
            Step(title=t, rationale="r", acceptance="a", relevant_paths=()) for t in titles
        ),
    )


def _repo(tmp_path: Path) -> RepoSummary:
    return RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def _wf(
    tmp_path: Path,
    *,
    steer_requested: bool,
    steer_prompt_returns: str | None,
) -> tuple[ImplementWorkflow, dict[str, Any]]:
    state: dict[str, Any] = {"clear_called": 0, "prompt_called": 0}

    def requested() -> bool:
        return steer_requested

    def clear() -> None:
        state["clear_called"] += 1

    def prompt() -> str | None:
        state["prompt_called"] += 1
        return steer_prompt_returns

    wf = ImplementWorkflow(
        root=tmp_path,
        config=MagicMock(),
        planner=MagicMock(),
        worker=MagicMock(),
        reviewer=MagicMock(),
        critic=MagicMock(),
        dispatcher=MagicMock(),
        logger=_silent,
        steer_requested=requested,
        steer_clear=clear,
        steer_prompt=prompt,
    )
    return wf, state


def test_maybe_steer_returns_none_when_flag_unset(tmp_path: Path) -> None:
    wf, state = _wf(tmp_path, steer_requested=False, steer_prompt_returns=None)
    out = wf._maybe_steer(  # pyright: ignore[reportPrivateUsage]
        plan=_plan("a", "b"),
        completed=1,
        root_node_id=None,
        remaining_node_ids=(),
        repo=_repo(tmp_path),
    )
    assert out is None
    assert state["prompt_called"] == 0
    assert state["clear_called"] == 0


def test_maybe_steer_blank_prompt_clears_and_continues(tmp_path: Path) -> None:
    wf, state = _wf(tmp_path, steer_requested=True, steer_prompt_returns="   ")
    out = wf._maybe_steer(  # pyright: ignore[reportPrivateUsage]
        plan=_plan("a", "b"),
        completed=1,
        root_node_id=None,
        remaining_node_ids=(),
        repo=_repo(tmp_path),
    )
    assert out is None
    assert state["clear_called"] == 1


def test_maybe_steer_abort_returns_string(tmp_path: Path) -> None:
    wf, state = _wf(tmp_path, steer_requested=True, steer_prompt_returns="abort")
    out = wf._maybe_steer(  # pyright: ignore[reportPrivateUsage]
        plan=_plan("a", "b"),
        completed=1,
        root_node_id=None,
        remaining_node_ids=(),
        repo=_repo(tmp_path),
    )
    assert out == "abort"
    assert state["clear_called"] == 1


def test_maybe_steer_invokes_planner_revise_and_returns_new_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_planner_revise(
        provider: Any,
        *,
        previous_plan: Plan,
        user_feedback: str,
        repo: RepoSummary,
        steer_instruction: str = "",
    ) -> Plan:
        captured["previous"] = previous_plan
        captured["steer"] = steer_instruction
        captured["feedback"] = user_feedback
        return _plan("new-1", "new-2")

    monkeypatch.setattr(impl_module, "planner_revise", fake_planner_revise)
    wf, _state = _wf(tmp_path, steer_requested=True, steer_prompt_returns="focus on docs")
    out = wf._maybe_steer(  # pyright: ignore[reportPrivateUsage]
        plan=_plan("a", "b", "c"),
        completed=1,
        root_node_id=None,  # no graph client, so splice_graph returns ()
        remaining_node_ids=(),
        repo=_repo(tmp_path),
    )
    assert isinstance(out, tuple)
    new_plan, new_ids = out
    assert [s.title for s in new_plan.steps] == ["new-1", "new-2"]
    assert new_ids == ()  # no graph client wired
    # Only the *remaining* steps (b, c) should be in the previous_plan handed to revise.
    assert [s.title for s in captured["previous"].steps] == ["b", "c"]
    assert captured["steer"] == "focus on docs"


def test_maybe_steer_revise_failure_falls_back_to_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*_a: Any, **_kw: Any) -> Plan:
        raise RuntimeError("provider down")

    monkeypatch.setattr(impl_module, "planner_revise", boom)
    wf, _state = _wf(tmp_path, steer_requested=True, steer_prompt_returns="do something")
    out = wf._maybe_steer(  # pyright: ignore[reportPrivateUsage]
        plan=_plan("a", "b"),
        completed=1,
        root_node_id=None,
        remaining_node_ids=(),
        repo=_repo(tmp_path),
    )
    # Failure path: do not crash the run, just continue with the original plan.
    assert out is None


def test_splice_plan_preserves_head_and_replaces_tail(tmp_path: Path) -> None:
    wf, _state = _wf(tmp_path, steer_requested=False, steer_prompt_returns=None)
    spliced = wf._splice_plan(  # pyright: ignore[reportPrivateUsage]
        _plan("a", "b", "c"),
        head_count=1,
        new_tail=(Step(title="x", rationale="r", acceptance="a", relevant_paths=()),),
    )
    assert [s.title for s in spliced.steps] == ["a", "x"]
