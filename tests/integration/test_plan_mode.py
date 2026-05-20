# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the plan-mode workflow.

The Anthropic-facing sub-agents are monkeypatched so the loop logic itself
can be exercised deterministically without a network call.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.graph.client import GraphClient, spawn_curator
from agent6.graph.storage import RunLayout, load_graph
from agent6.models import OpenQuestion, Plan, RefinedSpec, Step
from agent6.types import RepoSummary
from agent6.workflows import plan_mode as plan_mode_module
from agent6.workflows.plan_mode import PlanModeError, PlanModeWorkflow, read_manifest


def _make_critic(refined_task: str, open_questions: tuple[OpenQuestion, ...] = ()) -> Any:
    def _critic(provider: Any, *, user_task: str, agents_md: str) -> RefinedSpec:
        return RefinedSpec(refined_task=refined_task, open_questions=open_questions)

    return _critic


def _make_critic_sequence(specs: list[RefinedSpec]) -> Any:
    """Critic that returns each spec in order, one per call."""

    it = iter(specs)

    def _critic(provider: Any, *, user_task: str, agents_md: str) -> RefinedSpec:
        return next(it)

    return _critic


def _make_planner(plan: Plan) -> Any:
    def _planner(provider: Any, *, refined_task: str, repo: RepoSummary) -> Plan:
        return plan

    return _planner


def _make_reviser(plan: Plan) -> Any:
    def _reviser(
        provider: Any, *, previous_plan: Plan, user_feedback: str, repo: RepoSummary
    ) -> Plan:
        return plan

    return _reviser


def _scripted_prompt(answers: list[str]) -> Any:
    it = iter(answers)

    def _p(_q: str) -> str:
        return next(it)

    return _p


def _silent_logger(_msg: str) -> None:
    return None


def _plan(summary: str, *titles: str) -> Plan:
    return Plan(
        summary=summary,
        steps=tuple(Step(title=t, rationale="", acceptance="", relevant_paths=()) for t in titles),
    )


def _repo(root: Path) -> RepoSummary:
    return RepoSummary(
        root=root,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
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


def test_plan_mode_accept_first_pass_persists_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic("refined: x"))
    monkeypatch.setattr(
        plan_mode_module, "planner_plan", _make_planner(_plan("the plan", "a", "b"))
    )
    revise = MagicMock()
    monkeypatch.setattr(plan_mode_module, "planner_revise", revise)

    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt(["accept"]),
        logger=_silent_logger,
    )
    result = wf.run("write a feature")

    assert result.final_plan.summary == "the plan"
    assert len(result.step_node_ids) == 2
    revise.assert_not_called()
    layout = RunLayout(root=tmp_path, run_id="r1")
    nodes = load_graph(layout)
    assert len(nodes) == 3
    root = next(n for n in nodes.values() if n.parent_id is None)
    assert root.title == "the plan"
    assert {n.title for n in nodes.values() if n.parent_id == root.id} == {"a", "b"}


def test_plan_mode_revision_then_accept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic("r"))
    monkeypatch.setattr(plan_mode_module, "planner_plan", _make_planner(_plan("v1", "one")))
    monkeypatch.setattr(
        plan_mode_module, "planner_revise", _make_reviser(_plan("v2", "one", "two"))
    )

    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt(["add a second step", "accept"]),
        logger=_silent_logger,
    )
    result = wf.run("a task")
    assert result.final_plan.summary == "v2"
    assert len(result.step_node_ids) == 2


def test_plan_mode_abort_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic("r"))
    monkeypatch.setattr(plan_mode_module, "planner_plan", _make_planner(_plan("v1", "one")))
    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt(["abort"]),
        logger=_silent_logger,
    )
    with pytest.raises(PlanModeError, match="aborted"):
        wf.run("a task")


def test_plan_mode_open_questions_qa_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    """Critic raises questions, user answers, critic clears, plan proceeds."""

    q = OpenQuestion(question="what about X?", suggestions=("use foo", "use bar"))
    specs = [
        RefinedSpec(refined_task="r1", open_questions=(q,)),
        RefinedSpec(refined_task="r2 with answers", open_questions=()),
    ]
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic_sequence(specs))
    monkeypatch.setattr(plan_mode_module, "planner_plan", _make_planner(_plan("p", "a")))
    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        # First answer "1" picks suggestion #1 ("use foo"); then accept.
        prompt=_scripted_prompt(["1", "accept"]),
        logger=_silent_logger,
    )
    result = wf.run("a task")
    assert result.refined.refined_task == "r2 with answers"
    # Manifest should record the augmented task with the embedded Q&A.
    layout = RunLayout(root=tmp_path, run_id="r1")
    m = read_manifest(layout)
    assert "Answers to clarifying questions" in m.task
    assert "use foo" in m.task


def test_plan_mode_open_questions_abort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    q = OpenQuestion(question="what about X?", suggestions=("a", "b"))
    monkeypatch.setattr(
        plan_mode_module,
        "critic_refine",
        _make_critic("r", (q,)),
    )
    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt(["abort"]),
        logger=_silent_logger,
    )
    with pytest.raises(PlanModeError, match="aborted"):
        wf.run("a task")


def test_plan_mode_writes_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic("refined: do x"))
    plan = _plan("the plan", "step a", "step b")
    monkeypatch.setattr(plan_mode_module, "planner_plan", _make_planner(plan))
    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt(["accept"]),
        logger=_silent_logger,
    )
    wf.run("do x please")
    layout = RunLayout(root=tmp_path, run_id="r1")
    m = read_manifest(layout)
    assert m.kind == "plan"
    assert m.task == "do x please"
    assert m.refined_task == "refined: do x"
    assert m.plan is not None
    assert m.plan.summary == "the plan"
    assert [s.title for s in m.plan.steps] == ["step a", "step b"]
    assert m.parent_run_id == ""


def test_plan_mode_run_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    """run_revision re-runs the critic on (refined+feedback) then revises."""

    revised = _plan("revised", "newstep")
    monkeypatch.setattr(plan_mode_module, "planner_revise", _make_reviser(revised))
    # Critic is now called in revision mode to surface any new open
    # questions raised by the user's feedback. Here it returns a clean
    # (no-questions) refined spec, so no Q&A loop runs.
    critic_mock = MagicMock(wraps=_make_critic("refined+feedback"))
    monkeypatch.setattr(plan_mode_module, "critic_refine", critic_mock)

    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        parent_run_id="r0",
        prompt=_scripted_prompt(["accept"]),
        logger=_silent_logger,
    )
    previous = _plan("old", "old-step")
    result = wf.run_revision(
        previous_plan=previous,
        previous_task="old task",
        previous_refined_task="old refined",
        initial_feedback="please change it",
    )
    assert result.final_plan.summary == "revised"
    assert critic_mock.call_count == 1
    # The critic should see both the prior refined task and the new feedback.
    seed = critic_mock.call_args.kwargs["user_task"]
    assert "old refined" in seed
    assert "please change it" in seed
    layout = RunLayout(root=tmp_path, run_id="r1")
    m = read_manifest(layout)
    assert m.parent_run_id == "r0"
    assert m.plan is not None
    assert m.plan.summary == "revised"


def test_plan_mode_run_revision_qa_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    """run_revision drives the same Q&A loop as run() if the critic asks."""

    q = OpenQuestion(question="really?", suggestions=("yes", "no"))
    specs = [
        RefinedSpec(refined_task="refined", open_questions=(q,)),
        RefinedSpec(refined_task="refined+answers", open_questions=()),
    ]
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic_sequence(specs))
    revised = _plan("revised", "newstep")
    monkeypatch.setattr(plan_mode_module, "planner_revise", _make_reviser(revised))

    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        parent_run_id="r0",
        prompt=_scripted_prompt(["1", "accept"]),
        logger=_silent_logger,
    )
    previous = _plan("old", "old-step")
    result = wf.run_revision(
        previous_plan=previous,
        previous_task="old task",
        previous_refined_task="old refined",
        initial_feedback="tweak step 1",
    )
    assert result.final_plan.summary == "revised"


def test_plan_mode_run_edit(tmp_path: Path, curator_client: GraphClient) -> None:
    """run_edit makes no LLM calls and persists the supplied plan."""

    edited = _plan("hand-edited", "x", "y", "z")
    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        parent_run_id="r0",
        prompt=_scripted_prompt([]),
        logger=_silent_logger,
    )
    result = wf.run_edit(
        edited_plan=edited,
        previous_task="t",
        previous_refined_task="rt",
    )
    assert result.final_plan == edited
    layout = RunLayout(root=tmp_path, run_id="r1")
    m = read_manifest(layout)
    assert m.parent_run_id == "r0"
    assert m.plan == edited


def test_plan_mode_writes_questions_file_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    """With --questions-file set and no answers, writes stub and raises."""

    q1 = OpenQuestion(question="how?", suggestions=("a", "b"))
    q2 = OpenQuestion(question="why?", suggestions=("c",))
    monkeypatch.setattr(
        plan_mode_module,
        "critic_refine",
        _make_critic("r", (q1, q2)),
    )
    monkeypatch.setattr(plan_mode_module, "planner_plan", _make_planner(_plan("p", "s")))

    qpath = tmp_path / "questions.json"
    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt([]),
        logger=_silent_logger,
        questions_file=qpath,
    )
    with pytest.raises(plan_mode_module.PlanModeQuestionsPending) as exc_info:
        wf.run("do x")
    assert exc_info.value.path == qpath
    assert qpath.is_file()
    payload = json.loads(qpath.read_text())
    assert payload["schema"] == "agent6/plan-questions/v1"
    assert payload["run_id"] == "r1"
    assert len(payload["questions"]) == 2
    assert payload["questions"][0]["question"] == "how?"
    assert payload["questions"][0]["suggestions"] == ["a", "b"]
    assert payload["questions"][0]["answer"] == ""


def test_plan_mode_consumes_answers_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, curator_client: GraphClient
) -> None:
    """Offline answers bypass interactive prompt and re-refine succeeds."""

    q = OpenQuestion(question="which?", suggestions=("x", "y"))
    specs = [
        RefinedSpec(refined_task="r1", open_questions=(q,)),
        RefinedSpec(refined_task="r2", open_questions=()),
    ]
    monkeypatch.setattr(plan_mode_module, "critic_refine", _make_critic_sequence(specs))
    monkeypatch.setattr(plan_mode_module, "planner_plan", _make_planner(_plan("p", "s")))

    wf = PlanModeWorkflow(
        root=tmp_path,
        repo=_repo(tmp_path),
        critic=MagicMock(),
        planner=MagicMock(),
        graph_client=curator_client,
        run_id="r1",
        prompt=_scripted_prompt(["accept"]),
        logger=_silent_logger,
        offline_answers=("my custom answer",),
    )
    result = wf.run("do x")
    assert result.refined.refined_task == "r2"
    assert result.final_plan.summary == "p"


def test_read_answers_file_round_trip(tmp_path: Path) -> None:
    from agent6.workflows.plan_mode import read_answers_file, write_questions_file

    qpath = tmp_path / "q.json"
    qs = (
        OpenQuestion(question="a?", suggestions=("alpha", "beta")),
        OpenQuestion(question="b?", suggestions=("gamma",)),
    )
    write_questions_file(qpath, qs, run_id="rid")
    # Empty answers fall back to first suggestion.
    answers = read_answers_file(qpath)
    assert answers == ("alpha", "gamma")
    # Explicit answers win.
    payload = json.loads(qpath.read_text())
    payload["questions"][0]["answer"] = "custom"
    qpath.write_text(json.dumps(payload))
    assert read_answers_file(qpath) == ("custom", "gamma")
