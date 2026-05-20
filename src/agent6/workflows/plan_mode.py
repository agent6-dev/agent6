# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Interactive plan-mode workflow.

`agent6 plan "<task>"` runs this workflow. It refines the task with the
critic, asks the planner for an initial Plan, then loops on stdin/stdout
letting the user accept the plan, abort, or send free-text feedback that the
planner-revise sub-agent incorporates into a new revision.

On accept the workflow persists the frozen plan into the task graph (one
root TaskNode whose title is the plan summary, plus one child TaskNode per
plan step, all created_by="planner") so a subsequent
`agent6 run --resume <run-id>` can execute it.

The plan-mode loop is intentionally line-oriented: it mirrors `git
rebase -i`. A real TUI is explicitly out of scope for v1.1.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent6.agents import critic_refine, planner_plan, planner_revise
from agent6.events import UserInputSink
from agent6.graph.client import GraphClient
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.graph.storage import RunLayout
from agent6.models import OpenQuestion, Plan, RefinedSpec, RunManifest
from agent6.providers import Provider
from agent6.types import RepoSummary


class PlanModeError(Exception):
    """User aborted plan mode or planner produced an unresolvable result."""


class ManifestError(Exception):
    """Run manifest is missing, malformed, or of an unexpected kind."""


class PlanModeQuestionsPending(Exception):
    """Critic surfaced open questions in offline mode; stub written to disk.

    Raised by ``_ask_open_questions`` when ``questions_file`` is set and
    no usable ``offline_answers`` are available. The CLI catches this and
    exits with a dedicated code so the user can fill in the answers file
    and re-run with ``--answers-file <same path>``.
    """

    def __init__(self, path: Path, questions: tuple[OpenQuestion, ...]) -> None:
        super().__init__(f"open questions written to {path}; fill in 'answer' fields and re-run")
        self.path = path
        self.questions = questions


@dataclass(frozen=True, slots=True)
class PlanModeResult:
    refined: RefinedSpec
    final_plan: Plan
    root_node_id: str
    step_node_ids: tuple[str, ...]


def write_manifest(layout: RunLayout, manifest: RunManifest) -> None:
    """Atomically write ``manifest.json`` for the given run layout."""
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump_json(indent=2) + "\n"
    tmp = layout.manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(layout.manifest_path)
    dfd = os.open(layout.run_dir, os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def read_manifest(layout: RunLayout) -> RunManifest:
    """Load ``manifest.json``. Raises ``ManifestError`` if missing or malformed."""
    path = layout.manifest_path
    if not path.is_file():
        raise ManifestError(f"no manifest.json under {layout.run_dir}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc
    try:
        return RunManifest.model_validate(raw)
    except Exception as exc:
        raise ManifestError(f"manifest.json failed schema validation: {exc}") from exc


def format_plan(plan: Plan) -> str:
    """Render a ``Plan`` as the same human-readable text the plan loop prints."""
    return _format_plan(plan)


_QUESTIONS_SCHEMA = "agent6/plan-questions/v1"


def write_questions_file(path: Path, questions: tuple[OpenQuestion, ...], *, run_id: str) -> None:
    """Write a JSON stub of open questions for offline answering.

    The stub mirrors ``OpenQuestion`` plus an empty ``answer`` field per
    question. The user fills in ``answer`` and re-runs with
    ``--answers-file <same path>``.
    """

    payload = {
        "schema": _QUESTIONS_SCHEMA,
        "run_id": run_id,
        "questions": [
            {
                "question": q.question,
                "suggestions": list(q.suggestions),
                "answer": "",
            }
            for q in questions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_answers_file(path: Path) -> tuple[str, ...]:
    """Read answers from a questions-file stub previously written.

    Each entry's ``answer`` field is used. Empty answers default to the
    first suggestion (matching the interactive ``blank-defaults-to-first``
    behaviour); a missing first suggestion on an empty answer is an error.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != _QUESTIONS_SCHEMA:
        raise PlanModeError(f"answers file {path} is not an agent6 questions-file v1")
    items = raw.get("questions", [])
    if not isinstance(items, list):
        raise PlanModeError(f"answers file {path}: 'questions' must be a list")
    out: list[str] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise PlanModeError(f"answers file {path}: question {i} is not an object")
        answer = item.get("answer", "")
        if not isinstance(answer, str):
            raise PlanModeError(f"answers file {path}: question {i} 'answer' must be a string")
        answer = answer.strip()
        if not answer:
            suggestions = item.get("suggestions", [])
            if isinstance(suggestions, list) and suggestions:
                first = suggestions[0]
                if isinstance(first, str):
                    answer = first
        if not answer:
            raise PlanModeError(f"answers file {path}: question {i} has no answer and no fallback")
        out.append(answer)
    return tuple(out)


def _format_plan(plan: Plan) -> str:
    lines = [f"Summary: {plan.summary}", "Steps:"]
    for i, s in enumerate(plan.steps, 1):
        lines.append(f"  {i}. {s.title}")
        if s.rationale:
            lines.append(f"     why: {s.rationale}")
        if s.relevant_paths:
            lines.append(f"     paths: {', '.join(s.relevant_paths)}")
        if s.acceptance:
            lines.append(f"     accept: {s.acceptance}")
    return "\n".join(lines)


@dataclass
class PlanModeWorkflow:
    """The interactive loop.

    The `prompt` callable returns a user response per round; in production
    that's just `input()`, but tests inject a deterministic script.
    """

    root: Path
    repo: RepoSummary
    critic: Provider
    planner: Provider
    graph_client: GraphClient
    run_id: str
    prompt: Callable[[str], str] = field(default=input)
    logger: Callable[[str], None] = field(default=print)
    parent_run_id: str = ""
    offline_answers: tuple[str, ...] = ()
    questions_file: Path | None = None
    user_inputs: UserInputSink | None = None
    user_task: str = field(default="", init=False, repr=False)

    def run(self, user_task: str) -> PlanModeResult:
        self.user_task = user_task
        refined, effective_task = self._refine_with_qa(user_task)
        self.user_task = effective_task

        self._log("INITIAL PLAN")
        plan = planner_plan(self.planner, refined_task=refined.refined_task, repo=self.repo)
        self._log(_format_plan(plan))

        plan = self._accept_loop(plan)
        return self._persist(refined, plan)

    def run_revision(
        self,
        *,
        previous_plan: Plan,
        previous_task: str,
        previous_refined_task: str,
        initial_feedback: str,
    ) -> PlanModeResult:
        """Apply ``initial_feedback`` to ``previous_plan`` then accept-loop.

        Re-runs the critic on the combined (previous refined task + user
        feedback) seed, drives the same interactive Q&A loop ``run()`` uses
        for any open questions it surfaces, then asks the planner-revise
        sub-agent to produce a revised plan, and finally enters the same
        accept/abort/feedback loop. Used by ``agent6 plan revise``.
        """

        self.user_task = previous_task
        base = previous_refined_task or previous_task
        seed = f"{base}\n\nUser feedback for revision:\n{initial_feedback}"
        refined, effective_task = self._refine_with_qa(seed)
        self.user_task = effective_task
        self._log("REVISE")
        plan = planner_revise(
            self.planner,
            previous_plan=previous_plan,
            user_feedback=effective_task,
            repo=self.repo,
        )
        self._log(_format_plan(plan))
        plan = self._accept_loop(plan)
        return self._persist(refined, plan)

    def run_edit(
        self,
        *,
        edited_plan: Plan,
        previous_task: str,
        previous_refined_task: str,
    ) -> PlanModeResult:
        """Persist a user-edited Plan as a new run, no LLM calls."""

        self.user_task = previous_task
        refined = RefinedSpec(refined_task=previous_refined_task or previous_task)
        self._log("EDIT")
        self._log(_format_plan(edited_plan))
        return self._persist(refined, edited_plan)

    def _accept_loop(self, plan: Plan) -> Plan:
        revision = 0
        accept_prompt = "\n[plan] accept / abort / <feedback>: "
        while True:
            try:
                answer = self.prompt(accept_prompt).strip()
            except EOFError as exc:
                raise PlanModeError("plan-mode aborted (EOF)") from exc
            if not answer:
                continue
            self._audit(
                kind="plan_decision",
                prompt=accept_prompt.strip(),
                answer=answer,
                revision=revision,
            )
            low = answer.lower()
            if low in {"accept", "a", "y", "yes"}:
                return plan
            if low in {"abort", "q", "n", "no"}:
                raise PlanModeError("plan-mode aborted by user")
            revision += 1
            self._log(f"REVISION {revision}")
            plan = planner_revise(
                self.planner,
                previous_plan=plan,
                user_feedback=answer,
                repo=self.repo,
            )
            self._log(_format_plan(plan))

    # ---- internals ------------------------------------------------------

    def _refine_with_qa(self, seed_task: str) -> tuple[RefinedSpec, str]:
        """Refine ``seed_task`` via the critic, looping on open questions.

        Returns (refined, effective_task) where ``effective_task`` is the
        seed augmented with every Q&A round's answers. Raises on user
        abort or after 5 unproductive rounds.
        """

        self._log("REFINE")
        refined = critic_refine(self.critic, user_task=seed_task, agents_md=self.repo.agents_md)
        effective_task = seed_task
        rounds = 0
        while refined.open_questions:
            rounds += 1
            if rounds > 5:
                raise PlanModeError("critic keeps raising open questions after 5 rounds; aborting")
            answers = self._ask_open_questions(refined.open_questions)
            qa_lines = ["", "Answers to clarifying questions:"]
            for q, a in zip(refined.open_questions, answers, strict=True):
                qa_lines.append(f"- Q: {q.question}")
                qa_lines.append(f"  A: {a}")
            effective_task = effective_task + "\n" + "\n".join(qa_lines)
            self._log(f"REFINE (round {rounds + 1})")
            refined = critic_refine(
                self.critic, user_task=effective_task, agents_md=self.repo.agents_md
            )
        return refined, effective_task

    def _log(self, msg: str) -> None:
        self.logger(f"[agent6 plan] {msg}")

    def _audit(self, **fields: object) -> None:
        """Append a row to the per-run user-input audit log, if attached."""
        if self.user_inputs is None:
            return
        # Strict kwargs are enforced by UserInputSink.record; this wrapper
        # only exists to make the None-check ergonomic.
        kind = str(fields.pop("kind"))
        prompt = str(fields.pop("prompt"))
        answer = str(fields.pop("answer"))
        source = str(fields.pop("source", "stdin"))
        self.user_inputs.record(kind=kind, prompt=prompt, answer=answer, source=source, **fields)

    def _ask_open_questions(self, questions: tuple[OpenQuestion, ...]) -> list[str]:
        """Collect one answer per open question (offline or interactive).

        Three modes:

        * ``offline_answers`` is set and matches the question count:
          consume them in order and return. The field is then cleared so
          a subsequent round falls into the next branch.
        * ``questions_file`` is set and no usable offline answers: write
          a JSON stub of the current questions to that path and raise
          ``PlanModeQuestionsPending``. The CLI translates this to exit 4.
        * Otherwise: interactive prompt (numbered suggestions, free text,
          ``abort``, blank-defaults-to-first).
        """

        total = len(questions)
        if self.offline_answers:
            if len(self.offline_answers) != total:
                msg = (
                    f"answers file supplies {len(self.offline_answers)} answers but the critic "
                    f"raised {total} questions; cannot match"
                )
                if self.questions_file is not None:
                    write_questions_file(self.questions_file, questions, run_id=self.run_id)
                    raise PlanModeQuestionsPending(self.questions_file, questions)
                raise PlanModeError(msg)
            answers = list(self.offline_answers)
            self.offline_answers = ()
            self._log(f"OPEN QUESTIONS ({total}) — using answers from file")
            for q, a in zip(questions, answers, strict=True):
                self._audit(
                    kind="plan_qa_answer",
                    prompt=q.question,
                    answer=a,
                    source="answers_file",
                )
            return answers
        if self.questions_file is not None:
            write_questions_file(self.questions_file, questions, run_id=self.run_id)
            raise PlanModeQuestionsPending(self.questions_file, questions)
        answers: list[str] = []
        self._log(f"OPEN QUESTIONS ({total})")
        for idx, q in enumerate(questions, 1):
            answers.append(self._ask_one_question(idx, total, q))
        return answers

    def _ask_one_question(self, idx: int, total: int, q: OpenQuestion) -> str:
        self.logger("")
        self.logger(f"[Q {idx}/{total}] {q.question}")
        for i, s in enumerate(q.suggestions, 1):
            self.logger(f"  {i}. {s}")
        prompt = "> pick a number, type your own answer, or 'abort': "
        try:
            raw = self.prompt(prompt).strip()
        except EOFError as exc:
            raise PlanModeError("plan-mode aborted (EOF)") from exc
        if raw.lower() in {"abort", "q"}:
            raise PlanModeError("plan-mode aborted by user")
        chosen: str
        if raw.isdigit() and 1 <= int(raw) <= len(q.suggestions):
            chosen = q.suggestions[int(raw) - 1]
        elif not raw:
            if not q.suggestions:
                raise PlanModeError(f"no answer given for question {idx}")
            chosen = q.suggestions[0]
        else:
            chosen = raw
        self._audit(
            kind="plan_qa_answer",
            prompt=q.question,
            answer=chosen,
            question_index=idx,
            raw_input=raw,
        )
        return chosen

    def _persist(self, refined: RefinedSpec, plan: Plan) -> PlanModeResult:
        root = self.graph_client.add_subtask(
            AddSubtaskIntent(
                parent_id=None,
                draft=TaskNodeDraft(
                    title=plan.summary,
                    rationale=refined.refined_task,
                    created_by="planner",
                ),
            )
        )
        step_ids: list[str] = []
        for step in plan.steps:
            child = self.graph_client.add_subtask(
                AddSubtaskIntent(
                    parent_id=root.id,
                    draft=TaskNodeDraft(
                        title=step.title,
                        rationale=step.rationale,
                        acceptance=step.acceptance,
                        relevant_paths=step.relevant_paths,
                        created_by="planner",
                    ),
                )
            )
            step_ids.append(child.id)
        self._write_manifest(refined, plan)
        return PlanModeResult(
            refined=refined,
            final_plan=plan,
            root_node_id=root.id,
            step_node_ids=tuple(step_ids),
        )

    def _write_manifest(self, refined: RefinedSpec, plan: Plan) -> None:
        layout = RunLayout(root=self.root, run_id=self.run_id)
        manifest = RunManifest(
            run_id=self.run_id,
            kind="plan",
            created_at=datetime.now(tz=UTC).isoformat(),
            task=self.user_task,
            refined_task=refined.refined_task,
            plan=plan,
            parent_run_id=self.parent_run_id,
        )
        write_manifest(layout, manifest)
