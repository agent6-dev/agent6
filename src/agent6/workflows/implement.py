# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `implement` workflow — Python state machine that drives the sub-agents.

The control flow is in Python; the LLM is a typed component, not the orchestrator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from agent6.agents import (
    alignment_check,
    critic_refine,
    planner_plan,
    reviewer_review,
    worker_edit,
)
from agent6.agents.planner_revise import planner_revise
from agent6.config import Config
from agent6.events import EventSink, UserInputSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    commit_all,
    create_branch,
    diff_since,
    recent_log,
    reset_to,
    show_commit,
    slugify,
    stash_all,
    status,
)
from agent6.graph.client import GraphClient
from agent6.graph.curator import hash_uncommitted
from agent6.graph.models import (
    AddSubtaskIntent,
    ObsoleteIntent,
    RecordCommitIntent,
    SetCursorIntent,
    SnapshotNodeIntent,
    TaskNode,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.models import AlignmentAction, AlignmentVerdict, Edit, FileEdit, Plan, RefinedSpec, Step
from agent6.providers import Provider
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import CommandResult, FileContext, RepoSummary


class WorkflowError(Exception):
    """A workflow encountered an unrecoverable error."""


@dataclass(frozen=True, slots=True)
class StepResult:
    title: str
    status: str  # "passed" | "failed" | "skipped" | "obsolete"
    commit_sha: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    branch: str
    plan: Plan
    steps: tuple[StepResult, ...]
    refined: RefinedSpec

    @property
    def all_passed(self) -> bool:
        return all(s.status == "passed" for s in self.steps)


@dataclass
class ImplementWorkflow:
    """State machine driving the implement workflow.

    LOAD_CONTEXT → REFINE → PLAN → CONFIRM → for each step (EDIT, VERIFY, REVIEW, COMMIT).
    """

    root: Path
    config: Config
    planner: Provider
    worker: Provider
    reviewer: Provider
    critic: Provider
    dispatcher: ToolDispatcher
    confirm_plan: Callable[[Plan], bool] = field(default=lambda _p: True)
    logger: Callable[[str], None] = field(default=print)
    graph_client: GraphClient | None = None
    alignment_guard: Provider | None = None
    alignment_period: int = 5  # run drift check every N completed steps; 0 disables periodic.
    events: EventSink | None = None
    user_inputs: UserInputSink | None = None
    # ---- steering -----------------------------------------------------
    # Returns True iff the operator has asked to steer the run (e.g. via SIGINT
    # or the TUI). Polled at safe boundaries between steps. Default = never.
    steer_requested: Callable[[], bool] = field(default=lambda: False)
    # Reset the request flag (called after a successful prompt drain).
    steer_clear: Callable[[], None] = field(default=lambda: None)
    # Acquire steer text. Return ``None`` to mean "user backed out, just resume".
    # Return ``""`` (or whitespace) to mean the same. Return ``"abort"`` to halt
    # the run. Any other string is the steering instruction.
    steer_prompt: Callable[[], str | None] = field(default=lambda: None)

    def run(self, user_task: str) -> WorkflowResult:  # noqa: PLR0915, PLR0912
        self._emit("run.start", user_task=user_task)
        self._log("LOAD_CONTEXT")
        repo = self._load_context()

        self._log("PRE_FLIGHT_GIT")
        self._git_pre_flight()
        branch = self._make_branch_name(user_task)
        if self.config.git.branch_per_run:
            create_branch(self.root, branch)
            self._log(f"  branch created: {branch}")
        else:
            branch = repo.branch

        self._log("REFINE_SPEC")
        refined = critic_refine(self.critic, user_task=user_task, agents_md=repo.agents_md)
        if refined.open_questions:
            joined = "\n  - ".join(q.question for q in refined.open_questions)
            raise WorkflowError(
                "Open questions from critic — run `agent6 plan new` first to "
                f"answer them, then `agent6 run` again:\n  - {joined}"
            )

        self._log("PLAN")
        plan = planner_plan(self.planner, refined_task=refined.refined_task, repo=repo)
        self._log(f"  plan: {plan.summary}")
        for i, step in enumerate(plan.steps, 1):
            self._log(f"  {i}. {step.title}")
        self._emit(
            "plan.ready",
            summary=plan.summary,
            step_count=len(plan.steps),
            steps=[s.title for s in plan.steps],
        )

        if not self.confirm_plan(plan):
            raise WorkflowError("Plan rejected by user")

        # Mirror the plan into the task graph so the curator can drive resume.
        step_node_ids: tuple[str, ...] = ()
        root_node_id: str | None = None
        if self.graph_client is not None:
            root_node_id, step_node_ids = self._seed_graph_from_plan(plan)

        original_task = refined.refined_task
        results: list[StepResult] = []
        run_start_sha = repo.head_sha
        start_sha = run_start_sha
        passed_count = 0
        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            self._log(f"STEP {i + 1}/{len(plan.steps)}: {step.title}")
            self._emit(
                "step.start",
                index=i + 1,
                total=len(plan.steps),
                title=step.title,
                relevant_paths=list(step.relevant_paths),
            )
            node_id = step_node_ids[i] if step_node_ids else None

            # Pre-execute alignment check.
            guard_outcome = self._alignment_pre_execute(
                node_id=node_id,
                step=step,
                root_node_id=root_node_id,
                original_task=original_task,
            )
            if guard_outcome is not None:
                results.append(guard_outcome)
                if guard_outcome.status == "failed":
                    break
                i += 1
                continue  # obsolete / skipped — keep going with next step

            sibling_commits = tuple(
                (r.commit_sha, r.title) for r in results if r.status == "passed" and r.commit_sha
            )
            result = self._run_step(
                step,
                start_sha=start_sha,
                node_id=node_id,
                branch=branch,
                agents_md=repo.agents_md,
                parent_acceptance=plan.summary,
                sibling_commits=sibling_commits,
            )
            results.append(result)
            if result.commit_sha and result.commit_sha != start_sha:
                self._emit(
                    "step.diff",
                    index=i + 1,
                    title=step.title,
                    commit_sha=result.commit_sha,
                    patch=show_commit(self.root, result.commit_sha),
                )
            self._emit(
                "step.end",
                index=i + 1,
                title=step.title,
                status=result.status,
                commit_sha=result.commit_sha,
                notes=result.notes,
            )
            if result.status != "passed":
                self._log(f"  step failed: {result.notes}")
                break
            passed_count += 1
            if result.commit_sha:
                start_sha = result.commit_sha

            # Periodic drift check.
            drift = self._alignment_periodic(
                passed_count=passed_count,
                node_id=node_id,
                step=step,
                root_node_id=root_node_id,
                original_task=original_task,
            )
            if drift is not None:
                results.append(drift)
                break

            # Steering boundary: check between steps, after any commit. If the
            # operator typed a steering instruction, replan the remaining tail
            # and splice the new steps into the graph + the in-memory plan.
            steer_action = self._maybe_steer(
                plan=plan,
                completed=i + 1,
                root_node_id=root_node_id,
                remaining_node_ids=step_node_ids[i + 1 :] if step_node_ids else (),
                repo=repo,
            )
            if steer_action is not None:
                if isinstance(steer_action, str):
                    # "abort"
                    results.append(
                        StepResult(
                            title="(steer abort)",
                            status="failed",
                            notes="run aborted by operator via steering prompt",
                        )
                    )
                    break
                new_plan, new_tail_ids = steer_action
                plan = self._splice_plan(plan, head_count=i + 1, new_tail=new_plan.steps)
                step_node_ids = step_node_ids[: i + 1] + new_tail_ids
            i += 1

        self._log("DONE")
        wr = WorkflowResult(branch=branch, plan=plan, steps=tuple(results), refined=refined)
        final_sha = self._finalize_commit_strategy(
            wr, root_node_id=root_node_id, run_start_sha=run_start_sha
        )
        self._emit(
            "run.end",
            all_passed=wr.all_passed,
            step_count=len(wr.steps),
            passed=sum(1 for s in wr.steps if s.status == "passed"),
            final_commit_sha=final_sha,
        )
        return wr

    # ---------- internals ----------

    def _log(self, msg: str) -> None:
        self.logger(f"[agent6] {msg}")

    def _emit(self, event_type: str, /, **fields: object) -> None:
        if self.events is not None:
            self.events.emit(event_type, **fields)

    def _load_context(self) -> RepoSummary:
        st = status(self.root)
        top = tuple(
            sorted(
                p.name + ("/" if p.is_dir() else "")
                for p in self.root.iterdir()
                if not p.name.startswith(".")
            )
        )
        file_count = sum(1 for p in self.root.rglob("*") if p.is_file())
        agents_md_path = self.root / "AGENTS.md"
        agents_md = agents_md_path.read_text(encoding="utf-8") if agents_md_path.is_file() else ""
        return RepoSummary(
            root=self.root,
            branch=st.branch,
            head_sha=st.head_sha,
            file_count=file_count,
            top_level=top,
            agents_md=agents_md,
            recent_log=recent_log(self.root, n=20),
        )

    def _git_pre_flight(self) -> None:
        st = status(self.root)
        if not st.is_clean:
            if self.config.git.auto_stash:
                stash_all(
                    self.root,
                    f"agent6 pre-run {datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%S')}",
                )
                self._log("  auto-stashed dirty worktree")
            elif self.config.git.require_clean_worktree:
                raise WorkflowError(
                    "Worktree is dirty and git.auto_stash is false. Commit or stash first."
                )

    def _make_branch_name(self, user_task: str) -> str:
        ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        return f"agent6/{ts}-{slugify(user_task)}"

    def _commit_identity(self) -> CommitIdentity:
        c = self.config.git.commit
        return CommitIdentity(name=c.name, email=c.email, coauthor=c.coauthor)

    def _gather_files(self, step: Step) -> FileContext:
        items: list[tuple[Path, str]] = []
        for rel in step.relevant_paths:
            abs_path = (self.root / rel).resolve()
            try:
                abs_path.relative_to(self.root.resolve())
            except ValueError:
                continue
            if abs_path.is_file():
                try:
                    items.append((Path(rel), abs_path.read_text(encoding="utf-8")))
                except (OSError, UnicodeDecodeError):
                    items.append((Path(rel), "<binary or unreadable>"))
            elif abs_path.is_dir():
                for child in sorted(abs_path.rglob("*")):
                    if not child.is_file():
                        continue
                    try:
                        text = child.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    items.append((child.relative_to(self.root), text))
                    if len(items) >= 30:
                        break
        return FileContext(files=tuple(items))

    def _apply_edit_object(self, edit: Edit) -> None:
        for fe in edit.edits:
            self._apply_one_edit(fe)

    def _apply_one_edit(self, fe: FileEdit) -> None:
        self.dispatcher.dispatch(
            "apply_edit",
            {
                "path": fe.path,
                "edits": [
                    {
                        "kind": fe.kind,
                        "old_string": fe.old_string,
                        "new_string": fe.new_string,
                    }
                ],
            },
        )

    def _run_verify(self) -> CommandResult:
        out = self.dispatcher.dispatch("run_verify_command", {})
        return CommandResult(
            argv=tuple(self.config.workflow.verify_command),
            returncode=int(out["returncode"]),
            stdout=str(out["stdout"]),
            stderr=str(out["stderr"]),
            duration_s=float(out.get("duration_s", 0.0)),
        )

    def _run_step(
        self,
        step: Step,
        *,
        start_sha: str,
        node_id: str | None = None,
        branch: str = "",
        agents_md: str = "",
        parent_acceptance: str = "",
        sibling_commits: tuple[tuple[str, str], ...] = (),
    ) -> StepResult:
        if self.graph_client is not None and node_id is not None:
            self.graph_client.update_status(
                UpdateStatusIntent(id=node_id, new_status="in_progress")
            )
            self.graph_client.set_cursor(SetCursorIntent(id=node_id))
            touched = hash_uncommitted(self.root, step.relevant_paths)
            self.graph_client.snapshot_node(
                SnapshotNodeIntent(
                    id=node_id,
                    head_sha=start_sha,
                    branch=branch,
                    uncommitted_touched=touched,
                )
            )

        attempt_feedback = ""
        for attempt in range(2):
            # Re-gather file context EVERY attempt: previous attempts may have
            # partially modified files (some edits applied, then one failed),
            # so the worker must always see current on-disk content or its
            # next `old_string` will not match.
            ctx = self._gather_files(step)
            try:
                edit = worker_edit(
                    self.worker,
                    step=step,
                    file_context=ctx,
                    previous_attempt_feedback=attempt_feedback,
                    agents_md=agents_md,
                    parent_acceptance=parent_acceptance,
                    sibling_commits=sibling_commits,
                )
            except Exception as exc:
                self._mark_node_failed(node_id, f"worker error: {exc}")
                return StepResult(title=step.title, status="failed", notes=f"worker error: {exc}")

            try:
                self._apply_edit_object(edit)
            except Exception as exc:
                attempt_feedback = f"apply failed: {exc}"
                continue

            verify = self._run_verify()
            diff = diff_since(self.root, start_sha)
            # If the worker produced no changes and verify still passes, the
            # step's goal has already been satisfied by an earlier step (the
            # planner's decomposition turned out to be over-eager). Don't fail
            # the workflow just because a planned step is now vacuous.
            if verify.ok and not diff.strip():
                self._mark_node_passed(node_id, start_sha)
                return StepResult(
                    title=step.title,
                    status="passed",
                    commit_sha=start_sha,
                    notes="no-op: step already satisfied by prior work",
                )
            try:
                review = reviewer_review(
                    self.reviewer,
                    step=step,
                    diff=diff,
                    verify_output=(verify.stdout + verify.stderr),
                    verify_ok=verify.ok,
                    agents_md=agents_md,
                )
            except Exception as exc:
                review = None
                review_failed_reason = f"reviewer error: {exc}"
            else:
                review_failed_reason = ""

            if verify.ok and (review is None or review.verdict == "pass"):
                trailers = {
                    "agent6-step": step.title,
                    "agent6-attempt": str(attempt + 1),
                }
                # Every strategy commits per step — that's what gives clean
                # per-step reviewer diffs and crash-resistant resume. The
                # configured ``commit_strategy`` only governs end-of-run
                # finalization (squash / stage / none rewind the branch tip).
                try:
                    sha = commit_all(
                        self.root,
                        f"agent6: {step.title}",
                        trailers=trailers,
                        identity=self._commit_identity(),
                    )
                except GitError as exc:
                    self._mark_node_failed(node_id, f"commit failed: {exc}")
                    return StepResult(
                        title=step.title, status="failed", notes=f"commit failed: {exc}"
                    )
                self._mark_node_passed(node_id, sha)
                return StepResult(title=step.title, status="passed", commit_sha=sha)

            attempt_feedback = json.dumps(
                {
                    "verify_returncode": verify.returncode,
                    "verify_tail": (verify.stdout + verify.stderr)[-2000:],
                    "review": review.comments if review is not None else review_failed_reason,
                    "proposed_followup": (review.proposed_followup if review is not None else ""),
                }
            )

        msg = f"step failed after retries: {attempt_feedback[:400]}"
        self._mark_node_failed(node_id, msg)
        return StepResult(
            title=step.title,
            status="failed",
            notes=msg,
        )

    # ---- graph helpers ---------------------------------------------------

    def _seed_graph_from_plan(self, plan: Plan) -> tuple[str, tuple[str, ...]]:
        assert self.graph_client is not None
        root_node = self.graph_client.add_subtask(
            AddSubtaskIntent(
                parent_id=None,
                draft=TaskNodeDraft(
                    title=plan.summary or "implement",
                    rationale="root node for plan",
                    created_by="planner",
                ),
            )
        )
        ids: list[str] = []
        for step in plan.steps:
            child = self.graph_client.add_subtask(
                AddSubtaskIntent(
                    parent_id=root_node.id,
                    draft=TaskNodeDraft(
                        title=step.title,
                        rationale=step.rationale,
                        acceptance=step.acceptance,
                        relevant_paths=step.relevant_paths,
                        created_by="planner",
                    ),
                )
            )
            ids.append(child.id)
        return root_node.id, tuple(ids)

    def _mark_node_passed(self, node_id: str | None, sha: str) -> None:
        if self.graph_client is None or node_id is None:
            return
        self.graph_client.record_commit(RecordCommitIntent(id=node_id, sha=sha))
        self.graph_client.update_status(UpdateStatusIntent(id=node_id, new_status="passed"))

    def _mark_node_failed(self, node_id: str | None, note: str) -> None:
        if self.graph_client is None or node_id is None:
            return
        self.graph_client.update_status(
            UpdateStatusIntent(id=node_id, new_status="failed", note=note[:400])
        )

    def _finalize_commit_strategy(
        self,
        wr: WorkflowResult,
        *,
        root_node_id: str | None,
        run_start_sha: str,
    ) -> str:
        """Apply the end-of-run side of ``git.commit_strategy``.

        Every step has already committed (see ``_run_step``); this method
        rewinds the branch tip back to ``run_start_sha`` and then optionally
        produces a single combined commit. The per-step commits remain
        reachable via reflog for recovery.

        * ``per_step`` — nothing to do; leave N commits as-is.
        * ``squash``   — soft reset to ``run_start_sha``, then one commit
          whose body lists every passed step.
        * ``stage``    — soft reset only; all changes left staged for the
          operator to inspect and commit.
        * ``none``     — mixed reset; changes land in the worktree only,
          nothing staged. Operator reviews ``git diff`` and commits.

        Returns the final commit sha (or ``""`` for ``stage`` / ``none`` /
        when nothing passed).
        """
        strategy = self.config.git.commit_strategy
        if strategy == "per_step":
            return ""
        passed = [s for s in wr.steps if s.status == "passed"]
        if not passed:
            # No per-step commits were made — nothing to rewind, nothing to
            # combine. Leave the branch tip where it started.
            return ""
        if strategy == "squash":
            try:
                reset_to(self.root, run_start_sha, mode="soft")
            except GitError as exc:
                self._log(f"  final reset failed: {exc}")
                return ""
            body_lines = [f"- {s.title}" for s in passed]
            message = f"agent6: {wr.plan.summary or 'implement'}\n\n" + "\n".join(body_lines)
            trailers = {f"agent6-step-{i + 1}": s.title for i, s in enumerate(passed)}
            try:
                sha = commit_all(
                    self.root,
                    message,
                    trailers=trailers,
                    identity=self._commit_identity(),
                )
            except GitError as exc:
                self._log(f"  final commit failed: {exc}")
                return ""
            if self.graph_client is not None and root_node_id is not None:
                self.graph_client.record_commit(RecordCommitIntent(id=root_node_id, sha=sha))
            return sha
        # "stage" or "none": rewind the branch tip and stop. The mode
        # difference controls whether the worktree changes remain staged.
        mode = "soft" if strategy == "stage" else "mixed"
        try:
            reset_to(self.root, run_start_sha, mode=mode)
        except GitError as exc:
            self._log(f"  final reset failed: {exc}")
        return ""

    # ---- alignment guard -------------------------------------------------

    def _alignment_pre_execute(
        self,
        *,
        node_id: str | None,
        step: Step,
        root_node_id: str | None,
        original_task: str,
    ) -> StepResult | None:
        """Run the alignment guard before executing a step.

        Returns None to mean "proceed". Returns a StepResult to record into the
        result list when the guard rejects, escalates, or asks. The caller
        continues with the next step iff the returned status is "obsolete"
        or "skipped"; otherwise the workflow halts.
        """
        if self.alignment_guard is None:
            return None
        node = self._snapshot_step_as_node(node_id=node_id, step=step)
        parent_path = self._lookup_parent_path(root_node_id)
        verdict = self._call_guard(
            node=node,
            parent_path=parent_path,
            original_task=original_task,
            proposed_action="execute",
        )
        return self._apply_verdict(verdict, node_id=node_id, step=step)

    def _alignment_periodic(
        self,
        *,
        passed_count: int,
        node_id: str | None,
        step: Step,
        root_node_id: str | None,
        original_task: str,
    ) -> StepResult | None:
        """Periodic drift check after every N completed steps."""
        if self.alignment_guard is None or self.alignment_period <= 0:
            return None
        if passed_count == 0 or passed_count % self.alignment_period != 0:
            return None
        node = self._snapshot_step_as_node(node_id=node_id, step=step)
        parent_path = self._lookup_parent_path(root_node_id)
        verdict = self._call_guard(
            node=node,
            parent_path=parent_path,
            original_task=original_task,
            proposed_action="execute",
        )
        # On periodic check, only escalations are interesting (proceed = silent).
        if verdict.verdict == "proceed":
            return None
        return self._apply_verdict(verdict, node_id=node_id, step=step)

    def _call_guard(
        self,
        *,
        node: TaskNode,
        parent_path: tuple[TaskNode, ...],
        original_task: str,
        proposed_action: str,
    ) -> AlignmentVerdict:
        assert self.alignment_guard is not None
        return alignment_check(
            self.alignment_guard,
            node=node,
            parent_path=parent_path,
            original_task=original_task,
            proposed_action=cast(AlignmentAction, proposed_action),
        )

    def _apply_verdict(
        self, verdict: AlignmentVerdict, *, node_id: str | None, step: Step
    ) -> StepResult | None:
        v = verdict.verdict
        self._log(f"  alignment guard: {v} — {verdict.reasoning}")
        if v == "proceed":
            return None
        if v == "reject":
            if self.graph_client is not None and node_id is not None:
                self.graph_client.obsolete(ObsoleteIntent(id=node_id, reason=verdict.reasoning))
            return StepResult(
                title=step.title,
                status="obsolete",
                notes=f"alignment guard rejected: {verdict.reasoning}",
            )
        if v == "reorder":
            # v1.1: log + proceed. The plan is single-pass; full reorder
            # support hooks into the curator and a re-entrant loop, deferred.
            self._log(
                f"  reorder suggested ({list(verdict.suggested_reorder)}); proceeding for v1.1"
            )
            return None
        # re-plan-subtree, re-plan-root, ask  → halt with explanatory failure.
        self._mark_node_failed(node_id, f"alignment escalation {v}: {verdict.reasoning}")
        return StepResult(
            title=step.title,
            status="failed",
            notes=f"alignment escalation {v}: {verdict.reasoning}",
        )

    def _snapshot_step_as_node(self, *, node_id: str | None, step: Step) -> TaskNode:
        """Return a TaskNode view of the current step.

        Prefers the live graph node (so status/created_by reflect reality);
        falls back to a synthesized node when no graph client is configured.
        """
        if self.graph_client is not None and node_id is not None:
            state = self.graph_client.get_state()
            nodes = state.get("nodes", {})
            raw = nodes.get(node_id)
            if raw is not None:
                return TaskNode.model_validate(raw)
        return TaskNode(
            id=node_id or "0" * 26,
            parent_id=None,
            depends_on=(),
            children=(),
            title=step.title,
            rationale=step.rationale,
            acceptance=step.acceptance,
            relevant_paths=step.relevant_paths,
            status="in_progress",
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
            created_by="planner",
            commit_sha="",
            notes="",
        )

    def _lookup_parent_path(self, root_node_id: str | None) -> tuple[TaskNode, ...]:
        if self.graph_client is None or root_node_id is None:
            return ()
        state = self.graph_client.get_state()
        nodes = state.get("nodes", {})
        raw = nodes.get(root_node_id)
        if raw is None:
            return ()
        return (TaskNode.model_validate(raw),)

    # ---- steering ---------------------------------------------------------

    def _maybe_steer(
        self,
        *,
        plan: Plan,
        completed: int,
        root_node_id: str | None,
        remaining_node_ids: tuple[str, ...],
        repo: RepoSummary,
    ) -> tuple[Plan, tuple[str, ...]] | str | None:
        """Check the steer flag and, if set, drain a steering instruction.

        Returns:
        * ``None``        — no steering requested (continue as-is)
        * ``"abort"``     — operator asked to abort the run
        * ``(plan, ids)`` — replanned remaining tail + new graph node ids

        The boundary is *between* committed steps. Steering cannot preempt an
        in-flight sub-agent call; the flag is only inspected here.
        """
        if not self.steer_requested():
            return None
        self._emit("run.steer_requested", source="signal_or_tui", completed=completed)
        self._log("STEER: operator requested mid-run steering")
        try:
            text = self.steer_prompt()
        finally:
            self.steer_clear()
        prompt_text = "steer (blank=continue, 'abort'=stop, else=instruction): "
        if text is None or not text.strip():
            self._log("  (empty — continuing)")
            if self.user_inputs is not None:
                self.user_inputs.record(
                    kind="steer_input", prompt=prompt_text, answer="", source="signal"
                )
            return None
        steer_text = text.strip()
        if self.user_inputs is not None:
            self.user_inputs.record(
                kind="steer_input", prompt=prompt_text, answer=steer_text, source="signal"
            )
        if steer_text.lower() == "abort":
            self._emit("run.steer_aborted", text=steer_text)
            return "abort"
        # Replan the remaining tail. We hand planner_revise a *sub-plan* that
        # contains only the pending steps; the LLM returns the new tail.
        remaining = plan.steps[completed:]
        previous_tail = Plan(summary=plan.summary, steps=remaining)
        try:
            revised = planner_revise(
                self.planner,
                previous_plan=previous_tail,
                user_feedback=(
                    "The operator is steering the run mid-flight. Revise the "
                    "REMAINING steps only (already-committed steps are immutable)."
                ),
                repo=repo,
                steer_instruction=steer_text,
            )
        except Exception as exc:
            # If the revise call fails, fall back to continuing as planned;
            # do not crash the run.
            self._log(f"  steer revise failed: {exc} — continuing with original plan")
            self._emit("run.steer_failed", error=str(exc)[:400])
            return None
        new_tail_ids = self._splice_graph_for_steer(
            root_node_id=root_node_id,
            obsolete_node_ids=remaining_node_ids,
            new_steps=revised.steps,
            steer_text=steer_text,
        )
        self._emit(
            "run.steered",
            steer_text=steer_text,
            new_step_count=len(revised.steps),
            obsoleted_step_count=len(remaining),
        )
        self._log(f"  replanned: {len(remaining)} obsolete -> {len(revised.steps)} new")
        return revised, new_tail_ids

    def _splice_plan(self, plan: Plan, *, head_count: int, new_tail: tuple[Step, ...]) -> Plan:
        """Return a new Plan whose first ``head_count`` steps are preserved
        and whose tail is replaced by ``new_tail``."""
        return Plan(summary=plan.summary, steps=plan.steps[:head_count] + new_tail)

    def _splice_graph_for_steer(
        self,
        *,
        root_node_id: str | None,
        obsolete_node_ids: tuple[str, ...],
        new_steps: tuple[Step, ...],
        steer_text: str,
    ) -> tuple[str, ...]:
        """Mark pending step nodes obsolete and add fresh ``created_by="steering"``
        children under the same root. Returns the new node ids."""
        if self.graph_client is None or root_node_id is None:
            return ()
        reason = f"superseded by steering: {steer_text[:120]}"
        for nid in obsolete_node_ids:
            self.graph_client.obsolete(ObsoleteIntent(id=nid, reason=reason))
        new_ids: list[str] = []
        for step in new_steps:
            child = self.graph_client.add_subtask(
                AddSubtaskIntent(
                    parent_id=root_node_id,
                    draft=TaskNodeDraft(
                        title=step.title,
                        rationale=step.rationale,
                        acceptance=step.acceptance,
                        relevant_paths=step.relevant_paths,
                        created_by="steering",
                    ),
                )
            )
            new_ids.append(child.id)
        return tuple(new_ids)
