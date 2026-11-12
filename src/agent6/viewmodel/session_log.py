# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold session journals tolerantly for listings and session inspection."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
from collections.abc import Iterator
from typing import Any

from agent6 import budget, evidence, kinds, terminal
from agent6.sessions import journal
from agent6.viewmodel import events, format, review, session_core, session_status, state_types


@dataclasses.dataclass(frozen=True, slots=True)
class LogScan:
    """One tolerant pass over a session's journal, behind the hub listing and `sessions show`.

    One owner, so the resume rules (bank the cost of past executions, un-finish) and
    the torn-line tolerances cannot drift between consumers. Token counters are the
    current execution's; `cost_usd` and `plan_observed_points` are cumulative, matching the typed
    fold's BudgetView.

    Attributes:
        saw_start: A session.start or session.resume was seen; neither means unstarted.
        mode: The session's mode, "?" until a start event names it.
        task_label: The first task and the latest resume steer from the journal.
        finished: A session.stop was seen and no later execution began.
        verification: The final verification word, or None when unknown.
        verify_scoped: The judging gate ran scoped.
        stop_reason: The session.stop reason word.
        stranded: The latest execution finished with its edits uncommitted and no commit landed.
        setup_output: The workspace's setup command and retained output, or its failure diagnostic.
        end_summary: The journaled end sentence, "" while an execution runs or none was recorded.
        gate_swapped: The verify gate changed between executions, so the first baseline is
            another gate's verdict.
        baseline_ok: The start gate's verdict on the tree before the run's first turn; None
            when none was recorded.
        end_tree: The worktree's tree id that end recorded; "" when it recorded none or a
            later execution began.
        undone_to: The fork `/undo` continues the session as; "" when it was never undone.
        provider_error: The last failed driving call's error, the reason behind a
            `provider_error` end; "" when none failed.
        cost_usd: The spend across every execution; None when no budget.update was seen.
        usd_partial: Some execution had unpriced spend, so `cost_usd` is a lower bound.
        unmetered_tokens: Fallback tokens across every execution, including settled lanes.
        settled_lanes: Child sessions whose spend is durably included in the journal.
        executions: One plus the completed resume executions.
        input_tokens: This execution's input tokens, None until a budget.update carries them.
        output_tokens: This execution's output tokens.
        cache_read_tokens: The cached side of the input, None when the event lacked it.
        cache_creation_tokens: Cache-creation tokens, None when the event lacked it.
        plan_observed_points: The plan movement observed across every execution, in points; 0.0
            until a plan call runs.
        plan_prior_executions: The points banked by completed executions.
        plan_cap: `[budget].max_plan_points` for this execution, 0.0 until a plan call runs.
        plan_used_percent: The account's last reading, 0.0 until a plan call runs.
        plan_usage: That reading's windows, None until a plan call reports them.
        plan_usage_at: When the reading was taken, as epoch seconds; 0.0 without one.
        turn: The last event's turn.
        start_ep: session.start's epoch, else the first event's (a fork's log has none).
        last_ep: The last event's epoch.
        last_type: The last event's type.
        operator_blocked: A prompt is still unanswered on this execution.
        blocked_kind: The oldest unanswered prompt's kind, "approval" or "question".
        blocked_since_ep: That prompt's asked-at epoch.
        last_verify_rc: This execution's last verify.end exit code.
        verify_timeout_s: The limit that verify ran out, 0.0 when it ended on its own.
        pins: The operator's pinned instructions in force.
        unattended_questions: This execution's questions answered empty because nobody was
            attached.
        pending_questions: The questions nobody has answered, across every execution.
        oldest_pending_question_ep: When the oldest of them was asked, as epoch seconds.
        left_out: The (path, reason) of each edit the commits left out, across every
            execution; the reason is `ignored` or `untracked` (`loop.edit.left_out`).
        review: The shared fold of panel evidence and finding history.
        end_ep: The latest execution's end time as epoch seconds; None while it runs.
        execution_id: The current execution's recorded id, if present.
        source_tip: The most recent diff checkpoint's commit.
        verifications: Every recorded verification, including baseline observations.
        final_verify_id: The end event's verification id, absent if evidence is damaged.
        evidence_notes: Missing or unreadable execution evidence.
        targets: Immutable target observations, never inferred from current runtime state.
        current_target: The journal's current target binding, None when unproved.
        setups: Setup results with their recorded execution and target references.
        jobs: Job-settlement observations, separate from environment lifecycle.
        environments: Environment lifecycle observations, absent in historical journals.
        acquisitions: The container leases the run's executions held, in observation order.
        environment: The container the current execution holds; None outside one.
    """

    saw_start: bool = False
    mode: str = "?"
    task_label: session_status.TaskLabel = dataclasses.field(
        default_factory=session_status.TaskLabel
    )
    finished: bool = False
    verification: kinds.Verification | None = None
    verify_scoped: bool = False
    stop_reason: str = ""
    stranded: bool = False
    setup_output: str = ""
    end_summary: str = ""
    baseline_ok: bool | None = None
    gate_swapped: bool = False
    end_tree: str = ""
    undone_to: str = ""
    provider_error: str = ""
    cost_usd: float | None = None
    usd_partial: bool = False
    unmetered_tokens: int = 0
    settled_lanes: frozenset[str] = frozenset()
    executions: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    plan_observed_points: float = 0.0
    plan_prior_executions: float = 0.0
    plan_cap: float = 0.0
    plan_used_percent: float = 0.0
    plan_usage: budget.PlanUsage | None = None
    plan_usage_at: float = 0.0
    turn: int | None = None
    start_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    operator_blocked: bool = False
    blocked_kind: str = ""
    blocked_since_ep: float | None = None
    last_verify_rc: int | None = None
    verify_timeout_s: float = 0.0
    pins: tuple[str, ...] = ()
    unattended_questions: int = 0
    pending_questions: int = 0
    oldest_pending_question_ep: float | None = None
    left_out: tuple[tuple[str, str], ...] = ()
    review: review.ReviewView = dataclasses.field(default_factory=review.ReviewView)

    execution_id: str | None = None
    source_tip: str | None = None
    verifications: tuple[evidence.VerificationEvidence, ...] = ()
    final_verify_id: str | None = None
    end_ep: float | None = None
    evidence_notes: tuple[str, ...] = ()
    targets: tuple[evidence.ExecutionTargetEvidence, ...] = ()
    current_target: evidence.ExecutionTargetEvidence | None = None
    setups: tuple[evidence.SetupEvidence, ...] = ()
    jobs: tuple[evidence.JobsObservation, ...] = ()
    environments: tuple[evidence.EnvironmentObservation, ...] = ()
    acquisitions: tuple[evidence.AcquisitionObservation, ...] = ()
    environment: state_types.EnvironmentView | None = None

    def gate_word(self) -> str:
        """Return `verification`, with `timed_out` for a red that is the gate running out its limit.

        Returns:
            The gate's word for the final tree.
        """
        return (
            "timed_out"
            if self.verification == "failed" and self.verify_timeout_s
            else self.verification or ""
        )

    @property
    def usage_cell(self) -> str:
        """The session's cost as `usage_cell` spells it; "" before any budget.update."""
        if self.cost_usd is None:
            return ""
        return session_status.usage_cell(
            self.cost_usd,
            partial=self.usd_partial,
            plan_observed_points=self.plan_observed_points,
            plan_used_percent=self.plan_used_percent,
        )

    def status_facts(self) -> session_status.StatusFacts:
        """Return this scan's answers to the status questions.

        The typed fold's `state_view.status_facts` agrees on the same log.

        Returns:
            The facts `status_for_session_dir` reads.
        """
        return session_status.StatusFacts(
            started=self.saw_start,
            finished=self.finished,
            verification=self.verification,
            verify_scoped=self.verify_scoped,
            gate_red=self.last_verify_rc is not None and self.last_verify_rc != 0,
            gate_timed_out=self.verify_timeout_s > 0,
            stop_reason=self.stop_reason,
            operator_blocked=self.operator_blocked,
            blocked_kind=self.blocked_kind,
            blocked_since_ep=self.blocked_since_ep,
            unattended_questions=self.unattended_questions,
            pending_questions=self.pending_questions,
            stranded=self.stranded,
        )


def events_backward(logs: pathlib.Path) -> Iterator[dict[str, Any]]:
    """Yield the journal's parseable records newest first, reading from the tail.

    Args:
        logs: The journal path.

    Yields:
        Each record that parses as a JSON object; an unreadable journal yields nothing.
    """
    try:
        with logs.open("rb") as fh:
            end = fh.seek(0, os.SEEK_END)
            pending = b""
            while end:
                # Grow with a spanning record so assembling a large line stays linear.
                start = max(0, end - max(8192, len(pending)))
                fh.seek(start)
                lines = (fh.read(end - start) + pending).split(b"\n")
                pending = lines.pop(0) if start else b""
                for line in reversed(lines):
                    try:
                        ev = json.loads(line.decode("utf-8", errors="replace"))
                    except ValueError:
                        continue
                    if isinstance(ev, dict):
                        yield ev
                end = start
    except OSError:
        return


def last_event_epoch(logs: pathlib.Path) -> float | None:
    """Return the last parseable event time, reading backward from the journal's tail.

    Args:
        logs: The journal path.

    Returns:
        The epoch seconds, or None when no timestamp can be read.
    """
    for ev in events_backward(logs):
        ep = events.event_epoch(ev.get("ts"))
        if ep is not None:
            return ep
    return None


# The records that bind or clear the current execution target, read from the tail.
_TARGET_BOUNDARIES = frozenset(
    {"execution.target", "session.start", "session.resume", "session.stop"}
)


def current_target(logs: pathlib.Path) -> evidence.ExecutionTargetEvidence | None:
    """Return the journal's current execution target, reading backward from its tail.

    The same answer as `scan_session_log(logs).current_target`, at the cost of the current
    execution's records rather than a replay of the journal: the last target stamp binds
    the current execution unless another execution's boundary follows it or an earlier
    record of the same target contradicts it.

    Args:
        logs: The journal path.

    Returns:
        The current target stamp, or None when the journal proves none.
    """
    stamp: evidence.ExecutionTargetEvidence | None = None
    later_executions: set[object] = set()
    for ev in events_backward(logs):
        kind = ev.get("type")
        if kind not in _TARGET_BOUNDARIES:
            continue
        if stamp is None:
            if kind != "execution.target":
                later_executions.add(ev.get("execution_id"))
                continue
            try:
                stamp = evidence.TARGET.validate_python(ev)
            except (ValueError, TypeError):
                return None
            if not stamp.execution_id or later_executions - {stamp.execution_id}:
                return None
        elif kind == "execution.target":
            if ev.get("target_id") != stamp.target_id:
                continue
            with contextlib.suppress(ValueError, TypeError):
                if evidence.TARGET.validate_python(ev) != stamp:
                    return None
        elif ev.get("execution_id") != stamp.execution_id:
            break  # an earlier execution's boundary: its stamps are its own
    return stamp


def current_environment(logs: pathlib.Path) -> state_types.EnvironmentView | None:
    """Return the container the current execution holds, from the journal's tail.

    The same answer as `scan_session_log(logs).environment`: the last lease row binds the
    current execution unless another execution's boundary follows it.

    Args:
        logs: The journal path.

    Returns:
        The view, None before the execution acquires or starts a container.
    """
    outcome = ""
    later_executions: set[object] = set()
    for ev in events_backward(logs):
        kind = ev.get("type")
        if kind in ("session.start", "session.resume", "session.stop"):
            later_executions.add(ev.get("execution_id"))
            continue
        if kind not in ("environment.acquired", "environment.starting", "execution.environment"):
            continue
        record = events.parse_event(ev)
        if isinstance(record, journal.ExecutionEnvironment):
            outcome = outcome or record.outcome
        elif isinstance(record, journal.EnvironmentAcquired | journal.EnvironmentStarting):
            if later_executions - {record.execution_id}:
                return None
            return session_core.held_environment(record, outcome)
    return None


def scan_session_log(logs: pathlib.Path) -> LogScan:  # noqa: C901, PLR0912, PLR0915  # one branch per event type
    """Fold a session's journal into a `LogScan`.

    The lifecycle and the usage fold in `session_core`, the one reducer the viewers' fold shares;
    the rest is the listing's own. A live writer can leave a torn multibyte tail, so the file is
    decoded with replacement and the mangled line fails `json.loads` and is skipped.

    Args:
        logs: The journal path.

    Returns:
        The scan; an empty one when the file cannot be read.
    """
    core = state_types.SessionCore()
    setup_output = ""
    baseline_ok: bool | None = None
    gate_swapped = False
    end_tree = ""
    saw_budget = False
    turn: int | None = None
    start_ep: float | None = None
    first_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    # Only an answer or an execution boundary clears a prompt: a steer request leaves it waiting.
    pending_prompts: dict[str, tuple[str, float | None]] = {}
    # Every question nobody has answered yet, by id, across executions.
    open_questions: dict[str, float | None] = {}
    left_out: dict[str, str] = {}
    review_view = review.ReviewView()
    source_tip: str | None = None
    final_verify_id: str | None = None
    end_ep: float | None = None
    verifications: list[evidence.VerificationEvidence] = []
    pending_verifications: set[str] = set()
    evidence_notes: list[str] = []
    observations = evidence.ExecutionObservations()
    try:
        with logs.open(encoding="utf-8", errors="replace") as fh:
            for line_number, line in enumerate(fh, 1):
                try:
                    ev = json.loads(line)
                except ValueError as exc:
                    evidence_notes.append(f"journal line {line_number}: {exc}")
                    review_view = dataclasses.replace(
                        review_view,
                        read_failure=review_view.read_failure
                        or f"journal line {line_number}: {exc}",
                        completion="unknown",
                        freshness="unknown",
                    )
                    continue
                if not isinstance(ev, dict):
                    evidence_notes.append(f"journal line {line_number}: expected an event object")
                    continue
                review_view = review.apply_event(review_view, ev)
                etype = ev.get("type")
                try:
                    observations.observe(ev)
                except (ValueError, TypeError) as exc:
                    evidence_notes.append(f"journal line {line_number}: {exc}")
                if etype in ("session.start", "session.resume"):
                    final_verify_id = None
                elif etype == "diff.updated":
                    source_tip = ev.get("sha") or source_tip
                ep = events.event_epoch(ev.get("ts"))
                if ep is not None:
                    last_ep = ep
                    if first_ep is None:
                        first_ep = ep
                if isinstance(etype, str):
                    last_type = etype
                # Journals written before the turn rename stamp `iteration` and `iterations`.
                if isinstance(ev_turn := ev.get("turn", ev.get("iteration")), int):
                    turn = ev_turn
                if etype in session_status.OPERATOR_PROMPT_EVENTS:
                    # Keyed by str(id) like the answer side, so an int id still clears. A queued
                    # question (`wait` false) pauses nothing.
                    blocks = etype == "approval.prompt" or ev.get("wait") is not False
                    if blocks and (pid := ev.get("id")) is not None:
                        kind = "approval" if etype == "approval.prompt" else "question"
                        pending_prompts[str(pid)] = (kind, ep)
                elif etype in session_status.OPERATOR_ANSWER_EVENTS:
                    pending_prompts.pop(str(ev.get("id")), None)
                # A question outside an execution (the start gate) dies with its prompt.
                in_execution = core.lifecycle.started and not core.lifecycle.finished
                if (
                    etype == "question.prompt"
                    and in_execution
                    and (qid := ev.get("id")) is not None
                ):
                    open_questions[str(qid)] = ep
                elif etype == "question.answer":
                    open_questions.pop(str(ev.get("id")), None)
                if session_core.concerns(ev):
                    core = session_core.apply(core, events.parse_event(ev), ev)
                if etype == "session.start":
                    end_ep = None
                    end_tree = ""
                    # A new execution re-asks with restarted ids; a held-over prompt waits forever.
                    pending_prompts.clear()
                    if start_ep is None:
                        start_ep = ep
                elif etype == "session.stop":
                    if isinstance(end_turns := ev.get("turns", ev.get("iterations")), int):
                        turn = end_turns
                    if core.lifecycle.stop_reason == "setup_command_failed" and ev.get("summary"):
                        setup_output = terminal.scrub_terminal_controls(str(ev["summary"]))
                    end_tree = str(ev.get("tree") or "")
                    final_verify_id = ev.get("final_verify_id")
                    end_ep = ep
                elif etype == "session.resume":
                    end_ep = None
                    end_tree = ""
                    pending_prompts.clear()
                elif etype == "workspace.setup":
                    setup_output = format.setup_output(ev)
                elif etype == "verify.start":
                    if ev.get("verify_id"):
                        pending_verifications.add(str(ev["verify_id"]))
                elif etype == "verify.end":
                    try:
                        record = evidence.from_event(ev)
                    except (ValueError, TypeError) as exc:
                        evidence_notes.append(f"journal line {line_number}: {exc}")
                    else:
                        verifications.append(record)
                        if record.verify_id:
                            pending_verifications.discard(record.verify_id)
                elif etype == "loop.baseline":
                    if baseline_ok is None:
                        baseline_ok = bool(ev.get("ok"))
                    for i, record in enumerate(verifications):
                        matches = bool(record.verify_id) and record.verify_id == ev.get("verify_id")
                        legacy = not ev.get("verify_id") and i == len(verifications) - 1
                        if matches or legacy:
                            verifications[i] = dataclasses.replace(record, baseline=True)
                elif etype == "loop.verify_swapped":
                    gate_swapped = True
                elif etype == "loop.edit.left_out":
                    for reason in ("ignored", "untracked"):
                        raw_paths = ev.get(reason)
                        for p in raw_paths if isinstance(raw_paths, list) else []:
                            left_out[str(p)] = reason
                elif etype == "budget.update":
                    saw_budget = True
    except FileNotFoundError:
        evidence_notes.append("Journal missing.")
    except OSError as exc:
        evidence_notes.append(f"Journal unreadable: {exc}")
        review_view = dataclasses.replace(review_view, read_failure=str(exc))
    life, usage = core.lifecycle, core.budget
    account = usage.account
    oldest = min(
        pending_prompts.values(),
        key=lambda kv: kv[1] if kv[1] is not None else float("inf"),
        default=None,
    )
    return LogScan(
        saw_start=life.started,
        review=review_view,
        execution_id=life.execution_id or None,
        source_tip=source_tip,
        verifications=tuple(verifications),
        targets=observations.targets.recorded,
        current_target=observations.targets.current,
        setups=tuple(observations.setups),
        jobs=tuple(observations.jobs),
        environments=tuple(observations.environments),
        acquisitions=tuple(observations.acquisitions),
        environment=life.environment,
        final_verify_id=None if evidence_notes else final_verify_id,
        end_ep=end_ep,
        evidence_notes=tuple(evidence_notes)
        + tuple(f"Verification {v} has no recorded result." for v in sorted(pending_verifications)),
        mode=life.mode or "?",
        task_label=life.task_label,
        finished=life.finished,
        verification=life.verification,
        verify_scoped=life.verify_scoped,
        stop_reason=life.stop_reason,
        stranded=life.stranded,
        end_summary=life.end_summary,
        baseline_ok=baseline_ok,
        gate_swapped=gate_swapped,
        setup_output=setup_output,
        end_tree=end_tree,
        undone_to=life.undone_to,
        provider_error=life.provider_error,
        cost_usd=usage.session.usd if saw_budget else None,
        usd_partial=usage.session.usd_partial,
        unmetered_tokens=usage.session.unmetered_tokens,
        settled_lanes=frozenset(usage.session.settled_lanes),
        executions=life.executions,
        input_tokens=usage.execution.input_tokens,
        output_tokens=usage.execution.output_tokens,
        cache_read_tokens=usage.execution.cache_read_tokens,
        cache_creation_tokens=usage.execution.cache_creation_tokens,
        plan_observed_points=usage.session.plan_observed_points,
        plan_prior_executions=(
            usage.session.plan_observed_points - usage.execution.plan_observed_points
        ),
        plan_cap=usage.execution.plan_cap or 0.0,
        plan_used_percent=account.used_percent if account else 0.0,
        plan_usage=budget.PlanUsage(windows=account.windows)
        if account and account.windows
        else None,
        plan_usage_at=account.observed_at if account else 0.0,
        unattended_questions=life.unattended_questions,
        pending_questions=len(open_questions),
        oldest_pending_question_ep=min(
            (ep for ep in open_questions.values() if ep is not None), default=None
        ),
        turn=turn,
        start_ep=start_ep if start_ep is not None else first_ep,
        last_ep=last_ep,
        last_type=last_type,
        operator_blocked=bool(pending_prompts),
        blocked_kind=oldest[0] if oldest else "",
        blocked_since_ep=oldest[1] if oldest else None,
        last_verify_rc=life.last_verify_rc,
        verify_timeout_s=life.verify_timeout_s,
        pins=life.pins,
        left_out=tuple(left_out.items()),
    )
