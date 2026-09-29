# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's events into a `SessionState` and its wire form.

`session_state_as_dict` is the contract every external viewer reads (`attach --json`,
the web page): the state's fields plus `status`, `status_label`, `dead_state`,
`shells`, `live` and `operator_blocked`, with `log_tail` as plain strings; its keys
stay stable. The fold does no I/O: `apply_event` returns a new frozen state, so an
unchanged identity means nothing changed. With a session dir, the wire form also
reads the dir's status probes and manifest.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from agent6.graph.models import owner_note
from agent6.models.registry import context_window
from agent6.sessions.ipc import listening_ports
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.background import SHELLS_DIR, roster_from_dir
from agent6.viewmodel import events
from agent6.viewmodel.format import (
    TASK_STATUS_GLYPH,
    budget_usd_text,
    dead_run_note,
    short_task_id,
    status_label,
)
from agent6.viewmodel.listing import (
    LIVE_STATUS_WORDS,
    StatusFacts,
    needs_new_work,
    status_for_session_dir,
    status_word,
    task_snippet,
)
from agent6.viewmodel.log_line import format_log_line, render_args
from agent6.viewmodel.policy import session_policy
from agent6.viewmodel.transcript import scrub_terminal_controls

NodeStatus = Literal["pending", "in_progress", "passed", "failed", "skipped", "obsolete"]


@dataclass(frozen=True, slots=True)
class TaskNodeView:
    """One node of the live task tree, flattened in DFS pre-order with its depth.

    Mirrors `graph.models.TaskNode`; fed by the `graph.update` snapshot the worker
    emits whenever it changes its task breakdown.

    Attributes:
        id: The graph's id for the task.
        title: The task's title.
        status: The task's status word.
        depth: The nesting depth, for tree rendering.
        is_cursor: The worker is on this task.
        created_by: Who added the task; "" for a run dir written before the field existed.
        standing: The task is the standing goal.
        note: The operator's mark beside a task they own ("queued by you", "standing
            goal"), or "".
        short_id: The id as the operator reads and types it.
        glyph: The status as every surface draws it; the cursor's task draws as in progress.
    """

    id: str
    title: str
    status: NodeStatus = "pending"
    depth: int = 0
    is_cursor: bool = False
    created_by: str = ""
    standing: bool = False
    note: str = ""
    short_id: str = ""
    glyph: str = ""

    def __post_init__(self) -> None:
        """Fill `glyph` from the status when the caller left it empty."""
        if not self.glyph:
            status = "in_progress" if self.is_cursor else self.status
            object.__setattr__(self, "glyph", TASK_STATUS_GLYPH.get(status, "·"))


@dataclass(frozen=True, slots=True)
class ToolCallView:
    """One tool call in the bounded history.

    Attributes:
        name: The tool's name.
        args_preview: The args rendered with each value truncated, for the inline table.
        args_full: The args rendered with a generous per-value cap, for the detail view.
        result_summary: The result's summary once it landed.
        ok: The tool's verdict; None while in flight.
        task_id: The task in focus when the call ran, for filtering.
        call_id: The per-dispatch correlation id; None on a log without ids.
    """

    name: str
    args_preview: str
    args_full: str = ""
    result_summary: str = ""
    ok: bool | None = None
    task_id: str | None = None
    call_id: int | None = None


@dataclass(frozen=True, slots=True)
class LogLine:
    """One log line plus the task in focus when it was emitted, so a viewer can filter."""

    text: str
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiffView:
    """One auto-commit diff plus the task in focus when it landed."""

    patch: str
    task_id: str | None = None
    sha: str = ""


@dataclass(frozen=True, slots=True)
class VerifyView:
    """The last verify gate run.

    Attributes:
        cmd: The gate's argv.
        exit_code: The exit code; None while in flight.
        duration_s: How long it ran.
        stdout_tail: The capped stdout tail.
        stderr_tail: The capped stderr tail.
    """

    cmd: tuple[str, ...]
    exit_code: int | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True, slots=True)
class BudgetView:
    """The run's spend as every surface shows it.

    Token and plan counters and the caps are the current execution's, pairing with
    the per-execution enforcement; `usd_total` is cumulative across executions, as
    the listing scan sums it, so the surfaces agree on what a run cost.

    Attributes:
        input_total: This execution's input tokens.
        output_total: This execution's output tokens.
        cache_read_total: The cached side of the input.
        cache_creation_total: Cache-creation tokens.
        usd_total: The spend across every execution.
        usd_prior_executions: The spend banked by completed executions.
        usd_partial: Some model had no price, so `usd_total` is a lower bound.
        usd_cap: `[budget].max_usd` for this execution; -1 unlimited, 0 unknown.
        tokens_unmetered: Input and output tokens of calls the meter could not price.
        tokens_fallback_cap: `[budget].max_tokens_fallback`; -1 unlimited, 0 unknown.
        plan_used_percent: The account's reported plan usage; 0 until a plan call runs.
        plan_consumed: This execution's consumed plan points.
        plan_cap: `[budget].max_percent`; 0 until a plan call runs.
        plan_resets_at: When the plan window resets, as epoch seconds.
    """

    input_total: int = 0
    output_total: int = 0
    cache_read_total: int = 0
    cache_creation_total: int = 0
    usd_total: float = 0.0
    usd_prior_executions: float = 0.0
    usd_partial: bool = False
    usd_cap: float = 0.0
    tokens_unmetered: int = 0
    tokens_fallback_cap: int = 0
    plan_used_percent: float = 0.0
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_resets_at: float = 0.0


@dataclass(frozen=True, slots=True)
class RoleCall:
    """The last model call and what it streamed.

    Attributes:
        role: The role that called.
        model: The model called.
        in_flight: The call has not returned.
        provider: The provider that dialled the model; pairs with `model` for the
            registry's context-window lookup.
        ctx_tokens: The full prompt in tokens at the last completed call (fresh input
            plus cache reads and writes); 0 until a result lands.
        streamed_text: The live text, reset on every call, appended per delta, kept to
            the last `_STREAM_TAIL` characters.
        streamed_thinking: The live reasoning, with the same lifecycle.
    """

    role: str
    model: str
    in_flight: bool
    provider: str = ""
    ctx_tokens: int = 0
    streamed_text: str = ""
    streamed_thinking: str = ""


@dataclass(frozen=True, slots=True)
class CommitStep:
    """One per-step commit of the run, which the dashboards select among."""

    iteration: int
    sha: str
    subject: str

    @property
    def label(self) -> str:
        """The step picker's line: `iter N · sha7 · subject`, a part left out when empty."""
        return " · ".join(p for p in (f"iter {self.iteration}", self.sha[:7], self.subject) if p)


def approval_parts(prompt: str) -> tuple[str, str]:
    """Split an approval prompt into its head and payload.

    Every dispatch prompt is "Allow <tool>: <payload>"; the CLI, TUI and web render
    exactly these two parts.

    Args:
        prompt: The prompt's words.

    Returns:
        The head (the question) and the payload (the command under judgment); a prompt
        without a payload is all head.
    """
    head, sep, payload = prompt.partition(": ")
    if sep and payload.strip():
        return head, payload
    return prompt, ""


@dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    """One approval the run asked for.

    Attributes:
        id: The prompt's id.
        prompt: The prompt's words.
        standing: An "allow all" is on offer for this prompt.
        answered: An answer was folded.
        approved: The answer; None until answered.
        asked_ep: When it was asked, for the waiting status's age.
    """

    id: str
    prompt: str
    standing: bool = True
    answered: bool = False
    approved: bool | None = None
    asked_ep: float | None = None

    @property
    def head(self) -> str:
        """The question part of the prompt."""
        return approval_parts(self.prompt)[0]

    @property
    def payload(self) -> str:
        """The command under judgment, "" when the prompt has none."""
        return approval_parts(self.prompt)[1]


@dataclass(frozen=True, slots=True)
class Question:
    """One question within an `ask_user` prompt.

    Attributes:
        question: The question's text.
        options: Selectable presets; the operator may also type a free-text answer.
    """

    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionPrompt:
    """An `ask_user` prompt: related questions the operator answers together.

    Attributes:
        id: The prompt's id.
        questions: The questions asked.
        answered: An answer was folded.
        answers: One answer per question, aligned to `questions`.
        from_harness: agent6 itself asked (a start gate such as the dirty-tree
            question): it was asked before the session started or after it finished,
            when no model runs.
        asked_ep: When it was asked, for the waiting status's age.
    """

    id: str
    questions: tuple[Question, ...] = ()
    answered: bool = False
    answers: tuple[str, ...] = ()
    from_harness: bool = False
    asked_ep: float | None = None


@dataclass(frozen=True, slots=True)
class SessionState:
    """A session's folded state, the read model every front-end paints.

    Attributes:
        session_id: The session's id, "" until an event carries it.
        user_task: The task text.
        tasks: The live task tree in DFS pre-order.
        cursor_task_id: The task the worker is on.
        last_role: The last model call.
        tool_calls: The bounded tool-call history, most recent last.
        last_verify: The last verify gate run.
        budget: The spend.
        pending_approvals: Every approval asked this execution, answered or not.
        pending_questions: Every question prompt asked this execution, answered or not.
        log_tail: The bounded window of log lines, most recent last.
        log_count: The monotonic count of log lines ever; a live viewer diffs on it,
            since `len(log_tail)` freezes once the window saturates.
        recent_diffs: The bounded auto-commit diffs, for task filtering.
        started: An execution has begun; a parked or created run has none.
        finished: A session.end was folded and no later execution began.
        all_passed: The verify tri-state; None when no verify command gated the end.
        verify_scoped: The judging gate ran scoped.
        end_reason: The session.end reason word.
        unattended_questions: Questions the harness answered empty because nobody was
            attached, across every execution.
        undone_to: The child session /undo forked, which surfaces follow.
        undone_text: The message /undo took back, for the composer to refill.
        finish_summary: The finish tool's summary, the agent's closing statement.
        latest_diff: The patch of the most recent auto-commit.
        steps: The run's per-step commits, oldest first.
        steer_requests: The monotonic count of mid-run steer requests; the TUI
            compares it with its own count to react once per press.
        compact_elided: The elision markers in the current context.
        compact_gists_live: The live gists in the current context; a demoted gist is
            back to a bare marker, and a tier-2 restart resets both.
        pins: The operator's /pin instructions in force, most recent last.
        last_event_ep: The last folded event's own ts, the idle anchor every
            "working… Ns" timer measures from, so replayed history reads as its true age.
    """

    session_id: str = ""
    user_task: str = ""
    tasks: tuple[TaskNodeView, ...] = ()
    cursor_task_id: str | None = None
    last_role: RoleCall | None = None
    tool_calls: tuple[ToolCallView, ...] = ()
    last_verify: VerifyView | None = None
    budget: BudgetView = field(default_factory=BudgetView)
    pending_approvals: tuple[ApprovalPrompt, ...] = ()
    pending_questions: tuple[QuestionPrompt, ...] = ()
    log_tail: tuple[LogLine, ...] = ()
    log_count: int = 0
    recent_diffs: tuple[DiffView, ...] = ()
    started: bool = False
    finished: bool = False
    all_passed: bool | None = None
    verify_scoped: bool = False
    end_reason: str = ""
    unattended_questions: int = 0
    undone_to: str = ""
    undone_text: str = ""
    finish_summary: str = ""
    latest_diff: str = ""
    steps: tuple[CommitStep, ...] = ()
    steer_requests: int = 0
    compact_elided: int = 0
    compact_gists_live: int = 0
    pins: tuple[str, ...] = ()
    last_event_ep: float | None = None


def initial_state() -> SessionState:
    """Return the state before any event."""
    return SessionState()


_MAX_TOOL_HISTORY = 50
_MAX_DIFF_HISTORY = 30
# The inline log widget caps to this so it stays a gapless window.
MAX_LOG_TAIL = 400
# Only the tail of an in-flight stream travels; the transcript keeps the full turn.
_STREAM_TAIL = 6000

# Live-view only: a reasoning model would flood the log with contentless delta lines.
STREAM_DELTA_EVENTS = frozenset({"role.thinking_delta", "role.text_delta"})
# Loop-side mirrors of events the log already shows.
LOG_NOISE_EVENTS = frozenset({"loop.tool.call", "loop.budget"})


def _answered_only[PromptT: (ApprovalPrompt, QuestionPrompt)](
    prompts: tuple[PromptT, ...],
) -> tuple[PromptT, ...]:
    """Drop the unanswered prompts at an execution boundary.

    They belong to the execution that died holding them; the new one re-asks with
    restarted ids.

    Args:
        prompts: The prompts folded so far.

    Returns:
        The answered ones.
    """
    return tuple(p for p in prompts if p.answered)


def apply_event(state: SessionState, event: dict[str, Any]) -> SessionState:  # noqa: C901, PLR0911, PLR0912, PLR0915  # one branch per event type
    """Fold one event into the session state.

    The event is parsed once into a typed family; the log line and the session id
    peek read the raw dict, since they render every event. An unknown or telemetry
    type changes no state.

    Args:
        state: The state so far.
        event: The raw event dict.

    Returns:
        The new state, or `state` itself when nothing changed.
    """
    etype = event.get("type", "")
    if not state.session_id and event.get("session_id"):
        state = replace(state, session_id=str(event["session_id"]))
    # The event's own ts, so replayed history measures idle time from when the run last spoke.
    if (ep := events.event_epoch(event.get("ts"))) is not None:
        state = replace(state, last_event_ep=ep)
    if etype not in STREAM_DELTA_EVENTS and etype not in LOG_NOISE_EVENTS:
        # cursor_task_id is the focus task: graph.update lands before a turn's calls.
        entry = LogLine(format_log_line(event), state.cursor_task_id)
        new_log = _push_bounded(state.log_tail, entry, MAX_LOG_TAIL)
        state = replace(state, log_tail=new_log, log_count=state.log_count + 1)

    match events.parse_event(event):
        case events.SessionStart(user_task=task):
            # The ask REPL re-runs on one log, so a second start clears the prior end.
            # No banking, unlike ResumeStart: the REPL's one tracker is already cumulative.
            return replace(
                state,
                user_task=task,
                started=True,
                finished=False,
                end_reason="",
                last_role=None,
                finish_summary="",
                last_verify=None,
                pending_approvals=_answered_only(state.pending_approvals),
                pending_questions=_answered_only(state.pending_questions),
            )

        case events.ResumeStart():
            # The new execution's counters start fresh: bank the spend, zero the rest.
            return replace(
                state,
                started=True,
                finished=False,
                end_reason="",
                last_role=None,
                finish_summary="",
                last_verify=None,
                pending_approvals=_answered_only(state.pending_approvals),
                pending_questions=_answered_only(state.pending_questions),
                budget=replace(
                    state.budget,
                    usd_prior_executions=state.budget.usd_total,
                    input_total=0,
                    output_total=0,
                    usd_cap=0.0,
                    tokens_unmetered=0,
                    tokens_fallback_cap=0,
                    plan_used_percent=0.0,
                    plan_consumed=0.0,
                    plan_cap=0.0,
                    plan_resets_at=0.0,
                ),
            )

        case events.GraphUpdate(nodes=nodes, cursor=cursor):
            return replace(
                state,
                tasks=task_tree_views(nodes, cursor),
                cursor_task_id=cursor,
            )

        case events.AutoCommit(iteration=iteration, sha=sha, subject=subject):
            if not sha:
                return state
            step = CommitStep(iteration=iteration, sha=sha, subject=subject)
            return replace(state, steps=(*state.steps, step))

        case events.DiffUpdated(patch=patch, sha=sha):
            entry = DiffView(patch=patch, task_id=state.cursor_task_id, sha=sha)
            return replace(
                state,
                latest_diff=patch,
                recent_diffs=_push_bounded(state.recent_diffs, entry, _MAX_DIFF_HISTORY),
            )

        case events.RoleCall(role=role, model=model, provider=provider):
            prior = state.last_role
            return replace(
                state,
                last_role=RoleCall(
                    role=role,
                    model=model,
                    in_flight=True,
                    provider=provider,
                    # The last known context size stays until this call's result lands.
                    ctx_tokens=prior.ctx_tokens if prior is not None else 0,
                    streamed_text="",
                    streamed_thinking="",
                ),
            )

        case events.RoleTextDelta(text=piece):
            # Scrub the concatenation: an escape sequence can arrive split across deltas.
            last = state.last_role
            if last is None or not last.in_flight or not piece:
                return state
            joined = scrub_terminal_controls(last.streamed_text + piece)
            return replace(
                state,
                last_role=replace(last, streamed_text=joined[-_STREAM_TAIL:]),
            )

        case events.RoleThinkingDelta(text=piece):
            last = state.last_role
            if last is None or not last.in_flight or not piece:
                return state
            joined = scrub_terminal_controls(last.streamed_thinking + piece)
            return replace(
                state,
                last_role=replace(last, streamed_thinking=joined[-_STREAM_TAIL:]),
            )

        case events.RoleResult(tokens_in=tin, cache_read=cr, cache_creation=cc):
            last = state.last_role
            if last is None:
                return state
            ctx = tin + cr + cc
            return replace(
                state,
                last_role=replace(
                    last, in_flight=False, ctx_tokens=ctx if ctx > 0 else last.ctx_tokens
                ),
            )

        case events.ToolCall(name=name, args=raw_args, call_id=cid):
            tc = ToolCallView(
                name=name,
                args_preview=render_args(raw_args),
                args_full=render_args(raw_args, max_value=4000),
                ok=None,
                task_id=state.cursor_task_id,
                call_id=cid,
            )
            finish_summary = state.finish_summary
            if name in ("finish_session", "finish_planning") and isinstance(raw_args, dict):
                finish_summary = str(raw_args.get("summary", "")).strip() or finish_summary
            return replace(
                state,
                tool_calls=_push_bounded(state.tool_calls, tc, _MAX_TOOL_HISTORY),
                finish_summary=finish_summary,
            )

        case events.ToolResult(name=name, ok=ok, summary=summary, call_id=cid):
            if not state.tool_calls:
                return state
            if cid is not None:
                # Concurrent seats interleave events, so the matching call may not be the last.
                for i in range(len(state.tool_calls) - 1, -1, -1):
                    if state.tool_calls[i].call_id == cid:
                        updated = replace(state.tool_calls[i], ok=ok, result_summary=summary)
                        return replace(
                            state,
                            tool_calls=(
                                *state.tool_calls[:i],
                                updated,
                                *state.tool_calls[i + 1 :],
                            ),
                        )
                return state
            last = state.tool_calls[-1]
            if last.name != name:
                return state
            updated_last = replace(last, ok=ok, result_summary=summary)
            return replace(
                state,
                tool_calls=(*state.tool_calls[:-1], updated_last),
            )

        case events.VerifyStart(cmd=cmd):
            return replace(state, last_verify=VerifyView(cmd=cmd))

        case events.VerifyEnd(
            cmd=cmd, exit_code=code, duration_s=dur, stdout_tail=out, stderr_tail=err
        ):
            return replace(
                state,
                last_verify=VerifyView(
                    cmd=cmd, exit_code=code, duration_s=dur, stdout_tail=out, stderr_tail=err
                ),
            )

        case events.BudgetUpdate(
            input_total=it,
            output_total=ot,
            cache_read_total=cr,
            cache_creation_total=cc,
            usd_total=usd,
            usd_partial=partial,
            usd_cap=ucap,
            tokens_unmetered=unmet,
            tokens_fallback_cap=fcap,
            plan_used_percent=plan_pct,
            plan_consumed=plan_used,
            plan_cap=plan_cap,
            plan_resets_at=plan_resets,
        ):
            # The event's usd_total is this execution's; the view's is cumulative.
            return replace(
                state,
                budget=BudgetView(
                    input_total=it,
                    output_total=ot,
                    cache_read_total=cr,
                    cache_creation_total=cc,
                    usd_total=state.budget.usd_prior_executions + usd,
                    usd_prior_executions=state.budget.usd_prior_executions,
                    usd_partial=partial or state.budget.usd_partial,
                    usd_cap=ucap,
                    tokens_unmetered=unmet,
                    tokens_fallback_cap=fcap,
                    plan_used_percent=plan_pct,
                    plan_consumed=plan_used,
                    plan_cap=plan_cap,
                    plan_resets_at=plan_resets,
                ),
            )

        case events.ApprovalPrompt(id=aid, prompt=prompt, standing=standing, asked_ep=asked_ep):
            ap = ApprovalPrompt(id=aid, prompt=prompt, standing=standing, asked_ep=asked_ep)
            return replace(state, pending_approvals=(*state.pending_approvals, ap))

        case events.ApprovalAnswer(id=wanted_id, approved=approved):
            new = tuple(
                replace(a, answered=True, approved=approved) if a.id == wanted_id else a
                for a in state.pending_approvals
            )
            return replace(state, pending_approvals=new)

        case events.QuestionPrompt(id=qid, questions=qs, asked_ep=asked_ep):
            questions = tuple(Question(question=q.question, options=q.options) for q in qs)
            qp = QuestionPrompt(
                id=qid,
                questions=questions,
                from_harness=not state.started or state.finished,
                asked_ep=asked_ep,
            )
            return replace(state, pending_questions=(*state.pending_questions, qp))

        case events.QuestionAnswer(id=wanted, answers=answers, unseen=unseen):
            new_q = tuple(
                replace(q, answered=True, answers=answers) if q.id == wanted else q
                for q in state.pending_questions
            )
            # Counted per event: prompt ids restart on every execution.
            return replace(
                state,
                pending_questions=new_q,
                unattended_questions=state.unattended_questions + (1 if unseen else 0),
            )

        case events.PinAdded(text=text):
            return replace(state, pins=(*state.pins, text))

        case events.PinsRestored(pins=pins):
            return replace(state, pins=pins)

        case events.CompactRestored(elided=elided, gists=gists):
            return replace(state, compact_elided=elided, compact_gists_live=gists)

        case events.CompactDropped(n=n):
            return replace(state, compact_elided=state.compact_elided + n)

        case events.CompactGists(gisted=gisted, demoted=demoted):
            live = max(0, state.compact_gists_live + gisted - demoted)
            return replace(state, compact_gists_live=live)

        case events.CompactSummarised():
            return replace(state, compact_elided=0, compact_gists_live=0)

        case events.SteerRequested():
            return replace(state, steer_requests=state.steer_requests + 1)

        case events.SessionEnd(all_passed=all_passed, reason=reason, scoped=scoped):
            return replace(
                state,
                finished=True,
                all_passed=all_passed,
                end_reason=reason,
                verify_scoped=scoped,
            )

        case events.SessionUndone(new_session_id=new_id, undone_text=text):
            return replace(state, undone_to=new_id, undone_text=text)

        case events.RawEvent():
            return state


def task_tree_views(nodes: dict[str, Any], cursor: str | None) -> tuple[TaskNodeView, ...]:
    """Flatten the task node map into DFS pre-order with depths, for an indented tree.

    Args:
        nodes: The `graph.update` node map.
        cursor: The id of the task in progress.

    Returns:
        The views: roots first in sorted id order, each followed by its children in
        recorded order; a node whose parent is missing follows the roots; cycles and
        duplicates are visited once.
    """
    out: list[TaskNodeView] = []
    seen: set[str] = set()

    def visit(nid: str, depth: int) -> None:
        """Append the node and its subtree, skipping a malformed or already-seen node."""
        node = nodes.get(nid)
        if not isinstance(node, dict) or nid in seen:
            return
        seen.add(nid)
        created_by, standing = str(node.get("created_by", "")), bool(node.get("standing", False))
        parent_id = node.get("parent_id")
        out.append(
            TaskNodeView(
                id=nid,
                title=str(node.get("title", "")),
                status=node.get("status", "pending"),
                depth=depth,
                is_cursor=(nid == cursor),
                created_by=created_by,
                standing=standing,
                note=owner_note(
                    created_by=created_by,
                    parent_id=str(parent_id) if isinstance(parent_id, str) else None,
                    standing=standing,
                ),
                short_id=short_task_id(nid),
            )
        )
        children = node.get("children", ())
        if not isinstance(children, (list, tuple)):
            return
        for child in children:
            visit(str(child), depth + 1)

    # Sorted like `tree_order`, so the roots read in the same order on every surface.
    roots = [
        nid
        for nid in sorted(nodes)
        if not isinstance((n := nodes[nid]), dict) or n.get("parent_id") is None
    ]
    for nid in roots:
        visit(nid, 0)
    for nid in sorted(nodes):
        visit(nid, 0)
    return tuple(out)


def _push_bounded[T](existing: tuple[T, ...], item: T, cap: int) -> tuple[T, ...]:
    """Return the tuple with the item appended, the oldest dropped past the cap."""
    new = (*existing, item)
    if len(new) > cap:
        return new[-cap:]
    return new


def fold_session(events: Iterable[dict[str, Any]]) -> SessionState:
    """Fold a session's whole event stream into one state.

    Args:
        events: The raw events, in order.

    Returns:
        The state after the last event.
    """
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return state


def open_approval_of(
    state: SessionState, *, taken: Callable[[str], bool] = lambda _aid: False
) -> ApprovalPrompt | None:
    """Return the approval a surface answers now.

    The one rule behind the server's answer route, the run views' docked row and the
    web's box, so no surface offers an approval the run will refuse.

    Args:
        state: The folded state.
        taken: Whether the surface already took the approval with that id.

    Returns:
        The oldest unanswered approval not yet taken, or None.
    """
    return next(
        (ap for ap in state.pending_approvals if not ap.answered and not taken(ap.id)), None
    )


def open_approval(session_dir: Path) -> ApprovalPrompt | None:
    """Return the run's open approval from its journal, or None when none is open."""
    from agent6.viewmodel.tail import tail_events  # noqa: PLC0415  # cycle at import time

    return open_approval_of(fold_session(tail_events(session_dir / LOGS_NAME, follow=False)))


def open_question(session_dir: Path) -> QuestionPrompt | None:
    """Return the run's oldest unanswered `ask_user` prompt, or None when none is open.

    Every surface that writes an answer file checks against it, so an answer list of
    the wrong length is refused rather than thrown away by the asking side.
    """
    from agent6.viewmodel.tail import tail_events  # noqa: PLC0415  # cycle at import time

    state = fold_session(tail_events(session_dir / LOGS_NAME, follow=False))
    return next((q for q in state.pending_questions if not q.answered), None)


def fold_until_commit(events: Iterable[dict[str, Any]], sha: str) -> SessionState | None:
    """Fold the state as of one of the run's commits.

    Args:
        events: The raw events, in order.
        sha: The commit's full sha or a prefix of at least 7 hex digits; the first
            commit it matches wins.

    Returns:
        The state after that commit's `loop.auto_commit`, or None when no commit has it.
    """
    if len(sha) < 7:
        return None
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
        if state.steps and state.steps[-1].sha.startswith(sha):
            return state
    return None


def status_facts(state: SessionState) -> StatusFacts:
    """Return the fold's answers to the status questions.

    The typed twin of `LogScan.status_facts`; the two agree on the same log.

    Args:
        state: The folded state.

    Returns:
        The facts `status_for_session_dir` reads.
    """
    pending: list[tuple[str, float | None]] = [
        ("approval", a.asked_ep) for a in state.pending_approvals if not a.answered
    ] + [("question", q.asked_ep) for q in state.pending_questions if not q.answered]
    oldest = min(pending, key=lambda p: p[1] if p[1] is not None else float("inf"), default=None)
    return StatusFacts(
        started=state.started,
        finished=state.finished,
        all_passed=state.all_passed,
        verify_scoped=state.verify_scoped,
        gate_red=state.last_verify is not None
        and state.last_verify.exit_code is not None
        and state.last_verify.exit_code != 0,
        end_reason=state.end_reason,
        operator_blocked=bool(pending),
        blocked_kind=oldest[0] if oldest else "",
        blocked_since_ep=oldest[1] if oldest else None,
        unattended_questions=state.unattended_questions,
    )


@functools.lru_cache(maxsize=64)
def _window(provider: str, model: str) -> int | None:
    """Return the model's context window, memoised since every surface asks per heartbeat."""
    return context_window(provider, model)


def context_fill(state: SessionState) -> int | None:
    """Return the context-window fill in percent at the last completed model call.

    The one rule behind every surface's `ctx N%` readout.

    Args:
        state: The folded state.

    Returns:
        The call's full prompt tokens over the model's window, capped at 100; None
        until both sides are known.
    """
    role = state.last_role
    if role is None or role.ctx_tokens <= 0 or not role.model:
        return None
    window = _window(role.provider, role.model)
    if not window:
        return None
    return min(100, round(100 * role.ctx_tokens / window))


def session_state_as_dict(state: SessionState, session_dir: Path | None = None) -> dict[str, Any]:
    """Return the wire form of a `SessionState`, what `attach --json` and the web serialize.

    Args:
        state: The folded state.
        session_dir: The session's state dir; with it `status` is the dir-aware word,
            `live` says whether a steer or stop would reach anything, `ports` and
            `shells` are live probes, a plan's `plan_md` is its deliverable, and the
            dir fills the identity the fold left empty. Without it `live` is None,
            right only for a dir-less stream (the machine reasoning snapshot).

    Returns:
        The state's fields with tuples as lists, plus the computed `context_pct`,
        `needs_new_work`, `open_approval`, `status`, `status_label`, `task_line`,
        `dead_state`, `operator_blocked`, the rendered budget text, approval parts and
        step labels, and `log_tail` as plain strings.
    """
    d = asdict(state)
    d["context_pct"] = context_fill(state)
    d["needs_new_work"] = needs_new_work(
        finished=state.finished, end_reason=state.end_reason, all_passed=state.all_passed
    )
    d["budget"]["usd_text"] = budget_usd_text(
        state.budget.usd_total,
        partial=state.budget.usd_partial,
        usd_cap=state.budget.usd_cap,
        usd_prior_executions=state.budget.usd_prior_executions,
    )
    for ap, row in zip(state.pending_approvals, d["pending_approvals"], strict=True):
        row["head"], row["payload"] = approval_parts(ap.prompt)
    for step, row in zip(state.steps, d["steps"], strict=True):
        row["label"] = step.label
    current = open_approval_of(state)
    d["open_approval"] = None if current is None else current.id
    if session_dir is not None:
        word, reason = status_for_session_dir(session_dir, status_facts(state))
        d["live"] = word in LIVE_STATUS_WORDS
        d["policy"] = session_policy(session_dir).line()
        # A forked execution's log opens at loop.resume.start and folds its identity empty.
        d["session_id"] = d["session_id"] or session_dir.name
        d["mode"] = d.get("mode") or ""
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            d["user_task"] = d["user_task"] or manifest.user_task
            d["mode"] = d["mode"] or manifest.mode
        d["ports"] = listening_ports(session_dir)
        # On every frame: the run view streams from this dict.
        d["shells"] = roster_from_dir(session_dir / SHELLS_DIR)
        if d["mode"] == "plan":
            with contextlib.suppress(OSError):
                d["plan_md"] = (session_dir / "plan.md").read_text(encoding="utf-8")
    else:
        d["live"] = None
        word, reason = status_word(
            finished=state.finished,
            all_passed=state.all_passed,
            end_reason=state.end_reason,
            scoped=state.verify_scoped,
            gate_red=status_facts(state).gate_red,
        )
    d["status"] = word
    d["status_label"] = status_label(word, reason)
    d["task_line"] = task_snippet(d["user_task"])
    d["dead_state"] = dead_run_note(word, reason)[0]
    # From the fold, so a dir-less consumer still gets the "blocked, not working" signal.
    d["operator_blocked"] = status_facts(state).operator_blocked
    d["log_tail"] = [line.text for line in state.log_tail]
    return d
