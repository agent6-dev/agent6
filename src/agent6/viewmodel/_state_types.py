# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Frozen records for the session read model."""

from __future__ import annotations

import dataclasses
from typing import Literal

from agent6.viewmodel import _session_status, format, review

NodeStatus = Literal["pending", "in_progress", "passed", "failed", "skipped", "obsolete"]


@dataclasses.dataclass(frozen=True, slots=True)
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
            object.__setattr__(self, "glyph", format.TASK_STATUS_GLYPH.get(status, "·"))


@dataclasses.dataclass(frozen=True, slots=True)
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


@dataclasses.dataclass(frozen=True, slots=True)
class LogLine:
    """One log line plus the task in focus when it was emitted, so a viewer can filter."""

    text: str
    task_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DiffView:
    """One auto-commit diff plus the task in focus when it landed."""

    patch: str
    task_id: str | None = None
    sha: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyView:
    """The last verify gate run.

    Attributes:
        cmd: The gate's argv.
        exit_code: The exit code, None when no exit was observed.
        outcome: The finished attempt's outcome; None while in flight.
        reason: Why the command did not run, separate from its output.
        duration_s: How long it ran.
        stdout_tail: The capped stdout tail.
        stderr_tail: The capped stderr tail.
        timeout_s: The gate's wall-clock limit, 0.0 when the event carried none.
    """

    cmd: tuple[str, ...]
    exit_code: int | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    timeout_s: float = 0.0
    outcome: str | None = None
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetView:
    """The run's spend as every surface shows it.

    Token counters and the caps are the current execution's, pairing with the
    per-execution enforcement; `usd_total` and `plan_consumed` are cumulative across
    executions, as the listing scan sums them, so the surfaces agree on what a run cost.

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
        plan_consumed: The plan points consumed across every execution.
        plan_prior_executions: The points banked by completed executions.
        plan_cap: `[budget].max_plan_points`; 0 until a plan call runs.
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
    plan_prior_executions: float = 0.0
    plan_cap: float = 0.0
    plan_resets_at: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
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
            the last `state._STREAM_TAIL` characters.
        streamed_thinking: The live reasoning, with the same lifecycle.
    """

    role: str
    model: str
    in_flight: bool
    provider: str = ""
    ctx_tokens: int = 0
    streamed_text: str = ""
    streamed_thinking: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class CommitStep:
    """One per-step commit of the run, which the dashboards select among."""

    turn: int
    sha: str
    subject: str

    @property
    def label(self) -> str:
        """The step picker's line: `turn N · sha7 · subject`, a part left out when empty."""
        return " · ".join(p for p in (f"turn {self.turn}", self.sha[:7], self.subject) if p)


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


@dataclasses.dataclass(frozen=True, slots=True)
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


@dataclasses.dataclass(frozen=True, slots=True)
class Question:
    """One question within an `ask_user` prompt.

    Attributes:
        question: The question's text.
        options: Selectable presets; the operator may also type a free-text answer.
    """

    question: str
    options: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
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


@dataclasses.dataclass(frozen=True, slots=True)
class SessionState:
    """A session's folded state, the read model every front-end paints.

    Attributes:
        session_id: The session's id, "" until an event carries it.
        mode: The session's mode, "" until known.
        task_label: The first non-empty task and latest resume steer from the journal.
        tasks: The live task tree in DFS pre-order.
        cursor_task_id: The task the worker is on.
        last_role: The last model call.
        tool_calls: The bounded tool-call history, most recent last.
        last_verify: The last verify gate run.
        review: Durable panel evidence and finding history across executions.
        budget: The spend.
        pending_approvals: Every approval asked this execution, answered or not.
        pending_questions: Every question prompt asked this execution, answered or not.
        log_tail: The bounded window of log lines, most recent last.
        log_count: The monotonic count of log lines ever; a live viewer diffs on it,
            since `len(log_tail)` freezes once the window saturates.
        tool_count: The calls this execution dispatched, a finish tool apart, as the
            conversation's end line counts them; `len(tool_calls)` is a bounded window.
        recent_diffs: The bounded auto-commit diffs, for task filtering.
        started: An execution has begun; a parked or created run has none.
        finished: A session.end was folded and no later execution began.
        all_passed: The verify tri-state; None when no verify command gated the end.
        verify_scoped: The judging gate ran scoped.
        end_reason: The session.end reason word.
        stranded: This execution finished with its edits uncommitted and no commit landed.
        end_summary: The end sentence the journal carries, "" when it carries none.
        unattended_questions: Questions the harness answered empty because nobody was
            attached, in the current execution.
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
    mode: str = ""
    task_label: _session_status.TaskLabel = dataclasses.field(
        default_factory=_session_status.TaskLabel
    )
    tasks: tuple[TaskNodeView, ...] = ()
    cursor_task_id: str | None = None
    last_role: RoleCall | None = None
    tool_calls: tuple[ToolCallView, ...] = ()
    last_verify: VerifyView | None = None
    review: review.ReviewView = dataclasses.field(default_factory=review.ReviewView)
    budget: BudgetView = dataclasses.field(default_factory=BudgetView)
    pending_approvals: tuple[ApprovalPrompt, ...] = ()
    pending_questions: tuple[QuestionPrompt, ...] = ()
    log_tail: tuple[LogLine, ...] = ()
    log_count: int = 0
    tool_count: int = 0
    recent_diffs: tuple[DiffView, ...] = ()
    started: bool = False
    finished: bool = False
    all_passed: bool | None = None
    verify_scoped: bool = False
    end_reason: str = ""
    stranded: bool = False
    end_summary: str = ""
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
