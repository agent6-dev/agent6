# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Frozen records for the session read model."""

from __future__ import annotations

import dataclasses
from typing import Literal

from agent6 import budget, evidence, kinds
from agent6.viewmodel import format, review, session_status

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
    """One log line plus the task in focus when it was emitted, so a viewer can filter.

    Attributes:
        text: The rendered log line.
        task_id: The task in focus when the line was emitted, or null.
    """

    text: str
    task_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DiffView:
    """One auto-commit diff plus the task in focus when it landed.

    Attributes:
        patch: The auto-commit's unified patch.
        task_id: The task in focus when the commit landed, or null.
        sha: The auto-commit identifier.
    """

    patch: str
    task_id: str | None = None
    sha: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyView:
    """The last verify gate run.

    Attributes:
        command: The gate's argv.
        exit_code: The exit code, None when no exit was observed.
        outcome: The finished attempt's outcome; None while in flight.
        reason: Why the command did not run, separate from its output.
        duration_s: How long it ran.
        stdout_tail: The capped stdout tail.
        stderr_tail: The capped stderr tail.
        timeout_s: The gate's wall-clock limit, 0.0 when the event carried none.
    """

    command: tuple[str, ...]
    exit_code: int | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    timeout_s: float = 0.0
    outcome: str | None = None
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionUsage:
    """The current accounting lifetime's usage beside the caps that bound it.

    The lifetime is the tracker's: an execution's, or the whole REPL conversation's, which
    keeps one tracker across its starts. A cap is None until a `budget.update` carries it;
    a recorded -1 means unlimited and a recorded 0 means refused.

    Attributes:
        input_tokens: Input tokens; None until a budget row carries the counters.
        output_tokens: Output tokens; None until a budget row carries the counters.
        cache_read_tokens: The cached side of the input; None when the row lacked it.
        cache_write_tokens: Cache-write tokens; None when the row lacked it.
        usd: The metered spend.
        usd_partial: Some model had no price, so `usd` is a lower bound.
        unmetered_tokens: Input and output tokens of calls the meter could not price.
        plan_observed_points: The plan movement observed, in points.
        usd_cap: `[budget].max_usd`.
        tokens_fallback_cap: `[budget].max_tokens_fallback`.
        plan_cap: `[budget].max_plan_points`.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    usd: float = 0.0
    usd_partial: bool = False
    unmetered_tokens: int = 0
    plan_observed_points: float = 0.0
    usd_cap: float | None = None
    tokens_fallback_cap: int | None = None
    plan_cap: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SessionUsage:
    """The session's usage across every execution, what the listings' usage cell shows.

    Attributes:
        usd: The spend across every execution.
        usd_partial: Some execution's model had no price, so `usd` is a lower bound.
        plan_observed_points: The plan movement observed across every execution, in points.
        unmetered_tokens: Fallback tokens across every execution, including settled descendants.
        settled_lanes: Child sessions whose spend is durably included in the journal.
    """

    usd: float = 0.0
    usd_partial: bool = False
    plan_observed_points: float = 0.0
    unmetered_tokens: int = 0
    settled_lanes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class AccountReading:
    """The latest reading of a subscription account: an observation of the account, not the run.

    Attributes:
        provider: The provider entry whose account was read; "" in a journal written before
            the reading named it.
        used_percent: The binding window's used percent.
        resets_at: When the binding window resets, as epoch seconds; 0 when unknown.
        windows: Every window the backend reported, the primary first.
        credits_has: Whether the account holds purchased credits.
        credits_unlimited: Whether the credits are unlimited.
        credits_balance: The credit balance as the backend sent it; "" when none.
        observed_at: When the reading was journaled, as epoch seconds.
    """

    provider: str = ""
    used_percent: float = 0.0
    resets_at: float = 0.0
    windows: tuple[budget.PlanWindow, ...] = ()
    credits_has: bool = False
    credits_unlimited: bool = False
    credits_balance: str = ""
    observed_at: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetView:
    """The run's usage as every surface shows it, each amount with one lifetime.

    Attributes:
        session: The usage across every execution.
        execution: The current accounting lifetime's usage beside its caps.
        account: The latest subscription account reading; None before a plan call runs.
    """

    session: SessionUsage = dataclasses.field(default_factory=SessionUsage)
    execution: ExecutionUsage = dataclasses.field(default_factory=ExecutionUsage)
    account: AccountReading | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class EnvironmentView:
    """The run's container as the journal tells it: a start in progress, then the lease held.

    Attributes:
        execution_id: The execution that started or acquired it; the lease rows precede
            `session.start`, so the start keeps a view bound to its own execution.
        starting: The start command running, () when none runs.
        role: owner or borrower; "" before the run holds the container.
        reason: Why the run holds that role.
        started: This run's start command made the container appear.
        start_s: How long the start took; None when none ran.
        outcome: The lifecycle's end (left_running, stopped, unknown); "" while the run holds it.
    """

    execution_id: str | None = None
    starting: tuple[str, ...] = ()
    role: str = ""
    reason: str = ""
    started: bool = False
    start_s: float | None = None
    outcome: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Lifecycle:
    """Where a session stands, as every fold reads it.

    Attributes:
        started: An execution has begun; a parked or created run has none.
        finished: A session.stop was folded and no later execution began.
        mode: The session's mode, "" until a start names it.
        execution_id: The current execution's id, "" until its start is read.
        executions: One plus the resumed executions.
        task_label: The first task and the latest resume steer.
        stop_reason: The session.stop reason word, "" while an execution runs.
        verification: The final verification word, or None when unknown.
        verify_scoped: The judging gate ran scoped.
        stranded: This execution finished with its edits uncommitted and no commit landed.
        end_summary: The end sentence the journal carries, "" when it carries none.
        finish_summary: The finish tool's summary, the agent's closing statement.
        undone_to: The session `/undo` forked, "" when it was never undone.
        provider_error: The last failed driving call's error, "" when none failed.
        last_verify_rc: This execution's last verify.end exit code; None before one ends.
        verify_timeout_s: The limit that verify ran out, 0.0 when it ended on its own.
        unattended_questions: This execution's questions answered empty because nobody was
            attached.
        pins: The operator's pinned instructions in force, most recent last.
        environment: The container this execution holds; None outside one.
    """

    started: bool = False
    finished: bool = False
    mode: str = ""
    execution_id: str = ""
    executions: int = 1
    task_label: session_status.TaskLabel = dataclasses.field(
        default_factory=session_status.TaskLabel
    )
    stop_reason: str = ""
    verification: kinds.Verification | None = None
    verify_scoped: bool = False
    stranded: bool = False
    end_summary: str = ""
    finish_summary: str = ""
    undone_to: str = ""
    provider_error: str = ""
    last_verify_rc: int | None = None
    verify_timeout_s: float = 0.0
    unattended_questions: int = 0
    pins: tuple[str, ...] = ()
    environment: EnvironmentView | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SessionCore:
    """The facts both folds share: the lifecycle and the usage.

    Attributes:
        lifecycle: Where the session stands.
        budget: Its usage, each amount with one lifetime.
    """

    lifecycle: Lifecycle = dataclasses.field(default_factory=Lifecycle)
    budget: BudgetView = dataclasses.field(default_factory=BudgetView)


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
class RunCommit:
    """One per-turn commit of the run, which the dashboards select among.

    Attributes:
        turn: The turn that produced the commit.
        sha: The commit identifier.
        subject: The commit's subject.
    """

    turn: int
    sha: str
    subject: str

    @property
    def label(self) -> str:
        """The commit picker's line: `turn N · sha7 · subject`, a part left out when empty."""
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
        wait: Whether the call waited for the answers, as asked.
        blocking: Whether the run is paused on it now: a waiting question in the execution that
            asked it; a queued question, or one an earlier execution asked, blocks nothing.
        source: Who answered; `late` for a queued question's answers taken at a boundary.
    """

    id: str
    questions: tuple[Question, ...] = ()
    answered: bool = False
    answers: tuple[str, ...] = ()
    from_harness: bool = False
    asked_ep: float | None = None
    wait: bool = True
    blocking: bool = True
    source: str = ""

    def kind(self) -> str:
        """Return the word for the prompt's state, the same on every surface."""
        if self.answered:
            return "answered"
        if not self.wait:
            return "queued"
        return "waiting" if self.blocking else "asked by an earlier execution"


@dataclasses.dataclass(frozen=True, slots=True)
class SessionState:
    """A session's folded state, the read model every front-end paints.

    Attributes:
        core: The lifecycle and usage both folds share; the lifecycle and `budget` fields
            below mirror it for the wire and the surfaces.
        session_id: The session's id, "" until an event carries it.
        mode: The session's mode, "" until known.
        execution_id: The current execution's id, "" until its start is read.
        task_label: The first non-empty task and latest resume steer from the journal.
        tasks: The live task tree in DFS pre-order.
        cursor_task_id: The task the worker is on.
        last_role: The last model call.
        tool_calls: The bounded tool-call history, most recent last.
        last_verify: The last verify gate run.
        review: Durable panel evidence and finding history across executions.
        targets: Validated target history and the current execution binding.
        environment: The container this execution holds, its start and its end; None outside one.
        budget: The spend.
        approvals: Every approval asked this execution, answered or not.
        questions: Every question prompt asked this execution, answered or not.
        log_tail: The bounded window of log lines, most recent last.
        log_count: The monotonic count of log lines ever; a live viewer diffs on it,
            since `len(log_tail)` freezes once the window saturates.
        tool_count: The calls this execution dispatched, a finish tool apart, as the
            conversation's end line counts them; `len(tool_calls)` is a bounded window.
        recent_diffs: The bounded auto-commit diffs, for task filtering.
        started: An execution has begun; a parked or created run has none.
        finished: A session.stop was folded and no later execution began.
        verification: The final verification word, or None when unknown.
        verify_scoped: The judging gate ran scoped.
        stop_reason: The session.stop reason word.
        stranded: This execution finished with its edits uncommitted and no commit landed.
        end_summary: The end sentence the journal carries, "" when it carries none.
        unattended_questions: Questions the harness answered empty because nobody was
            attached, in the current execution.
        undone_to: The child session /undo forked, which surfaces follow.
        undone_text: The message /undo took back, for the composer to refill.
        finish_summary: The finish tool's summary, the agent's closing statement.
        latest_diff: The patch of the most recent auto-commit.
        commits: The run's per-turn commits, oldest first.
        steer_requests: The monotonic count of mid-run steer requests; the TUI
            compares it with its own count to react once per press.
        compact_elided: The elision markers in the current context.
        compact_gists_live: The live gists in the current context; a demoted gist is
            back to a bare marker, and a tier-2 restart resets both.
        pins: The operator's /pin instructions in force, most recent last.
        last_event_ep: The last folded event's own ts, the idle anchor every
            "working… Ns" timer measures from, so replayed history reads as its true age.
    """

    core: SessionCore = dataclasses.field(default_factory=SessionCore)
    session_id: str = ""
    mode: str = ""
    execution_id: str = ""
    task_label: session_status.TaskLabel = dataclasses.field(
        default_factory=session_status.TaskLabel
    )
    tasks: tuple[TaskNodeView, ...] = ()
    cursor_task_id: str | None = None
    last_role: RoleCall | None = None
    tool_calls: tuple[ToolCallView, ...] = ()
    last_verify: VerifyView | None = None
    review: review.ReviewView = dataclasses.field(default_factory=review.ReviewView)
    environment: EnvironmentView | None = None
    targets: evidence.ExecutionTargets = dataclasses.field(
        default_factory=evidence.ExecutionTargets
    )
    budget: BudgetView = dataclasses.field(default_factory=BudgetView)
    approvals: tuple[ApprovalPrompt, ...] = ()
    questions: tuple[QuestionPrompt, ...] = ()
    log_tail: tuple[LogLine, ...] = ()
    log_count: int = 0
    tool_count: int = 0
    recent_diffs: tuple[DiffView, ...] = ()
    started: bool = False
    finished: bool = False
    verification: kinds.Verification | None = None
    verify_scoped: bool = False
    stop_reason: str = ""
    stranded: bool = False
    end_summary: str = ""
    unattended_questions: int = 0
    undone_to: str = ""
    undone_text: str = ""
    finish_summary: str = ""
    latest_diff: str = ""
    commits: tuple[RunCommit, ...] = ()
    steer_requests: int = 0
    compact_elided: int = 0
    compact_gists_live: int = 0
    pins: tuple[str, ...] = ()
    last_event_ep: float | None = None
