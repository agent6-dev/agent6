# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure event-fold: list[event_dict] -> RunState.

Mirroring this in TypeScript (for the VS Code extension) is the intended
extension path. The shape of `RunState` IS the data contract for any
external viewer; keep field names stable.

No I/O, no textual, no async, just dataclasses and a `apply_event`
function that returns a new `RunState` (frozen so the TUI can rely on
"if state is state_prev, nothing changed").
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from agent6.viewmodel.listing import status_word

NodeStatus = Literal["pending", "in_progress", "passed", "failed", "skipped", "obsolete"]


@dataclass(frozen=True, slots=True)
class TaskNodeView:
    """One node of the live task DAG, flattened (DFS pre-order) with a depth for
    tree rendering. Mirrors graph.models.TaskNode; fed by the `graph.update`
    snapshot the worker emits whenever it mutates its task breakdown."""

    id: str
    title: str
    status: NodeStatus = "pending"
    depth: int = 0
    is_cursor: bool = False


@dataclass(frozen=True, slots=True)
class ToolCallView:
    name: str
    args_preview: str  # rendered, per-value truncated, for the inline table
    args_full: str = ""  # rendered with a generous per-value cap, for the detail modal
    result_summary: str = ""
    ok: bool | None = None  # None = in-flight
    task_id: str | None = None  # DAG task in focus when the call ran (for filtering)


@dataclass(frozen=True, slots=True)
class LogLine:
    """One audit-log line plus the DAG task in focus when it was emitted, so a
    viewer can filter the log to a selected task."""

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
    cmd: tuple[str, ...]
    exit_code: int | None = None  # None = in-flight
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True, slots=True)
class BudgetView:
    input_total: int = 0
    output_total: int = 0
    input_cap: int = 0
    output_cap: int = 0
    per_model_tokens: dict[str, int] = field(default_factory=dict)
    usd_total: float = 0.0
    usd_partial: bool = False  # True if some models had no price (under-estimate)


@dataclass(frozen=True, slots=True)
class RoleCall:
    role: str
    model: str
    in_flight: bool
    # Live SSE text accumulator. Reset on every role.call,
    # appended-to on each role.text_delta, frozen on role.result.
    streamed_text: str = ""
    # Live reasoning accumulator, fed by role.thinking_delta. Same
    # lifecycle as streamed_text; shown in the TUI's "thinking" view so a
    # long reasoning burst reads as progress rather than a hang.
    streamed_thinking: str = ""


@dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    id: str
    prompt: str
    answered: bool = False
    approved: bool | None = None


@dataclass(frozen=True, slots=True)
class Question:
    """One question within an `ask_user` prompt. `options` are selectable presets;
    the user may also type a free-text answer."""

    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionPrompt:
    """An agent->user `ask_user` prompt: one or more related questions the operator
    answers together (reviewing before submitting). `answers` align to `questions`."""

    id: str
    questions: tuple[Question, ...] = ()
    answered: bool = False
    answers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunState:
    run_id: str = ""
    user_task: str = ""
    tasks: tuple[TaskNodeView, ...] = ()  # live task DAG, DFS pre-order
    cursor_task_id: str | None = None
    last_role: RoleCall | None = None
    tool_calls: tuple[ToolCallView, ...] = ()  # most-recent-last, bounded
    last_verify: VerifyView | None = None
    budget: BudgetView = field(default_factory=BudgetView)
    pending_approvals: tuple[ApprovalPrompt, ...] = ()
    pending_questions: tuple[QuestionPrompt, ...] = ()
    log_tail: tuple[LogLine, ...] = ()  # most-recent-last, bounded
    log_count: int = 0  # monotonic total log lines ever (log_tail is windowed)
    recent_diffs: tuple[DiffView, ...] = ()  # auto-commit diffs, bounded, for task filtering
    finished: bool = False
    all_passed: bool | None = None
    end_reason: str = ""  # run.end reason: finish_run | steer_abort | provider_error | ...
    finish_summary: str = ""  # the finish tool's summary: the agent's closing statement
    latest_diff: str = ""  # patch of the most recent auto-commit (diff.updated)
    # Monotonic count of mid-run steer requests (Ctrl-C). The TUI compares it
    # against its own "seen" count to pop a steer modal exactly once per press.
    steer_requests: int = 0


def initial_state() -> RunState:
    return RunState()


_MAX_TOOL_HISTORY = 50
_MAX_DIFF_HISTORY = 30  # auto-commit diffs retained for per-task filtering
MAX_LOG_TAIL = 400  # public: the inline log RichLog caps to this so it stays a gapless window
# Live streamed reasoning/text is the frontier of an in-flight call; keep only the
# tail so a 25k-char reasoning burst doesn't bloat every SSE frame or re-render.
# The full turn is preserved in the transcript, which the conversation view folds.
_STREAM_TAIL = 6000

# Streaming deltas are ephemeral live-view events -- the reasoning shows in the
# stream/conversation panes as it arrives. They are NOT audit-log events, so the
# log_tail and the full LogScreen skip them; otherwise a reasoning model floods the
# log with thousands of contentless "role.thinking_delta" lines.
STREAM_DELTA_EVENTS = frozenset({"role.thinking_delta", "role.text_delta"})
# Loop-side mirrors of events already rendered (tool.call carries the args,
# budget.update the totals); they doubled every tool call and budget tick in
# the log view without adding a field worth reading.
LOG_NOISE_EVENTS = frozenset({"loop.tool.call", "loop.budget"})


def apply_event(state: RunState, event: dict[str, Any]) -> RunState:  # noqa: PLR0911, PLR0912, PLR0915
    """Fold one event into the run state. Pure function."""
    etype = event.get("type", "")
    if not state.run_id and event.get("run_id"):
        state = replace(state, run_id=str(event["run_id"]))
    if etype not in STREAM_DELTA_EVENTS and etype not in LOG_NOISE_EVENTS:
        # Deltas are live-stream only; noise mirrors add no readable field.
        # cursor_task_id is the focus task (graph.update lands before a turn's calls).
        entry = LogLine(format_log_line(event), state.cursor_task_id)
        new_log = _push_bounded(state.log_tail, entry, MAX_LOG_TAIL)
        # log_count is monotonic; log_tail is a sliding window. A live viewer must
        # diff on the count (which keeps growing) -- diffing on len(log_tail) freezes
        # the panel once the window saturates at MAX_LOG_TAIL.
        state = replace(state, log_tail=new_log, log_count=state.log_count + 1)

    match etype:
        case "run.start":
            return replace(state, user_task=str(event.get("user_task", "")))

        case "loop.resume.start":
            # A resume restarts a finished/stopped run in place (it appends to the
            # same log): it is running again, so clear the terminal state.
            return replace(state, finished=False, end_reason="")

        case "graph.update":
            nodes = event.get("nodes", {}) or {}
            cursor = event.get("cursor")
            return replace(
                state,
                tasks=_build_task_tree(nodes, cursor if isinstance(cursor, str) else None),
                cursor_task_id=cursor if isinstance(cursor, str) else None,
            )

        case "diff.updated":
            patch = str(event.get("patch", ""))
            entry = DiffView(
                patch=patch, task_id=state.cursor_task_id, sha=str(event.get("sha", ""))
            )
            return replace(
                state,
                latest_diff=patch,
                recent_diffs=_push_bounded(state.recent_diffs, entry, _MAX_DIFF_HISTORY),
            )

        case "role.call":
            return replace(
                state,
                last_role=RoleCall(
                    role=str(event.get("role", "")),
                    model=str(event.get("model", "")),
                    in_flight=True,
                    streamed_text="",
                    streamed_thinking="",
                ),
            )

        case "role.text_delta":
            # Append SSE delta to the in-flight RoleCall.
            last = state.last_role
            if last is None or not last.in_flight:
                return state
            piece = str(event.get("text", ""))
            if not piece:
                return state
            return replace(
                state,
                last_role=replace(last, streamed_text=(last.streamed_text + piece)[-_STREAM_TAIL:]),
            )

        case "role.thinking_delta":
            # Append a reasoning delta to the in-flight RoleCall.
            last = state.last_role
            if last is None or not last.in_flight:
                return state
            piece = str(event.get("text", ""))
            if not piece:
                return state
            return replace(
                state,
                last_role=replace(
                    last, streamed_thinking=(last.streamed_thinking + piece)[-_STREAM_TAIL:]
                ),
            )

        case "role.result":
            last = state.last_role
            if last is None:
                return state
            return replace(state, last_role=replace(last, in_flight=False))

        case "tool.call":
            raw_args = event.get("args", {}) or {}
            name = str(event.get("name", ""))
            tc = ToolCallView(
                name=name,
                args_preview=_render_args(raw_args),
                args_full=_render_args(raw_args, max_value=4000),
                ok=None,
                task_id=state.cursor_task_id,
            )
            # The finish tools' summary is the agent's closing statement; keep it
            # so an ended run's panes can render the end story, not a dead one.
            finish_summary = state.finish_summary
            if name in ("finish_run", "finish_planning") and isinstance(raw_args, dict):
                finish_summary = str(raw_args.get("summary", "")).strip() or finish_summary
            return replace(
                state,
                tool_calls=_push_bounded(state.tool_calls, tc, _MAX_TOOL_HISTORY),
                finish_summary=finish_summary,
            )

        case "tool.result":
            if not state.tool_calls:
                return state
            last = state.tool_calls[-1]
            if last.name != str(event.get("name", "")):
                return state
            updated_last = replace(
                last,
                ok=bool(event.get("ok", False)),
                result_summary=str(event.get("summary", "")),
            )
            return replace(
                state,
                tool_calls=(*state.tool_calls[:-1], updated_last),
            )

        case "verify.start":
            cmd = tuple(str(x) for x in event.get("cmd", []) or [])
            return replace(state, last_verify=VerifyView(cmd=cmd))

        case "verify.end":
            cmd = tuple(str(x) for x in event.get("cmd", []) or [])
            return replace(
                state,
                last_verify=VerifyView(
                    cmd=cmd,
                    exit_code=int(event.get("exit_code", -1)),
                    duration_s=float(event.get("duration_s", 0.0)),
                    stdout_tail=str(event.get("stdout_tail", "")),
                    stderr_tail=str(event.get("stderr_tail", "")),
                ),
            )

        case "budget.update":
            per_model = event.get("per_model_tokens", {}) or {}
            return replace(
                state,
                budget=BudgetView(
                    input_total=int(event.get("input_total", 0)),
                    output_total=int(event.get("output_total", 0)),
                    input_cap=int(event.get("input_cap", 0)),
                    output_cap=int(event.get("output_cap", 0)),
                    per_model_tokens={str(k): int(v) for k, v in per_model.items()},
                    usd_total=float(event.get("usd_total", 0.0)),
                    usd_partial=bool(event.get("usd_partial", False)),
                ),
            )

        case "approval.prompt":
            ap = ApprovalPrompt(
                id=str(event.get("id", "")),
                prompt=str(event.get("prompt", "")),
            )
            return replace(
                state,
                pending_approvals=(*state.pending_approvals, ap),
            )

        case "approval.answer":
            wanted_id = str(event.get("id", ""))
            new = tuple(
                replace(a, answered=True, approved=bool(event.get("approved", False)))
                if a.id == wanted_id
                else a
                for a in state.pending_approvals
            )
            return replace(state, pending_approvals=new)

        case "question.prompt":
            raw_qs = event.get("questions", ()) or ()
            questions = tuple(
                Question(
                    question=str(q.get("question", "")),
                    options=tuple(str(o) for o in (q.get("options", ()) or ())),
                )
                for q in raw_qs
                if isinstance(q, dict)
            )
            qp = QuestionPrompt(id=str(event.get("id", "")), questions=questions)
            return replace(state, pending_questions=(*state.pending_questions, qp))

        case "question.answer":
            wanted = str(event.get("id", ""))
            raw_ans = event.get("answers", ()) or ()
            answers = tuple(str(a) for a in raw_ans) if isinstance(raw_ans, (list, tuple)) else ()
            new_q = tuple(
                replace(q, answered=True, answers=answers) if q.id == wanted else q
                for q in state.pending_questions
            )
            return replace(state, pending_questions=new_q)

        case "run.steer_requested":
            return replace(state, steer_requests=state.steer_requests + 1)

        case "run.end":
            return replace(
                state,
                finished=True,
                all_passed=bool(event.get("all_passed", False)),
                end_reason=str(event.get("reason", "") or ""),
            )

        case _:
            return state


def _build_task_tree(nodes: dict[str, Any], cursor: str | None) -> tuple[TaskNodeView, ...]:
    """Flatten the curator's node map into a DFS pre-order list with depths, so
    the TUI can render the DAG as an indented tree. Roots are nodes with no
    parent (or whose parent is missing); children follow their parent's recorded
    order. Cycles/dupes are guarded by a visited set."""
    out: list[TaskNodeView] = []
    seen: set[str] = set()

    def visit(nid: str, depth: int) -> None:
        node = nodes.get(nid)
        # isinstance (not `is None`) so a malformed non-dict value is skipped
        # rather than crashing .get(), consistent with the roots filter below.
        if not isinstance(node, dict) or nid in seen:
            return
        seen.add(nid)
        out.append(
            TaskNodeView(
                id=nid,
                title=str(node.get("title", "")),
                status=node.get("status", "pending"),
                depth=depth,
                is_cursor=(nid == cursor),
            )
        )
        for child in node.get("children", ()) or ():
            visit(str(child), depth + 1)

    roots = [
        nid
        for nid, n in nodes.items()
        if not isinstance(n, dict) or n.get("parent_id") is None or n.get("parent_id") not in nodes
    ]
    for nid in roots:
        visit(nid, 0)
    # Any node not reachable from a root (shouldn't happen) still gets shown.
    for nid in nodes:
        visit(nid, 0)
    return tuple(out)


def _push_bounded[T](existing: tuple[T, ...], item: T, cap: int) -> tuple[T, ...]:
    new = (*existing, item)
    if len(new) > cap:
        return new[-cap:]
    return new


def _render_arg_value(key: str, value: Any) -> str:
    """One arg value, human-shaped: argv as a shell line, ask_user's questions as
    their text, everything else as its string / repr."""
    if key == "argv" and isinstance(value, (list, tuple)) and value:
        return shlex.join(str(a) for a in value)
    if key == "questions" and isinstance(value, (list, tuple)) and value:
        first = value[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        return str(q) + (f" (+{len(value) - 1})" if len(value) > 1 else "")
    return value if isinstance(value, str) else repr(value)


def _render_args(args: dict[str, Any], *, max_value: int = 80) -> str:
    """Render an args dict as `k=v, ...`, truncating each value to *max_value*
    chars. The inline table uses the tight default; the detail modal renders with
    a generous cap so a long arg (a command, a path, a payload) is readable while
    one pathological value still can't bloat the bounded history."""
    pairs: list[str] = []
    for k, v in args.items():
        s = _render_arg_value(k, v)
        if len(s) > max_value:
            s = s[:max_value] + "…"
        pairs.append(f"{k}={s}")
    return ", ".join(pairs)


def format_log_line(event: dict[str, Any]) -> str:  # noqa: PLR0912, PLR0915
    ts = str(event.get("ts", ""))
    etype = str(event.get("type", "?"))
    # Compact one-line representation: timestamp, type, salient field.
    salient = ""
    match etype:
        case "graph.update":
            nodes = event.get("nodes", {})
            salient = f"{len(nodes)} tasks" if isinstance(nodes, dict) else ""
        case "diff.updated":
            salient = f"{len(str(event.get('patch', '')).splitlines())} lines"
        case "tool.call":
            salient = f"{event.get('name', '')}({_render_args(event.get('args', {}) or {})})"
        case "tool.result":
            salient = f"{event.get('name', '')} ok={event.get('ok')} {event.get('summary', '')}"
            # Execution tools carry capped output tails; show a one-line hint of
            # the latest stderr (else stdout) so a command's outcome reads in the
            # log without opening the transcript. The full tail is in the event.
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "")
            snippet = " ".join(tail.split())[:100]
            if snippet:
                salient = f"{salient.rstrip()} | {snippet}"
        case "role.call":
            salient = f"{event.get('role', '')}/{event.get('model', '')}"
        case "role.result":
            role = event.get("role", "")
            if event.get("error"):
                # The error is the load-bearing field on a failed turn: this
                # line is how a dead run gets diagnosed from the log view.
                salient = f"{role} error: {str(event.get('error'))[:160]}"
            else:
                tin = event.get("tokens_in")
                tout = event.get("tokens_out")
                salient = f"{role} in={tin} out={tout}"
        case "loop.provider.retry":
            salient = f"attempt {event.get('attempt')}: {str(event.get('error', ''))[:160]}"
        case "loop.resume.start":
            salient = f"iteration={event.get('iteration')} messages={event.get('messages')}"
        case "budget.update":
            salient = (
                f"in={event.get('input_total')} out={event.get('output_total')}"
                f" ${event.get('usd_total')}"
            )
        case "run.start":
            salient = str(event.get("user_task", ""))[:80]
        case "verify.end":
            salient = f"exit={event.get('exit_code')} dur={event.get('duration_s')}s"
        case "approval.prompt":
            salient = str(event.get("prompt", ""))[:80]
        case "approval.answer":
            salient = f"id={event.get('id')} approved={event.get('approved')}"
        case "question.prompt":
            qs = event.get("questions", []) or []
            first = str(qs[0].get("question", "")) if qs and isinstance(qs[0], dict) else ""
            salient = (f"[{len(qs)}] " if len(qs) > 1 else "") + first[:80]
        case "question.answer":
            ans = event.get("answers", []) or []
            salient = f"id={event.get('id')} answers={len(ans)}"
        case "run.end":
            salient = f"{event.get('reason', '')} all_passed={event.get('all_passed')}"
        case _:
            salient = ""
    line = f"{ts[11:23] if len(ts) > 23 else ts}  {etype:<18}"
    return f"{line} {salient}" if salient else line


def fold_run(events: Iterable[dict[str, Any]]) -> RunState:
    """Reduce a run's whole event stream to one RunState (apply_event from the
    initial state). The snapshot a one-shot viewer or the JSON wire form builds
    on; the TUI folds incrementally and a CLI tail renders line-by-line instead."""
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return state


def run_status_label(state: RunState) -> str:
    """The header status word, distinguishing a stop from a finish from an error --
    all three set finished=True, so the reason is what tells them apart. A user who
    stopped a run must not see a bare 'finished' (which reads as 'it completed').
    The decision lives in ``viewmodel.summary.status_word`` so listings and
    headers can never disagree about how a run ended."""
    word, reason = status_word(
        finished=state.finished, all_passed=bool(state.all_passed), end_reason=state.end_reason
    )
    labels = {
        "running": "running",
        "stopped": "stopped",
        "passed": "finished · all passed",
        "finished": "finished",
    }
    return labels.get(word) or f"ended · {reason.replace('_', ' ')}"


def run_state_as_dict(state: RunState) -> dict[str, Any]:
    """The JSON-able wire form of a RunState, stable field names: what
    `agent6 watch --json` and a web client serialize. Tuples become lists, nested
    view dataclasses become dicts. `status_label` is a computed convenience the
    web/CLI render verbatim so the label logic lives in one place."""
    d = asdict(state)
    d["status_label"] = run_status_label(state)
    # log_tail is LogLine objects now; the wire form stays a flat list of strings
    # (web + `watch --json` consumers render lines verbatim). task_id filtering is
    # a TUI-local concern that reads the RunState directly.
    d["log_tail"] = [line.text for line in state.log_tail]
    return d
