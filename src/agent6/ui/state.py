# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure event-fold: list[event_dict] -> RunState.

Mirroring this in TypeScript (for the VS Code extension) is the intended
extension path. The shape of `RunState` IS the data contract for any
external viewer; keep field names stable.

No I/O, no textual, no async — just dataclasses and a `apply_event`
function that returns a new `RunState` (frozen so the TUI can rely on
"if state is state_prev, nothing changed").
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

StepStatus = Literal["pending", "running", "passed", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class StepView:
    index: int
    title: str
    status: StepStatus = "pending"
    commit_sha: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallView:
    name: str
    args_preview: str  # rendered, not raw dict
    result_summary: str = ""
    ok: bool | None = None  # None = in-flight


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
class RunState:
    run_id: str = ""
    user_task: str = ""
    plan_summary: str = ""
    steps: tuple[StepView, ...] = ()
    current_step_index: int | None = None
    last_role: RoleCall | None = None
    tool_calls: tuple[ToolCallView, ...] = ()  # most-recent-last, bounded
    last_verify: VerifyView | None = None
    budget: BudgetView = field(default_factory=BudgetView)
    pending_approvals: tuple[ApprovalPrompt, ...] = ()
    log_tail: tuple[str, ...] = ()  # most-recent-last, bounded
    log_count: int = 0  # monotonic total log lines ever (log_tail is windowed)
    finished: bool = False
    all_passed: bool | None = None
    diffs: dict[int, str] = field(default_factory=dict)  # step_index -> patch text
    # Monotonic count of mid-run steer requests (Ctrl-C). The TUI compares it
    # against its own "seen" count to pop a steer modal exactly once per press.
    steer_requests: int = 0


def initial_state() -> RunState:
    return RunState()


_MAX_TOOL_HISTORY = 50
_MAX_LOG_TAIL = 400


def apply_event(state: RunState, event: dict[str, Any]) -> RunState:  # noqa: PLR0911, PLR0912, PLR0915
    """Fold one event into the run state. Pure function."""
    etype = event.get("type", "")
    log_line = _format_log_line(event)
    new_log = _push_bounded(state.log_tail, log_line, _MAX_LOG_TAIL)
    # log_count is monotonic; log_tail is a sliding window. A live viewer must
    # diff on the count (which keeps growing) -- diffing on len(log_tail) freezes
    # the panel once the window saturates at _MAX_LOG_TAIL.
    state = replace(state, log_tail=new_log, log_count=state.log_count + 1)

    match etype:
        case "run.start":
            return replace(state, user_task=str(event.get("user_task", "")))

        case "plan.ready":
            steps_raw = event.get("steps", []) or []
            steps = tuple(StepView(index=i + 1, title=str(t)) for i, t in enumerate(steps_raw))
            return replace(
                state,
                plan_summary=str(event.get("summary", "")),
                steps=steps,
            )

        case "step.start":
            idx = int(event.get("index", 0))
            updated = _update_step(state.steps, idx, status="running")
            return replace(state, steps=updated, current_step_index=idx)

        case "step.end":
            idx = int(event.get("index", 0))
            status_raw = str(event.get("status", "failed"))
            valid_statuses: set[str] = {
                "pending",
                "running",
                "passed",
                "failed",
                "skipped",
            }
            status: StepStatus = (
                cast("StepStatus", status_raw) if status_raw in valid_statuses else "failed"
            )
            updated = _update_step(
                state.steps,
                idx,
                status=status,
                commit_sha=str(event.get("commit_sha", "")),
                notes=str(event.get("notes", "")),
            )
            return replace(state, steps=updated)

        case "step.diff":
            idx = int(event.get("index", 0))
            new_diffs = dict(state.diffs)
            new_diffs[idx] = str(event.get("patch", ""))
            return replace(state, diffs=new_diffs)

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
                last_role=replace(last, streamed_text=last.streamed_text + piece),
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
                last_role=replace(last, streamed_thinking=last.streamed_thinking + piece),
            )

        case "role.result":
            last = state.last_role
            if last is None:
                return state
            return replace(state, last_role=replace(last, in_flight=False))

        case "tool.call":
            tc = ToolCallView(
                name=str(event.get("name", "")),
                args_preview=_render_args(event.get("args", {}) or {}),
                ok=None,
            )
            return replace(
                state,
                tool_calls=_push_bounded(state.tool_calls, tc, _MAX_TOOL_HISTORY),
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

        case "run.steer_requested":
            return replace(state, steer_requests=state.steer_requests + 1)

        case "run.end":
            return replace(
                state,
                finished=True,
                all_passed=bool(event.get("all_passed", False)),
            )

        case _:
            return state


def _update_step(steps: tuple[StepView, ...], index: int, **changes: Any) -> tuple[StepView, ...]:
    return tuple(replace(s, **changes) if s.index == index else s for s in steps)


def _push_bounded[T](existing: tuple[T, ...], item: T, cap: int) -> tuple[T, ...]:
    new = (*existing, item)
    if len(new) > cap:
        return new[-cap:]
    return new


def _render_args(args: dict[str, Any]) -> str:
    pairs: list[str] = []
    for k, v in args.items():
        s = repr(v) if not isinstance(v, str) else v
        if len(s) > 80:
            s = s[:80] + "…"
        pairs.append(f"{k}={s}")
    return ", ".join(pairs)


def _format_log_line(event: dict[str, Any]) -> str:
    ts = str(event.get("ts", ""))
    etype = str(event.get("type", "?"))
    # Compact one-line representation: timestamp, type, salient field.
    salient = ""
    match etype:
        case "step.start" | "step.end":
            idx = event.get("index")
            title = event.get("title", "")
            status = event.get("status", "")
            salient = f"#{idx} {title} {status}".strip()
        case "tool.call":
            salient = f"{event.get('name', '')}({_render_args(event.get('args', {}) or {})})"
        case "tool.result":
            salient = f"{event.get('name', '')} ok={event.get('ok')} {event.get('summary', '')}"
        case "role.call":
            salient = f"{event.get('role', '')}/{event.get('model', '')}"
        case "role.result":
            role = event.get("role", "")
            tin = event.get("tokens_in")
            tout = event.get("tokens_out")
            salient = f"{role} in={tin} out={tout}"
        case "verify.end":
            salient = f"exit={event.get('exit_code')} dur={event.get('duration_s')}s"
        case "approval.prompt":
            salient = str(event.get("prompt", ""))[:80]
        case "approval.answer":
            salient = f"id={event.get('id')} approved={event.get('approved')}"
        case "run.end":
            salient = f"all_passed={event.get('all_passed')}"
        case _:
            salient = ""
    line = f"{ts[11:23] if len(ts) > 23 else ts}  {etype:<18}"
    return f"{line} {salient}" if salient else line
