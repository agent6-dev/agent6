# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Type the logs.jsonl event families the session fold consumes.

The write side (`agent6.events.EventSink`) appends free-form `{"type", "ts", **fields}`
dicts and never validates; about 90 types exist. `parse_event` turns one raw dict into
one of the frozen families here, or a `RawEvent` for every other type, which the fold
drops: an old run dir folds without a crash whatever it holds.

Frozen dataclasses, not pydantic: logs.jsonl is append-only history, so each family
holds the exact coercion old run dirs were written against (`str()`, `int()`, `bool()`
with per-field defaults, `as_int`'s swallow-to-zero, the isinstance guards) in one
place, where a pydantic model would impose its own coercion and failure semantics.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from typing import Any

# A fresh run emits session.start; a resumed execution emits only loop.resume.start.
SESSION_START_EVENTS = frozenset({"session.start", "loop.resume.start"})


def event_epoch(value: object) -> float | None:
    """Return an event `ts` as epoch seconds, None when unparseable.

    Accepts the ISO-8601 string `EventSink` writes as well as a bare number.

    Args:
        value: The raw `ts` field.

    Returns:
        The epoch seconds, or None for a bool, a non-ISO string or any other type.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def tool_result_ok(value: Any) -> bool:
    """Return the persisted `tool.result.ok` flag, tolerating the stringified form.

    Both folds use this one coercion, so a run-state surface and the conversation
    never disagree on a tool's verdict.

    Args:
        value: The raw `ok` field.

    Returns:
        True for True or "True"; False for everything else.
    """
    return value in (True, "True")


def readable_summary(value: Any) -> str:
    """Return a tool result's `summary` as text.

    Args:
        value: The raw `summary` field.

    Returns:
        The string as is; a dict or list as JSON rather than a Python repr; anything
        else through `str()`.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def as_int(value: object) -> int:
    """Return an event field as an int, 0 for anything unusable."""
    try:
        return int(value)  # type: ignore[arg-type]  # int() rejects bad types itself
    except (TypeError, ValueError):
        return 0


@dataclasses.dataclass(frozen=True, slots=True)
class SessionStart:
    """session.start: a fresh run began on the task text."""

    user_task: str


@dataclasses.dataclass(frozen=True, slots=True)
class ResumeStart:
    """loop.resume.start: a finished or stopped run restarts in place."""


@dataclasses.dataclass(frozen=True, slots=True)
class GraphUpdate:
    """graph.update: the task tree and its cursor.

    Attributes:
        nodes: The raw node map; the tree builder walks it with isinstance guards
            for cycles, duplicates and malformed values, so it is not coerced here.
        cursor: The id of the task in progress, None when none is.
    """

    nodes: Any
    cursor: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class DiffUpdated:
    """diff.updated: the run's cumulative patch and the commit it reaches."""

    patch: str
    sha: str


@dataclasses.dataclass(frozen=True, slots=True)
class AutoCommit:
    """One per-step commit on the run's chain (`loop.auto_commit`)."""

    iteration: int
    sha: str
    subject: str


@dataclasses.dataclass(frozen=True, slots=True)
class RoleCall:
    """role.call: a model call began for a role."""

    role: str
    model: str
    provider: str


@dataclasses.dataclass(frozen=True, slots=True)
class RoleResult:
    """role.result: a model call returned, with its input token counts."""

    tokens_in: int
    cache_read: int
    cache_creation: int


@dataclasses.dataclass(frozen=True, slots=True)
class RoleTextDelta:
    """role.text_delta: a streamed piece of the assistant's text."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class RoleThinkingDelta:
    """role.thinking_delta: a streamed piece of the assistant's thinking."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCall:
    """tool.call: a tool was dispatched.

    Attributes:
        name: The tool's name.
        args: The raw args; rendered per value and isinstance-checked downstream, so a
            garbled value degrades instead of raising.
        call_id: The correlation id stamped per dispatch; None on a log without ids.
    """

    name: str
    args: Any
    call_id: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResult:
    """tool.result: a tool returned.

    Attributes:
        name: The tool's name.
        ok: The tool's verdict.
        summary: The result's one-line summary.
        call_id: The correlation id of the matching call; None on a log without ids.
    """

    name: str
    ok: bool
    summary: str
    call_id: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyStart:
    """verify.start: the verify gate began running its command."""

    cmd: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyEnd:
    """verify.end: the verify gate finished, with its exit code and output tails."""

    cmd: tuple[str, ...]
    exit_code: int
    duration_s: float
    stdout_tail: str
    stderr_tail: str


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetUpdate:
    """budget.update: the cumulative token and spend figures.

    Attributes:
        input_total: Input tokens so far.
        output_total: Output tokens so far.
        cache_read_total: The cached side of the input; 0 in a journal written before
            it was recorded.
        cache_creation_total: Cache-creation tokens so far.
        usd_total: Spend so far.
        usd_partial: Some of the spend is unpriced, so `usd_total` is a lower bound.
        usd_cap: The execution's spend cap; -1 for unlimited.
        tokens_unmetered: Tokens no price list covered.
        tokens_fallback_cap: The token cap that stands in for an unpriceable run.
        plan_used_percent: Subscription plan usage, 0 when the provider is not plan-metered.
        plan_consumed: Plan points consumed by this run.
        plan_cap: The plan's points cap.
        plan_resets_at: When the plan window resets, as epoch seconds.
    """

    input_total: int
    output_total: int
    cache_read_total: int
    cache_creation_total: int
    usd_total: float
    usd_partial: bool
    usd_cap: float
    tokens_unmetered: int
    tokens_fallback_cap: int
    plan_used_percent: float = 0.0
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_resets_at: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    """approval.prompt: the run is waiting for an operator's yes or no.

    Attributes:
        id: The prompt's id, matched by the answer.
        prompt: The words shown to the operator.
        standing: An "allow all" would cover calls beyond this one, so a front-end
            offers the button; a log written before the field existed folds True.
        asked_ep: When it was asked, as epoch seconds, for the waiting status's age;
            None when the line carried no parseable ts.
        call_id: The dispatched tool call the prompt gates; None for one gating no call
            (a verify the harness runs itself) or a log written before the field.
    """

    id: str
    prompt: str
    standing: bool = True
    asked_ep: float | None = None
    call_id: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    """approval.answer: an operator answered an approval prompt."""

    id: str
    approved: bool


@dataclasses.dataclass(frozen=True, slots=True)
class EventQuestion:
    """One question of a question.prompt, with its offered options."""

    question: str
    options: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class QuestionPrompt:
    """question.prompt: the run is waiting for an operator's answers.

    Attributes:
        id: The prompt's id, matched by the answer.
        questions: The questions asked.
        asked_ep: When it was asked, as epoch seconds, for the waiting status's age.
        call_id: The gated `ask_user` call; None for a pre-run question.
    """

    id: str
    questions: tuple[EventQuestion, ...]
    asked_ep: float | None = None
    call_id: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class QuestionAnswer:
    """question.answer: the answers to a question prompt.

    Attributes:
        id: The prompt's id.
        answers: One answer per question.
        unseen: Nobody was attached, so the harness answered empty.
    """

    id: str
    answers: tuple[str, ...]
    unseen: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class PinAdded:
    """loop.pin.added: an operator /pin instruction was recorded."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class PinsRestored:
    """loop.pin.restored: a resume or fork execution restored the snapshot's pins.

    The full list replaces the fold's pins: a fork's fresh log carries only this event.
    """

    pins: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CompactRestored:
    """loop.compact.restored: the elision markers a restored context carries.

    The counts replace the fold's: a fork's fresh log has no compact.dropped events.
    """

    elided: int
    gists: int


@dataclasses.dataclass(frozen=True, slots=True)
class CompactDropped:
    """loop.compact.dropped: tier-1 elision, with the elided call identities."""

    n: int
    calls: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CompactGists:
    """loop.compact.gists: gists created and demoted in a tier-1 pass."""

    gisted: int
    demoted: int


@dataclasses.dataclass(frozen=True, slots=True)
class CompactSummarised:
    """loop.compact.summarise.done: a tier-2 restart replaced the history.

    Every elision marker and gist the context held went with it.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class SteerRequested:
    """session.steer_requested: an operator Ctrl-C mid-run."""


@dataclasses.dataclass(frozen=True, slots=True)
class SessionEnd:
    """session.end: the execution ended.

    Attributes:
        all_passed: True when the final tree was observed verify-green, False when not
            (red, stale or error), None when no verify command gated it.
        reason: The end reason word.
        scoped: The gate ran scoped to the tests nearest the run's diff because the
            full command overran `verify_timeout_s`, so a pass is qualified.
    """

    all_passed: bool | None
    reason: str
    scoped: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class SessionUndone:
    """session.undone: /undo forked this run.

    Surfaces follow the child with the undone text back in the composer.
    """

    new_session_id: str
    undone_text: str


@dataclasses.dataclass(frozen=True, slots=True)
class RawEvent:
    """Any event the fold does not consume: telemetry, unknown types, a line with no type.

    Carries the raw dict so the log-line renderer still reads it; the fold drops it.
    """

    type: str
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)


Event = (
    SessionStart
    | ResumeStart
    | GraphUpdate
    | DiffUpdated
    | AutoCommit
    | RoleCall
    | RoleResult
    | RoleTextDelta
    | RoleThinkingDelta
    | ToolCall
    | ToolResult
    | VerifyStart
    | VerifyEnd
    | BudgetUpdate
    | ApprovalPrompt
    | ApprovalAnswer
    | QuestionPrompt
    | QuestionAnswer
    | PinAdded
    | PinsRestored
    | CompactRestored
    | CompactDropped
    | CompactGists
    | CompactSummarised
    | SteerRequested
    | SessionEnd
    | SessionUndone
    | RawEvent
)


def _call_id(raw: dict[str, Any]) -> int | None:
    """Return the event's `call_id` when it is an int, else None."""
    cid = raw.get("call_id")
    return cid if isinstance(cid, int) else None


def parse_event(raw: dict[str, Any]) -> Event:
    """Parse one raw logs.jsonl event into its typed family.

    A malformed field inside a known family (a torn numeric in `verify.end`) degrades
    to `RawEvent` like an unknown type: the fold runs unwrapped inside live tails, so
    it never raises on a line an interrupted writer left behind.

    Args:
        raw: The event dict as read from the journal.

    Returns:
        The typed family, or `RawEvent` for every other type.
    """
    try:
        return _parse_known(raw)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
        return RawEvent(type=str(raw.get("type", "")), raw=raw)


def _parse_known(raw: dict[str, Any]) -> Event:  # noqa: C901, PLR0911, PLR0912  # one branch per event type
    """Parse a raw event by type, one coercion per family.

    Args:
        raw: The event dict as read from the journal.

    Returns:
        The typed family, or `RawEvent` for an unknown type.

    Raises:
        ValueError: A known family's field cannot be coerced.
        TypeError: A known family's field has an unusable type.
    """
    match raw.get("type", ""):
        case "session.start":
            return SessionStart(user_task=str(raw.get("user_task", "")))
        case "loop.resume.start":
            return ResumeStart()
        case "graph.update":
            nodes = raw.get("nodes", {}) or {}
            if not isinstance(nodes, dict):
                # An empty-dict fold would replace the task tree; RawEvent keeps the last one.
                raise ValueError("graph.update nodes must be an object")
            cursor = raw.get("cursor")
            return GraphUpdate(
                nodes=nodes,
                cursor=cursor if isinstance(cursor, str) else None,
            )
        case "diff.updated":
            return DiffUpdated(patch=str(raw.get("patch", "")), sha=str(raw.get("sha", "")))
        case "loop.auto_commit":
            return AutoCommit(
                iteration=as_int(raw.get("iteration")),
                sha=str(raw.get("sha") or ""),
                subject=str(raw.get("subject") or ""),
            )
        case "role.call":
            return RoleCall(
                role=str(raw.get("role", "")),
                model=str(raw.get("model", "")),
                provider=str(raw.get("provider", "")),
            )
        case "role.result":
            return RoleResult(
                tokens_in=as_int(raw.get("tokens_in")),
                cache_read=as_int(raw.get("cache_read")),
                cache_creation=as_int(raw.get("cache_creation")),
            )
        case "role.text_delta":
            return RoleTextDelta(text=str(raw.get("text", "")))
        case "role.thinking_delta":
            return RoleThinkingDelta(text=str(raw.get("text", "")))
        case "tool.call":
            args = raw.get("args")
            return ToolCall(
                name=str(raw.get("name", "")),
                # The call happened even with garbled args, and args is display-only downstream.
                args=args if isinstance(args, dict) else {},
                call_id=_call_id(raw),
            )
        case "tool.result":
            return ToolResult(
                name=str(raw.get("name", "")),
                ok=tool_result_ok(raw.get("ok")),
                summary=readable_summary(raw.get("summary", "")),
                call_id=_call_id(raw),
            )
        case "verify.start":
            return VerifyStart(cmd=tuple(str(x) for x in raw.get("cmd", []) or []))
        case "verify.end":
            return VerifyEnd(
                cmd=tuple(str(x) for x in raw.get("cmd", []) or []),
                exit_code=int(raw.get("exit_code", -1)),
                duration_s=float(raw.get("duration_s", 0.0)),
                stdout_tail=str(raw.get("stdout_tail", "")),
                stderr_tail=str(raw.get("stderr_tail", "")),
            )
        case "budget.update":
            return BudgetUpdate(
                input_total=int(raw.get("input_total", 0)),
                output_total=int(raw.get("output_total", 0)),
                cache_read_total=int(raw.get("cache_read_total", 0)),
                cache_creation_total=int(raw.get("cache_creation_total", 0)),
                usd_total=float(raw.get("usd_total", 0.0)),
                usd_partial=bool(raw.get("usd_partial", False)),
                # A log written without these keys folds 0.
                usd_cap=float(raw.get("usd_cap", 0.0)),
                tokens_unmetered=int(raw.get("tokens_unmetered", 0)),
                tokens_fallback_cap=int(raw.get("tokens_fallback_cap", 0)),
                plan_used_percent=float(raw.get("plan_used_percent", 0.0)),
                plan_consumed=float(raw.get("plan_consumed", 0.0)),
                plan_cap=float(raw.get("plan_cap", 0.0)),
                plan_resets_at=float(raw.get("plan_resets_at", 0.0)),
            )
        case "approval.prompt":
            return ApprovalPrompt(
                id=str(raw.get("id", "")),
                prompt=str(raw.get("prompt", "")),
                standing=bool(raw.get("standing", True)),
                asked_ep=event_epoch(raw.get("ts")),
                call_id=_call_id(raw),
            )
        case "approval.answer":
            return ApprovalAnswer(
                id=str(raw.get("id", "")), approved=bool(raw.get("approved", False))
            )
        case "question.prompt":
            questions = tuple(
                EventQuestion(
                    question=str(q.get("question", "")),
                    options=tuple(str(o) for o in (q.get("options", ()) or ())),
                )
                for q in (raw.get("questions", ()) or ())
                if isinstance(q, dict)
            )
            return QuestionPrompt(
                id=str(raw.get("id", "")),
                questions=questions,
                asked_ep=event_epoch(raw.get("ts")),
                call_id=_call_id(raw),
            )
        case "question.answer":
            raw_ans = raw.get("answers", ()) or ()
            answers = tuple(str(a) for a in raw_ans) if isinstance(raw_ans, (list, tuple)) else ()
            return QuestionAnswer(
                id=str(raw.get("id", "")), answers=answers, unseen=raw.get("unseen") is True
            )
        case "loop.pin.added":
            return PinAdded(text=str(raw.get("text", "")))
        case "loop.pin.restored":
            raw_pins = raw.get("pins", ()) or ()
            pins = tuple(str(x) for x in raw_pins) if isinstance(raw_pins, (list, tuple)) else ()
            return PinsRestored(pins=pins)
        case "loop.compact.restored":
            return CompactRestored(elided=as_int(raw.get("elided")), gists=as_int(raw.get("gists")))
        case "loop.compact.dropped":
            raw_calls = raw.get("calls", ()) or ()
            calls = tuple(str(c) for c in raw_calls) if isinstance(raw_calls, (list, tuple)) else ()
            return CompactDropped(n=as_int(raw.get("n")), calls=calls)
        case "loop.compact.gists":
            return CompactGists(
                gisted=as_int(raw.get("gisted")), demoted=as_int(raw.get("demoted"))
            )
        case "loop.compact.summarise.done":
            return CompactSummarised()
        case "session.steer_requested":
            return SteerRequested()
        case "session.undone":
            return SessionUndone(
                new_session_id=str(raw.get("new_session_id", "") or ""),
                undone_text=str(raw.get("undone_text", "") or ""),
            )
        case "session.end":
            # An explicit null is the ungated tri-state; an absent key folds False.
            raw_ap = raw.get("all_passed", False)
            return SessionEnd(
                all_passed=None if raw_ap is None else bool(raw_ap),
                reason=str(raw.get("reason", "") or ""),
                scoped=bool(raw.get("scoped", False)),
            )
        case other:
            return RawEvent(type=str(other), raw=raw)
