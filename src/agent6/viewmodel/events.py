# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Typed read model for the ~19 logs.jsonl event families the RunState fold consumes.

The write side (`agent6.events.EventSink`) appends free-form `{"type", "ts",
**fields}` dicts and never validates; ~90 distinct types exist. The RunState fold
(`viewmodel.state.apply_event`) structurally consumes only the 19 families defined
here. `parse_event` turns one raw event dict into exactly one of those frozen
families, or a `RawEvent` passthrough for every other type -- the compatibility
surface that keeps old run dirs folding: a type this module does not know becomes
`RawEvent`, which the fold drops (its old `case _`), never a crash.

Why hand-rolled frozen dataclasses, not pydantic (unlike `machine/journal.py`):
logs.jsonl is append-only history, so the fold MUST reproduce byte-for-byte the
coercion the fold did inline before this module existed (`str()`/`int()`/`bool()`
with per-field defaults, `_as_int`'s swallow-to-zero, the isinstance guards). A
pydantic model would impose pydantic's own coercion and validation-failure
semantics, changing how a malformed old line folds; these parsers instead move the
existing coercion verbatim into one place per family. `parse_event` is total for
unknown types (RawEvent) but preserves the fold's pre-existing latent raises on a
non-coercible known field (e.g. `verify.end` exit_code) -- "degrade exactly as
today", not "never raise".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def event_epoch(value: object) -> float | None:
    """Parse an event ``ts`` to epoch seconds, or None if unparseable.

    EventSink writes ``ts`` as an ISO-8601 string (``datetime.isoformat``),
    so the elapsed-time anchor must parse that, not only bare numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def readable_summary(value: Any) -> str:
    """A tool result's `summary` should be a string; a malformed dict/list value
    renders as neutral JSON, not the single-quoted Python repr `str()` produces
    (which leaked `{'unexpected': ...}` into the web/TUI tool detail + log tail)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _as_int(value: object) -> int:
    """An event field as an int; 0 for anything unusable (untrusted log data)."""
    try:
        return int(value)  # type: ignore[arg-type]  # int() rejects bad types itself
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class RunStart:
    user_task: str


@dataclass(frozen=True, slots=True)
class ResumeStart:
    """loop.resume.start: a finished/stopped run restarts in place."""


@dataclass(frozen=True, slots=True)
class GraphUpdate:
    # The node map is walked defensively by the tree builder (isinstance guards for
    # cycles, dupes, and malformed non-dict values), so it stays raw here.
    nodes: Any
    cursor: str | None


@dataclass(frozen=True, slots=True)
class DiffUpdated:
    patch: str
    sha: str


@dataclass(frozen=True, slots=True)
class RoleCall:
    role: str
    model: str
    provider: str


@dataclass(frozen=True, slots=True)
class RoleResult:
    tokens_in: int
    cache_read: int
    cache_creation: int


@dataclass(frozen=True, slots=True)
class RoleTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class RoleThinkingDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    # Raw args: rendered per-value and isinstance-checked for the finish summary,
    # so a non-dict value degrades exactly as the fold did inline.
    args: Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    summary: str


@dataclass(frozen=True, slots=True)
class VerifyStart:
    cmd: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifyEnd:
    cmd: tuple[str, ...]
    exit_code: int
    duration_s: float
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True, slots=True)
class BudgetUpdate:
    input_total: int
    output_total: int
    input_cap: int
    output_cap: int
    usd_total: float
    usd_partial: bool


@dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    id: str
    approved: bool


@dataclass(frozen=True, slots=True)
class EventQuestion:
    question: str
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuestionPrompt:
    id: str
    questions: tuple[EventQuestion, ...]


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    id: str
    answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SteerRequested:
    """run.steer_requested: an operator Ctrl-C mid-run."""


@dataclass(frozen=True, slots=True)
class RunEnd:
    all_passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RawEvent:
    """Any event the fold does not structurally consume (the ~65 loop.* telemetry
    types, unknown/future types, a line with no `type`). Carries the raw dict so the
    log-line renderer still reads it; the fold drops it (its old `case _`)."""

    type: str
    raw: dict[str, Any] = field(default_factory=dict)


Event = (
    RunStart
    | ResumeStart
    | GraphUpdate
    | DiffUpdated
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
    | SteerRequested
    | RunEnd
    | RawEvent
)


def parse_event(raw: dict[str, Any]) -> Event:
    """One raw logs.jsonl event dict -> one typed family, or RawEvent for the rest.

    A malformed field inside a KNOWN family (a torn numeric in ``verify.end`` or
    ``budget.update``) degrades to RawEvent exactly like an unknown type: the
    fold runs unwrapped inside live tails (web SSE, TUI reader), so it must
    never raise on a line an interrupted writer left behind."""
    try:
        return _parse_known(raw)
    except (ValueError, TypeError):
        return RawEvent(type=str(raw.get("type", "")), raw=raw)


def _parse_known(raw: dict[str, Any]) -> Event:  # noqa: PLR0911, PLR0912
    """The per-family arms. Each reproduces, field-for-field, the coercion the
    RunState fold applied inline before this module existed, so the fold output
    is byte-identical for every historical event."""
    match raw.get("type", ""):
        case "run.start":
            return RunStart(user_task=str(raw.get("user_task", "")))
        case "loop.resume.start":
            return ResumeStart()
        case "graph.update":
            cursor = raw.get("cursor")
            return GraphUpdate(
                nodes=raw.get("nodes", {}) or {},
                cursor=cursor if isinstance(cursor, str) else None,
            )
        case "diff.updated":
            return DiffUpdated(patch=str(raw.get("patch", "")), sha=str(raw.get("sha", "")))
        case "role.call":
            return RoleCall(
                role=str(raw.get("role", "")),
                model=str(raw.get("model", "")),
                provider=str(raw.get("provider", "")),
            )
        case "role.result":
            return RoleResult(
                tokens_in=_as_int(raw.get("tokens_in")),
                cache_read=_as_int(raw.get("cache_read")),
                cache_creation=_as_int(raw.get("cache_creation")),
            )
        case "role.text_delta":
            return RoleTextDelta(text=str(raw.get("text", "")))
        case "role.thinking_delta":
            return RoleThinkingDelta(text=str(raw.get("text", "")))
        case "tool.call":
            return ToolCall(name=str(raw.get("name", "")), args=raw.get("args", {}) or {})
        case "tool.result":
            return ToolResult(
                name=str(raw.get("name", "")),
                ok=bool(raw.get("ok", False)),
                summary=readable_summary(raw.get("summary", "")),
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
                input_cap=int(raw.get("input_cap", 0)),
                output_cap=int(raw.get("output_cap", 0)),
                usd_total=float(raw.get("usd_total", 0.0)),
                usd_partial=bool(raw.get("usd_partial", False)),
            )
        case "approval.prompt":
            return ApprovalPrompt(id=str(raw.get("id", "")), prompt=str(raw.get("prompt", "")))
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
            return QuestionPrompt(id=str(raw.get("id", "")), questions=questions)
        case "question.answer":
            raw_ans = raw.get("answers", ()) or ()
            answers = tuple(str(a) for a in raw_ans) if isinstance(raw_ans, (list, tuple)) else ()
            return QuestionAnswer(id=str(raw.get("id", "")), answers=answers)
        case "run.steer_requested":
            return SteerRequested()
        case "run.end":
            return RunEnd(
                all_passed=bool(raw.get("all_passed", False)),
                reason=str(raw.get("reason", "") or ""),
            )
        case other:
            return RawEvent(type=str(other), raw=raw)
