# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read logs.jsonl rows into the journal's records, with the coercions old run dirs need.

The vocabulary is `agent6.sessions.journal`; this module is its tolerant reader. Each
family keeps the exact coercion old run dirs were written against (`str()`, `int()`,
`bool()` with per-field defaults, `as_int`'s swallow-to-zero, the isinstance guards), so a
run dir folds without a crash whatever it holds. Every other row is a `RawEvent`, which
the fold drops.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Mapping
from typing import Any

from agent6 import evidence, kinds
from agent6.sessions import journal

# The calls that end a session rather than work on it; literals, so no tools import.
FINISH_TOOLS = frozenset({"finish_session", "finish_planning"})


def verification_word(value: object) -> kinds.Verification | None:
    """Return the verification word a journal row carries, or None for anything else."""
    if value in ("passed", "failed", "unverified", "not_applicable"):
        return value
    return None


def event_epoch(value: object) -> float | None:
    """Return a row's `ts` as epoch seconds: ISO 8601 text or a number; None otherwise.

    Args:
        value: The envelope's timestamp as read.

    Returns:
        The epoch seconds, or None for a value no clock wrote.
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


def verify_limit_hit(outcome: str | None, timeout_s: object) -> float:
    """Return the limit a verify.end's fields say the gate ran out.

    The one rule the listing scan, the typed fold and the transcript fold share.

    Args:
        outcome: The recorded verification outcome.
        timeout_s: The event's wall-clock limit.

    Returns:
        The positive limit for a recorded timeout, else 0.0.
    """
    if (
        outcome == "timed_out"
        and isinstance(timeout_s, (int, float))
        and not isinstance(timeout_s, bool)
        and timeout_s > 0
    ):
        return float(timeout_s)
    return 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class EventQuestion:
    """One question of a question.prompt, with its offered options."""

    question: str
    options: tuple[str, ...]


def question_rows(questions: Any) -> tuple[EventQuestion, ...]:
    """Return a question.prompt's questions as typed rows; a malformed entry is skipped."""
    if not isinstance(questions, (list, tuple)):
        return ()
    return tuple(
        EventQuestion(
            question=str(q.get("question", "")),
            options=tuple(str(o) for o in (q.get("options", ()) or ())),
        )
        for q in questions
        if isinstance(q, dict)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RawEvent:
    """Any event the fold does not consume: telemetry, unknown types, a line with no type.

    Carries the raw dict so the log-line renderer still reads it; the fold drops it.
    """

    type: str
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)


Event = journal.Event | RawEvent


def _text(raw: dict[str, Any], key: str) -> str:
    return str(raw.get(key, "") or "")


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in value) if isinstance(value, (list, tuple)) else ()


def _seat_index(raw: dict[str, Any]) -> int | None:
    """Return the panel seat index a role event carries, None outside a panel."""
    value = raw.get("seat_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def call_args(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a tool.call row's argument preview; an older journal keys it `args`."""
    args = raw.get("args_preview", raw.get("args"))
    return args if isinstance(args, dict) else {}  # pyright: ignore[reportUnknownVariableType]


_OUTCOME_KEYS = (
    "exit_code",
    "duration_s",
    "timed_out",
    "exec_failed",
    "background_id",
    "paths",
    "stdout_tail",
    "stderr_tail",
    "stdout_truncated",
    "stderr_truncated",
    "stopped_by_operator",
    "attachment",
    "head_tail",
    "lines_total",
)


_IDENTITY_KEYS = ("execution_id", "model_call_id", "panel_id", "provider_tool_call_id")


def _call_identity(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a row's model-call identity fields, each when present."""
    fields: dict[str, Any] = {key: raw[key] for key in _IDENTITY_KEYS if key in raw}
    if "seat_index" in raw:
        fields["seat_index"] = _seat_index(raw)
    return fields


def _count(value: object) -> int | None:
    """Return a recorded token counter; None when the row carries none.

    Args:
        value: The row's value.

    Returns:
        The counter.

    Raises:
        ValueError: The value is not an integer, so the row degrades to `RawEvent`.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"not a counter: {value!r}")
    return value


def _amount(value: object) -> float:
    """Return a recorded money figure.

    Args:
        value: The row's value.

    Returns:
        The figure.

    Raises:
        ValueError: The value is not a number (a torn or hand-edited row), so the row degrades
            to `RawEvent` and the fold keeps its last good figure.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"not an amount: {value!r}")
    return float(value)


def _cap(value: object) -> float | None:
    """Return a recorded cap; None when the row carries none."""
    return None if value is None else float(value)  # pyright: ignore[reportArgumentType]


def _account_reading(raw: dict[str, Any]) -> journal.AccountReading | None:
    """Return a budget row's account reading; a journal written before the record keys it flat."""
    nested = raw.get("account")
    if isinstance(nested, dict):
        return journal.AccountReading.model_validate(nested)
    windows = [
        journal.AccountWindow.model_validate(w)
        for w in raw.get("plan_windows") or ()
        if isinstance(w, dict)
    ]
    used = float(raw.get("plan_used_percent") or 0.0)
    if not windows and used <= 0:
        return None
    return journal.AccountReading(
        provider="",
        used_percent=used,
        resets_at=float(raw.get("plan_resets_at") or 0.0),
        windows=windows,
        credits_has=bool(raw.get("credits_has", False)),
        credits_unlimited=bool(raw.get("credits_unlimited", False)),
        credits_balance=str(raw.get("credits_balance") or ""),
    )


def _call_id(raw: dict[str, Any]) -> int | None:
    """Return the event's `call_id` when it is an int, else None."""
    cid = raw.get("call_id")
    return cid if isinstance(cid, int) and not isinstance(cid, bool) else None


def _turn(raw: dict[str, Any]) -> int | None:
    """Return the event's `turn` when it is an int, else None."""
    turn = raw.get("turn")
    return turn if isinstance(turn, int) and not isinstance(turn, bool) else None


def parse_event(raw: dict[str, Any]) -> Event:
    """Parse one raw logs.jsonl row into its journal record.

    A malformed field inside a known family (a torn numeric in `verify.end`) degrades
    to `RawEvent` like an unknown type: the fold runs unwrapped inside live tails, so
    it never raises on a line an interrupted writer left behind.

    Args:
        raw: The row as read from the journal.

    Returns:
        The record, or `RawEvent` for every other type.
    """
    try:
        return _parse_known(raw)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
        return RawEvent(type=str(raw.get("type", "")), raw=raw)


def _parse_known(raw: dict[str, Any]) -> Event:  # noqa: C901, PLR0911, PLR0912  # one branch per event type
    """Parse a row by type, one coercion per family.

    Args:
        raw: The row as read from the journal.

    Returns:
        The record, or `RawEvent` for an unknown type.

    Raises:
        ValueError: A known family's field cannot be coerced.
        TypeError: A known family's field has an unusable type.
    """
    match raw.get("type", ""):
        case "session.start":
            return journal.SessionStart(
                user_task=_text(raw, "user_task"),
                mode=_text(raw, "mode"),
                execution_id=_text(raw, "execution_id"),
            )
        case "session.resume":
            return journal.SessionResume(
                session_id=_text(raw, "session_id"),
                user_task=_text(raw, "user_task"),
                steer=_text(raw, "steer"),
                mode=_text(raw, "mode"),
                turn=as_int(raw.get("turn")),
                messages=as_int(raw.get("messages")),
                execution_id=_text(raw, "execution_id"),
            )
        case "graph.update":
            nodes = raw.get("nodes", {}) or {}
            if not isinstance(nodes, dict):
                # An empty-dict fold would replace the task tree; RawEvent keeps the last one.
                raise ValueError("graph.update nodes must be an object")
            cursor = raw.get("cursor")
            return journal.GraphUpdate(
                nodes=nodes, cursor=cursor if isinstance(cursor, str) else None
            )
        case "diff.updated":
            return journal.DiffUpdated(patch=_text(raw, "patch"), sha=_text(raw, "sha"))
        case "checkpoint.committed":
            return journal.CheckpointCommitted(
                # Journals written before the turn rename stamp `iteration`.
                turn=as_int(raw.get("turn", raw.get("iteration"))),
                sha=_text(raw, "sha"),
                subject=_text(raw, "subject"),
            )
        case "model.call":
            return journal.ModelCall(
                role=_text(raw, "role"),
                model=_text(raw, "model"),
                provider=_text(raw, "provider"),
                model_call_id=_text(raw, "model_call_id"),
                seat_index=_seat_index(raw),
                execution_id=raw.get("execution_id"),
            )
        case "model.attempt":
            return journal.ModelAttempt(
                role=_text(raw, "role"),
                model=_text(raw, "model"),
                provider=_text(raw, "provider"),
                model_call_id=_text(raw, "model_call_id"),
                seat_index=_seat_index(raw),
                execution_id=raw.get("execution_id"),
                attempt=as_int(raw.get("attempt")) or 1,
                request_id=raw.get("request_id"),
                ok=raw.get("ok") is not False,
                input_tokens=as_int(raw.get("input_tokens")),
                output_tokens=as_int(raw.get("output_tokens")),
                cache_read_tokens=as_int(raw.get("cache_read_tokens")),
                cache_write_tokens=as_int(raw.get("cache_write_tokens")),
            )
        case "model.result":
            return journal.ModelResult(
                role=_text(raw, "role"),
                model=_text(raw, "model"),
                provider=_text(raw, "provider"),
                model_call_id=_text(raw, "model_call_id"),
                seat_index=_seat_index(raw),
                execution_id=raw.get("execution_id"),
                ok=raw.get("ok") is not False,
                input_tokens=as_int(raw.get("input_tokens")),
                output_tokens=as_int(raw.get("output_tokens")),
                cache_read_tokens=as_int(raw.get("cache_read_tokens")),
                cache_write_tokens=as_int(raw.get("cache_write_tokens")),
                error=str(raw["error"]) if raw.get("error") else None,
            )
        case "model.text_delta":
            return journal.ModelTextDelta(
                role=_text(raw, "role"), text=_text(raw, "text"), **_call_identity(raw)
            )
        case "model.thinking_delta":
            return journal.ModelThinkingDelta(
                role=_text(raw, "role"), text=_text(raw, "text"), **_call_identity(raw)
            )
        case "tool.call":
            # The call happened even with garbled args, and the preview is display-only downstream.
            return journal.ToolCall(
                name=_text(raw, "name"),
                args_preview=call_args(raw),
                call_id=_call_id(raw),
                turn=_turn(raw),
                **_call_identity(raw),
            )
        case "tool.result":
            return journal.ToolResult(
                name=_text(raw, "name"),
                ok=tool_result_ok(raw.get("ok")),
                summary=readable_summary(raw.get("summary", "")),
                call_id=_call_id(raw),
                turn=_turn(raw),
                **_call_identity(raw),
                **{key: raw[key] for key in _OUTCOME_KEYS if key in raw},
            )
        case "verify.start":
            return journal.VerifyStart(
                command=_strings(raw.get("command", []) or []),
                scoped=bool(raw.get("scoped", False)),
                selected_paths=_strings(raw.get("selected_paths", []) or []),
                timeout_s=float(raw.get("timeout_s", 0.0) or 0.0),
            )
        case "verify.end":
            return journal.VerifyEnd(
                command=_strings(raw.get("command", []) or []),
                scoped=bool(raw.get("scoped", False)),
                selected_paths=_strings(raw.get("selected_paths", []) or []),
                exit_code=int(raw["exit_code"]) if raw.get("exit_code") is not None else None,
                duration_s=float(raw.get("duration_s", 0.0)),
                stdout_tail=_text(raw, "stdout_tail"),
                stderr_tail=_text(raw, "stderr_tail"),
                timeout_s=float(raw.get("timeout_s", 0.0) or 0.0),
                outcome=evidence.outcome(raw),
                reason=_text(raw, "reason"),
                baseline=bool(raw.get("baseline", False)),
            )
        case "budget.update":
            return journal.BudgetUpdate(
                settled_lanes=_strings(raw.get("settled_lanes")),
                # A counter the row lacks stays unknown; the four travel as one group.
                input_total=_count(raw.get("input_total")),
                output_total=_count(raw.get("output_total")),
                cache_read_total=_count(raw.get("cache_read_total")),
                cache_write_total=_count(raw.get("cache_write_total")),
                usd_total=_amount(raw.get("usd_total", 0.0)),
                usd_partial=bool(raw.get("usd_partial", False)),
                # A row without a cap leaves it unknown; a recorded 0 (refuse) or -1 stays.
                usd_cap=_cap(raw.get("usd_cap")),
                usd_running_lanes=_amount(raw.get("usd_running_lanes", 0.0)),
                tokens_unmetered=int(raw.get("tokens_unmetered", 0)),
                tokens_running_lanes=int(raw.get("tokens_running_lanes", 0)),
                tokens_fallback_cap=(
                    None
                    if raw.get("tokens_fallback_cap") is None
                    else int(raw["tokens_fallback_cap"])
                ),
                # Journals written before the rename stamp `plan_consumed`.
                plan_observed_points=float(
                    raw.get("plan_observed_points", raw.get("plan_consumed", 0.0))
                ),
                plan_cap=_cap(raw.get("plan_cap")),
                account=_account_reading(raw),
            )
        case "environment.starting":
            return journal.EnvironmentStarting(
                execution_id=raw.get("execution_id"),
                command=_strings(raw.get("command", []) or []),
            )
        case "environment.acquired":
            return journal.EnvironmentAcquired(
                execution_id=raw.get("execution_id"),
                lifecycle_ref=_text(raw, "lifecycle_ref"),
                role=_text(raw, "role"),
                reason=_text(raw, "reason"),
                started=bool(raw.get("started", False)),
                start_s=_cap(raw.get("start_s")),
            )
        case "execution.environment":
            return journal.ExecutionEnvironment(
                execution_id=raw.get("execution_id"),
                target_id=raw.get("target_id"),
                outcome=_text(raw, "outcome"),
                reason=_text(raw, "reason"),
            )
        case "approval.prompt":
            return journal.ApprovalPrompt(
                id=_text(raw, "id"),
                prompt=_text(raw, "prompt"),
                standing=bool(raw.get("standing", True)),
                scope=raw["scope"] if isinstance(raw.get("scope"), str) else None,
                call_id=_call_id(raw),
            )
        case "approval.answer":
            return journal.ApprovalAnswer(
                id=_text(raw, "id"),
                approved=bool(raw.get("approved", False)),
                source=_text(raw, "source"),
            )
        case "question.prompt":
            questions = raw.get("questions", ()) or ()
            return journal.QuestionPrompt(
                id=_text(raw, "id"),
                questions=[q for q in questions if isinstance(q, dict)]
                if isinstance(questions, (list, tuple))
                else (),
                call_id=_call_id(raw),
                wait=raw.get("wait") is not False,
            )
        case "question.answer":
            return journal.QuestionAnswer(
                id=_text(raw, "id"),
                answers=_strings(raw.get("answers", ()) or ()),
                unseen=raw.get("unseen") is True,
                source=_text(raw, "source"),
            )
        case "pin.added":
            return journal.PinAdded(
                text=_text(raw, "text"),
                chars=as_int(raw.get("chars")),
                count=as_int(raw.get("count")),
            )
        case "pin.restored":
            return journal.PinRestored(
                pins=_strings(raw.get("pins", ()) or ()), count=as_int(raw.get("count"))
            )
        case "compact.restored":
            return journal.CompactRestored(
                elided=as_int(raw.get("elided")), gists=as_int(raw.get("gists"))
            )
        case "compact.dropped":
            return journal.CompactDropped(
                count=as_int(raw.get("count")), calls=_strings(raw.get("calls", ()) or ())
            )
        case "compact.gist.done":
            return journal.CompactGistDone(
                gisted=as_int(raw.get("gisted")),
                demoted=as_int(raw.get("demoted")),
                paths=_strings(raw.get("paths", ()) or ()),
                demoted_paths=_strings(raw.get("demoted_paths", ()) or ()),
            )
        case "compact.summarise.done":
            return journal.CompactSummariseDone(
                summary_chars=as_int(raw.get("summary_chars")),
                summary=_text(raw, "summary"),
                kept_messages=as_int(raw.get("kept_messages")),
            )
        case "steer.signalled":
            return journal.SteerSignalled(source=_text(raw, "source"))
        case "session.undone":
            return journal.SessionUndone(
                new_session_id=_text(raw, "new_session_id"), undone_text=_text(raw, "undone_text")
            )
        case "session.stranded":
            return journal.SessionStranded()
        case "session.stop":
            return journal.SessionStop(
                execution_id=_text(raw, "execution_id"),
                reason=_text(raw, "reason"),
                turns=as_int(raw.get("turns")),
                verification=verification_word(raw.get("verification")),
                summary=_text(raw, "summary"),
                scoped=bool(raw.get("scoped", False)),
            )
        case other:
            return RawEvent(type=str(other), raw=raw)
