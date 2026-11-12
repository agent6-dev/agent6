# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The completion record a `read_only` session leaves: the deliverable of a bounded conversation."""

from __future__ import annotations

import json
import pathlib

import pydantic

from agent6 import portable, review_records

COMPLETION_NAME = "completion.json"


class CompletionError(ValueError):
    """The completion record is unreadable or malformed."""


class CompletionRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """What a bounded conversation delivered, written once at its end.

    Attributes:
        text: The last response's text, or the finish summary when the model finished.
        stop_reason: The session's end reason, as the journal's `session.stop` carries it.
        output_tokens: The output tokens of every attempt.
        calls: The provider attempts made, retries included.
        diagnostics: What the ceiling or the contract cut, one line each; empty when nothing was.
        session_id: The producing read-only session, empty in legacy records.
        review_binding: The parent-assigned panel, seat and target, or null outside a panel.
        response_stop_reason: The last provider response's stop reason.
        response_output_tokens: The last response's output tokens.
        reasoning_chars: The last response's reasoning character count.
        tool_uses: Whether the last response requested tools.
    """

    text: str
    stop_reason: str
    output_tokens: int = pydantic.Field(ge=0)
    calls: int = pydantic.Field(ge=0)
    diagnostics: tuple[str, ...] = ()
    session_id: str = ""
    review_binding: review_records.ReviewBinding | None = None
    response_stop_reason: str = ""
    response_output_tokens: int = 0
    reasoning_chars: int = 0
    tool_uses: bool = False


def write_completion(session_dir: pathlib.Path, record: CompletionRecord) -> pathlib.Path:
    """Write the session's completion record.

    Args:
        session_dir: The session directory.
        record: The record.

    Returns:
        The file's path.
    """
    path = session_dir / COMPLETION_NAME
    portable.atomic_write(path, (record.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return path


def read_completion(session_dir: pathlib.Path) -> CompletionRecord | None:
    """Read the session's completion record.

    Args:
        session_dir: The session directory.

    Returns:
        The record, or None when the session left none.

    Raises:
        CompletionError: The file is unreadable or malformed.
    """
    path = session_dir / COMPLETION_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CompletionError(f"cannot read {path}: {exc}") from exc
    try:
        return CompletionRecord.model_validate(json.loads(text))
    except (json.JSONDecodeError, pydantic.ValidationError) as exc:
        raise CompletionError(f"malformed completion record {path}: {exc}") from exc


__all__ = [
    "COMPLETION_NAME",
    "CompletionError",
    "CompletionRecord",
    "read_completion",
    "write_completion",
]
