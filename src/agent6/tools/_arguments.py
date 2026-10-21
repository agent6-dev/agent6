# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Argument repair: recover stringified containers and word validation errors for the model."""

from __future__ import annotations

import json
import re
from typing import Any

import pydantic

# A backslash JSON defines no escape for: a regex the model typed inside a JSON string.
_LONE_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')


def coerce_stringified_args(
    raw_input: dict[str, Any], exc: pydantic.ValidationError
) -> dict[str, Any] | None:
    """Recover a tool call whose array or object argument arrived as a JSON string.

    For each top-level field the validation error names whose value is a str, the string's head
    is parsed as JSON (`raw_decode` tolerates trailing junk; a lone backslash reads literally)
    and substituted when it is a container. A field the schema declares as a string is safe: a
    wrong substitution fails re-validation and the caller re-raises the original error.

    Args:
        raw_input: The arguments as they arrived.
        exc: The validation error they raised.

    Returns:
        A coerced copy, or None when nothing was coercible.
    """
    decoder = json.JSONDecoder()
    coerced: dict[str, Any] | None = None
    for err in exc.errors():
        loc = err.get("loc") or ()
        key = loc[0] if loc else None
        if not isinstance(key, str):
            continue
        val = raw_input.get(key)
        if not isinstance(val, str):
            continue
        try:
            parsed, _ = decoder.raw_decode(val.strip())
        except ValueError:
            try:
                parsed, _ = decoder.raw_decode(_LONE_BACKSLASH.sub(r"\\\\", val.strip()))
            except ValueError:
                continue
        if not isinstance(parsed, dict | list):
            continue
        if coerced is None:
            coerced = dict(raw_input)
        coerced[key] = parsed
    return coerced


# pydantic's container words in JSON's, for a model that writes JSON.
_JSON_WORDS = {
    "tuple_type": "expected an array",
    "list_type": "expected an array",
    "too_short": "expected a non-empty array",
    "dict_type": "expected an object",
    "model_type": "expected an object",
}


def invalid_arguments(exc: pydantic.ValidationError) -> str:
    """Return the real argument problems as one line, for the model and the log.

    A container error caused by an invalid item is dropped: the item's own problem says what to fix.

    Args:
        exc: The validation error the tool call's arguments raised.

    Returns:
        "invalid arguments: " then the problems joined by "; ". Each problem is the dotted field
        and the message without pydantic's "Value error, " lead and docs URL, a container's in
        JSON's words.
    """
    errors = exc.errors(include_url=False)
    item_locs = [tuple(e["loc"]) for e in errors]
    parts: list[str] = []
    for err in errors:
        loc = tuple(err["loc"])
        if err["type"] == "too_short" and any(
            other != loc and other[: len(loc)] == loc for other in item_locs
        ):
            continue
        field = ".".join(str(p) for p in loc) or "arguments"
        msg = _JSON_WORDS.get(err["type"]) or str(err["msg"]).removeprefix("Value error, ")
        if err["type"] == "too_long":
            ctx = err.get("ctx") or {}
            msg = f"expected at most {ctx['max_length']} items in the array"
        parts.append(f"{field}: {msg}")
    return "invalid arguments: " + "; ".join(parts)
