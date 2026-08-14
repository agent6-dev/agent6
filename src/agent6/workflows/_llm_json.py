# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pull one structured JSON object out of an LLM's prose reply.

Shared by the review seats and the compare judge: a model asked for a JSON
verdict often wraps it in markdown fences, prose, or a stray example object,
so the parse tolerates all three and picks the object that carries the
caller's expected keys.
"""

from __future__ import annotations

import json
from typing import Any


def balanced_objects(text: str) -> list[str]:
    """Every top-level balanced ``{...}`` span in *text* (brace depth,
    honoring string literals + escapes)."""
    spans: list[str] = []
    depth = 0
    start: int | None = None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
    return spans


def extract_json(text: str, *, prefer: tuple[str, ...]) -> dict[str, Any] | None:
    """The LAST balanced object carrying any *prefer* key, else the last
    parseable dict, else None."""
    objs: list[dict[str, Any]] = []
    for span in balanced_objects(text):
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    for obj in reversed(objs):
        if any(key in obj for key in prefer):
            return obj
    return objs[-1] if objs else None
