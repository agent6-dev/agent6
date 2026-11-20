# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pull one structured JSON object out of an LLM's prose reply.

Shared by the reviewers and the compare judge: a model asked for a JSON
verdict often wraps it in markdown fences, prose, or a stray example object,
so the parse tolerates all three and picks the object that carries the
caller's expected keys.
"""

from __future__ import annotations

import json
from typing import Any

_DECODER = json.JSONDecoder()


def extract_json(text: str, *, prefer: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the last JSON object in the text carrying any preferred key.

    Every `{` is tried as an object's start. An object ending inside the text a broken
    object had read is skipped: a cut-off verdict's findings are not the reply.

    Args:
        text: The reply to scan.
        prefer: The keys the caller expects.

    Returns:
        That object, else the last object found, else None.
    """
    objs: list[dict[str, Any]] = []
    read_to = 0  # where the furthest-reading broken object stopped
    at = text.find("{")
    while at != -1:
        try:
            obj, end = _DECODER.raw_decode(text, at)
        except json.JSONDecodeError as exc:
            read_to = max(read_to, exc.pos)
            at = text.find("{", at + 1)
            continue
        if end > read_to:
            objs.append(obj)
        at = text.find("{", end)
    for obj in reversed(objs):
        if any(key in obj for key in prefer):
            return obj
    return objs[-1] if objs else None
