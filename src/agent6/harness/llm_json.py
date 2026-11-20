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
import re
from typing import Any

_DECODER = json.JSONDecoder()
# Where a JSON object can start: a brace that opens a key or closes at once. Any other is prose.
_OBJECT_START = re.compile(r'\{\s*["}]')


def extract_json(text: str, *, prefer: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the last JSON object in the text carrying any preferred key.

    A brace that opens no key is passed over as prose. An object that opens a key and then
    breaks is the reply's own JSON gone wrong, so nothing is returned: an object quoted
    inside it, or cut off with it, must not stand in for the answer.

    Args:
        text: The reply to scan.
        prefer: The keys the caller expects.

    Returns:
        That object, else the last object found; None for no object or a broken one.
    """
    objs: list[dict[str, Any]] = []
    at = text.find("{")
    while at != -1:
        if _OBJECT_START.match(text, at) is None:
            at = text.find("{", at + 1)
            continue
        try:
            obj, end = _DECODER.raw_decode(text, at)
        except (json.JSONDecodeError, RecursionError):
            return None  # broken, or nested past the decoder's depth
        objs.append(obj)
        at = text.find("{", end)
    for obj in reversed(objs):
        if any(key in obj for key in prefer):
            return obj
    return objs[-1] if objs else None
