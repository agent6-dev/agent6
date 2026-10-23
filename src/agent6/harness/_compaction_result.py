# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Bound tool results and render compaction markers and distilled gists."""

from __future__ import annotations

import dataclasses
import json
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from agent6.harness import _panel, _verify_gate
from agent6.providers import CLAUDE_CODE_PERSIST_BYTES

# Every placeholder variant shares the prefix: idempotency checks and tests key on it.
ELISION_PREFIX = "<elided by context compaction"


ELISION_PLACEHOLDER = (
    "<elided by context compaction: this tool_result has been replaced "
    "with this short marker to keep the loop's cumulative input bounded. "
    "If you still need it, re-read only the part you need with a targeted "
    "read_file start_line/limit; do not re-issue the identical call.>"
)


# Distinguishable from the bare marker, so continued pressure can demote a gist to it.
ELISION_GIST_PREFIX = ELISION_PREFIX + " (distilled)"


# Placeholders stay in context, so the identity hint stays short.
ELISION_HINT_MAX_CHARS = 120


# The argument that identifies a call, tried in order; keyed on arguments, not tool
# names, so a tool added later is never anonymous in a compacted transcript.
IDENTIFYING_KEYS: Final = ("path", "argv", "symbol", "name", "id", "url", "query")


def call_label(tool_name: str, tool_input: Any) -> str:
    """Return a short identity for a tool call, such as "read_file src/foo.py".

    The placeholder hint and the `loop.compact.*` event payloads share it, so
    every surface can say what left the model's context.

    Args:
        tool_name: The tool's name.
        tool_input: The model's input to the tool.

    Returns:
        The name plus the identifying argument, clipped.
    """
    if not tool_name or not isinstance(tool_input, dict):
        return tool_name
    hint = ""
    if tool_name == "read_file":
        hint = str(tool_input.get("path", ""))
        start_line = tool_input.get("start_line")
        limit = tool_input.get("limit")
        if start_line or limit:
            hint += f" (start_line={start_line}, limit={limit})"
    else:
        for key in IDENTIFYING_KEYS:
            value = tool_input.get(key)
            if not value:  # absent, or present-but-empty: no identity to show
                continue
            hint = (
                shlex.join(str(a) for a in value) if isinstance(value, list | tuple) else str(value)
            )
            break
    if len(hint) > ELISION_HINT_MAX_CHARS:
        hint = hint[:ELISION_HINT_MAX_CHARS] + "..."
    return f"{tool_name} {hint}".rstrip()


def elision_placeholder(tool_name: str, tool_input: Any) -> str:
    """Return the tier-1 placeholder naming the elided call.

    The model can re-issue or skip the call without scanning up for the paired
    tool_use block.

    Args:
        tool_name: The tool's name; "" for an orphan result.
        tool_input: The model's input to the tool.

    Returns:
        The placeholder, the generic marker for an unknown tool.
    """
    if not tool_name or not isinstance(tool_input, dict):
        return ELISION_PLACEHOLDER
    described = call_label(tool_name, tool_input)
    # Only read_file takes a range, so only it can be told to re-read one.
    retry = (
        "re-read only the part you need (read_file with a targeted start_line/limit)"
        if tool_name == "read_file"
        else "re-run it with a narrower scope"
    )
    return (
        f"{ELISION_PREFIX}: the result of {described} was replaced with this "
        f"short marker to keep the loop's cumulative input bounded. If you "
        f"still need it, {retry}; do not re-issue the identical call.>"
    )


# Measured (bench/longhorizon FINDINGS #1): under a small window, bare elision of
# reference docs halves a retention task's score (0.921 -> 0.425), so a large read
# decays content -> gist -> bare marker. The caps bound the distiller call per drop
# event; a hot file (protect_paths) is never gisted, since a stale gist misleads.
GIST_MIN_SOURCE_CHARS = 2_000  # below this the content is nearly gist-sized


GIST_MAX_CHARS = 400  # per gist, clipped


GIST_FILE_SLICE_CHARS = 8_000  # per-file head sent to the distiller


GIST_INPUT_CAP_CHARS = 24_000  # total distiller input per drop event


GIST_MAX_FILES_PER_CALL = 12


@dataclasses.dataclass(frozen=True, slots=True)
class GistRequest:
    """One file whose about-to-be-elided read_file content is to be distilled.

    Attributes:
        path: The file's path as the call named it.
        content: The head of the file text sent to the distiller.
    """

    path: str
    content: str


# The impure seam, called once per drop event; a path it misses gets the bare marker.
Gister = Callable[[tuple[GistRequest, ...]], Mapping[str, str]]


def elision_gist_placeholder(described: str, gist: str) -> str:
    """Return the tier-1 placeholder that keeps a distilled gist of the elided read.

    The gist and bare markers carry the same `call_label`, so the conversation
    differ reads a demotion as one, not as a fresh elision.

    Args:
        described: The call's `call_label`.
        gist: The distilled gist.

    Returns:
        The placeholder.
    """
    return (
        f"{ELISION_GIST_PREFIX}: the result of {described} was replaced "
        f"by this distilled gist; if the gist is not enough, re-read only "
        f"the part you need (read_file with a targeted start_line/limit).\ngist: {gist}>"
    )


def read_file_text_from_result(raw: str) -> str:
    """Return the file text inside a serialized read_file tool_result.

    The distiller sees file text, not JSON escapes.

    Args:
        raw: The serialized result.

    Returns:
        The `content` field, the truncation envelope's `head`, "" for an error
        result, else the raw payload.
    """
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, str):
            return content
        head = data.get("head")
        if data.get("_tool_result_truncated") and isinstance(head, str):
            return head
        if "error" in data:
            return ""
    return raw


def parse_gist_lines(text: str, paths: Sequence[str]) -> dict[str, str]:
    """Return path to gist from the distiller's one-line-per-file reply.

    List markers and backticks around the path are tolerated.

    Args:
        text: The distiller's reply.
        paths: The paths asked for; any other is ignored.

    Returns:
        The gists for the paths the reply names.
    """
    wanted = set(paths)
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        head, sep, rest = stripped.partition(":")
        if not sep:
            continue
        head = head.strip().strip("`")
        rest = rest.strip()
        if head in wanted and rest:
            out[head] = rest
    return out


# 60_000 bytes of UTF-8 (~15k tokens) fits most source files whole; bytes, because the
# Claude Code provider's persistence threshold that displaces it is measured in bytes.
TOOL_RESULT_CAP_BYTES = 60_000


# A Claude Code turn's tool_result carries the trailing notices too, and the whole
# stays under the persist threshold with the notices at their largest.
CLAUDE_CODE_NOTICE_ROOM_BYTES = (
    4 * _verify_gate.VERIFY_TAIL_CHARS + _panel.REVIEW_NOTICE_BYTES + 4_000
)


CLAUDE_CODE_RESULT_CAP_BYTES = CLAUDE_CODE_PERSIST_BYTES - CLAUDE_CODE_NOTICE_ROOM_BYTES


def cap_tool_result(content: str, *, tool_name: str, cap: int = TOOL_RESULT_CAP_BYTES) -> str:
    """Cap a serialized tool_result payload without producing malformed JSON.

    A payload over the cap becomes a JSON envelope that says it was truncated,
    how many chars were shown of the total, the head of the content, and what
    to call next; a raw mid-JSON slice reads as a partial result the model
    re-calls for.

    Args:
        content: The serialized result.
        tool_name: The tool's name; picks the guidance.
        cap: The bound in bytes of UTF-8.

    Returns:
        The payload unchanged when it fits, else the envelope.
    """
    if len(content.encode()) <= cap:
        return content
    if tool_name == "read_file":
        guidance = (
            "Use `read_file` again with `start_line` and `limit` to read the rest"
            " of the file in chunks. Do NOT re-call with identical arguments"
            " expecting a different result - you will get the same truncated"
            " head and waste budget."
        )
    elif tool_name in ("run_command", "run_verify_command"):
        guidance = (
            "Re-run with a narrower scope (e.g. a single test, a narrower"
            " search, head/tail) to get a result that fits. Do NOT re-call"
            " with identical arguments expecting different output."
        )
    else:
        guidance = (
            "Re-call with arguments that produce less output. Do NOT re-call"
            " with identical arguments expecting different output."
        )

    def envelope(head_len: int) -> str:
        head = content[:head_len]
        return json.dumps(
            {
                "_tool_result_truncated": True,
                "tool": tool_name,
                "shown_chars": len(head),
                "total_chars": len(content),
                "head": head,
                "guidance": guidance,
            },
            ensure_ascii=False,
        )

    # Bisect on encoded length: escapes and wide characters make a raw-char budget
    # overshoot the cap (118k emitted against 60k observed).
    lo, hi = 0, min(len(content), cap)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(envelope(mid).encode()) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return envelope(lo)
