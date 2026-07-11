# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context-window management for the agent loop.

Two tiers keep a long run inside the model's context window:

- tier 1 (`compact_old_tool_results`): at `DROP_BLOCKS_AT_CHARS` the oldest
  tool_result blocks are replaced by `ELISION_PLACEHOLDER`.
- tier 2 (`context_chars` vs `SUMMARISE_AT_CHARS`): the elided history is
  summarised and the conversation restarts from (task + summary).

`cap_tool_result` separately bounds a single tool_result so one huge payload
cannot blow the budget on the turn it arrives. All three are pure functions of
the message list; the loop owns the policy of when to call them.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Stable prefix shared by every placeholder variant: idempotency checks and
# tests key on it.
ELISION_PREFIX = "<elided by context compaction"

ELISION_PLACEHOLDER = (
    "<elided by context compaction: this tool_result has been replaced "
    "with this short marker to keep the loop's cumulative input bounded. "
    "Re-call the tool with the same args if you still need the content.>"
)

# How much of a tool arg the placeholder echoes. Placeholders stay in context,
# so the identity hint must stay short.
_ELISION_HINT_MAX_CHARS = 120


def elision_placeholder(tool_name: str, tool_input: Any) -> str:
    """Identity-bearing tier-1 placeholder.

    Names the elided call (tool + its key argument) so the model can re-issue
    or skip it without scanning up for the paired tool_use block; a bare
    marker made weak models lose track of WHAT was elided and re-read the
    wrong files. Unknown tool (orphan result) falls back to the generic
    marker.
    """
    if not tool_name or not isinstance(tool_input, dict):
        return ELISION_PLACEHOLDER
    hint = ""
    if tool_name == "read_file":
        hint = str(tool_input.get("path", ""))
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        if offset or limit:
            hint += f" (offset={offset}, limit={limit})"
    elif tool_name == "grep":
        hint = f"pattern {str(tool_input.get('pattern', ''))!r}"
    elif tool_name in ("list_dir", "outline"):
        hint = str(tool_input.get("path", ""))
    elif tool_name in ("find_definition", "find_references"):
        hint = str(tool_input.get("name", ""))
    if len(hint) > _ELISION_HINT_MAX_CHARS:
        hint = hint[:_ELISION_HINT_MAX_CHARS] + "..."
    described = f"{tool_name} {hint}".rstrip()
    return (
        f"{ELISION_PREFIX}: the result of {described} was replaced with this "
        f"short marker to keep the loop's cumulative input bounded. Re-call "
        f"{tool_name} with the same args if you still need the content.>"
    )


# per-tool-result cap. was a hard 20_000 char slice
# applied mid-JSON, which produced a malformed result the model could
# not parse. Weak models (Kimi K2.6 observed live) then concluded the
# tool result was "cut off" and re-called `read_file` repeatedly trying
# to see the rest, latching the loop-guard. The fix: lift the cap to
# 60_000 chars (~15k tokens, comfortably fits most source files) AND
# when truncation is unavoidable, wrap the result in a fresh,
# well-formed JSON object that explicitly tells the model what
# happened and how to get the rest.
TOOL_RESULT_CHAR_CAP = 60_000

# compaction thresholds (chars, not tokens - approximate; tokens
# are roughly chars/4 for English-shaped content).
DROP_BLOCKS_AT_CHARS = 256_000  # ~64k tokens of tool_result content
SUMMARISE_AT_CHARS = 768_000  # ~192k tokens: full context restart


def cap_tool_result(content: str, *, tool_name: str) -> str:
    """Cap a serialized tool_result payload at ``TOOL_RESULT_CHAR_CAP``
    chars without producing malformed JSON. If the payload is over the
    cap, wrap it in a new JSON envelope that tells the model:
    (a) the result was truncated, (b) how many chars were shown vs
    total, (c) the head of the original content, (d) actionable next
    steps. This prevents weak models from inferring "the tool itself
    returned a partial result, let me call it again"."""
    if len(content) <= TOOL_RESULT_CHAR_CAP:
        return content
    if tool_name == "read_file":
        guidance = (
            "Use `read_file` again with `offset` and `limit` to read the rest"
            " of the file in chunks. Do NOT re-call with identical arguments"
            " expecting a different result - you will get the same truncated"
            " head and waste budget."
        )
    elif tool_name in ("run_command", "run_verify_command"):
        guidance = (
            "Re-run with a narrower scope (e.g. a single test, smaller grep"
            " pattern, head/tail) to get a result that fits. Do NOT re-call"
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

    # Size the head by ENCODED length: json.dumps re-escapes quotes/backslashes,
    # so a raw-char budget overshoots the cap on escape-heavy content (observed
    # 118k emitted against the 60k cap). Encoded length is monotone in head
    # length and the empty head always fits, so bisect for the largest head
    # whose envelope fits (~16 dumps passes).
    lo, hi = 0, TOOL_RESULT_CHAR_CAP
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(envelope(mid)) <= TOOL_RESULT_CHAR_CAP:
            lo = mid
        else:
            hi = mid - 1
    return envelope(lo)


_CHECKOFF_FENCE_RE = re.compile(r"```checkoff\s*\n(.*?)\n```", re.DOTALL)


def parse_checkoff(text: str) -> tuple[list[str], list[str]]:
    """Extract a tier-2 compaction check-off from the summariser's output.

    The summariser is asked to append a fenced ```checkoff block holding
    ``{"completed_ids": [...], "new_tasks": [...]}`` so agent6 can mark finished
    tasks done and queue newly-discovered ones in the curator-owned DAG (the
    model rarely calls update_task itself -- observed live). Returns
    ``(completed_ids, new_task_titles)``. Best-effort and total: a missing or
    malformed block yields ``([], [])`` so a bad summary never breaks the run.
    """
    m = _CHECKOFF_FENCE_RE.search(text)
    if m is None:
        return [], []
    try:
        data = json.loads(m.group(1))
    except (ValueError, TypeError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    completed = [s for s in data.get("completed_ids", []) if isinstance(s, str) and s]
    new_tasks = [s.strip() for s in data.get("new_tasks", []) if isinstance(s, str) and s.strip()]
    return completed, new_tasks


def strip_checkoff(text: str) -> str:
    """Remove the ```checkoff block from a summary before it re-enters context;
    it is agent6 bookkeeping, not narrative the restarted worker should re-read."""
    return _CHECKOFF_FENCE_RE.sub("", text).strip()


def context_chars(messages: list[dict[str, Any]]) -> int:
    """Approximate the full character size of the conversation context.

    Sums string content plus, for structured content blocks, their text,
    tool_result content, and tool_use inputs -- i.e. everything that grows the
    context and is sent back to the model each turn, not just tool_result bytes.
    Used as the tier-2 (summarise-and-restart) trigger, which must measure
    something tier-1 elision does not already cap.
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                total += len(str(item.get("text", "") or ""))
                total += len(str(item.get("content", "") or ""))
                if item.get("type") == "tool_use":
                    total += len(str(item.get("input", "") or ""))
    return total


# First target header in a unified diff (`+++ b/PATH`) or a v4a patch
# (`*** Update|Add File: PATH`). apply_patch is one-file-per-call, so the
# first match is the file.
_PATCH_TARGET_RE = re.compile(
    r"^(?:\+\+\+ b/(?P<u>\S+)|\*\*\* (?:Update|Add) File: (?P<v>.+))$", re.MULTILINE
)


def recently_edited_paths(messages: list[dict[str, Any]], *, last_turns: int = 8) -> frozenset[str]:
    """Paths targeted by apply_edit / apply_patch in the last *last_turns*
    assistant messages: the files the worker is actively editing. Tier-1
    elision deprioritises their read_file results (see
    ``compact_old_tool_results``), because a placeholder there triggers a paid
    re-read before the very next edit. Best-effort: an apply_patch without a
    ``path`` argument falls back to the patch headers; an unparseable patch
    just goes unprotected.
    """
    out: set[str] = set()
    seen_assistant = 0
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        seen_assistant += 1
        if seen_assistant > last_turns:
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            name = item.get("name")
            tool_input = item.get("input")
            if name not in ("apply_edit", "apply_patch") or not isinstance(tool_input, dict):
                continue
            path = str(tool_input.get("path", "") or "")
            if not path and name == "apply_patch":
                m = _PATCH_TARGET_RE.search(str(tool_input.get("patch", "")))
                path = ((m.group("u") or m.group("v") or "") if m else "").strip()
            if path:
                out.add(path)
    return frozenset(out)


def _tool_use_index(messages: list[dict[str, Any]]) -> dict[str, tuple[str, Any]]:
    """id -> (tool name, input) for every tool_use block, pairing each
    tool_result to the call that produced it (placeholder identity + hot-file
    protection)."""
    out: dict[str, tuple[str, Any]] = {}
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                out[str(item.get("id", ""))] = (str(item.get("name", "")), item.get("input"))
    return out


def _tool_result_pointers(
    messages: list[dict[str, Any]],
) -> tuple[list[tuple[int, int, int]], int]:
    """((msg_idx, item_idx, size) per tool_result, total size) in order."""
    pointers: list[tuple[int, int, int]] = []
    total = 0
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item_idx, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            raw_content = item.get("content")
            size = len(raw_content) if isinstance(raw_content, str) else len(str(raw_content))
            pointers.append((msg_idx, item_idx, size))
            total += size
    return pointers, total


def compact_old_tool_results(
    messages: list[dict[str, Any]],
    *,
    max_total_bytes: int,
    keep_recent: int = 2,
    protect_paths: frozenset[str] = frozenset(),
) -> int:
    """Elide old tool_result blocks once cumulative content exceeds the
    threshold. Walks messages oldest-first, replaces each tool_result's
    ``content`` with a short identity-bearing placeholder, stops once total
    size is back under ``max_total_bytes``. The most recent ``keep_recent``
    are always preserved, as is every tool_result in the last
    tool_result-bearing message: the loop compacts at top-of-iteration, before
    the provider call that would deliver that batch, so the model has never
    seen it and the placeholder's "re-call the tool" guidance would trigger a
    paid re-call cycle. (Keying on the final message alone is not enough: a
    trailing steer or nudge user message pushes the fresh, still undelivered
    results off the final index, and one turn can carry several such blocks.)

    ``protect_paths`` (the actively-edited set from ``recently_edited_paths``)
    deprioritises rather than exempts: read_file results for those paths are
    elided only after every other candidate, so the hot file's content
    survives as long as the budget allows but the hard bound still holds.

    Idempotent on already-elided entries. Returns the number of entries elided
    (for telemetry).
    """
    tool_uses = _tool_use_index(messages)
    pointers, total = _tool_result_pointers(messages)
    if total <= max_total_bytes or len(pointers) <= keep_recent:
        return 0

    def _is_protected(msg_idx: int, item_idx: int) -> bool:
        item = messages[msg_idx]["content"][item_idx]
        name, tool_input = tool_uses.get(str(item.get("tool_use_id", "")), ("", None))
        if name != "read_file" or not isinstance(tool_input, dict):
            return False
        return str(tool_input.get("path", "")) in protect_paths

    # The undelivered batch is always in the last tool_result-bearing message:
    # at top-of-iteration only text-only steer/nudge user messages can trail the
    # fresh results, and the delivering provider call runs after this compaction.
    # Exempt that whole message -- see docstring. Keying on the final message
    # index alone missed a trailing steer/nudge, and one turn can carry several
    # such blocks, so this is broader than keep_recent or the final index.
    last_tool_result_idx = max(msg_idx for msg_idx, _, _ in pointers)
    candidates = pointers[:-keep_recent]
    if protect_paths:
        # Protected reads go last, each group staying oldest-first.
        candidates = [c for c in candidates if not _is_protected(c[0], c[1])] + [
            c for c in candidates if _is_protected(c[0], c[1])
        ]
    elided_count = 0
    for msg_idx, item_idx, size in candidates:
        if total <= max_total_bytes:
            break
        if msg_idx == last_tool_result_idx:
            continue
        item = messages[msg_idx]["content"][item_idx]
        current = item.get("content")
        if isinstance(current, str) and current.startswith(ELISION_PREFIX):
            continue
        name, tool_input = tool_uses.get(str(item.get("tool_use_id", "")), ("", None))
        placeholder = elision_placeholder(name, tool_input)
        if size <= len(placeholder):
            # Replacing content already smaller than the placeholder would GROW
            # the total, defeating the point; skip it.
            continue
        item["content"] = placeholder
        total -= size - len(placeholder)
        elided_count += 1
    return elided_count
