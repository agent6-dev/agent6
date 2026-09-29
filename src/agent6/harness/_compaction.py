# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context-window management for the agent loop: the pure compaction rules.

Tier 1 (`compact_old_tool_results`) replaces the oldest tool_result blocks
with placeholders at `DROP_BLOCKS_AT_CHARS`; a large read_file result decays
through a distilled-gist placeholder first when the caller provides a
`gister`. Tier 2 (`context_chars` against `SUMMARISE_AT_CHARS`) is the
summarise-and-restart the driver in `_compactor` runs. `cap_tool_result`
bounds a single tool_result on the turn it arrives. The loop owns when to
call these and supplies the one impure seam, the `gister`.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agent6.harness._conversation import (
    AssistantTurn,
    Conversation,
    ToolResultItem,
    Turn,
    UserTurn,
)
from agent6.harness._panel import REVIEW_NOTICE_BYTES
from agent6.harness._verify_gate import VERIFY_TAIL_CHARS
from agent6.providers import CLAUDE_CODE_PERSIST_BYTES, Provider
from agent6.providers.types import ToolDefinition
from agent6.tools.schema import AskUserInput

# Every placeholder variant shares the prefix: idempotency checks and tests key on it.
ASK_USER_TOOL = AskUserInput.TOOL_NAME
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
_ELISION_HINT_MAX_CHARS = 120


# The argument that identifies a call, tried in order; keyed on arguments, not tool
# names, so a tool added later is never anonymous in a compacted transcript.
_IDENTIFYING_KEYS: Final = ("path", "argv", "symbol", "name", "id", "url", "query")


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
        for key in _IDENTIFYING_KEYS:
            value = tool_input.get(key)
            if not value:  # absent, or present-but-empty: no identity to show
                continue
            hint = (
                shlex.join(str(a) for a in value) if isinstance(value, list | tuple) else str(value)
            )
            break
    if len(hint) > _ELISION_HINT_MAX_CHARS:
        hint = hint[:_ELISION_HINT_MAX_CHARS] + "..."
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class CompactionStats:
    """What one tier-1 pass did; a count is the length of its tuple.

    Attributes:
        elided_calls: The `call_label` of each tool_result elided.
        gist_paths: The paths whose elision kept a gist.
        demoted_paths: The paths whose gist was demoted to the bare marker.
        deduped_calls: The `call_label` of each duplicate result replaced.
    """

    elided_calls: tuple[str, ...] = ()
    gist_paths: tuple[str, ...] = ()
    demoted_paths: tuple[str, ...] = ()
    deduped_calls: tuple[str, ...] = ()


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
CLAUDE_CODE_NOTICE_ROOM_BYTES = 4 * VERIFY_TAIL_CHARS + REVIEW_NOTICE_BYTES + 4_000
CLAUDE_CODE_RESULT_CAP_BYTES = CLAUDE_CODE_PERSIST_BYTES - CLAUDE_CODE_NOTICE_ROOM_BYTES

# Chars, not tokens: tokens are roughly chars/4 for English-shaped content.
DROP_BLOCKS_AT_CHARS = 256_000  # ~64k tokens of tool_result content
SUMMARISE_AT_CHARS = 768_000  # ~192k tokens: full context restart


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


_CHECKOFF_FENCE_RE = re.compile(r"```checkoff\s*\n(.*?)\n```", re.DOTALL)


def parse_checkoff(text: str) -> tuple[list[str], list[str]]:
    """Extract the tier-2 check-off block from the summariser's reply.

    The summariser appends a fenced checkoff block holding `completed_ids` and
    `new_tasks`, so the DAG stays accurate without the worker calling
    update_task.

    Args:
        text: The summariser's reply.

    Returns:
        (completed ids, new task titles); ([], []) for a missing or malformed
        block, so a bad summary never breaks the run.
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
    return _nonempty_strs(data.get("completed_ids")), _nonempty_strs(data.get("new_tasks"))


def _nonempty_strs(value: object) -> list[str]:
    """Return the stripped, non-empty strings in a JSON value.

    Args:
        value: The parsed field; `null`, a number or a bool are natural
            summariser output.

    Returns:
        The strings, [] when the value is not a list.
    """
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def strip_checkoff(text: str) -> str:
    """Remove the checkoff block from a summary before it re-enters context.

    Args:
        text: The summariser's reply.

    Returns:
        The narrative alone, stripped.
    """
    return _CHECKOFF_FENCE_RE.sub("", text).strip()


def context_chars(conversation: Conversation) -> int:
    """Return the character size of the conversation context.

    Notice text, tool_result content and every value of every assistant raw
    block count, since `to_wire` sends those blocks verbatim; counting known
    keys alone would score a thinking block as zero. The tier-2 trigger
    measures this against about 80% of the model's window.

    Args:
        conversation: The loop's history.

    Returns:
        The sum over the turns.
    """
    return sum(turn_chars(turn) for turn in conversation.turns)


def turn_chars(turn: Turn) -> int:
    """Return one turn's contribution to `context_chars`.

    Args:
        turn: The turn.

    Returns:
        Its character count, "type" keys excluded.
    """
    if isinstance(turn, AssistantTurn):
        total = 0
        for item in turn.raw_content:
            if not isinstance(item, dict):
                total += len(str(item))
                continue
            # "type" is the discriminator, not payload.
            total += sum(
                len(v if isinstance(v, str) else str(v))
                for k, v in item.items()
                if k != "type" and v is not None
            )
        return total
    return sum(
        len(item.content if isinstance(item, ToolResultItem) else item.text) for item in turn.items
    )


# The verbatim tail a tier-2 restart keeps, about 20k tokens; `[context]
# keep_recent_chars` overrides.
KEEP_RECENT_CHARS = 80_000


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Context compaction as the run configures it.

    Attributes:
        drop_at_chars: Tier 1 turns the oldest tool results into placeholders
            past this.
        summarise_at_chars: Tier 2 summarises and restarts past this.
        tool_result_cap_bytes: The bound on one result before it enters the
            conversation.
        keep_recent_chars: The verbatim tail a restart keeps.
        keep_thinking_turns: Thinking blocks are dropped from assistant turns
            older than this many at tier-1 moments; 0 keeps all.
        elision_gists: A large read decays through a model-written gist first.
        summary_max_tokens: The summariser's output cap.
        summariser: The reviewer role's provider; the worker's when None.
    """

    drop_at_chars: int = DROP_BLOCKS_AT_CHARS
    summarise_at_chars: int = SUMMARISE_AT_CHARS
    tool_result_cap_bytes: int = TOOL_RESULT_CAP_BYTES
    keep_recent_chars: int = KEEP_RECENT_CHARS
    keep_thinking_turns: int = 0
    elision_gists: bool = True
    summary_max_tokens: int = 2048
    summariser: Provider | None = None


def strip_old_thinking(conversation: Conversation, *, keep_turns: int) -> tuple[int, int]:
    """Drop thinking blocks from assistant turns older than the newest few.

    Anthropic requires the signed thinking block of a tool_use still being
    answered, so callers pass at least 1.

    Args:
        conversation: The loop's history.
        keep_turns: How many of the newest assistant turns keep their thinking.

    Returns:
        (turns stripped, chars removed).
    """
    assistant_idxs = [
        i for i, turn in enumerate(conversation.turns) if isinstance(turn, AssistantTurn)
    ]
    n = chars = 0
    for idx in assistant_idxs[:-keep_turns]:
        removed = conversation.strip_thinking(idx)
        if removed:
            n += 1
            chars += removed
    return n, chars


def request_prefix_chars(system: str, tools: Sequence[ToolDefinition]) -> int:
    """Return the chars every request carries besides the conversation.

    The window bounds the whole request, so a threshold on the conversation
    alone leaves a band the size of this prefix where the loop sees room and
    the provider refuses. AGENTS.md rides in the system prompt whole.

    Args:
        system: The system prompt.
        tools: The tool definitions.

    Returns:
        The system prompt's length plus each tool's name, description and
        compact schema.
    """
    return len(system) + sum(
        len(t.name) + len(t.description) + len(json.dumps(t.input_schema, separators=(",", ":")))
        for t in tools
    )


def recent_tail_start(turns: Sequence[Turn], cap_chars: int) -> int:
    """Return the index where a tier-2 restart's verbatim tail begins.

    A safe start is any turn except a user turn carrying tool_results, which
    answers the assistant turn before it. Turn 0, the task, is never part of
    the tail; the restart keeps it separately.

    Args:
        turns: The conversation's turns.
        cap_chars: The tail's size cap.

    Returns:
        The start of the largest tail of whole turns within the cap on a safe
        boundary; `len(turns)` when nothing is kept.
    """
    if cap_chars <= 0:
        return len(turns)
    total = 0
    start = len(turns)
    i = len(turns) - 1
    while i >= 1:
        size = turn_chars(turns[i])
        if total + size > cap_chars:
            break
        total += size
        start = i
        i -= 1
    while start < len(turns) and _starts_with_results(turns[start]):
        start += 1
    if start == len(turns):
        # The newest exchange exceeds the cap alone: keep it anyway, since paraphrasing
        # undelivered results away is the one loss the tail exists to prevent.
        assistant_idxs = [i for i in range(1, len(turns)) if isinstance(turns[i], AssistantTurn)]
        if assistant_idxs:
            start = assistant_idxs[-1]
    return start


def _starts_with_results(turn: Turn) -> bool:
    return isinstance(turn, UserTurn) and any(
        isinstance(item, ToolResultItem) for item in turn.items
    )


# Target headers in a unified diff (`+++ b/PATH`) or a v4a patch (`*** Update File:`).
_PATCH_TARGET_RE = re.compile(
    r"^(?:\+\+\+ b/(?P<u>\S+)|\*\*\* (?:Update|Add) File: (?P<v>.+))$", re.MULTILINE
)


def recently_edited_paths(conversation: Conversation, *, last_turns: int = 8) -> frozenset[str]:
    """Return the paths apply_edit and apply_patch targeted in the newest turns.

    Tier-1 elision deprioritises these files' reads, since a placeholder there
    triggers a paid re-read before the next edit. An apply_patch without a
    `path` argument falls back to the patch headers; an unparseable patch goes
    unprotected.

    Args:
        conversation: The loop's history.
        last_turns: How many of the newest assistant turns count.

    Returns:
        The paths.
    """
    out: set[str] = set()
    seen_assistant = 0
    for turn in reversed(conversation.turns):
        if not isinstance(turn, AssistantTurn):
            continue
        seen_assistant += 1
        if seen_assistant > last_turns:
            break
        for tu in turn.tool_uses:
            if tu.name not in ("apply_edit", "apply_patch") or not isinstance(tu.input, dict):
                continue
            path = str(tu.input.get("path", "") or "")
            if not path and tu.name == "apply_patch":
                for match in _PATCH_TARGET_RE.finditer(str(tu.input.get("patch", ""))):
                    target = (match.group("u") or match.group("v") or "").strip()
                    if target:
                        out.add(target)
            if path:
                out.add(path)
    return frozenset(out)


def _tool_result_pointers(
    conversation: Conversation,
) -> tuple[list[tuple[int, int, int]], int]:
    """Return every tool_result's position and size, in order.

    Args:
        conversation: The loop's history.

    Returns:
        ((turn index, item index, size) per result, total size).
    """
    pointers: list[tuple[int, int, int]] = []
    total = 0
    for turn_idx, turn in enumerate(conversation.turns):
        if isinstance(turn, AssistantTurn):
            continue
        for item_idx, item in enumerate(turn.items):
            if not isinstance(item, ToolResultItem):
                continue
            pointers.append((turn_idx, item_idx, len(item.content)))
            total += len(item.content)
    return pointers, total


def count_elisions(conversation: Conversation) -> tuple[int, int]:
    """Count the elision markers in the context, and the live gists among them.

    A resumed or forked execution re-announces these, since its fresh log has
    no compaction events to fold.

    Args:
        conversation: The loop's history.

    Returns:
        (markers, gists).
    """
    elided = gists = 0
    for turn in conversation.turns:
        for item in getattr(turn, "items", ()):
            body = getattr(item, "content", "")
            if isinstance(body, str) and body.startswith(ELISION_PREFIX):
                elided += 1
                gists += body.startswith(ELISION_GIST_PREFIX)
    return elided, gists


def compact_old_tool_results(
    conversation: Conversation,
    *,
    max_total_bytes: int,
    keep_recent: int = 2,
    protect_paths: frozenset[str] = frozenset(),
    gister: Gister | None = None,
) -> CompactionStats:
    """Elide old tool_result blocks once their content exceeds the threshold.

    The walk is oldest-first and stops once the total is back under the bound.
    The newest `keep_recent` results stay, as does every result in the newest
    result turn until an assistant turn has consumed it: the loop compacts
    before the provider call that would deliver it, and a placeholder there
    triggers a paid re-call. Protected reads are elided only after every other
    candidate. With a gister, a large unprotected read decays to a gist
    placeholder; when the total still exceeds the bound, gists are demoted
    oldest-first to the bare marker, after even the protected reads: losing a
    gist costs correctness, losing a hot read one paid re-read. Idempotent on
    already-elided entries.

    Args:
        conversation: The loop's history, rewritten in place.
        max_total_bytes: The bound on tool_result content.
        keep_recent: How many of the newest results always stay.
        protect_paths: The actively-edited paths from `recently_edited_paths`.
        gister: The distiller; None elides to bare markers.

    Returns:
        What the pass did.
    """
    pointers, total = _tool_result_pointers(conversation)
    if total <= max_total_bytes or len(pointers) <= keep_recent:
        return CompactionStats()

    # Dedup first: freeing duplicate bytes is lossless and may spare real content.
    deduped_calls = _dedupe_identical_results(conversation, pointers, keep_recent=keep_recent)
    if deduped_calls:
        pointers, total = _tool_result_pointers(conversation)
        if total <= max_total_bytes:
            return CompactionStats(deduped_calls=deduped_calls)

    def _is_protected(turn_idx: int, item_idx: int) -> bool:
        call = _result_at(conversation, turn_idx, item_idx).for_call
        if call.name != "read_file" or not isinstance(call.input, dict):
            return False
        return str(call.input.get("path", "")) in protect_paths

    undelivered_turn = _undelivered_result_turn(conversation, pointers)
    older = pointers[:-keep_recent] if keep_recent else pointers
    candidates = [
        c
        for c in older
        if c[0] != undelivered_turn
        and not _is_operator_answer(_result_at(conversation, c[0], c[1]))
    ]
    if protect_paths:
        # Protected reads go last, each group staying oldest-first.
        candidates = [c for c in candidates if not _is_protected(c[0], c[1])] + [
            c for c in candidates if _is_protected(c[0], c[1])
        ]

    walk = _Tier1Pass(
        conversation=conversation,
        max_total_bytes=max_total_bytes,
        protect_paths=protect_paths,
        candidates=candidates,
        total=total,
    )
    walk.plan()
    if gister is not None:
        walk.distill(gister)
    walk.apply()
    walk.demote()
    return CompactionStats(
        elided_calls=tuple(walk.elided_calls),
        gist_paths=tuple(walk.gist_paths),
        demoted_paths=tuple(walk.demoted_paths),
        deduped_calls=deduped_calls,
    )


def _is_operator_answer(item: ToolResultItem) -> bool:
    """Return whether the result is the operator's answer to an `ask_user`.

    An answer is exempt from elision and dedup: it is a binding ruling that
    exists nowhere else in the context, and a re-run would re-ask the operator.

    Args:
        item: The result.

    Returns:
        True for an `ask_user` result.
    """
    return item.for_call.name == ASK_USER_TOOL


def _result_at(conversation: Conversation, turn_idx: int, item_idx: int) -> ToolResultItem:
    turn = conversation.turns[turn_idx]
    assert not isinstance(turn, AssistantTurn)
    item = turn.items[item_idx]
    assert isinstance(item, ToolResultItem)  # pointers only ever index tool_results
    return item


def _undelivered_result_turn(
    conversation: Conversation, pointers: list[tuple[int, int, int]]
) -> int | None:
    """Return the newest result turn when no assistant turn has consumed it yet.

    Args:
        conversation: The loop's history.
        pointers: The result positions from `_tool_result_pointers`.

    Returns:
        The turn index, or None once an assistant turn follows it.
    """
    last_result = max(turn_idx for turn_idx, _, _ in pointers)
    if any(isinstance(turn, AssistantTurn) for turn in conversation.turns[last_result + 1 :]):
        return None
    return last_result


# Below this a duplicate's placeholder is barely smaller than the content it replaces.
_DEDUP_MIN_CHARS = 200


def _dedupe_identical_results(
    conversation: Conversation,
    pointers: list[tuple[int, int, int]],
    *,
    keep_recent: int,
) -> tuple[str, ...]:
    """Replace every copy but the newest of a byte-identical result with a placeholder.

    It runs only where history is being rewritten anyway, so it adds no
    cache-invalidation points. The placeholder points at no other block, since
    the elision pass can take the newest copy in the same call. The undelivered
    final batch, the newest `keep_recent` results, already-elided placeholders,
    operator answers and results under `_DEDUP_MIN_CHARS` are never rewritten.

    Args:
        conversation: The loop's history, rewritten in place.
        pointers: The result positions from `_tool_result_pointers`.
        keep_recent: How many of the newest results always stay.

    Returns:
        The `call_label` of each copy replaced.
    """
    if len(pointers) <= keep_recent:
        return ()
    undelivered_turn = _undelivered_result_turn(conversation, pointers)
    recent = pointers[-keep_recent:] if keep_recent else []
    exempt = {(t, i) for t, i, _ in recent}
    by_key: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for turn_idx, item_idx, _size in pointers:
        item = _result_at(conversation, turn_idx, item_idx)
        if (
            item.content.startswith(ELISION_PREFIX)
            or len(item.content) < _DEDUP_MIN_CHARS
            or _is_operator_answer(item)
        ):
            continue
        call = item.for_call
        try:
            input_key = json.dumps(call.input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            continue
        by_key.setdefault((call.name, input_key, item.content), []).append((turn_idx, item_idx))
    labels: list[str] = []
    for locs in by_key.values():
        for turn_idx, item_idx in locs[:-1]:  # every copy but the newest
            if turn_idx == undelivered_turn or (turn_idx, item_idx) in exempt:
                continue
            item = _result_at(conversation, turn_idx, item_idx)
            label = call_label(item.for_call.name, item.for_call.input)
            marker = (
                f"{ELISION_PREFIX} (duplicate): {label} returned byte-identical"
                " content again later in this conversation. If you still need it,"
                " re-read only the part you need; do not re-issue the identical call.>"
            )
            if len(marker) >= len(item.content):
                # A long path can make the marker bigger than the result it replaces.
                continue
            conversation.set_result_content(turn_idx, item_idx, marker)
            labels.append(label)
    return tuple(labels)


@dataclass(slots=True)
class _Tier1Pass:
    """The state shared by the phases of one tier-1 pass.

    Attributes:
        conversation: The loop's history, rewritten in place.
        max_total_bytes: The bound on tool_result content.
        protect_paths: The actively-edited paths, never gisted.
        candidates: The results the pass may rewrite, oldest-first, protected
            reads last.
        total: The tool_result content size as the pass stands.
        victims: The candidates the plan elides.
        gist_headroom: What a gist may add back on top of the bare plan.
        gists: The distilled gists by victim position.
        elided_calls: The `call_label` of each result elided.
        gist_paths: The paths whose elision kept a gist.
        demoted_paths: The paths whose gist was demoted.
    """

    conversation: Conversation
    max_total_bytes: int
    protect_paths: frozenset[str]
    candidates: list[tuple[int, int, int]]
    total: int
    victims: list[tuple[int, int, int]] = field(default_factory=list)
    gist_headroom: int = 0
    gists: dict[tuple[int, int], str] = field(default_factory=dict)
    elided_calls: list[str] = field(default_factory=list)
    gist_paths: list[str] = field(default_factory=list)
    demoted_paths: list[str] = field(default_factory=list)

    def _item(self, turn_idx: int, item_idx: int) -> ToolResultItem:
        return _result_at(self.conversation, turn_idx, item_idx)

    def plan(self) -> None:
        """Pick the victims under bare-placeholder accounting, the maximum shrink.

        Nothing is mutated yet, so the distiller can still read the content.
        """
        planned = self.total
        for turn_idx, item_idx, size in self.candidates:
            if planned <= self.max_total_bytes:
                break
            item = self._item(turn_idx, item_idx)
            if item.content.startswith(ELISION_PREFIX):
                continue
            placeholder = elision_placeholder(item.for_call.name, item.for_call.input)
            if size <= len(placeholder):
                # Content smaller than the placeholder would grow the total.
                continue
            self.victims.append((turn_idx, item_idx, size))
            planned -= size - len(placeholder)
        # What a gist may add back on top of the bare-placeholder plan.
        self.gist_headroom = self.max_total_bytes - planned

    def distill(self, gister: Gister) -> None:
        """Make one batched distiller call over the eligible victims.

        Eligible is a large unprotected read_file result, the newest read per
        path, largest files first under the input caps.

        Args:
            gister: The distiller.
        """
        newest_by_path: dict[str, tuple[int, int, int]] = {}
        for turn_idx, item_idx, size in self.victims:
            call = self._item(turn_idx, item_idx).for_call
            if call.name != "read_file" or not isinstance(call.input, dict):
                continue
            path = str(call.input.get("path", ""))
            if not path or path in self.protect_paths or size < GIST_MIN_SOURCE_CHARS:
                continue
            newest_by_path[path] = (turn_idx, item_idx, size)  # victims are oldest-first
        batch: list[GistRequest] = []
        keys: dict[str, tuple[int, int]] = {}
        input_budget = GIST_INPUT_CAP_CHARS
        by_size = sorted(newest_by_path.items(), key=lambda kv: kv[1][2], reverse=True)
        for path, (turn_idx, item_idx, _size) in by_size:
            if len(batch) >= GIST_MAX_FILES_PER_CALL or input_budget <= 0:
                break
            text = read_file_text_from_result(self._item(turn_idx, item_idx).content)
            if len(text) < GIST_MIN_SOURCE_CHARS:
                continue
            excerpt = text[: min(GIST_FILE_SLICE_CHARS, input_budget)]
            input_budget -= len(excerpt)
            batch.append(GistRequest(path=path, content=excerpt))
            keys[path] = (turn_idx, item_idx)
        if not batch:
            return
        for path, gist in gister(tuple(batch)).items():
            flat = " ".join(gist.split())
            if path in keys and flat:
                self.gists[keys[path]] = flat[:GIST_MAX_CHARS]

    def _landing_gists(self) -> dict[tuple[int, int], str]:
        """Return the gist placeholders that land, chosen newest-first.

        The newest read of a path is the one a later turn needs. A gist no
        smaller than the content it replaces never lands, nor one costing more
        than the plan's headroom, which `demote` would strip in this same pass.

        Returns:
            The placeholder by victim position.
        """
        headroom = max(self.gist_headroom, 0)
        landing: dict[tuple[int, int], str] = {}
        for turn_idx, item_idx, size in reversed(self.victims):
            gist = self.gists.get((turn_idx, item_idx))
            call = self._item(turn_idx, item_idx).for_call
            if gist is None or not isinstance(call.input, dict):
                continue
            candidate = elision_gist_placeholder(call_label(call.name, call.input), gist)
            extra = len(candidate) - len(elision_placeholder(call.name, call.input))
            if len(candidate) < size and extra <= headroom:
                headroom -= extra
                landing[(turn_idx, item_idx)] = candidate
        return landing

    def apply(self) -> None:
        """Rewrite every victim with its gist or bare placeholder."""
        landing = self._landing_gists()
        for turn_idx, item_idx, size in self.victims:
            call = self._item(turn_idx, item_idx).for_call
            gist_placeholder = landing.get((turn_idx, item_idx))
            placeholder = gist_placeholder or elision_placeholder(call.name, call.input)
            if gist_placeholder is not None and isinstance(call.input, dict):
                self.gist_paths.append(str(call.input.get("path", "")))
            self.conversation.set_result_content(turn_idx, item_idx, placeholder)
            self.total -= size - len(placeholder)
            self.elided_calls.append(call_label(call.name, call.input))

    def demote(self) -> None:
        """Demote gist placeholders oldest-first to the bare marker while over budget."""
        if self.total <= self.max_total_bytes:
            return
        for turn_idx, item_idx, _size in self.candidates:
            if self.total <= self.max_total_bytes:
                break
            item = self._item(turn_idx, item_idx)
            if not item.content.startswith(ELISION_GIST_PREFIX):
                continue
            bare = elision_placeholder(item.for_call.name, item.for_call.input)
            if len(item.content) <= len(bare):
                continue
            self.conversation.set_result_content(turn_idx, item_idx, bare)
            self.total -= len(item.content) - len(bare)
            call = item.for_call
            self.demoted_paths.append(
                str(call.input.get("path", "")) if isinstance(call.input, dict) else ""
            )
