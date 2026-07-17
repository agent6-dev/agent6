# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context-window management for the agent loop.

Two tiers keep a long run inside the model's context window:

- tier 1 (`compact_old_tool_results`): at `DROP_BLOCKS_AT_CHARS` the oldest
  tool_result blocks are replaced by `ELISION_PLACEHOLDER`; large read_file
  results decay through a distilled-gist placeholder first when the caller
  provides a `gister` (see below).
- tier 2 (`context_chars` vs `SUMMARISE_AT_CHARS`): the elided history is
  summarised and the conversation restarts from (task + summary).

`cap_tool_result` separately bounds a single tool_result so one huge payload
cannot blow the budget on the turn it arrives. Everything here is a pure
function of the conversation; the loop owns the policy of when to call them
and supplies the one impure seam (the `gister` callable that distills
about-to-be-elided file reads with the summariser model).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent6.workflows._conversation import (
    AssistantTurn,
    Conversation,
    ToolResultItem,
)

# Stable prefix shared by every placeholder variant: idempotency checks and
# tests key on it.
ELISION_PREFIX = "<elided by context compaction"

ELISION_PLACEHOLDER = (
    "<elided by context compaction: this tool_result has been replaced "
    "with this short marker to keep the loop's cumulative input bounded. "
    "If you still need it, re-read only the part you need with a targeted "
    "read_file offset/limit; do not re-issue the identical call.>"
)

# Gist placeholders share ELISION_PREFIX (idempotency walks key on it) but are
# distinguishable so continued pressure can demote them to the bare marker.
ELISION_GIST_PREFIX = ELISION_PREFIX + " (distilled)"

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
        f"short marker to keep the loop's cumulative input bounded. If you "
        f"still need it, re-read only the part you need ({tool_name} with a "
        f"targeted offset/limit); do not re-issue the identical call.>"
    )


# Distilled-gist elision. Measured (bench/longhorizon FINDINGS #1): under a
# small-window regime tier-1 elision of reference docs halves a retention
# task's score (0.921 -> 0.425) and every redundant read is post-drop, while
# code files are cheaply re-readable. So a large read_file result about to be
# elided decays in two stages: first to a placeholder carrying a model-written
# gist of the file's load-bearing facts, then (under continued pressure) to
# the bare identity marker, so the hard byte bound always holds. The caps
# bound the distiller call per drop event; hot files (protect_paths) are never
# gisted because their content is changing under edits and a stale gist would
# mislead.
GIST_MIN_SOURCE_CHARS = 2_000  # below this the content is nearly gist-sized
GIST_MAX_CHARS = 400  # per gist, clipped
GIST_FILE_SLICE_CHARS = 8_000  # per-file head sent to the distiller
GIST_INPUT_CAP_CHARS = 24_000  # total distiller input per drop event
GIST_MAX_FILES_PER_CALL = 12


@dataclass(frozen=True, slots=True)
class GistRequest:
    """One file whose about-to-be-elided read_file content should be distilled."""

    path: str
    content: str


# The impure seam: called once per drop event with the batch of eligible
# reads; returns path -> distilled gist (missing paths fall back to the bare
# placeholder). The loop binds this to the summariser model; on provider
# failure it returns {}.
Gister = Callable[[tuple[GistRequest, ...]], Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class CompactionStats:
    """One tier-1 pass: fresh tool_results elided (of which gisted), plus
    gist placeholders demoted to the bare marker."""

    elided: int = 0
    gisted: int = 0
    demoted: int = 0


def elision_gist_placeholder(path: str, gist: str) -> str:
    """Tier-1 placeholder that keeps a distilled gist of the elided read."""
    return (
        f"{ELISION_GIST_PREFIX}: the result of read_file {path} was replaced "
        f"by this distilled gist; if the gist is not enough, re-read only "
        f"the part you need (read_file with a targeted offset/limit).\ngist: {gist}>"
    )


def read_file_text_from_result(raw: str) -> str:
    """The file text inside a serialized read_file tool_result.

    Unwraps the {"content": ...} result shape (and the truncation envelope's
    "head") so the distiller sees file text, not JSON escapes. An error result
    returns "" (nothing worth distilling); any other shape falls back to the
    raw payload.
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
    """path -> gist from the distiller's one-line-per-file reply.

    Tolerant of list markers and backticks around the path; a file the reply
    misses simply keeps the bare placeholder, and unknown paths are ignored.
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


def context_chars(conversation: Conversation) -> int:
    """Approximate the full character size of the conversation context.

    Sums notice text, tool_result content, and -- for assistant turns' raw
    blocks -- their text and tool_use inputs; i.e. everything that grows the
    context and is sent back to the model each turn, not just tool_result bytes.
    Used as the tier-2 (summarise-and-restart) trigger, which must measure
    something tier-1 elision does not already cap.
    """
    total = 0
    for turn in conversation.turns:
        if isinstance(turn, AssistantTurn):
            for item in turn.raw_content:
                if not isinstance(item, dict):
                    continue
                total += len(str(item.get("text", "") or ""))
                total += len(str(item.get("content", "") or ""))
                if item.get("type") == "tool_use":
                    total += len(str(item.get("input", "") or ""))
        else:
            for item in turn.items:
                total += len(item.content if isinstance(item, ToolResultItem) else item.text)
    return total


# First target header in a unified diff (`+++ b/PATH`) or a v4a patch
# (`*** Update|Add File: PATH`). apply_patch is one-file-per-call, so the
# first match is the file.
_PATCH_TARGET_RE = re.compile(
    r"^(?:\+\+\+ b/(?P<u>\S+)|\*\*\* (?:Update|Add) File: (?P<v>.+))$", re.MULTILINE
)


def recently_edited_paths(conversation: Conversation, *, last_turns: int = 8) -> frozenset[str]:
    """Paths targeted by apply_edit / apply_patch in the last *last_turns*
    assistant turns: the files the worker is actively editing. Tier-1
    elision deprioritises their read_file results (see
    ``compact_old_tool_results``), because a placeholder there triggers a paid
    re-read before the very next edit. Best-effort: an apply_patch without a
    ``path`` argument falls back to the patch headers; an unparseable patch
    just goes unprotected.
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
                m = _PATCH_TARGET_RE.search(str(tu.input.get("patch", "")))
                path = ((m.group("u") or m.group("v") or "") if m else "").strip()
            if path:
                out.add(path)
    return frozenset(out)


def _tool_result_pointers(
    conversation: Conversation,
) -> tuple[list[tuple[int, int, int]], int]:
    """((turn_idx, item_idx, size) per tool_result, total size) in order."""
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


def compact_old_tool_results(
    conversation: Conversation,
    *,
    max_total_bytes: int,
    keep_recent: int = 2,
    protect_paths: frozenset[str] = frozenset(),
    gister: Gister | None = None,
) -> CompactionStats:
    """Elide old tool_result blocks once cumulative content exceeds the
    threshold. Walks the conversation oldest-first, replaces each tool_result's
    ``content`` with a short identity-bearing placeholder, stops once total
    size is back under ``max_total_bytes``. The most recent ``keep_recent``
    are always preserved, as is every tool_result in the last
    tool_result-bearing turn: the loop compacts at top-of-iteration, before
    the provider call that would deliver that batch, so the model has never
    seen it and the placeholder's "re-call the tool" guidance would trigger a
    paid re-call cycle. (Keying on the final turn alone is not enough: a
    trailing steer or nudge user turn pushes the fresh, still undelivered
    results off the final index, and one turn can carry several such blocks.)

    ``protect_paths`` (the actively-edited set from ``recently_edited_paths``)
    deprioritises rather than exempts: read_file results for those paths are
    elided only after every other candidate, so the hot file's content
    survives as long as the budget allows but the hard bound still holds.

    With a ``gister``, each large unprotected read_file victim decays to a
    placeholder carrying a distilled gist of the file (one batched distiller
    call per pass, newest read per path, caps above); everything else gets the
    bare marker. Gists make the pass land slightly OVER the bare-accounting
    plan, so when the applied total still exceeds the budget, existing gist
    placeholders are demoted oldest-first to the bare marker: content decays
    content -> gist -> bare marker, and the spec facts survive the longest
    while the byte bound still holds (in the limit everything is bare, exactly
    the pre-gist behavior). Demotion runs after even the protected reads are
    elided: losing a gist costs correctness (the file is gone from context),
    losing a hot read costs one paid re-read.

    Idempotent on already-elided entries. Returns per-pass counts.
    """
    pointers, total = _tool_result_pointers(conversation)
    if total <= max_total_bytes or len(pointers) <= keep_recent:
        return CompactionStats()

    def _is_protected(turn_idx: int, item_idx: int) -> bool:
        call = _result_at(conversation, turn_idx, item_idx).for_call
        if call.name != "read_file" or not isinstance(call.input, dict):
            return False
        return str(call.input.get("path", "")) in protect_paths

    # The undelivered batch is always in the last tool_result-bearing turn:
    # at top-of-iteration only text-only steer/nudge user turns can trail the
    # fresh results, and the delivering provider call runs after this compaction.
    # Exempt that whole turn -- see docstring. Keying on the final turn
    # index alone missed a trailing steer/nudge, and one turn can carry several
    # such blocks, so this is broader than keep_recent or the final index.
    last_tool_result_idx = max(turn_idx for turn_idx, _, _ in pointers)
    candidates = pointers[:-keep_recent]
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
        last_tool_result_idx=last_tool_result_idx,
        total=total,
    )
    walk.plan()
    if gister is not None:
        walk.distill(gister)
    walk.apply()
    walk.demote()
    return CompactionStats(elided=walk.elided, gisted=walk.gisted, demoted=walk.demoted)


def _result_at(conversation: Conversation, turn_idx: int, item_idx: int) -> ToolResultItem:
    turn = conversation.turns[turn_idx]
    assert not isinstance(turn, AssistantTurn)
    item = turn.items[item_idx]
    assert isinstance(item, ToolResultItem)  # pointers only ever index tool_results
    return item


@dataclass(slots=True)
class _Tier1Pass:
    """State shared by the phases of one tier-1 pass (the loop's ``_TurnState``
    pattern: one mutable object instead of six hand-threaded locals)."""

    conversation: Conversation
    max_total_bytes: int
    protect_paths: frozenset[str]
    candidates: list[tuple[int, int, int]]
    last_tool_result_idx: int
    total: int
    victims: list[tuple[int, int, int]] = field(default_factory=list)
    gists: dict[tuple[int, int], str] = field(default_factory=dict)
    elided: int = 0
    gisted: int = 0
    demoted: int = 0

    def _item(self, turn_idx: int, item_idx: int) -> ToolResultItem:
        return _result_at(self.conversation, turn_idx, item_idx)

    def plan(self) -> None:
        """Pick the victim set under bare-placeholder accounting (the maximum
        shrink); nothing is mutated yet so the distiller can still read the
        content."""
        planned = self.total
        for turn_idx, item_idx, size in self.candidates:
            if planned <= self.max_total_bytes:
                break
            if turn_idx == self.last_tool_result_idx:
                continue
            item = self._item(turn_idx, item_idx)
            if item.content.startswith(ELISION_PREFIX):
                continue
            placeholder = elision_placeholder(item.for_call.name, item.for_call.input)
            if size <= len(placeholder):
                # Replacing content already smaller than the placeholder would
                # GROW the total, defeating the point; skip it.
                continue
            self.victims.append((turn_idx, item_idx, size))
            planned -= size - len(placeholder)

    def distill(self, gister: Gister) -> None:
        """One batched distiller call over the eligible victims: large
        unprotected read_file results, the newest read per path, largest files
        first under the input caps."""
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

    def apply(self) -> None:
        """Apply the whole plan (``plan`` already chose the minimal set; gist
        placeholders only add back a bounded extra on top of it)."""
        for turn_idx, item_idx, size in self.victims:
            call = self._item(turn_idx, item_idx).for_call
            placeholder = elision_placeholder(call.name, call.input)
            gist = self.gists.get((turn_idx, item_idx))
            if gist is not None and isinstance(call.input, dict):
                candidate = elision_gist_placeholder(str(call.input.get("path", "")), gist)
                if len(candidate) < size:  # a gist longer than the content is useless
                    placeholder = candidate
                    self.gisted += 1
            self.conversation.set_result_content(turn_idx, item_idx, placeholder)
            self.total -= size - len(placeholder)
            self.elided += 1

    def demote(self) -> None:
        """Still over budget (gist extras, or a shrunken budget with nothing
        fresh left): demote gist placeholders oldest-first to the bare marker
        until the bound holds or none remain."""
        if self.total <= self.max_total_bytes:
            return
        for turn_idx, item_idx, _size in self.candidates:
            if self.total <= self.max_total_bytes:
                break
            if turn_idx == self.last_tool_result_idx:
                continue
            item = self._item(turn_idx, item_idx)
            if not item.content.startswith(ELISION_GIST_PREFIX):
                continue
            bare = elision_placeholder(item.for_call.name, item.for_call.input)
            if len(item.content) <= len(bare):
                continue
            self.conversation.set_result_content(turn_idx, item_idx, bare)
            self.total -= len(item.content) - len(bare)
            self.demoted += 1
