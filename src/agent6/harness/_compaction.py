# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context-window management for the agent loop: the pure compaction rules.

Tier 1 (`compact_old_tool_results`) replaces the oldest tool_result blocks
with placeholders at `_compaction_window.DROP_BLOCKS_AT_CHARS`; a large read_file result decays
through a distilled-gist placeholder first when the caller provides a
`gister`. Tier 2 compares `_compaction_window.context_chars` against
`_compaction_window.SUMMARISE_AT_CHARS` for the restart `_compactor` runs.
`_compaction_result.cap_tool_result`
bounds a single tool_result on the turn it arrives. The loop owns when to
call these and supplies the one impure seam, the `gister`.
"""

from __future__ import annotations

import dataclasses
import json
import re

from agent6.harness import (
    _compaction_result,
    _conversation,
)
from agent6.tools import schema

ASK_USER_TOOL = schema.AskUserInput.TOOL_NAME


@dataclasses.dataclass(frozen=True, slots=True)
class CompactionStats:
    """What one tier-1 pass did; a count is the length of its tuple.

    Attributes:
        elided_calls: The `_compaction_result.call_label` of each tool_result elided.
        gist_paths: The paths whose elision kept a gist.
        demoted_paths: The paths whose gist was demoted to the bare marker.
        deduped_calls: The `_compaction_result.call_label` of each duplicate result replaced.
    """

    elided_calls: tuple[str, ...] = ()
    gist_paths: tuple[str, ...] = ()
    demoted_paths: tuple[str, ...] = ()
    deduped_calls: tuple[str, ...] = ()


# Target headers in a unified diff (`+++ b/PATH`) or a v4a patch (`*** Update File:`).
_PATCH_TARGET_RE = re.compile(
    r"^(?:\+\+\+ b/(?P<u>\S+)|\*\*\* (?:Update|Add) File: (?P<v>.+))$", re.MULTILINE
)


def recently_edited_paths(
    conversation: _conversation.Conversation, *, last_messages: int = 8
) -> frozenset[str]:
    """Return the paths apply_edit and apply_patch targeted in the newest messages.

    Tier-1 elision deprioritises these files' reads, since a placeholder there
    triggers a paid re-read before the next edit. An apply_patch without a
    `path` argument falls back to the patch headers; an unparseable patch goes
    unprotected.

    Args:
        conversation: The loop's history.
        last_messages: How many of the newest assistant messages count.

    Returns:
        The paths.
    """
    out: set[str] = set()
    seen_assistant = 0
    for message in reversed(conversation.messages):
        if not isinstance(message, _conversation.AssistantMessage):
            continue
        seen_assistant += 1
        if seen_assistant > last_messages:
            break
        for tu in message.tool_uses:
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
    conversation: _conversation.Conversation,
) -> tuple[list[tuple[int, int, int]], int]:
    """Return every tool_result's position and size, in order.

    Args:
        conversation: The loop's history.

    Returns:
        ((message index, item index, size) per result, total size).
    """
    pointers: list[tuple[int, int, int]] = []
    total = 0
    for message_idx, message in enumerate(conversation.messages):
        if isinstance(message, _conversation.AssistantMessage):
            continue
        for item_idx, item in enumerate(message.items):
            if not isinstance(item, _conversation.ToolResultItem):
                continue
            pointers.append((message_idx, item_idx, len(item.content)))
            total += len(item.content)
    return pointers, total


def count_elisions(conversation: _conversation.Conversation) -> tuple[int, int]:
    """Count the elision markers in the context, and the live gists among them.

    A resumed or forked execution re-announces these, since its fresh log has
    no compaction events to fold.

    Args:
        conversation: The loop's history.

    Returns:
        (markers, gists).
    """
    elided = gists = 0
    for message in conversation.messages:
        for item in getattr(message, "items", ()):
            body = getattr(item, "content", "")
            if isinstance(body, str) and body.startswith(_compaction_result.ELISION_PREFIX):
                elided += 1
                gists += body.startswith(_compaction_result.ELISION_GIST_PREFIX)
    return elided, gists


def compact_old_tool_results(
    conversation: _conversation.Conversation,
    *,
    max_total_bytes: int,
    keep_recent: int = 2,
    protect_paths: frozenset[str] = frozenset(),
    gister: _compaction_result.Gister | None = None,
) -> CompactionStats:
    """Elide old tool_result blocks once their content exceeds the threshold.

    The walk is oldest-first and stops once the total is back under the bound.
    The newest `keep_recent` results stay, as does every result in the newest
    result message until an assistant message has consumed it: the loop compacts
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

    def _is_protected(message_idx: int, item_idx: int) -> bool:
        call = _result_at(conversation, message_idx, item_idx).for_call
        if call.name != "read_file" or not isinstance(call.input, dict):
            return False
        return str(call.input.get("path", "")) in protect_paths

    undelivered_message = _undelivered_result_message(conversation, pointers)
    older = pointers[:-keep_recent] if keep_recent else pointers
    candidates = [
        c
        for c in older
        if c[0] != undelivered_message
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


def _is_operator_answer(item: _conversation.ToolResultItem) -> bool:
    """Return whether the result is the operator's answer to an `ask_user`.

    An answer is exempt from elision and dedup: it is a binding ruling that
    exists nowhere else in the context, and a re-run would re-ask the operator.

    Args:
        item: The result.

    Returns:
        True for an `ask_user` result.
    """
    return item.for_call.name == ASK_USER_TOOL


def _result_at(
    conversation: _conversation.Conversation, message_idx: int, item_idx: int
) -> _conversation.ToolResultItem:
    message = conversation.messages[message_idx]
    assert not isinstance(message, _conversation.AssistantMessage)
    item = message.items[item_idx]
    assert isinstance(item, _conversation.ToolResultItem)  # pointers only ever index tool_results
    return item


def _undelivered_result_message(
    conversation: _conversation.Conversation, pointers: list[tuple[int, int, int]]
) -> int | None:
    """Return the newest result message when no assistant message has consumed it yet.

    Args:
        conversation: The loop's history.
        pointers: The result positions from `_tool_result_pointers`.

    Returns:
        The message index, or None once an assistant message follows it.
    """
    last_result = max(message_idx for message_idx, _, _ in pointers)
    if any(
        isinstance(message, _conversation.AssistantMessage)
        for message in conversation.messages[last_result + 1 :]
    ):
        return None
    return last_result


# Below this a duplicate's placeholder is barely smaller than the content it replaces.
_DEDUP_MIN_CHARS = 200


def _dedupe_identical_results(
    conversation: _conversation.Conversation,
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
        The `_compaction_result.call_label` of each copy replaced.
    """
    if len(pointers) <= keep_recent:
        return ()
    undelivered_message = _undelivered_result_message(conversation, pointers)
    recent = pointers[-keep_recent:] if keep_recent else []
    exempt = {(t, i) for t, i, _ in recent}
    by_key: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for message_idx, item_idx, _size in pointers:
        item = _result_at(conversation, message_idx, item_idx)
        if (
            item.content.startswith(_compaction_result.ELISION_PREFIX)
            or len(item.content) < _DEDUP_MIN_CHARS
            or _is_operator_answer(item)
        ):
            continue
        call = item.for_call
        try:
            input_key = json.dumps(call.input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            continue
        by_key.setdefault((call.name, input_key, item.content), []).append((message_idx, item_idx))
    labels: list[str] = []
    for locs in by_key.values():
        for message_idx, item_idx in locs[:-1]:  # every copy but the newest
            if message_idx == undelivered_message or (message_idx, item_idx) in exempt:
                continue
            item = _result_at(conversation, message_idx, item_idx)
            label = _compaction_result.call_label(item.for_call.name, item.for_call.input)
            marker = (
                f"{_compaction_result.ELISION_PREFIX} (duplicate): {label} returned byte-identical"
                " content again later in this conversation. If you still need it,"
                " re-read only the part you need; do not re-issue the identical call.>"
            )
            if len(marker) >= len(item.content):
                # A long path can make the marker bigger than the result it replaces.
                continue
            conversation.set_result_content(message_idx, item_idx, marker)
            labels.append(label)
    return tuple(labels)


@dataclasses.dataclass(slots=True)
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
        elided_calls: The `_compaction_result.call_label` of each result elided.
        gist_paths: The paths whose elision kept a gist.
        demoted_paths: The paths whose gist was demoted.
    """

    conversation: _conversation.Conversation
    max_total_bytes: int
    protect_paths: frozenset[str]
    candidates: list[tuple[int, int, int]]
    total: int
    victims: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    gist_headroom: int = 0
    gists: dict[tuple[int, int], str] = dataclasses.field(default_factory=dict)
    elided_calls: list[str] = dataclasses.field(default_factory=list)
    gist_paths: list[str] = dataclasses.field(default_factory=list)
    demoted_paths: list[str] = dataclasses.field(default_factory=list)

    def _item(self, message_idx: int, item_idx: int) -> _conversation.ToolResultItem:
        return _result_at(self.conversation, message_idx, item_idx)

    def plan(self) -> None:
        """Pick the victims under bare-placeholder accounting, the maximum shrink.

        Nothing is mutated yet, so the distiller can still read the content.
        """
        planned = self.total
        for message_idx, item_idx, size in self.candidates:
            if planned <= self.max_total_bytes:
                break
            item = self._item(message_idx, item_idx)
            if item.content.startswith(_compaction_result.ELISION_PREFIX):
                continue
            placeholder = _compaction_result.elision_placeholder(
                item.for_call.name, item.for_call.input
            )
            if size <= len(placeholder):
                # Content smaller than the placeholder would grow the total.
                continue
            self.victims.append((message_idx, item_idx, size))
            planned -= size - len(placeholder)
        # What a gist may add back on top of the bare-placeholder plan.
        self.gist_headroom = self.max_total_bytes - planned

    def distill(self, gister: _compaction_result.Gister) -> None:
        """Make one batched distiller call over the eligible victims.

        Eligible is a large unprotected read_file result, the newest read per
        path, largest files first under the input caps.

        Args:
            gister: The distiller.
        """
        newest_by_path: dict[str, tuple[int, int, int]] = {}
        for message_idx, item_idx, size in self.victims:
            call = self._item(message_idx, item_idx).for_call
            if call.name != "read_file" or not isinstance(call.input, dict):
                continue
            path = str(call.input.get("path", ""))
            if (
                not path
                or path in self.protect_paths
                or size < _compaction_result.GIST_MIN_SOURCE_CHARS
            ):
                continue
            newest_by_path[path] = (message_idx, item_idx, size)  # victims are oldest-first
        batch: list[_compaction_result.GistRequest] = []
        keys: dict[str, tuple[int, int]] = {}
        input_budget = _compaction_result.GIST_INPUT_CAP_CHARS
        by_size = sorted(newest_by_path.items(), key=lambda kv: kv[1][2], reverse=True)
        for path, (message_idx, item_idx, _size) in by_size:
            if len(batch) >= _compaction_result.GIST_MAX_FILES_PER_CALL or input_budget <= 0:
                break
            text = _compaction_result.read_file_text_from_result(
                self._item(message_idx, item_idx).content
            )
            if len(text) < _compaction_result.GIST_MIN_SOURCE_CHARS:
                continue
            excerpt = text[: min(_compaction_result.GIST_FILE_SLICE_CHARS, input_budget)]
            input_budget -= len(excerpt)
            batch.append(_compaction_result.GistRequest(path=path, content=excerpt))
            keys[path] = (message_idx, item_idx)
        if not batch:
            return
        for path, gist in gister(tuple(batch)).items():
            flat = " ".join(gist.split())
            if path in keys and flat:
                self.gists[keys[path]] = flat[: _compaction_result.GIST_MAX_CHARS]

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
        for message_idx, item_idx, size in reversed(self.victims):
            gist = self.gists.get((message_idx, item_idx))
            call = self._item(message_idx, item_idx).for_call
            if gist is None or not isinstance(call.input, dict):
                continue
            candidate = _compaction_result.elision_gist_placeholder(
                _compaction_result.call_label(call.name, call.input), gist
            )
            extra = len(candidate) - len(
                _compaction_result.elision_placeholder(call.name, call.input)
            )
            if len(candidate) < size and extra <= headroom:
                headroom -= extra
                landing[(message_idx, item_idx)] = candidate
        return landing

    def apply(self) -> None:
        """Rewrite every victim with its gist or bare placeholder."""
        landing = self._landing_gists()
        for message_idx, item_idx, size in self.victims:
            call = self._item(message_idx, item_idx).for_call
            gist_placeholder = landing.get((message_idx, item_idx))
            placeholder = gist_placeholder or _compaction_result.elision_placeholder(
                call.name, call.input
            )
            if gist_placeholder is not None and isinstance(call.input, dict):
                self.gist_paths.append(str(call.input.get("path", "")))
            self.conversation.set_result_content(message_idx, item_idx, placeholder)
            self.total -= size - len(placeholder)
            self.elided_calls.append(_compaction_result.call_label(call.name, call.input))

    def demote(self) -> None:
        """Demote gist placeholders oldest-first to the bare marker while over budget."""
        if self.total <= self.max_total_bytes:
            return
        for message_idx, item_idx, _size in self.candidates:
            if self.total <= self.max_total_bytes:
                break
            item = self._item(message_idx, item_idx)
            if not item.content.startswith(_compaction_result.ELISION_GIST_PREFIX):
                continue
            bare = _compaction_result.elision_placeholder(item.for_call.name, item.for_call.input)
            if len(item.content) <= len(bare):
                continue
            self.conversation.set_result_content(message_idx, item_idx, bare)
            self.total -= len(item.content) - len(bare)
            call = item.for_call
            self.demoted_paths.append(
                str(call.input.get("path", "")) if isinstance(call.input, dict) else ""
            )
