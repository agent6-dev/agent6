# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-owned conversation: typed turns over the provider wire.

`Conversation` holds the history as typed turns, one shape for every
consumer, and produces the Anthropic-wire dict list only at the boundary.
`to_wire` builds the list providers and snapshots take, with `cache_control`
stamped from the mark positions and assistant blocks verbatim, so thinking
blocks and unknown block types round-trip. `from_wire` accepts exactly the
shapes the loop writes and fails loudly on anything else.

Pair safety is structural: a `tool_use` turn can only be followed by
`results` covering exactly its ids, `pop_quiet_assistant` removes only a
turn with no tool calls, `restart` keeps whole turns, and compaction
rewrites result content in place. No operation strands a `tool_use`.

Rolling cache breakpoints: Anthropic bills a request's prefix up to a
`cache_control` breakpoint at 0.1x once cached (1.25x to write), and the
conversation dominates input tokens in a long run. The roll keeps two marks,
the previous call's position (the guaranteed hit) and the last user turn's
final block (the next write), for four breakpoints with the provider's two
static ones, Anthropic's per-request maximum. Marks persist into resume
snapshots; tier-1 elision costs one re-write on the next call. OpenAI-format
providers never forward the field.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

_EPHEMERAL = {"type": "ephemeral"}


@dataclass(frozen=True, slots=True)
class ToolUse:
    """One tool call from an assistant turn, parsed once from the raw blocks.

    Attributes:
        id: The provider's tool_use id.
        name: The tool's name.
        input: Whatever the provider parsed; the dispatcher's schema validation
            owns its shape.
    """

    id: str
    name: str
    input: Any


@dataclass(frozen=True, slots=True)
class ToolResultItem:
    """One tool_result block.

    Attributes:
        tool_use_id: The id the wire carries.
        content: The result text.
        for_call: The ToolUse it answers, paired at construction so compaction
            never rebuilds an id index; in-memory only.
    """

    tool_use_id: str
    content: str
    for_call: ToolUse


@dataclass(frozen=True, slots=True)
class Notice:
    """Harness or operator text in a user turn.

    The initial task, nudges, critiques, steering and the tier-2 restart summary
    are notices.

    Attributes:
        text: The text.
    """

    text: str


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """One assistant message.

    Attributes:
        raw_content: The verbatim block tuple the provider returned; tool_use ids,
            thinking blocks and unknown block types round-trip untouched.
        tool_uses: The parsed tool_use view of those blocks.
    """

    raw_content: tuple[Any, ...]
    tool_uses: tuple[ToolUse, ...]

    def is_substantive(self) -> bool:
        """Return whether the turn carries visible text or a tool call.

        An empty or thinking-only turn is dead context: Anthropic rejects empty
        assistant content and strict OpenAI-compatible backends refuse the
        translation.

        Returns:
            True when a text block has content or a tool_use block is present.
        """
        return any(
            isinstance(b, dict)
            and (
                (b.get("type") == "text" and str(b.get("text", "")).strip())
                or b.get("type") == "tool_use"
            )
            for b in self.raw_content
        )


@dataclass(frozen=True, slots=True)
class UserTurn:
    """One user message.

    Attributes:
        items: Tool results and notices in wire order, results first.
    """

    items: tuple[ToolResultItem | Notice, ...]


Turn = AssistantTurn | UserTurn


def _parse_tool_uses(blocks: Sequence[Any]) -> tuple[ToolUse, ...]:
    return tuple(
        ToolUse(id=str(b.get("id", "")), name=str(b.get("name", "")), input=b.get("input", {}))
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )


def _result_ids(turn: UserTurn) -> list[str]:
    return [it.tool_use_id for it in turn.items if isinstance(it, ToolResultItem)]


def _results_first[T](items: Sequence[T], *, key: Callable[[T], object] = lambda it: it) -> list[T]:
    """Return the items with tool results first and notices after, each in order.

    Args:
        items: A user turn's items, or pairs the key maps to an item.
        key: Reads the item from an element.

    Returns:
        The stable partition.
    """
    return sorted(items, key=lambda it: isinstance(key(it), Notice))


class Conversation:
    """The history as frozen turns plus the rolling cache-mark pair.

    Marks are (turn index, item index) positions into user turns; `to_wire`
    stamps `cache_control` there. They live here rather than on the items
    because breakpoint placement moves as the tail grows, while the turns are
    history.
    """

    __slots__ = ("_marks", "_turns")

    def __init__(self) -> None:
        self._turns: list[Turn] = []
        self._marks: list[tuple[int, int]] = []

    # ---- reads ----------------------------------------------------------

    @property
    def turns(self) -> tuple[Turn, ...]:
        """The turns, as a copy: no caller appends around the guarded mutators."""
        return tuple(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    # ---- appends (each preserves pair safety) ---------------------------

    def _require_no_open_call(self, what: str) -> None:
        if self._turns and isinstance(prev := self._turns[-1], AssistantTurn) and prev.tool_uses:
            raise ValueError(
                f"conversation invariant: cannot append {what} after an unanswered"
                " tool_use turn; append its results first"
            )

    def assistant(self, raw_blocks: Any) -> AssistantTurn:
        """Append the assistant turn exactly as the provider returned it.

        Args:
            raw_blocks: The provider's content blocks.

        Returns:
            The appended turn.

        Raises:
            ValueError: A previous turn's tool calls have no results yet.
        """
        self._require_no_open_call("an assistant turn")
        blocks = tuple(raw_blocks)
        turn = AssistantTurn(raw_content=blocks, tool_uses=_parse_tool_uses(blocks))
        self._turns.append(turn)
        return turn

    def results(self, items: Sequence[ToolResultItem | Notice]) -> None:
        """Append the user turn answering the preceding tool_use turn.

        The wire requires a user message to lead with its tool_result blocks, so
        notices are moved after the results here rather than trusted per call
        site.

        Args:
            items: The results, covering exactly the pending tool_use ids in
                order, plus any notices.

        Raises:
            ValueError: The result ids do not answer the pending tool_use ids.
        """
        prev = self._turns[-1] if self._turns else None
        want = [tu.id for tu in prev.tool_uses] if isinstance(prev, AssistantTurn) else []
        turn = UserTurn(items=tuple(_results_first(items)))
        if _result_ids(turn) != want:
            raise ValueError(
                f"conversation invariant: tool_result ids {_result_ids(turn)} do not"
                f" answer the pending tool_use ids {want}"
            )
        self._turns.append(turn)

    def notice(self, text: str) -> None:
        """Append harness or operator text as its own user turn.

        Args:
            text: The notice.
        """
        self._append_notices((Notice(text),))

    def _append_notices(self, items: tuple[ToolResultItem | Notice, ...]) -> None:
        self._require_no_open_call("a notice")
        self._turns.append(UserTurn(items=items))

    def pop_quiet_assistant(self) -> None:
        """Drop a trailing non-substantive assistant turn.

        Such a turn has no tool_uses, so no pair can split; any other tail is left
        alone.
        """
        if (
            self._turns
            and isinstance(last := self._turns[-1], AssistantTurn)
            and not last.is_substantive()
        ):
            self._turns.pop()

    def restart(self, summary_text: str, keep: Sequence[Turn] = ()) -> None:
        """Replace everything after the initial turn with a summary notice and a tail.

        Marks outside the kept turns are dropped with their blocks.

        Args:
            summary_text: The restart notice.
            keep: The most recent turns, already balanced. A tail leading with
                tool results would answer a turn the restart summarised away.

        Raises:
            ValueError: The first turn is a tool_use turn, or the tail leads with
                tool results.
        """
        first = self._turns[0]
        if isinstance(first, AssistantTurn) and first.tool_uses:
            raise ValueError("conversation invariant: cannot restart from a tool_use turn")
        if keep and isinstance(keep[0], UserTurn) and _result_ids(keep[0]):
            raise ValueError("conversation invariant: a kept tail cannot lead with tool_results")
        self._turns[:] = [first, UserTurn(items=(Notice(summary_text),)), *keep]
        self._marks = [m for m in self._marks if m[0] == 0]

    def strip_thinking(self, turn_idx: int) -> int:
        """Drop the thinking blocks from one assistant turn, in place.

        The tool_use blocks survive verbatim, so pairing is untouched.

        Args:
            turn_idx: The assistant turn's index.

        Returns:
            The characters removed, 0 when the turn had no thinking.

        Raises:
            ValueError: The index names a user turn.
        """
        turn = self._turns[turn_idx]
        if not isinstance(turn, AssistantTurn):
            raise ValueError("strip_thinking targets an assistant turn")
        dropped = [
            b
            for b in turn.raw_content
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
        ]
        if not dropped:
            return 0
        kept = tuple(b for b in turn.raw_content if b not in dropped)
        removed = sum(
            len(v if isinstance(v, str) else str(v))
            for b in dropped
            for k, v in b.items()
            if k != "type" and v is not None
        )
        self._turns[turn_idx] = AssistantTurn(raw_content=kept, tool_uses=turn.tool_uses)
        return removed

    def set_result_content(self, turn_idx: int, item_idx: int, content: str) -> None:
        """Rewrite one tool_result's content in place.

        The id and pairing are untouched, so the wire stays balanced.

        Args:
            turn_idx: The user turn's index.
            item_idx: The result's index in the turn.
            content: The new content.

        Raises:
            ValueError: The position is not a tool_result in a user turn.
        """
        turn = self._turns[turn_idx]
        if not isinstance(turn, UserTurn):
            raise ValueError("set_result_content targets a user turn")
        item = turn.items[item_idx]
        if not isinstance(item, ToolResultItem):
            raise ValueError("set_result_content targets a tool_result")
        items = list(turn.items)
        items[item_idx] = replace(item, content=content)
        self._turns[turn_idx] = UserTurn(items=tuple(items))

    # ---- rolling cache breakpoints --------------------------------------

    def roll_cache_marks(self) -> None:
        """Advance the rolling cache-mark pair.

        The newest existing mark stays (the previous call's write, the guaranteed hit)
        and the final item of the newest user turn is marked (the new write).
        Idempotent, so a crash-resume re-issuing the same call keeps its
        positions; positions survive content rewrites.
        """
        target: tuple[int, int] | None = None
        for t_idx in range(len(self._turns) - 1, -1, -1):
            turn = self._turns[t_idx]
            # Assistant raw blocks are verbatim history and are never stamped.
            if isinstance(turn, UserTurn) and turn.items:
                target = (t_idx, len(turn.items) - 1)
                break
        if target is None:
            self._marks = []
            return
        # The newest non-target mark is the previous call's breakpoint; the newest mark
        # is the target itself when nothing was appended.
        prev = next((m for m in reversed(self._marks) if m != target), None)
        self._marks = ([prev] if prev is not None else []) + [target]

    # ---- the wire boundary ----------------------------------------------

    def to_wire(self) -> list[dict[str, Any]]:
        """Return the provider and snapshot message list.

        Returns:
            Fresh dicts for user turns with `cache_control` stamped at the mark
            positions, the verbatim raw blocks for assistant turns.
        """
        marks = set(self._marks)
        out: list[dict[str, Any]] = []
        for t_idx, turn in enumerate(self._turns):
            if isinstance(turn, AssistantTurn):
                out.append({"role": "assistant", "content": list(turn.raw_content)})
                continue
            blocks: list[dict[str, Any]] = []
            for i_idx, item in enumerate(turn.items):
                if isinstance(item, ToolResultItem):
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": item.tool_use_id,
                        "content": item.content,
                    }
                else:
                    block = {"type": "text", "text": item.text}
                if (t_idx, i_idx) in marks:
                    block["cache_control"] = dict(_EPHEMERAL)
                blocks.append(block)
            out.append({"role": "user", "content": blocks})
        return out

    @classmethod
    def from_wire(cls, messages: Sequence[Any]) -> Conversation:
        """Parse a persisted message list.

        A turn in canonical order round-trips byte-for-byte through `to_wire`; a
        notice ahead of its results is moved after them on load, each mark kept
        on its own block.

        Args:
            messages: The snapshot's message list.

        Returns:
            The conversation with its marks.

        Raises:
            ValueError: A message has a shape this loop cannot have written.
        """
        conv = cls()
        marks: list[tuple[int, int]] = []
        for t_idx, msg in enumerate(messages):
            where = f"message {t_idx}"
            if not isinstance(msg, dict) or set(msg) != {"role", "content"}:
                raise ValueError(f"malformed conversation: {where} is not a role/content object")
            role, content = msg["role"], msg["content"]
            if not isinstance(content, list):
                raise ValueError(f"malformed conversation: {where} content is not a block list")
            if role == "assistant":
                conv.assistant(content)
                continue
            if role != "user":
                raise ValueError(f"malformed conversation: {where} has role {role!r}")
            prev = conv._turns[-1] if conv._turns else None
            pending = list(prev.tool_uses) if isinstance(prev, AssistantTurn) else []
            parsed: list[tuple[ToolResultItem | Notice, bool]] = []
            for i_idx, block in enumerate(content):
                item = _parse_user_block(block, pending, where=f"{where} block {i_idx}")
                parsed.append((item, _block_mark(block, where=f"{where} block {i_idx}")))
            # Canonicalize before recording positions, so each mark stays on its block.
            parsed = _results_first(parsed, key=lambda pair: pair[0])
            for i_idx, (_, marked) in enumerate(parsed):
                if marked:
                    marks.append((t_idx, i_idx))
            items = [item for item, _ in parsed]
            if any(isinstance(it, ToolResultItem) for it in items):
                conv.results(items)  # validates the pairing against the tool_use turn
            else:
                conv._append_notices(tuple(items))
        conv._marks = marks
        return conv


def _block_mark(block: dict[str, Any], *, where: str) -> bool:
    """Return whether a user block carries the cache mark.

    Args:
        block: The wire block.
        where: The block's position, for the error.

    Returns:
        True when the block is marked.

    Raises:
        ValueError: The mark is not the ephemeral one.
    """
    if "cache_control" not in block:
        return False
    if block["cache_control"] != _EPHEMERAL:
        raise ValueError(f"malformed conversation: {where} has a non-ephemeral cache_control")
    return True


def _parse_user_block(block: Any, pending: list[ToolUse], *, where: str) -> ToolResultItem | Notice:
    """Return the typed item for one user-turn wire block.

    Args:
        block: The wire block.
        pending: The preceding assistant turn's unanswered tool_uses; a result
            consumes the first.
        where: The block's position, for the error.

    Returns:
        The notice or the result.

    Raises:
        ValueError: The block is not a plain text or tool_result block, or answers
            a tool_use out of order.
    """
    if not isinstance(block, dict):
        raise ValueError(f"malformed conversation: {where} is not a block object")
    keys = set(block) - {"cache_control"}
    if block.get("type") == "text":
        if keys != {"type", "text"} or not isinstance(block["text"], str):
            raise ValueError(f"malformed conversation: {where} is not a plain text block")
        return Notice(text=block["text"])
    if block.get("type") == "tool_result":
        if (
            keys != {"type", "tool_use_id", "content"}
            or not isinstance(block["tool_use_id"], str)
            or not isinstance(block["content"], str)
        ):
            raise ValueError(f"malformed conversation: {where} is not a plain tool_result block")
        if not pending or pending[0].id != block["tool_use_id"]:
            raise ValueError(
                f"malformed conversation: {where} answers tool_use"
                f" {block['tool_use_id']!r} out of order"
            )
        return ToolResultItem(
            tool_use_id=block["tool_use_id"], content=block["content"], for_call=pending.pop(0)
        )
    raise ValueError(f"malformed conversation: {where} has unsupported type {block.get('type')!r}")


def format_transcript_tail(
    turns: Sequence[Turn], *, max_messages: int = 6, max_chars: int = 6000
) -> str:
    """Render the last few turns as a plain-text transcript for a summariser call.

    Tool calls and results are clipped, and thinking blocks are skipped.

    Args:
        turns: The turns to render, newest last.
        max_messages: How many of the newest turns to render.
        max_chars: The cap, applied from the end.

    Returns:
        The transcript.
    """
    parts: list[str] = []
    for turn in turns[-max_messages:]:
        if isinstance(turn, AssistantTurn):
            for block in turn.raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(f"[assistant:text] {str(block.get('text', ''))[:1500]}")
                elif btype == "tool_use":
                    inp = json.dumps(block.get("input") or {}, ensure_ascii=False)
                    parts.append(f"[assistant:tool_use {block.get('name', '')}] {inp[:800]}")
            continue
        for item in turn.items:
            if isinstance(item, Notice):
                parts.append(f"[user:text] {item.text[:1500]}")
            else:
                parts.append(f"[user:tool_result] {item.content[:800]}")
    joined = "\n".join(parts)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def last_assistant_prose(conversation: Conversation) -> str:
    """Return the text of the newest assistant turn, for pairing a steer with it.

    Harness notices after it do not hide it; a tool result does, since the
    model went on working.

    Args:
        conversation: The loop's history.

    Returns:
        The text, "" when a tool result follows the turn or there is none.
    """
    for turn in reversed(conversation.turns):
        if isinstance(turn, AssistantTurn):
            return "".join(
                str(b.get("text", ""))
                for b in turn.raw_content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if any(isinstance(item, ToolResultItem) for item in turn.items):
            return ""
    return ""
