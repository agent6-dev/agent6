# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-owned conversation: typed turns over the provider wire.

The agent loop's history used to be a raw ``list[dict]`` in Anthropic wire
shape, re-parsed defensively by every consumer (compaction, cache roll, critic
tail). ``Conversation`` owns that history as typed turns and produces the wire
form only at the boundary:

- ``to_wire()`` builds the exact dict list providers and resume snapshots
  take: same keys, same order, ``cache_control`` stamped from the mark
  positions. Assistant blocks are the verbatim tuple the provider returned,
  so thinking blocks / signatures / unknown block types round-trip untouched.
- ``from_wire()`` is the one guarded parser (snapshot load): it accepts
  exactly the shapes the loop writes and reproduces them byte-for-byte, and
  it fails loudly on anything else.

Pair safety is structural, not disciplined: a ``tool_use`` turn can only be
followed by ``results()`` covering exactly its ids, ``pop_quiet_assistant``
removes only a turn with no tool calls, ``restart`` keeps whole turns, and
compaction rewrites result *content* in place via ``set_result_content``. No
operation can strand a ``tool_use`` without its ``tool_result``.

Rolling cache breakpoints (``roll_cache_marks``): Anthropic prompt caching
bills a request's prefix up to a ``cache_control`` breakpoint at 0.1x once
cached (1.25x to write). The provider marks the system prompt and the tool
list, but the conversation dominates input tokens in a long run and grows
every turn: without a breakpoint near the tail, each turn re-bills the whole
history at full price (quadratic in run length). The roll keeps exactly two
marks: the previous call's position (the guaranteed cache HIT) and the final
block of the last user turn (the WRITE the next call's hit lands on) --
4 breakpoints total with the provider's two static ones, Anthropic's
per-request maximum. Marks persist through ``to_wire()`` into resume
snapshots, so continuity survives crash-resume; tier-1 elision rewrites old
blocks and costs one 1.25x re-write on the next call, and the rolling pair
keeps caching from there. OpenAI-format providers rebuild content
block-by-block and never forward the field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

_EPHEMERAL = {"type": "ephemeral"}


@dataclass(frozen=True, slots=True)
class ToolUse:
    """One tool call from an assistant turn, parsed once from the raw blocks.

    ``input`` is whatever the provider parsed (both providers guarantee a
    dict in practice); the dispatcher's schema validation owns its shape.
    """

    id: str
    name: str
    input: Any


@dataclass(frozen=True, slots=True)
class ToolResultItem:
    """One tool_result block. ``for_call`` is the ToolUse it answers, paired
    at construction, so compaction never rebuilds an id index. In-memory
    only: the wire carries ``tool_use_id``."""

    tool_use_id: str
    content: str
    for_call: ToolUse


@dataclass(frozen=True, slots=True)
class Notice:
    """Harness/operator text injected as (or into) a user turn: the initial
    task, nudges, critiques, steering, the tier-2 restart summary."""

    text: str


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """One assistant message. ``raw_content`` is the verbatim block tuple the
    provider returned (exact round-trip; tool_use IDs, thinking blocks and
    unknown block types survive untouched); ``tool_uses`` is its parsed
    tool_use view."""

    raw_content: tuple[Any, ...]
    tool_uses: tuple[ToolUse, ...]

    def is_substantive(self) -> bool:
        """True when the turn carries visible text or a tool call. A turn
        that is neither (empty, or thinking-only reasoning starvation) is
        dead context: Anthropic rejects empty assistant content and strict
        OpenAI-compatible backends 400 on the translation."""
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
    """One user message: tool results and/or notices, in wire order (a
    harness notice may sit between two results in the same turn)."""

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


class Conversation:
    """Mutable container of frozen turns plus the rolling cache-mark pair.

    Marks are (turn index, item index) positions into user turns; ``to_wire``
    stamps ``cache_control`` there. They live here (not on the frozen items)
    because breakpoint placement is a wire concern that moves as the tail
    grows, while the turns themselves are history.
    """

    __slots__ = ("_marks", "_turns")

    def __init__(self) -> None:
        self._turns: list[Turn] = []
        self._marks: list[tuple[int, int]] = []

    # ---- reads ----------------------------------------------------------

    @property
    def turns(self) -> tuple[Turn, ...]:
        # A copy, not the live list: pair safety is structural only if no
        # caller can append around the guarded mutators.
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
        """Append the assistant turn exactly as the provider returned it."""
        self._require_no_open_call("an assistant turn")
        blocks = tuple(raw_blocks)
        turn = AssistantTurn(raw_content=blocks, tool_uses=_parse_tool_uses(blocks))
        self._turns.append(turn)
        return turn

    def results(self, items: Sequence[ToolResultItem | Notice]) -> None:
        """Append the user turn answering the preceding tool_use turn. The
        result items must cover exactly its tool_use ids, in order (notices
        may interleave); anything else is a split pair and raises."""
        prev = self._turns[-1] if self._turns else None
        want = [tu.id for tu in prev.tool_uses] if isinstance(prev, AssistantTurn) else []
        turn = UserTurn(items=tuple(items))
        if _result_ids(turn) != want:
            raise ValueError(
                f"conversation invariant: tool_result ids {_result_ids(turn)} do not"
                f" answer the pending tool_use ids {want}"
            )
        self._turns.append(turn)

    def notice(self, text: str) -> None:
        """Append harness/operator text as its own user turn."""
        self._append_notices((Notice(text),))

    def _append_notices(self, items: tuple[ToolResultItem | Notice, ...]) -> None:
        self._require_no_open_call("a notice")
        self._turns.append(UserTurn(items=items))

    def pop_quiet_assistant(self) -> None:
        """Drop a trailing non-substantive assistant turn (went-quiet repair).
        Such a turn has no tool_uses by definition, so no pair can split; a
        substantive tail or a non-assistant tail is left alone."""
        if (
            self._turns
            and isinstance(last := self._turns[-1], AssistantTurn)
            and not last.is_substantive()
        ):
            self._turns.pop()

    def restart(self, summary_text: str) -> None:
        """Tier-2 restart: keep the initial turn, replace everything after it
        with one summary notice. Marks outside the kept turn are dropped (the
        blocks they pointed at are gone)."""
        first = self._turns[0]
        if isinstance(first, AssistantTurn) and first.tool_uses:
            raise ValueError("conversation invariant: cannot restart from a tool_use turn")
        self._turns[:] = [first, UserTurn(items=(Notice(summary_text),))]
        self._marks = [m for m in self._marks if m[0] == 0]

    def set_result_content(self, turn_idx: int, item_idx: int, content: str) -> None:
        """Rewrite one tool_result's content in place (tier-1 elision). The
        id and pairing are untouched, so the wire stays balanced."""
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
        """Advance the rolling pair (see module docstring): keep the newest
        existing mark (the previous call's write position, now the guaranteed
        hit) and mark the final item of the newest user turn (the new write).
        Idempotent, so a crash-resume re-issuing the same call keeps its
        positions; safe after compaction (positions survive content rewrites,
        and ``restart`` already dropped any that lost their block)."""
        target: tuple[int, int] | None = None
        for t_idx in range(len(self._turns) - 1, -1, -1):
            turn = self._turns[t_idx]
            # Only user-turn items carry positional marks. The loop always
            # rolls with a user turn at the tail; assistant raw blocks are
            # verbatim history and are never stamped.
            if isinstance(turn, UserTurn) and turn.items:
                target = (t_idx, len(turn.items) - 1)
                break
        if target is None:
            self._marks = []
            return
        # _marks is in message order, so its newest non-target entry is the
        # previous call's breakpoint: the position whose prefix the cache
        # already holds. Keeping it (rather than the newest mark, which is the
        # target itself when nothing was appended, e.g. a crash-resume
        # re-issuing the same call) makes the roll idempotent.
        prev = next((m for m in reversed(self._marks) if m != target), None)
        self._marks = ([prev] if prev is not None else []) + [target]

    # ---- the wire boundary ----------------------------------------------

    def to_wire(self) -> list[dict[str, Any]]:
        """The provider/snapshot message list: fresh dicts for user turns
        (with ``cache_control`` stamped at the mark positions), the verbatim
        raw blocks for assistant turns."""
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
        """Parse a persisted message list (resume/fork snapshot load). Accepts
        exactly the shapes the loop writes -- anything it accepts round-trips
        byte-for-byte through ``to_wire`` -- and raises ValueError loudly on
        every other shape (a snapshot this loop cannot have written)."""
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
            items: list[ToolResultItem | Notice] = []
            for i_idx, block in enumerate(content):
                item = _parse_user_block(block, pending, where=f"{where} block {i_idx}")
                if _block_mark(block, where=f"{where} block {i_idx}"):
                    marks.append((t_idx, i_idx))
                items.append(item)
            if any(isinstance(it, ToolResultItem) for it in items):
                conv.results(items)  # validates the pairing against the tool_use turn
            else:
                conv._append_notices(tuple(items))
        conv._marks = marks
        return conv


def _block_mark(block: dict[str, Any], *, where: str) -> bool:
    """Whether a parsed user block carries the (validated) cache mark."""
    if "cache_control" not in block:
        return False
    if block["cache_control"] != _EPHEMERAL:
        raise ValueError(f"malformed conversation: {where} has a non-ephemeral cache_control")
    return True


def _parse_user_block(block: Any, pending: list[ToolUse], *, where: str) -> ToolResultItem | Notice:
    """One user-turn wire block -> its typed item. ``pending`` is the
    preceding assistant turn's unanswered tool_uses; results consume it in
    order (Conversation.results re-validates the full pairing)."""
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
