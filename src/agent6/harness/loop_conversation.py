# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-owned conversation: typed messages over the provider wire.

`Conversation` holds the history as typed messages, one shape for every
consumer, and produces the Anthropic-wire dict list only at the boundary.
`to_wire` builds the list providers and snapshots take, with `cache_control`
stamped from the mark positions and assistant blocks verbatim, so thinking
blocks and unknown block types round-trip. `from_wire` accepts exactly the
shapes the loop writes and fails loudly on anything else.

Pair safety is structural: a `tool_use` message can only be followed by
`results` covering exactly its ids, `pop_quiet_assistant` removes only a
message with no tool calls, `restart` keeps whole messages, and compaction
rewrites result content in place. No operation strands a `tool_use`.

Rolling cache breakpoints: Anthropic bills a request's prefix up to a
`cache_control` breakpoint at 0.1x once cached (1.25x to write), and the
conversation dominates input tokens in a long run. The roll keeps two marks,
the previous call's position (the guaranteed hit) and the last user message's
final block (the next write), for four breakpoints with the provider's two
static ones, Anthropic's per-request maximum. Marks persist into resume
snapshots; tier-1 elision costs one re-write on the next call. OpenAI-format
providers never forward the field.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import json
from collections.abc import Callable, Sequence
from typing import Any, Literal

from agent6 import kinds, media, operator_input
from agent6.harness import nudges

TaskSource = Literal["harness", "operator", "model", "panel"]
NoticeOrigin = TaskSource | Literal["plan", "restart", "contract", "reply_count"]
# Reloads an operator's image a snapshot names; a string says why it cannot be shown.
ImageLoader = Callable[[operator_input.ImageRef], "kinds.MediaAttachment | str"]

_EPHEMERAL = {"type": "ephemeral"}
# Why an older image leaves a request: `media.AttachmentBudget` counts the newest first.
_OVER_BUDGET = "newer images took the request's limit of four images and 4 MiB"


def image_block(image: kinds.MediaAttachment) -> dict[str, Any]:
    """Return the wire block that carries an image or a document inline.

    Args:
        image: The media.

    Returns:
        The block `read_file` results and operator messages share.
    """
    return {
        "type": image.modality,
        "source": {
            "type": "base64",
            "media_type": image.media_type,
            "data": base64.b64encode(image.data).decode("ascii"),
        },
    }


def unshown_image_line(ref: operator_input.ImageRef, reason: str) -> str:
    """Return the line that stands in a message for an operator image the model is not shown.

    Args:
        ref: The image.
        reason: Why it is not shown, as a clause.

    Returns:
        The line, naming the image as its label in the message does.
    """
    return f"[{ref.label()} is not shown: {reason}]"


def image_ref(image: kinds.MediaAttachment) -> operator_input.ImageRef:
    """Return the reference that names an operator's image."""
    return operator_input.ImageRef.model_validate(
        {"media_type": image.media_type, "byte_size": image.byte_size, "sha256": image.sha256}
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ToolUse:
    """One tool call from an assistant message, parsed once from the raw blocks.

    Attributes:
        id: The provider's tool_use id.
        name: The tool's name.
        input: Whatever the provider parsed; the dispatcher's schema validation
            owns its shape.
    """

    id: str
    name: str
    input: Any


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResultItem:
    """One tool_result block.

    Attributes:
        tool_use_id: The id the wire carries.
        content: The result text.
        for_call: The ToolUse it answers, paired at construction so compaction
            never rebuilds an id index; in-memory only.
        attachment: The optional attachment, kept separately from the text.
        unshown_answer: Operator words the tool payload cap or repeat stub omitted.
    """

    tool_use_id: str
    content: str
    for_call: ToolUse
    attachment: kinds.MediaAttachment | None = None
    unshown_answer: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Notice:
    """Harness or operator text in a user message.

    The initial task, nudges, critiques, steering and the tier-2 restart summary
    are notices. Snapshots keep the origin. Providers receive plain text.

    Attributes:
        text: The text.
        origin: Who delivered the notice. A local summary is a `restart`.
        operator_words: Verbatim inputs within a rendered task, steer, answer notice or restart.
        result_index: Zero-based position of this answer's capped result in the same message.
        images: The images the operator sent with the text, which names each by its label.
    """

    text: str
    origin: NoticeOrigin = "harness"
    operator_words: tuple[str, ...] = ()
    result_index: int | None = None
    images: tuple[kinds.MediaAttachment, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class AssistantMessage:
    """One assistant message.

    Attributes:
        raw_content: The verbatim block tuple the provider returned; tool_use ids,
            thinking blocks and unknown block types round-trip untouched.
        tool_uses: The parsed tool_use view of those blocks.
    """

    raw_content: tuple[Any, ...]
    tool_uses: tuple[ToolUse, ...]

    def without_thinking(self) -> tuple[AssistantMessage, int]:
        """Remove thinking and redacted thinking without changing any tool calls.

        Returns:
            The stripped reply and the number of characters removed.
        """
        kept: list[Any] = []
        removed = 0
        for b in self.raw_content:
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
                removed += sum(len(str(v)) for k, v in b.items() if k != "type" and v is not None)
            else:
                kept.append(b)
        if len(kept) == len(self.raw_content):
            return self, 0
        return AssistantMessage(tuple(kept), self.tool_uses), removed

    def is_substantive(self) -> bool:
        """Return whether the message carries visible text or a tool call.

        An empty or thinking-only message is dead context: Anthropic rejects empty
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


@dataclasses.dataclass(frozen=True, slots=True)
class UserMessage:
    """One user message.

    Attributes:
        items: Tool results and notices in wire order, results first.
    """

    items: tuple[ToolResultItem | Notice, ...]

    @property
    def plan_text(self) -> str | None:
        """Return the standalone planner notice, including its header."""
        if len(self.items) != 1 or not isinstance(notice := self.items[0], Notice):
            return None
        if notice.origin == "plan" or (
            notice.origin == "harness"
            and notice.text.startswith(nudges.PLAN_ON_DISK_HEADER + "\n\n")
        ):
            return notice.text
        return None


Message = AssistantMessage | UserMessage


def latest_plan(messages: Sequence[Message]) -> UserMessage | None:
    """Return the most recent saved plan notice, if the planner has seen one."""
    return next(
        (m for m in reversed(messages) if isinstance(m, UserMessage) and m.plan_text is not None),
        None,
    )


def _parse_tool_uses(blocks: Sequence[Any]) -> tuple[ToolUse, ...]:
    return tuple(
        ToolUse(id=str(b.get("id", "")), name=str(b.get("name", "")), input=b.get("input", {}))
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )


def _result_ids(message: UserMessage) -> list[str]:
    return [it.tool_use_id for it in message.items if isinstance(it, ToolResultItem)]


def _results_first[T](items: Sequence[T], *, key: Callable[[T], object] = lambda it: it) -> list[T]:
    """Return the items with tool results first and notices after, each in order.

    Args:
        items: A user message's items, or pairs the key maps to an item.
        key: Reads the item from an element.

    Returns:
        The stable partition.
    """
    return sorted(items, key=lambda it: isinstance(key(it), Notice))


class Conversation:
    """The history as frozen messages plus the rolling cache-mark pair.

    Marks are (message index, item index) positions into user messages; `to_wire`
    stamps `cache_control` there. They live here rather than on the items
    because breakpoint placement moves as the tail grows, while the messages are
    history.
    """

    __slots__ = ("_after_results", "_marks", "_messages")

    def __init__(self) -> None:
        self._after_results: str | None = None
        self._messages: list[Message] = []
        self._marks: list[tuple[int, int]] = []

    # ---- reads ----------------------------------------------------------

    @property
    def messages(self) -> tuple[Message, ...]:
        """The messages, as a copy: no caller appends around the guarded mutators."""
        return tuple(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def complete(self, *, keep_reply: bool = False) -> Conversation:
        """Return the history up to the last complete turn, with its cache marks.

        Args:
            keep_reply: Leave even a prose reply after an endpoint summary of its request.

        Returns:
            This conversation, or its prefix without the assistant awaiting tool results.
        """
        if (
            self._messages
            and isinstance(last := self._messages[-1], AssistantMessage)
            and (keep_reply or last.tool_uses)
        ):
            prefix = Conversation()
            prefix._messages = self._messages[:-1]
            # Only user blocks carry marks.
            prefix._marks = self._marks.copy()
            return prefix
        return self

    # ---- appends (each preserves pair safety) ---------------------------

    def _require_no_open_call(self, what: str) -> None:
        if (
            self._messages
            and isinstance(prev := self._messages[-1], AssistantMessage)
            and prev.tool_uses
        ):
            raise ValueError(
                f"conversation invariant: cannot append {what} after an unanswered"
                " tool_use message; append its results first"
            )

    def assistant(self, raw_blocks: Any) -> AssistantMessage:
        """Append the assistant message exactly as the provider returned it.

        Args:
            raw_blocks: The provider's content blocks.

        Returns:
            The appended message.

        Raises:
            ValueError: A previous message's tool calls have no results yet.
        """
        self._require_no_open_call("an assistant message")
        blocks = tuple(raw_blocks)
        message = AssistantMessage(raw_content=blocks, tool_uses=_parse_tool_uses(blocks))
        self._messages.append(message)
        if not message.tool_uses and message.is_substantive():
            self._deliver_restart_notice()
        return message

    def results(self, items: Sequence[ToolResultItem | Notice]) -> None:
        """Append the user message answering the preceding tool_use message.

        The wire requires a user message to lead with its tool_result blocks, so
        notices are moved after the results here rather than trusted per call
        site.

        Args:
            items: The results, covering exactly the pending tool_use ids in
                order, plus any notices.

        Raises:
            ValueError: The result ids do not answer the pending tool_use ids.
        """
        prev = self._messages[-1] if self._messages else None
        want = [tu.id for tu in prev.tool_uses] if isinstance(prev, AssistantMessage) else []
        message = UserMessage(items=tuple(_results_first(items)))
        if _result_ids(message) != want:
            raise ValueError(
                f"conversation invariant: tool_result ids {_result_ids(message)} do not"
                f" answer the pending tool_use ids {want}"
            )
        self._messages.append(message)
        self._deliver_restart_notice()

    @property
    def pending_notice_chars(self) -> int:
        """Count the restart text reserved after the waiting reply or tool result."""
        return len(self._after_results or "")

    def _deliver_restart_notice(self) -> None:
        if self._after_results is not None:
            self.notice(self._after_results)
            self._after_results = None

    def adopt(
        self,
        block: dict[str, Any],
        keep: Sequence[Message],
        notice: str,
        *,
        awaiting_reply: bool = False,
    ) -> None:
        """Put the endpoint's block before the kept turns and the restart notice after them.

        Args:
            block: The unmodified summary block.
            keep: Messages after the request the block covers.
            awaiting_reply: The response to the summarized request has yet to enter history.
            notice: The task and restart instructions to repeat after the last complete turn.
        """
        self._messages.clear()
        self._marks.clear()
        self.assistant([block])
        self._messages.extend(keep)
        if awaiting_reply or (
            keep and isinstance(last := keep[-1], AssistantMessage) and last.tool_uses
        ):
            self._after_results = notice
        else:
            self.notice(notice)

    def unsign(self, notice: Callable[[str], str]) -> None:
        """Replace a leading summary block with a notice any route can read.

        Args:
            notice: Build the notice from the summary text.
        """
        first = self._messages[0] if self._messages else None
        if (
            isinstance(first, AssistantMessage)
            and len(first.raw_content) == 1
            and first.raw_content[0].get("type") == "compaction"
        ):
            text = notice(str(first.raw_content[0].get("content", "")))
            self._messages[0] = UserMessage(items=(Notice(text, "restart"),))

    def notice(
        self,
        text: str,
        *,
        origin: NoticeOrigin = "harness",
        operator_words: tuple[str, ...] = (),
        images: tuple[kinds.MediaAttachment, ...] = (),
    ) -> None:
        """Append harness or operator text as its own user message.

        Args:
            text: The notice.
            origin: `operator` for the task, a steer or an answer.
            operator_words: The operator input without its harness framing.
            images: The images the operator sent with the text.
        """
        self._append_notices((Notice(text, origin, operator_words, images=images),))

    def show_plan(self, text: str, *, restarted: bool) -> bool:
        """Show a changed plan without duplicating the notice kept by a restart.

        Args:
            text: The plan notice from disk, including its header.
            restarted: The history holds at most one retained plan notice.

        Returns:
            Whether the plan notice changed.
        """
        saved = latest_plan(self._messages)
        if saved is not None and saved.plan_text == text:
            return False
        if restarted and saved is not None:
            self._messages[self._messages.index(saved)] = UserMessage((Notice(text, "plan"),))
        else:
            self.notice(text, origin="plan")
        return True

    def latest_notice(self, prefix: str) -> str | None:
        """Return the newest notice starting with the prefix, as far as the history still holds it.

        Args:
            prefix: The notice's leading words.

        Returns:
            The notice's text, or None when a restart replaced it or none was shown.
        """
        for message in reversed(self._messages):
            if isinstance(message, UserMessage):
                for item in reversed(message.items):
                    if isinstance(item, Notice) and item.text.startswith(prefix):
                        return item.text
        return None

    @property
    def initiator(self) -> str:
        """Who the next call answers: `user` when operator text arrived since the last reply.

        A harness notice after the operator's text (a gate baseline, a presence note) does
        not make the turn the harness's; a nudge after the model's own reply does.
        """
        for message in reversed(self._messages):
            if isinstance(message, AssistantMessage):
                break
            if any(isinstance(it, Notice) and it.origin == "operator" for it in message.items):
                return "user"
        return "agent"

    def _append_notices(self, items: tuple[ToolResultItem | Notice, ...]) -> None:
        self._require_no_open_call("a notice")
        self._messages.append(UserMessage(items=items))

    def pop_quiet_assistant(self) -> None:
        """Drop a trailing non-substantive assistant message.

        Such a message has no tool_uses, so no pair can split; any other tail is left
        alone.
        """
        if (
            self._messages
            and isinstance(last := self._messages[-1], AssistantMessage)
            and not last.is_substantive()
        ):
            self._messages.pop()
            self._deliver_restart_notice()

    def restart(
        self,
        summary_text: str,
        keep: Sequence[Message] = (),
        *,
        quotes: tuple[str, ...] = (),
        endpoint_task: Notice | None = None,
    ) -> None:
        """Replace the history with a restart notice and paired messages.

        Args:
            summary_text: The rendered restart, including any quoted operator words.
            keep: The turns and pending inputs to replay.
            quotes: The quotes in the notice, for selection on a subsequent restart.
            endpoint_task: Repeat the opening task on a failed endpoint summary.

        Raises:
            ValueError: The kept tail starts with results whose calls are missing.
        """
        if keep and isinstance(keep[0], UserMessage) and _result_ids(keep[0]):
            raise ValueError("conversation invariant: a kept tail cannot lead with tool_results")
        notice = UserMessage(items=(Notice(summary_text, "restart", quotes),))
        opening: list[Message] = []
        if endpoint_task is not None:
            first = self._messages[0]
            opening.append(
                first if isinstance(first, UserMessage) else UserMessage((endpoint_task,))
            )
        self._messages[:] = [*opening, notice, *keep]
        self._marks = [m for m in self._marks if opening and m[0] == 0]

    def strip_thinking(self, message_idx: int) -> int:
        """Drop the thinking blocks from one assistant message, in place.

        The tool_use blocks survive verbatim, so pairing is untouched.

        Args:
            message_idx: The assistant message's index.

        Returns:
            The characters removed, 0 when the message had no thinking.

        Raises:
            ValueError: The index names a user message.
        """
        message = self._messages[message_idx]
        if not isinstance(message, AssistantMessage):
            raise ValueError("strip_thinking targets an assistant message")
        self._messages[message_idx], removed = message.without_thinking()
        return removed

    def set_result_content(
        self, message_idx: int, item_idx: int, content: str, *, drop_attachment: bool = False
    ) -> None:
        """Rewrite one tool_result's content in place.

        The id and pairing are untouched, so the wire stays balanced.

        Args:
            message_idx: The user message's index.
            item_idx: The result's index in the message.
            content: The new content.
            drop_attachment: Remove the image when replacing it with a placeholder or refusal.

        Raises:
            ValueError: The position is not a tool_result in a user message.
        """
        message = self._messages[message_idx]
        if not isinstance(message, UserMessage):
            raise ValueError("set_result_content targets a user message")
        item = message.items[item_idx]
        if not isinstance(item, ToolResultItem):
            raise ValueError("set_result_content targets a tool_result")
        items = list(message.items)
        items[item_idx] = dataclasses.replace(
            item, content=content, attachment=None if drop_attachment else item.attachment
        )
        self._messages[message_idx] = UserMessage(items=tuple(items))

    def drop_notice_images(
        self,
        message_idx: int,
        item_idx: int,
        drop: Callable[[kinds.MediaAttachment], str],
    ) -> tuple[str, ...]:
        """Take images out of an operator's notice, leaving a line that says why for each.

        Args:
            message_idx: The user message's index.
            item_idx: The notice's index in the message.
            drop: Returns why an image is not shown, or "" to keep it; called newest first.

        Returns:
            The lines added to the notice, one per dropped image.

        Raises:
            ValueError: The position is not a notice in a user message.
        """
        message = self._messages[message_idx]
        if not isinstance(message, UserMessage):
            raise ValueError("drop_notice_images targets a user message")
        item = message.items[item_idx]
        if not isinstance(item, Notice):
            raise ValueError("drop_notice_images targets a notice")
        reasons = {id(image): drop(image) for image in reversed(item.images)}
        lines = tuple(
            unshown_image_line(image_ref(image), reasons[id(image)])
            for image in item.images
            if reasons[id(image)]
        )
        if not lines:
            return ()
        items = list(message.items)
        items[item_idx] = dataclasses.replace(
            item,
            text="\n".join((item.text, *lines)),
            images=tuple(image for image in item.images if not reasons[id(image)]),
        )
        self._messages[message_idx] = UserMessage(items=tuple(items))
        return lines

    def limit_attachments(self, *, append_only: bool = False) -> tuple[tuple[str, str], ...]:
        """Keep attachments inside the outbound request limits.

        Args:
            append_only: Give previously delivered media priority over new results.

        Returns:
            The call IDs and refusal texts of attachments replaced in canonical history; an
            operator's image has no call ID.
        """
        budget = media.AttachmentBudget()
        refused: list[tuple[str, str]] = []
        positions = range(len(self._messages))
        for mi in positions if append_only else reversed(positions):
            message = self._messages[mi]
            if not isinstance(message, UserMessage):
                continue
            indices = range(len(message.items))
            for ii in indices if append_only else reversed(indices):
                item = message.items[ii]
                if isinstance(item, Notice):
                    lines = self.drop_notice_images(
                        mi, ii, lambda image: _OVER_BUDGET if budget.refusal(image) else ""
                    )
                    refused.extend(("", line) for line in lines)
                elif item.attachment is not None and (reason := budget.refusal(item.attachment)):
                    self.set_result_content(mi, ii, reason, drop_attachment=True)
                    refused.append((item.tool_use_id, reason))
        return tuple(refused)

    # ---- rolling cache breakpoints --------------------------------------

    def roll_cache_marks(self) -> None:
        """Advance the rolling cache-mark pair.

        The newest existing mark stays (the previous call's write, the guaranteed hit)
        and the final item of the newest user message is marked (the new write).
        Idempotent, so a crash-resume re-issuing the same call keeps its
        positions; positions survive content rewrites.
        """
        target: tuple[int, int] | None = None
        for m_idx in range(len(self._messages) - 1, -1, -1):
            message = self._messages[m_idx]
            # Assistant raw blocks are verbatim history and are never stamped.
            if isinstance(message, UserMessage) and message.items:
                target = (m_idx, len(message.items) - 1)
                break
        if target is None:
            self._marks = []
            return
        # The newest non-target mark is the previous call's breakpoint; the newest mark
        # is the target itself when nothing was appended.
        prev = next((m for m in reversed(self._marks) if m != target), None)
        self._marks = ([prev] if prev is not None else []) + [target]

    # ---- the wire boundary ----------------------------------------------

    def to_wire(self, *, snapshot: bool = False) -> list[dict[str, Any]]:
        """Return the provider or snapshot message list.

        Args:
            snapshot: Include notice origins and operator words for resume.

        Returns:
            Fresh dicts for user messages with `cache_control` stamped at the mark
            positions, the verbatim raw blocks for assistant messages.
        """
        marks = set(self._marks)
        out: list[dict[str, Any]] = []
        for m_idx, message in enumerate(self._messages):
            if isinstance(message, AssistantMessage):
                out.append({"role": "assistant", "content": list(message.raw_content)})
                continue
            blocks: list[dict[str, Any]] = []
            for i_idx, item in enumerate(message.items):
                if isinstance(item, ToolResultItem):
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": item.tool_use_id,
                        "content": (
                            [{"type": "text", "text": item.content}, image_block(item.attachment)]
                            if item.attachment is not None
                            else item.content
                        ),
                    }
                    if snapshot and item.unshown_answer is not None:
                        block["unshown_answer"] = item.unshown_answer
                    blocks.append(block)
                else:
                    block = {"type": "text", "text": item.text}
                    if snapshot and item.origin != "harness":
                        block["origin"] = item.origin
                    if snapshot and item.operator_words:
                        block["operator_words"] = list(item.operator_words)
                    if snapshot and item.result_index is not None:
                        block["result_index"] = item.result_index
                    blocks.append(block)
                    blocks.extend(image_block(image) for image in item.images)
                if (m_idx, i_idx) in marks:
                    # The item's last block, so the cached prefix covers its images.
                    blocks[-1]["cache_control"] = dict(_EPHEMERAL)
            out.append({"role": "user", "content": blocks})
        return out

    @classmethod
    def from_wire(
        cls, messages: Sequence[Any], *, load_image: ImageLoader | None = None
    ) -> Conversation:
        """Parse a persisted message list.

        A message in canonical order round-trips byte-for-byte through `to_wire`; a
        notice ahead of its results is moved after them on load, each mark kept
        on its own block. A snapshot holds an operator's image as its reference: the
        loader's image returns to its notice, and one it cannot load becomes a line there.

        Args:
            messages: The snapshot's message list.
            load_image: Reloads an operator's image from the session's attachments; without
                one, a referenced image is not shown.

        Returns:
            The conversation with its marks.

        Raises:
            ValueError: A message has a shape this loop cannot have written.
        """
        conv = cls()
        marks: list[tuple[int, int]] = []
        for m_idx, msg in enumerate(messages):
            where = f"message {m_idx}"
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
            prev = conv._messages[-1] if conv._messages else None
            pending = list(prev.tool_uses) if isinstance(prev, AssistantMessage) else []
            parsed: list[tuple[ToolResultItem | Notice, bool]] = []
            for i_idx, block in enumerate(content):
                at = f"{where} block {i_idx}"
                last = parsed[-1][0] if parsed else None
                is_image = isinstance(block, dict) and block.get("type") == "image"
                # An operator's image follows the text it came with.
                if is_image and isinstance(last, Notice):
                    notice = _with_image(last, block, load_image, where=at)
                    parsed[-1] = (notice, parsed[-1][1] or _block_mark(block, where=at))
                    continue
                item = _parse_user_block(block, pending, where=at)
                parsed.append((item, _block_mark(block, where=at)))
            # Canonicalize before recording positions, so each mark stays on its block.
            parsed = _results_first(parsed, key=lambda pair: pair[0])
            for i_idx, (_, marked) in enumerate(parsed):
                if marked:
                    marks.append((m_idx, i_idx))
            items = [item for item, _ in parsed]
            if any(isinstance(it, ToolResultItem) for it in items):
                conv.results(items)  # validates the pairing against the tool_use message
            else:
                conv._append_notices(tuple(items))
        conv._marks = marks
        return conv


def _snapshot_ref(source: dict[str, Any]) -> operator_input.ImageRef:
    """Return the reference a snapshot keeps in place of an operator image's bytes."""
    return operator_input.ImageRef.model_validate(
        {key: source.get(key) for key in ("media_type", "byte_size", "sha256")}
    )


def referenced_images(messages: Sequence[Any]) -> tuple[operator_input.ImageRef, ...]:
    """Return the operator images a snapshot's messages name, in order.

    Args:
        messages: A snapshot's or a checkpoint's message list.

    Returns:
        The references; a block that is not a well-formed reference is left out.
    """
    refs: list[operator_input.ImageRef] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else ():
            source = block.get("source") if isinstance(block, dict) else None
            if (
                isinstance(source, dict)
                and block.get("type") == "image"
                and source.get("payload_omitted") is True
            ):
                with contextlib.suppress(ValueError):
                    refs.append(_snapshot_ref(source))
    return tuple(refs)


def _with_image(
    notice: Notice, block: dict[str, Any], load_image: ImageLoader | None, *, where: str
) -> Notice:
    """Return the notice with the image a following wire block carries or names.

    Args:
        notice: The notice the image came with.
        block: The image block: inline base64, or the reference a snapshot keeps.
        load_image: Reloads a referenced image.
        where: The block's position, for the error.

    Returns:
        The notice with the image, or with the line that says why it is not shown.

    Raises:
        ValueError: The block is neither shape.
    """
    source = block.get("source")
    if set(block) - {"cache_control"} != {"type", "source"} or not isinstance(source, dict):
        raise ValueError(f"malformed conversation: {where} is not an image block")
    try:
        if source.get("payload_omitted") is True:
            ref = _snapshot_ref(source)
            loaded = load_image(ref) if load_image is not None else "its file was not reloaded"
            if isinstance(loaded, str):
                line = unshown_image_line(ref, loaded)
                return dataclasses.replace(notice, text=f"{notice.text}\n{line}")
        else:
            if set(source) != {"type", "media_type", "data"} or source["type"] != "base64":
                raise ValueError("not a base64 source")
            loaded = kinds.MediaAttachment(
                base64.b64decode(source["data"], validate=True), source["media_type"], ""
            )
            image_ref(loaded)  # an operator's image is never a document
    except (ValueError, TypeError) as exc:
        raise ValueError(f"malformed conversation: {where} is not an image block") from exc
    return dataclasses.replace(notice, images=(*notice.images, loaded))


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
    """Return the typed item for one user-message wire block.

    Args:
        block: The wire block.
        pending: The preceding assistant message's unanswered tool_uses; a result
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
        origin = block.get("origin", "harness")
        words = block.get("operator_words", [])
        result_index = block.get("result_index")
        if (
            keys - {"origin", "operator_words", "result_index"} != {"type", "text"}
            or not isinstance(block["text"], str)
            or ("result_index" in keys and (type(result_index) is not int or result_index < 0))
            or origin
            not in (
                "harness",
                "operator",
                "model",
                "panel",
                "plan",
                "restart",
                "contract",
                "reply_count",
            )
            or not isinstance(words, list)
            or not all(isinstance(w, str) for w in words)
        ):
            raise ValueError(f"malformed conversation: {where} is not a plain text block")
        return Notice(block["text"], origin, tuple(words), result_index)
    if block.get("type") == "tool_result":
        if (
            keys - {"unshown_answer"} != {"type", "tool_use_id", "content"}
            or ("unshown_answer" in block and not isinstance(block["unshown_answer"], str))
            or not isinstance(block["tool_use_id"], str)
            or not isinstance(block["content"], (str, list))
        ):
            raise ValueError(f"malformed conversation: {where} is not a plain tool_result block")
        if not pending or pending[0].id != block["tool_use_id"]:
            raise ValueError(
                f"malformed conversation: {where} answers tool_use"
                f" {block['tool_use_id']!r} out of order"
            )
        call = pending.pop(0)
        text, image = _parse_result_content(block["content"], call)
        return ToolResultItem(block["tool_use_id"], text, call, image, block.get("unshown_answer"))
    raise ValueError(f"malformed conversation: {where} has unsupported type {block.get('type')!r}")


def _parse_result_content(content: Any, call: ToolUse) -> tuple[str, kinds.MediaAttachment | None]:
    """Parse text or a text-plus-media result at the wire boundary.

    Args:
        content: The result's wire content.
        call: The answered tool call, carrying the source path.

    Returns:
        The text and optional image.

    Raises:
        ValueError: The content is not a supported result shape.
    """
    if isinstance(content, str):
        return content, None
    if (
        not isinstance(content, list)
        or len(content) != 2
        or not isinstance(content[0], dict)
        or set(content[0]) != {"type", "text"}
        or content[0]["type"] != "text"
        or not isinstance(content[0]["text"], str)
        or not isinstance(content[1], dict)
        or set(content[1]) != {"type", "source"}
        or content[1]["type"] not in ("image", "document")
    ):
        raise ValueError("malformed conversation: not a text-plus-media tool_result")
    source = content[1]["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"type", "media_type", "data"}
        or source["type"] != "base64"
        or not isinstance(source["data"], str)
        or not isinstance(source["media_type"], str)
    ):
        raise ValueError("malformed conversation: invalid image source")
    try:
        data = base64.b64decode(source["data"], validate=True)
    except ValueError as exc:
        raise ValueError("malformed conversation: invalid image base64") from exc
    path = str(call.input.get("path", "")) if isinstance(call.input, dict) else ""
    attachment = kinds.MediaAttachment(data, source["media_type"], path)
    if attachment.modality != content[1]["type"]:
        raise ValueError("malformed conversation: media type disagrees with block type")
    return content[0]["text"], attachment


def format_transcript_tail(
    messages: Sequence[Message], *, max_messages: int = 6, max_chars: int = 6000
) -> str:
    """Render the last few messages as a plain-text transcript for a summarizer call.

    Tool calls and results are clipped, and thinking blocks are skipped.

    Args:
        messages: The messages to render, newest last.
        max_messages: How many of the newest messages to render.
        max_chars: The cap, applied from the end.

    Returns:
        The transcript.
    """
    parts: list[str] = []
    for message in messages[-max_messages:]:
        if isinstance(message, AssistantMessage):
            for block in message.raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(f"[assistant:text] {str(block.get('text', ''))[:1500]}")
                elif btype == "tool_use":
                    inp = json.dumps(block.get("input") or {}, ensure_ascii=False)
                    parts.append(f"[assistant:tool_use {block.get('name', '')}] {inp[:800]}")
            continue
        for item in message.items:
            if isinstance(item, Notice):
                parts.append(f"[user:text] {item.text[:1500]}")
            else:
                parts.append(f"[user:tool_result] {item.content[:800]}")
    joined = "\n".join(parts)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def last_assistant_prose(conversation: Conversation) -> str:
    """Return the text of the newest assistant message, for pairing a steer with it.

    Harness notices after it do not hide it; a tool result does, since the
    model went on working.

    Args:
        conversation: The loop's history.

    Returns:
        The text, "" when a tool result follows the message or there is none.
    """
    for message in reversed(conversation.messages):
        if isinstance(message, AssistantMessage):
            return "".join(
                str(b.get("text", ""))
                for b in message.raw_content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if any(isinstance(item, ToolResultItem) for item in message.items):
            return ""
    return ""
