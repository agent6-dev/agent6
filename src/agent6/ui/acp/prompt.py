# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read one `session/prompt`, and tell the editor how its turn goes.

v2 answers a prompt once its message is in the conversation and v1 when its turn ends:
`Answer` holds that difference, and `Turn` is the form the rest of the front-end calls.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import threading
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from agent6 import directive
from agent6.sessions import ipc
from agent6.ui import operator_images
from agent6.ui.acp import model, rpc, updates

if TYPE_CHECKING:
    from agent6.ui.acp import session as acp_session


@dataclasses.dataclass(frozen=True, slots=True)
class Turn:
    """How one prompt's turn reaches the editor.

    v2 answers a prompt once its message is in the conversation and reports the work
    through the session's state; v1 answers when the work ends.

    Attributes:
        finish: Answers for the end of the turn's work: once, with the stop reason or the
            failed turn's error, `cancelled` after a cancel whatever the run returned.
        inserted: Says the prompt's message is in the conversation; safe to repeat.
        notify: Sends an update to the prompt's conversation.
        message_id: The id of the prompt's message.
    """

    finish: Callable[[acp_session.TurnEnd], None]
    inserted: Callable[[], None] = lambda: None
    notify: Callable[[model.Update], None] = lambda _update: None
    message_id: str = ""

    def begin(self, text: str) -> None:
        """Report a prompt whose message is in the conversation as its work begins.

        Args:
            text: The prompt's text.
        """
        self.inserted()
        if text:
            message = updates.printable(text)
            self.notify(model.Message("user", self.message_id, message, prompt=True))

    def end(self, end: acp_session.TurnEnd) -> None:
        """Report that the turn's work ended, then answer for it.

        Args:
            end: The stop reason, or the error that ended the work.
        """
        if isinstance(end, rpc.RpcError):
            self.notify(model.StateUpdate("idle", "error", end))
        else:
            self.notify(model.StateUpdate("idle", end))
        self.finish(end)


class Reply(Protocol):
    """Send the reply to one request."""

    def __call__(
        self,
        req_id: object,
        *,
        result: dict[str, Any] | None = None,
        error: rpc.RpcError | None = None,
    ) -> None:
        """Send the reply to one request: its result, or its error.

        Args:
            req_id: The request the reply answers.
            result: The result of a request that succeeded.
            error: The error of one that did not.
        """
        ...


@dataclasses.dataclass
class Answer:
    """Answer one `session/prompt` as the connection's version orders it.

    v1 answers when the turn's work ends, with its stop reason or its error. v2 answers
    once the prompt's message is in the conversation, with the message's id; an error
    answers only a prompt whose message never got in. A retry that arrives before the
    answer shares it.

    Attributes:
        version: The connection's protocol version.
        session: The session the prompt came on.
        message_id: The id of the prompt's message.
        request_ids: The requests the answer goes to: the prompt's, then each retry's.
        reply: Sends the answer to one request.
        answered: Called once, as the answer goes out.
    """

    version: int
    session: acp_session.Session
    message_id: str
    request_ids: list[object]
    reply: Reply
    answered: Callable[[], object]
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _answered: bool = False

    def inserted(self) -> None:
        """Record the prompt's message as in the conversation, and answer a v2 prompt."""
        self.session.messages.add(self.message_id)
        if self.version >= 2:
            self._once(result={"messageId": self.message_id})

    def ended(self, end: acp_session.TurnEnd) -> None:
        """Answer a v1 prompt for its turn's end, or a v2 prompt whose message never got in."""
        if isinstance(end, rpc.RpcError):
            self._once(error=end)
        elif self.version == 1:
            self._once(result={"stopReason": end})
        elif end == "cancelled":
            why = "the prompt was cancelled before its message was added"
            self._once(error=rpc.RpcError(rpc.REQUEST_CANCELLED, why))
        else:
            self.inserted()

    def join(self, request_id: object) -> bool:
        """Add a retry's request to the answer.

        Args:
            request_id: The retry's request.

        Returns:
            False when the answer already went out.
        """
        with self._lock:
            if not self._answered:
                self.request_ids.append(request_id)
            return not self._answered

    def _once(
        self, *, result: dict[str, Any] | None = None, error: rpc.RpcError | None = None
    ) -> None:
        with self._lock:
            if self._answered:
                return
            self._answered = True
        self.answered()
        for request_id in self.request_ids:
            self.reply(request_id, result=result, error=error)


def message_id(acp_session_id: str, key: object) -> str:
    """Return the id of a prompt's message, derived from its idempotency key when it has one.

    The same key on the same session names the same message, across restarts too, and
    the key itself is never stored.

    Args:
        acp_session_id: The session the prompt came on.
        key: The prompt's `idempotencyKey`, or None.

    Returns:
        A fresh id for a prompt without a key.

    Raises:
        RpcError: The key is not a string of 1 to 255 characters.
    """
    if key is None:
        return uuid.uuid4().hex
    if not isinstance(key, str) or not 1 <= len(key) <= 255:
        raise rpc.RpcError(
            rpc.INVALID_PARAMS, "idempotencyKey must be a string of 1 to 255 characters"
        )
    return uuid.uuid5(uuid.NAMESPACE_OID, f"agent6.acp.prompt\n{acp_session_id}\n{key}").hex


def steer(
    session: acp_session.Session,
    state_dir: pathlib.Path,
    text: str,
    *,
    turn: Turn,
    images: tuple[operator_images.EncodedImage, ...] = (),
) -> None:
    """Deliver a prompt that arrived during the session's run as a steer.

    The prompt joins the run's steer queue, as a message typed during a run does on
    every other surface, and is in the conversation from then: the run takes it at its
    next turn boundary, or a later execution does when this one ends first.

    Args:
        session: The session the prompt came on, with an execution in flight.
        state_dir: The state dir of the session's repository.
        text: The prompt, as `words` joined it.
        turn: How the turn reaches the editor.
        images: The prompt's image blocks, as `image_blocks` read them.

    Raises:
        RpcError: The execution has not chosen its run yet, the text is a composer
            command or a malformed directive, an image is refused, or the steer could
            not be written.
    """
    if not session.steerable:
        raise rpc.RpcError(
            rpc.INTERNAL_ERROR, "this session's run is still starting; send the prompt again"
        )
    if (problem := directive.steer_problem(text)) is not None:
        raise rpc.RpcError(rpc.INVALID_PARAMS, problem)
    # Terminal verbs have no ACP meaning; only session/cancel stops the execution.
    task = f"Task: {text}" if text.strip().lower() in ipc.STEER_VERBS else text
    session_dir = session.layout(state_dir).session_dir
    try:
        message = operator_images.accept(session_dir, task, images)
    except (operator_images.ImageRefusedError, OSError) as exc:
        raise rpc.RpcError(rpc.INVALID_PARAMS, f"not sent: {exc}") from exc
    if not ipc.submit_steer(
        session_dir, message.text, images=message.images, message_id=turn.message_id
    ):
        raise rpc.RpcError(rpc.INTERNAL_ERROR, "could not write the steer request")
    turn.begin(message.text)


def holds(
    session: acp_session.Session,
    state_dir_for: Callable[[pathlib.Path], pathlib.Path],
    message_id: str,
) -> bool:
    """Return whether the conversation holds the message a prompt's id names.

    Args:
        session: The session the prompt came on.
        state_dir_for: The state dir for a working directory, read for a recorded run.
        message_id: The id, which a prompt with an idempotency key derives from the key.

    Returns:
        True for a message this connection added, one the run's journal records, or
        one still waiting in the run's steer queue: a retry of its prompt adds nothing.
    """
    if message_id in session.messages:
        return True
    if not session.session_id:
        return False
    saved = session.layout(state_dir_for(session.cwd))
    if any(entry.message_id == message_id for entry in ipc.queued_steers(saved.session_dir)):
        return True
    needle = message_id.encode()
    try:
        slot = (saved.session_dir / ipc.STEER_MESSAGE_ID_FILE).read_bytes()
    except OSError:
        slot = b""
    if slot == needle:
        return True
    try:
        with saved.logs_path.open("rb") as rows:
            # The id is 32 hex digits: a row without those bytes cannot hold it.
            hits = [row for row in rows if needle in row]
    except OSError:
        return False
    return any(_row_names(row, message_id) for row in hits)


def _row_names(row: bytes, message_id: str) -> bool:
    """Return whether a journal row records the message with this id."""
    try:
        event = json.loads(row)
    except ValueError:
        return False
    return isinstance(event, dict) and event.get("message_id") == message_id


def image_blocks(params: dict[str, Any]) -> tuple[operator_images.EncodedImage, ...]:
    """Return the prompt's image blocks, in order, as the editor sent them.

    A block's `uri` is the editor's and is never opened: only the inline `data` is read.

    Args:
        params: The `session/prompt` params.

    Returns:
        Each image's declared media type and base64 data; `operator_images` checks them.

    Raises:
        RpcError: An image block lacks its `mimeType` or its `data`.
    """
    blocks: Any = params.get("prompt")
    images: list[operator_images.EncodedImage] = []
    for index, b in enumerate(blocks if isinstance(blocks, list) else [], start=1):
        if not isinstance(b, dict) or b.get("type") != "image":
            continue
        for field in ("mimeType", "data"):
            if not isinstance(b.get(field), str):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, f"prompt content block {index} {field} must be a string"
                )
        images.append((b["mimeType"], b["data"]))
    return tuple(images)


def words(params: dict[str, Any], *, allow_empty: bool = False, strict: bool = False) -> str:
    """Join the prompt's text and resource_link blocks into the task.

    A `resource_link` renders as its uri, which the model reads through the
    ordinary tools; an image block is read by `image_blocks`.

    Args:
        params: The `session/prompt` params.
        allow_empty: The prompt carries images, so it needs no text.
        strict: Refuse a block of a type `initialize` did not offer, as v2 rules; v1
            drops one.

    Returns:
        The task text.

    Raises:
        RpcError: The blocks are malformed, carry no text, or under `strict` hold a type
            agent6 does not take.
    """
    blocks = params.get("prompt")
    if not isinstance(blocks, list):
        raise rpc.RpcError(rpc.INVALID_PARAMS, "prompt must be a list of content blocks")
    parts: list[str] = []
    for index, b in enumerate(blocks, start=1):
        if not isinstance(b, dict):
            raise rpc.RpcError(
                rpc.INVALID_PARAMS, f"prompt content block {index} must be an object"
            )
        if b.get("type") == "text":
            text = b.get("text")
            if not isinstance(text, str):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, f"prompt content block {index} text must be a string"
                )
            if text:
                parts.append(text)
        elif b.get("type") == "resource_link":
            uri = b.get("uri")
            if not isinstance(uri, str):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, f"prompt content block {index} uri must be a string"
                )
            if uri:
                parts.append(f"Attached: {uri}")
        elif strict and b.get("type") != "image":
            raise rpc.RpcError(
                rpc.INVALID_PARAMS,
                f"prompt content block {index} is {b.get('type')!r};"
                " agent6 takes text, resource_link and image blocks",
            )
    text = "\n\n".join(parts).strip()
    if not text and not allow_empty:
        raise rpc.RpcError(rpc.INVALID_PARAMS, "the prompt carried no text")
    return text
