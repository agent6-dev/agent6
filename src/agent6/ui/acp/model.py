# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Name a session's updates in ACP v2's vocabulary.

`updates` projects the journal into these, and `wire` renders each one as the frame the
connection's protocol version reads. Text is wire-ready: its builder scrubbed it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from agent6.ui.acp import rpc

# Whose message an update belongs to: v2's `user_message`, `agent_message` and `agent_thought`.
Role = Literal["user", "agent", "thought"]
ToolCallStatus = Literal["pending", "in_progress", "paused", "completed", "failed", "cancelled"]
# The session's foreground work: under way, blocked on the operator, or none.
State = Literal["running", "requires_action", "idle"]


@dataclasses.dataclass(frozen=True, slots=True)
class Message:
    """A whole message, which replaces what its id held.

    Attributes:
        role: Whose message it is.
        message_id: The id every update of the message carries.
        text: The message's text.
        prompt: The message is the prompt this connection sent, which v1 never echoes.
    """

    role: Role
    message_id: str
    text: str
    prompt: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class MessageChunk:
    """One piece of a message in flight, appended to what its id holds.

    Attributes:
        role: Whose message it is.
        message_id: The id every update of the message carries.
        text: The piece.
    """

    role: Role
    message_id: str
    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class DiffChange:
    """One file a tool call changed, with the text of both sides.

    Attributes:
        path: The file's absolute path.
        old_text: The text before the call; None for a file the call added.
        new_text: The text after the call; None for a file the call deleted.
    """

    path: str
    old_text: str | None
    new_text: str | None

    @property
    def operation(self) -> Literal["add", "delete", "modify"]:
        """Return what the call did to the file."""
        if self.old_text is None:
            return "add"
        return "delete" if self.new_text is None else "modify"


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallUpdate:
    """One upsert of a tool call; the first for an id creates the call.

    Attributes:
        tool_call_id: The call's id, unique for the life of the session.
        status: The call's status.
        title: The call's title, on the update that creates the call; "" after.
        name: The tool's name, sent with the title.
        kind: ACP's kind for the tool, sent with the title.
        text: What the tool said, or why the call waits; "" sends none.
        changes: The files a settled edit changed.
        locations: The absolute paths an editor can follow, each once.
    """

    tool_call_id: str
    status: ToolCallStatus
    title: str = ""
    name: str = ""
    kind: str = ""
    text: str = ""
    changes: tuple[DiffChange, ...] = ()
    locations: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class PlanEntry:
    """One task of the plan.

    Attributes:
        content: The task's title.
        status: `pending`, `in_progress` or `completed`.
    """

    content: str
    status: str


@dataclasses.dataclass(frozen=True, slots=True)
class PlanUpdate:
    """The session's complete task list, which replaces the one before.

    Attributes:
        entries: The tasks, in display order.
    """

    entries: tuple[PlanEntry, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Command:
    """One slash command an editor can offer.

    Attributes:
        name: The command's word, without the slash.
        description: What it does.
        hint: What to type after it; "" for a command that takes nothing.
    """

    name: str
    description: str
    hint: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class AvailableCommandsUpdate:
    """The session's complete command list.

    Attributes:
        commands: The commands.
    """

    commands: tuple[Command, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigOptionUpdate:
    """The session's complete config options, after agent6 changed them.

    Attributes:
        config_options: The options, as `options.config_options` builds them.
    """

    config_options: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class SessionInfoUpdate:
    """The session's title.

    Attributes:
        title: The task's headline.
    """

    title: str


@dataclasses.dataclass(frozen=True, slots=True)
class UsageUpdate:
    """The context in use and the run's spend.

    Attributes:
        used: The tokens of the last call's whole prompt.
        size: The model's context window.
        cost_usd: The run's cumulative USD; None when it has no priced total.
    """

    used: int
    size: int
    cost_usd: float | None


@dataclasses.dataclass(frozen=True, slots=True)
class StateUpdate:
    """A snapshot of the session's foreground work, which v1 reads from the prompt's reply.

    Attributes:
        state: What the foreground work is doing.
        stop_reason: Why the work ended, on the `idle` that ends it; "" on a snapshot.
        error: The failure that ended the work, with the `error` stop reason.
    """

    state: State
    stop_reason: str = ""
    error: rpc.RpcError | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayComplete:
    """The end of the history a `session/resume` replayed, which v1 reads from its reply."""


Update = (
    Message
    | MessageChunk
    | ToolCallUpdate
    | PlanUpdate
    | AvailableCommandsUpdate
    | ConfigOptionUpdate
    | SessionInfoUpdate
    | UsageUpdate
    | StateUpdate
    | ReplayComplete
)


@dataclasses.dataclass(frozen=True, slots=True)
class PermissionOption:
    """One answer a permission request offers; its place in the request is its id.

    Attributes:
        name: The answer's label.
        kind: `allow_once`, `allow_always`, `reject_once` or `reject_always`.
    """

    name: str
    kind: str


@dataclasses.dataclass(frozen=True, slots=True)
class PermissionRequest:
    """One approval or question put to the editor.

    Attributes:
        title: The prompt's text.
        tool_call_id: The tool call the answer gates; "" for a prompt that gates none.
        options: The answers, in order.
    """

    title: str
    tool_call_id: str
    options: tuple[PermissionOption, ...]
