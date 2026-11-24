"""Define the in-process compaction strategy and its per-turn inputs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from agent6.graph import curator as graph_curator
from agent6.harness import loop_conversation
from agent6.harness.compaction import cold_start, window
from agent6.providers import provider_types
from agent6.sessions import journal
from agent6.tools import waits


class RestartNotice(Protocol):
    """Build the header for a local restart."""

    def __call__(self, quotes: Sequence[str], /, *, truncated: bool = False) -> str:
        """State the run facts and the retained operator messages.

        Args:
            quotes: Older operator messages outside the kept turns.
            truncated: No summary follows the notice.

        Returns:
            The restart notice.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class Context:
    """Give a strategy the prepared turn without exposing loop state.

    Attributes:
        conversation: The live history, with the current turn's notices and cache marks.
        settings: The resolved compaction settings.
        prefix_chars: The size of the system text and tool definitions.
        task: The original task.
        pins: Operator instructions that survive a restart.
        endpoint_notice: Build the notice that follows a signed summary and its reply.
        restart_notice: Build the restart header with operator quotes, pins and run facts.
        call_endpoint: Ask the endpoint to summarise the given request.
        call_worker: Append one instruction to the worker request using the supplied call id.
        call_helper: Send system and user text to the helper using the supplied call id.
        check_stop: Raise on a pending stop or steer.
        check_budget: Raise when the run cannot afford another call.
        compact_requested: Read the operator's pending focus, or None for no request.
        compact_clear: Consume that request.
        log: Write a run log line.
        emit: Append a journal event.
        curator: The task graph, if wired.
        root_task_id: The parent for tasks discovered during compaction.
        emit_graph_snapshot: Record a graph change after a check-off.
        execution_id: The execution that owns this turn.
        awaiting_reply: The reply to this request has not entered the conversation.
        cold_start: The rule that fired for the first request after restore.
    """

    conversation: loop_conversation.Conversation
    settings: window.CompactionSettings
    prefix_chars: int
    task: str
    pins: Sequence[str]
    endpoint_notice: Callable[[], str]
    restart_notice: RestartNotice
    call_endpoint: Callable[[str, str, int], provider_types.ProviderResponse]
    call_worker: Callable[[str, str], provider_types.ProviderResponse]
    call_helper: Callable[[str, str, str], provider_types.ProviderResponse]
    check_stop: Callable[[], None]
    check_budget: Callable[[], None]
    compact_requested: Callable[[], str | None]
    compact_clear: Callable[[], None]
    log: Callable[[str], None]
    emit: Callable[[journal.Event], None]
    curator: graph_curator.GraphCurator | None
    root_task_id: str | None
    emit_graph_snapshot: Callable[[], None]
    execution_id: str
    awaiting_reply: bool = False
    cold_start: cold_start.Trigger | None = None


@runtime_checkable
class IdleStrategy(Protocol):
    """Supply a deadline for waits after a worker response."""

    def idle(self, context: Context, request_started: float) -> waits.Deadline | None:
        """Return the deadline and callback, or None to leave this turn's waits alone.

        Args:
            context: The turn with the worker reply appended.
            request_started: The monotonic start of the worker's last provider attempt.

        Returns:
            One deadline shared by the turn's waits, or None.
        """
        ...


@runtime_checkable
class ClearingStrategy(Protocol):
    """Name the tool results the endpoint clears on a worker request."""

    def clearing(self, context: Context) -> provider_types.ClearToolResults | None:
        """Return the edit for the request about to be sent.

        Args:
            context: The prepared turn.

        Returns:
            The edit, or None to send the history whole.
        """
        ...


class Strategy(Protocol):
    """Compact the history of one worker request."""

    def compact(self, context: Context) -> bool:
        """Keep a valid history with complete tool exchanges.

        Args:
            context: The live conversation and this turn's services.

        Returns:
            True after a restart that needs fresh focus notices.
        """
        ...
