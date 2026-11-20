"""Define the in-process compaction strategy and its per-turn inputs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Protocol

from agent6.graph import curator as graph_curator
from agent6.harness import loop_conversation
from agent6.harness.compaction import window
from agent6.providers import provider_types
from agent6.sessions import journal


@dataclasses.dataclass(frozen=True, slots=True)
class Context:
    """Give a strategy the prepared turn without exposing loop state.

    Attributes:
        conversation: The live history, with the current turn's notices and cache marks.
        settings: The resolved compaction settings.
        prefix_chars: The size of the system text and tool definitions.
        task: The original task.
        pins: Operator instructions that survive a restart.
        restart_notice: Build the restart header with current pins, rulings and verify fact.
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
    """

    conversation: loop_conversation.Conversation
    settings: window.CompactionSettings
    prefix_chars: int
    task: str
    pins: Sequence[str]
    restart_notice: Callable[[], str]
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


class Strategy(Protocol):
    """Compact one prepared turn before its ordinary provider call."""

    def compact(self, context: Context) -> bool:
        """Keep a valid history with the original task and complete tool exchanges.

        Args:
            context: The live conversation and this turn's services.

        Returns:
            True after a restart that needs fresh focus notices.
        """
        ...
