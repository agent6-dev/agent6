"""Exercise a compaction strategy supplied by a harness caller."""

from __future__ import annotations

import json
import pathlib

from agent6.harness import loop, loop_conversation, run_chain
from agent6.harness.compaction import strategy
from tests.unit import test_loop_wire_golden as scripted


class KeepLastExchange:
    """Keep the newest complete tool exchange."""

    def __init__(self) -> None:
        self.turns = 0

    def compact(self, context: strategy.Context) -> bool:
        self.turns += 1
        messages = context.conversation.messages
        if len(messages) < 5:
            return False
        context.check_stop()
        context.conversation.restart(
            context.restart_notice(()) + "Kept the newest exchange.", keep=messages[-2:]
        )
        return True


def test_standalone_strategy_replaces_the_history_the_worker_receives(
    tmp_path: pathlib.Path,
) -> None:
    snap = tmp_path / "loop_state.json"
    worker = scripted._WorkerScript([scripted._RESPONSES[i] for i in (0, 2, 4)], snap)
    keep = KeepLastExchange()
    harness = loop.Harness(
        chain=run_chain.RunChain(tmp_path),
        config=scripted._config(),
        provider=worker,
        dispatcher=scripted._Dispatcher([False]),  # type: ignore[arg-type]
        logger=lambda _: None,
        compactor=keep,
        resume_state_path=snap,
    )
    conversation = loop_conversation.Conversation()
    conversation.notice("TASK: keep the newest exchange")
    result = harness._drive_loop(
        system="SYSTEM",
        conversation=conversation,
        tool_calls=0,
        start_turn=1,
        root_task_id=None,
        original_task="keep the newest exchange",
    )
    assert result.completed and keep.turns == len(worker.captured) == 3
    last = json.loads(worker.captured[-1]["messages"])
    assert "TASK: keep the newest exchange" not in str(last)
    assert "Kept the newest exchange." in str(last)
    assert "tu-1a" not in str(last) and "tu-1b" not in str(last)
    assert last[-2]["content"][0]["id"] == last[-1]["content"][0]["tool_use_id"] == "tu-3"
