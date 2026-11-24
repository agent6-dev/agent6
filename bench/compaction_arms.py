"""Define the compaction treatments without changing product settings."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from agent6 import kinds
from agent6.harness import loop_conversation as history
from agent6.harness.compaction import elision, retention, window


@dataclasses.dataclass(frozen=True, slots=True)
class Arm:
    """The four retention caps and the local cold-start rule."""

    operator: int = -1
    turns: int = -1
    chars: int = 80_000
    thinking: int = -1
    cold: kinds.ColdStart = "resend"


KEEP = {
    "baseline": Arm(),
    "no-operator": Arm(operator=0),
    "chars-20000": Arm(chars=20_000),
    "turns-1": Arm(turns=1),
    "operator-only": Arm(turns=0),
    "pointers": Arm(),
}
COLD = {
    "resend": Arm(),
    "drop": Arm(cold="drop"),
    "drop-thinking-2": Arm(cold="drop", thinking=2),
    "summarize": Arm(cold="summarize"),
    "truncate": Arm(cold="truncate"),
}
SELECT = retention.select


def pointer_tail(
    messages: Sequence[history.Message],
    settings: window.CompactionSettings,
    task: str = "",
    pins: Sequence[str] = (),
) -> retention.Replay:
    """Replace selected tool output after charging the same turns as the baseline."""
    replay = SELECT(messages, settings, task, pins)
    tail: list[history.Message] = []
    for msg in replay.tail:
        if isinstance(msg, history.UserMessage):
            msg = history.UserMessage(
                tuple(
                    dataclasses.replace(
                        item,
                        content=elision.elision_placeholder(
                            item.for_call.name, item.for_call.input
                        ),
                        attachment=None,
                    )
                    if isinstance(item, history.ToolResultItem) and item.for_call.name != "ask_user"
                    else item
                    for item in msg.items
                )
            )
        tail.append(msg)
    return dataclasses.replace(replay, tail=tuple(tail))
