# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a composer line does when it is not a steer.

`/btw`, `/task` and `/compact` act beside a live run: they open a side session,
add to the task graph, or ask for a compaction, and none of them reaches the
model as a message. Every entry point that takes a typed line routes it through
here (the TUI composer, the web composer, the CLI pause menu, `agent6 steer`),
so the same words do the same thing wherever they are typed.

`/pin` and `/parallel` are not here: they are steers the loop parses itself.
`/now` is not either: it is an ordinary steer carrying urgency, which each
surface submits its own way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent6.directive import parse_btw, parse_compact, parse_task
from agent6.sessions.ipc import queue_task, request_compact
from agent6.ui.btw import open_btw


@dataclass(frozen=True, slots=True)
class _Directive:
    """One directive: how to recognize it, what to say when it arrives bare
    (`""` when it is complete on its own, as `/compact` is), and what it does."""

    parse: Callable[[str], str | None]
    empty: str
    act: Callable[[Path, str], tuple[bool, str]]


def _btw(session_dir: Path, question: str) -> tuple[bool, str]:
    opened, line = open_btw(session_dir, question)
    return opened, line.removeprefix("[agent6] ")


def _task(session_dir: Path, text: str) -> tuple[bool, str]:
    queue_task(session_dir, text)
    return True, "task queued; it runs once the open tasks drain"


def _compact(session_dir: Path, focus: str) -> tuple[bool, str]:
    if not request_compact(session_dir, focus=focus):
        return False, "could not write the compaction request"
    return True, "compaction requested; it applies before the next model call"


_DIRECTIVES: tuple[_Directive, ...] = (
    _Directive(parse_btw, "/btw needs a question: /btw <question>", _btw),
    _Directive(parse_task, "/task needs the work: /task <text>", _task),
    _Directive(parse_compact, "", _compact),
)


def act_on_directive(session_dir: Path, text: str) -> tuple[bool, str] | None:
    """Act on *text* when it is a directive that works beside the run, and
    report `(did_it, what_to_say)`. None means *text* is an ordinary steer.

    The message is bare prose: a caller prefixes or styles it as its surface
    does."""
    for directive in _DIRECTIVES:
        arg = directive.parse(text)
        if arg is None:
            continue
        if not arg and directive.empty:
            return False, directive.empty
        return directive.act(session_dir, arg)
    return None
