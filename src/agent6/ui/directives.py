# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Act on a composer line that is not a steer.

`/btw`, `/task`, `/standing`, `/retire` and `/compact` act beside a live run and
never reach the model as a message; `/now` is a steer that interrupts the call
in flight. Every entry point that takes a typed line routes it through here, so
the same words do the same thing wherever they are typed. `/pin` and
`/parallel` are steers the loop parses itself.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

from agent6 import directive as agent6_directive
from agent6.graph import order, storage
from agent6.sessions import ipc, layout
from agent6.ui import btw
from agent6.viewmodel import format


@dataclasses.dataclass(frozen=True, slots=True)
class _Directive:
    """One directive.

    Attributes:
        parse: Recognizes it, returning its argument or None.
        empty: What to say when it arrives bare; "" when it is complete on its own.
        act: Does it, reporting whether it did and what to say.
    """

    parse: Callable[[str], str | None]
    empty: str
    act: Callable[[pathlib.Path, str], tuple[bool, str]]


def _btw(session_dir: pathlib.Path, question: str) -> tuple[bool, str]:
    """Open a side question beside the run.

    Returns:
        Whether it opened, and what to say.
    """
    opened, line = btw.open_btw(session_dir, question)
    return opened, line.removeprefix("[agent6] ")


def _task(session_dir: pathlib.Path, text: str) -> tuple[bool, str]:
    """Queue work into the task graph.

    Returns:
        True, and what to say.
    """
    ipc.queue_request(session_dir, "task", text)
    return True, "task queued; it runs once the open tasks drain"


def _standing(session_dir: pathlib.Path, goal: str) -> tuple[bool, str]:
    """Set the run's standing goal.

    Returns:
        True, and what to say.
    """
    ipc.queue_request(session_dir, "standing", goal)
    return True, "standing goal set; it replaces any the run had, at the next step"


def _retire(session_dir: pathlib.Path, named: str) -> tuple[bool, str]:
    """Retire a task, by the id the task tree prints.

    Returns:
        Whether the task was found, and what to say.
    """
    wanted = named.strip().upper()
    nodes = storage.load_graph(layout.layout_of(session_dir))
    matches = [nid for nid in nodes if format.short_task_id(nid) == wanted or nid == wanted]
    if not matches:
        if not nodes:
            return False, "this run has no tasks yet"
        shown = "  ".join(
            f"{format.short_task_id(nid)} {nodes[nid].title[:30]}"
            for nid in sorted(nodes, key=order.id_order)
        )
        return False, f"no task {named!r} here. This run has: {shown}"
    task_id = matches[0]
    ipc.queue_request(session_dir, "retire", task_id)
    return True, f"retiring {nodes[task_id].title[:60]!r} at the next step"


def _compact(session_dir: pathlib.Path, focus: str) -> tuple[bool, str]:
    """Ask for a compaction before the next model call.

    Returns:
        Whether the request was written, and what to say.
    """
    if not ipc.request_compact(session_dir, focus=focus):
        return False, "could not write the compaction request"
    return True, "compaction requested; it applies before the next model call"


_DIRECTIVES: tuple[_Directive, ...] = (
    _Directive(agent6_directive.parse_btw, "/btw needs a question: /btw <question>", _btw),
    _Directive(agent6_directive.parse_task, "/task needs the work: /task <text>", _task),
    _Directive(
        agent6_directive.parse_standing,
        "/standing needs the goal: /standing <text> (the task pane shows the current one)",
        _standing,
    ),
    _Directive(
        agent6_directive.parse_retire,
        "/retire needs the task: /retire <task id>, the number /tasks prints",
        _retire,
    ),
    _Directive(agent6_directive.parse_compact, "", _compact),
)


def submit_composer_line(
    session_dir: pathlib.Path, text: str, *, now: bool = False
) -> tuple[bool, str]:
    """Act on a composer line: a directive when it is one, else the steer it is.

    Args:
        session_dir: The live session.
        text: The line.
        now: Force the urgency `/now` spells, for `agent6 steer --now`.

    Returns:
        Whether it acted, and what to say.
    """
    handled = act_on_directive(session_dir, text)
    if handled is not None:
        return handled
    urgent = agent6_directive.parse_now(text)
    if urgent == "":
        return False, "/now needs the instruction: /now <text>"
    now = now or urgent is not None
    sent = urgent or text
    if not ipc.submit_steer(session_dir, sent, now=now):
        return False, "could not write the steer request"
    said = "steering now, interrupting the call in flight" if now else "steering"
    # A token inside the sent text travelled as words, in case it was meant as a directive.
    if (stray := agent6_directive.stray_directive(sent)) is not None:
        said += f" (`{stray}` mid-line is text; a directive has to start the line)"
    return True, said


def act_on_directive(session_dir: pathlib.Path, text: str) -> tuple[bool, str] | None:
    """Act on the line when it is a directive that works beside the run.

    Args:
        session_dir: The live session.
        text: The line.

    Returns:
        Whether it acted and what to say, as bare prose; None for an ordinary steer.
    """
    for directive in _DIRECTIVES:
        arg = directive.parse(text)
        if arg is None:
            continue
        if not arg and directive.empty:
            return False, directive.empty
        return directive.act(session_dir, arg)
    return None
