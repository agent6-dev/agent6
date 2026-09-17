# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a composer line does when it is not a steer.

`/btw`, `/task`, `/standing` and `/compact` act beside a live run: they open a
side session, add to or re-aim the task graph, or ask for a compaction, and
none of them reaches the model as a message. Every entry point that takes a
typed line routes it through here (the TUI composer, the web composer, the CLI
pause menu, `agent6 steer`), so the same words do the same thing wherever they
are typed.

`/now` is the fourth thing a line can be: an ordinary steer that interrupts the
call in flight, so `submit_steer` takes it here too and every surface reports
the same outcome.

`/pin` and `/parallel` are not here: they are steers the loop parses itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent6.directive import (
    parse_btw,
    parse_compact,
    parse_now,
    parse_retire,
    parse_standing,
    parse_task,
)
from agent6.graph.storage import load_graph
from agent6.sessions.ipc import (
    queue_task,
    request_compact,
    retire_task,
    set_standing_goal,
    submit_steer,
)
from agent6.sessions.layout import layout_of
from agent6.ui.btw import open_btw
from agent6.viewmodel.format import SHORT_TASK_ID, short_task_id


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


def _standing(session_dir: Path, goal: str) -> tuple[bool, str]:
    set_standing_goal(session_dir, goal)
    return True, "standing goal set; it replaces any the run had, at the next step"


def _retire(session_dir: Path, named: str) -> tuple[bool, str]:
    """Retire the task *named* names: its id, or the end of it that `/tasks`
    prints. Ids made in one turn differ only in their last characters, so the
    match is on the tail; an ambiguous one is refused by name, and a typo
    matches nothing rather than a neighbour."""
    wanted = named.strip().upper()
    if len(wanted) < SHORT_TASK_ID:
        return False, f"name at least {SHORT_TASK_ID} characters of the task id, as /tasks shows it"
    nodes = load_graph(layout_of(session_dir))
    matches = sorted(nid for nid in nodes if nid.endswith(wanted))
    if not matches:
        return False, f"no task here ends with {named!r}"
    if len(matches) > 1:
        shown = "  ".join(f"{short_task_id(nid)} {nodes[nid].title[:40]}" for nid in matches[:5])
        return False, f"{named!r} names {len(matches)} tasks: {shown}"
    task_id = matches[0]
    retire_task(session_dir, task_id)
    return True, f"retiring {nodes[task_id].title[:60]!r} at the next step"


def _compact(session_dir: Path, focus: str) -> tuple[bool, str]:
    if not request_compact(session_dir, focus=focus):
        return False, "could not write the compaction request"
    return True, "compaction requested; it applies before the next model call"


_DIRECTIVES: tuple[_Directive, ...] = (
    _Directive(parse_btw, "/btw needs a question: /btw <question>", _btw),
    _Directive(parse_task, "/task needs the work: /task <text>", _task),
    _Directive(
        parse_standing,
        "/standing needs the goal: /standing <text> (the task pane shows the current one)",
        _standing,
    ),
    _Directive(
        parse_retire,
        "/retire needs the task: /retire <task id> (the short id /tasks prints)",
        _retire,
    ),
    _Directive(parse_compact, "", _compact),
)


def submit_composer_line(session_dir: Path, text: str, *, now: bool = False) -> tuple[bool, str]:
    """Act on *text* and report `(did_it, what_to_say)`: a directive if it is
    one, else the steer it is. *now* forces the urgency `/now` spells, for the
    flag that says the same thing (`agent6 steer --now`)."""
    handled = act_on_directive(session_dir, text)
    if handled is not None:
        return handled
    urgent = parse_now(text)
    if urgent == "":
        return False, "/now needs the instruction: /now <text>"
    now = now or urgent is not None
    if not submit_steer(session_dir, urgent or text, now=now):
        return False, "could not write the steer request"
    return True, "steering now, interrupting the call in flight" if now else "steering"


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
