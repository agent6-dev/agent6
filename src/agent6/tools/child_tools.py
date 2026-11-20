# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run's children as the tools reach them: read one, wait for one, steer one, stop one.

A child is background work of its parent, named by its session id beside a command's
`bg<N>`. The loop owns the children and hands the tools this view of them.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Protocol

from agent6.sessions import layout as sessions_layout
from agent6.tools import results, schema, sessions, waits

# The characters of a child's transcript a read renders before it keeps the last lines.
_TRANSCRIPT_CHARS = 20_000


@dataclasses.dataclass(frozen=True, slots=True)
class ChildLine:
    """One child of the run, as the background roster shows it.

    Attributes:
        id: The child's session id, the handle `parallel` returned.
        attempt: Its number within this task.
        state: Where the child stands, including its review, ranking or recorded result.
        task: The first line of its task.
        running: Whether a wait has something to wait for: the child, or the judges ranking it.
        mergeable: Whether it is stopped with unmerged work.
        commits: Its measured commits beyond start, None when unknown.
    """

    id: str
    attempt: int
    state: str
    task: str
    running: bool
    mergeable: bool = False
    commits: int | None = None

    def line(self) -> str:
        """Return the roster line."""
        count = (
            ""
            if self.commits is None
            else (f" · {self.commits} commit{'' if self.commits == 1 else 's'} beyond start")
        )
        return f"[{self.id}] attempt {self.attempt} · {self.state}{count}: {self.task}"


class Children(Protocol):
    """The children of one run, as its loop hands them to the tools."""

    def start(self, task: str, spec: str) -> results.StartedResult:
        """Start one task's children and return their ids at once."""
        ...

    def roster(self) -> tuple[ChildLine, ...]:
        """Return every child the run's tasks started, in the order started."""
        ...

    def wait(self, child: str, wait_s: float, interrupted: Callable[[], bool]) -> None:
        """Wait up to `wait_s` for the child to end; with "", for the next one to end."""
        ...

    def stop(self, child: str) -> None:
        """Ask a running child to stop now; an ended one is left as it is."""
        ...

    def steer(self, child: str, text: str) -> str:
        """Steer a running child or continue a stopped child; a refusal raises."""
        ...

    def merge(self, child: str) -> str:
        """Request a stopped child's merge at the next boundary; a refusal raises."""
        ...


def addressed(args: schema.ReadBackgroundInput, roster: tuple[ChildLine, ...]) -> bool:
    """Return whether a read is of the children: a child's id, or a wait for the next to end.

    Args:
        args: The read's arguments.
        roster: The run's children.

    Returns:
        True for a child's id, and for no id with a wait while a child runs.
    """
    if args.id:
        return any(child.id == args.id for child in roster)
    return bool(args.wait_s) and any(child.running for child in roster)


def read(
    args: schema.ReadBackgroundInput,
    children: Children,
    *,
    shells: tuple[str, ...],
    wait_s: float,
    interrupted: Callable[[], bool],
    state_dir: pathlib.Path | None,
) -> results.BackgroundResult:
    """Read one child after waiting for it, or wait for the next child to end.

    Args:
        args: The read's arguments, which `addressed` took for the children's.
        children: The run's children.
        shells: The command roster for a read without an id.
        wait_s: The resolved maximum seconds to wait for the child to end.
        interrupted: Cuts the wait short.
        state_dir: The project's state dir, where a child's journal is; None renders none.

    Returns:
        Both rosters without an id. With an id, only that child's row and transcript tail.
    """
    with waits.during(wait_s):
        children.wait(args.id, wait_s, interrupted)
    roster = children.roster()
    if not args.id:
        running = any(child.running for child in roster)
        lines = tuple(child.line() for child in roster)
        return results.BackgroundResult(shells=shells, children=lines, running=running)
    child = next(child for child in roster if child.id == args.id)
    return results.BackgroundResult(
        shells=shells,
        children=(child.line(),),
        output=f"{child.line()}\n\n{_transcript(state_dir, child.id, args.tail_lines)}",
        running=child.running,
    )


def _transcript(state_dir: pathlib.Path | None, child: str, tail_lines: int) -> str:
    """Return the last lines of a child's conversation, as `read_session` renders one.

    Args:
        state_dir: The project's state dir.
        child: The child's session id.
        tail_lines: The most lines returned.

    Returns:
        The tail, or a note when the child has no journal yet.
    """
    found = sessions_layout.session_layout(state_dir, child) if state_dir is not None else None
    if found is None:
        return "(no transcript yet)"
    lines = sessions.conversation(found, max_chars=_TRANSCRIPT_CHARS).splitlines()
    if len(lines) > tail_lines:
        lines = [f"... {len(lines) - tail_lines} earlier lines ...", *lines[-tail_lines:]]
    return "\n".join(lines)
