# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory add/list/show/rm/decisions` commands.

Store refusals (a bad name, an unreadable store) raise MemoryStoreError, an
OperatorError the cli_main boundary presents; no per-command arms.
"""

from __future__ import annotations

from pathlib import Path

from agent6.memory import (
    MemoryUse,
    add,
    decisions_path,
    index_name,
    index_text,
    memory_dir,
    read_use,
    remove,
    show,
    unindexed_names,
)
from agent6.paths import state_dir


def _cmd_memory_add(name: str, body: str) -> int:
    path = add(state_dir(Path.cwd()), name, body)
    print(f"wrote {path}")
    return 0


def _cmd_memory_list() -> int:
    """The whole index (a run's prompt gets it clipped), each entry followed
    by its use record (who wrote it, who read it) when the harness has one,
    then the files the index does not list."""
    state = state_dir(Path.cwd())
    text = index_text(state)
    orphans = unindexed_names(state)
    if not text:
        print(f"(no memories; files live under {memory_dir(state)})")
    use = read_use(state)
    for line in text.splitlines():
        print(line)
        name = index_name(line)
        if name is not None and name in use:
            print(f"    {format_use(use[name])}")
    if orphans:
        print(f"not in the index (no run sees them; `memory rm` deletes): {', '.join(orphans)}")
    return 0


def format_use(use: MemoryUse) -> str:
    """One line: `written <date> by <session>[, edited <date> by <session>],
    read once|N times, last <date> by <session>` or `never read`; a part the
    record does not hold is left out."""
    parts: list[str] = []
    if use.created_by:
        parts.append(f"written {use.created_at[:10]} by {use.created_by}")
        if use.updated_by and (use.updated_by, use.updated_at) != (use.created_by, use.created_at):
            parts.append(f"edited {use.updated_at[:10]} by {use.updated_by}")
    if use.reads:
        times = "once" if use.reads == 1 else f"{use.reads} times"
        parts.append(f"read {times}, last {use.read_at[:10]} by {use.read_by}")
    else:
        parts.append("never read")
    return ", ".join(parts)


def _cmd_memory_show(name: str) -> int:
    print(show(state_dir(Path.cwd()), name), end="")
    return 0


def _cmd_memory_decisions() -> int:
    state = state_dir(Path.cwd())
    path = decisions_path(state)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        print(f"(no rulings recorded; the harness writes them to {path})")
        return 0
    print(text, end="")
    return 0


def _cmd_memory_rm(name: str) -> int:
    remove(state_dir(Path.cwd()), name)
    print(f"removed {name}")
    return 0
