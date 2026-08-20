# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory add/list/show/rm` commands.

Store refusals (a bad name, an unreadable store) raise MemoryStoreError, an
OperatorError the cli_main boundary presents; no per-command arms.
"""

from __future__ import annotations

from pathlib import Path

from agent6.config.layer import resolved_state_dir
from agent6.memory import add, index_text, memory_dir, remove, show


def _cmd_memory_add(name: str, body: str) -> int:
    path = add(resolved_state_dir(Path.cwd()), name, body)
    print(f"wrote {path}")
    return 0


def _cmd_memory_list() -> int:
    state = resolved_state_dir(Path.cwd())
    text = index_text(state)
    if not text:
        print(f"(no memories; files live under {memory_dir(state)})")
        return 0
    print(text)
    return 0


def _cmd_memory_show(name: str) -> int:
    print(show(resolved_state_dir(Path.cwd()), name), end="")
    return 0


def _cmd_memory_rm(name: str) -> int:
    remove(resolved_state_dir(Path.cwd()), name)
    print(f"removed {name}")
    return 0
