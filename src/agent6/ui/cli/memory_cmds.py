# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory add/list/invalidate` commands."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.memory import (
    MemoryError as Agent6MemoryError,
)
from agent6.memory import (
    MemoryScope,
)
from agent6.memory import (
    add as memory_add,
)
from agent6.memory import (
    invalidate as memory_invalidate,
)
from agent6.memory import (
    list_entries as memory_list,
)
from agent6.ui.cli._common import _state_dir


def _cmd_memory_add(scope: MemoryScope, body: str) -> int:
    try:
        entry = memory_add(_state_dir(Path.cwd()), scope, body)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{entry.scope} {entry.id} created at {entry.created_at}")
    return 0


def _cmd_memory_list(scope: MemoryScope | None, *, include_invalidated: bool) -> int:
    try:
        entries = memory_list(_state_dir(Path.cwd()), scope)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print("(no memories)")
        return 0
    for e in entries:
        if not include_invalidated and not e.is_active:
            continue
        flag = "" if e.is_active else " [INVALIDATED]"
        print(f"[{e.scope}] {e.id} {e.created_at}{flag}")
        if not e.is_active and e.invalidation_reason:
            print(f"    invalidated_at: {e.invalidated_at}  reason: {e.invalidation_reason}")
        for line in e.body.splitlines():
            print(f"    {line}")
        print()
    return 0


def _cmd_memory_invalidate(memory_id: str, reason: str) -> int:
    try:
        entry = memory_invalidate(_state_dir(Path.cwd()), memory_id, reason)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"invalidated {entry.scope} {entry.id} at {entry.invalidated_at}")
    return 0
