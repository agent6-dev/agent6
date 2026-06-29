# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Find the agent6 executable and spawn it detached.

Shared by the hub (new work / merge) and the machines page (run / create) so a
TUI action shells out to the same CLI a user would run, never doing the work
in-process. A leaf module (no other ui imports) so both pages depend on it
without a cycle."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def agent6_exe() -> str:
    """The agent6 executable that launched this TUI (so a spawned child uses the
    same install), falling back to the entry on PATH."""
    argv0 = Path(sys.argv[0])
    if argv0.name.startswith("agent6") and argv0.exists():
        return str(argv0.resolve())
    return shutil.which("agent6") or "agent6"


def spawn_detached(argv: list[str], cwd: Path) -> str:
    """Spawn *argv* detached (non-TTY stdout, new session). Returns "" on a clean
    launch or an error string. Detached + non-TTY so the child never opens its own
    TUI and the launcher does not block on it."""
    try:
        subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return f"failed to start {argv[0]}: {exc}"
    return ""
