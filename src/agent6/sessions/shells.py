# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run's background command records, as the surfaces read them off disk.

`tools/background.py` writes `<session dir>/shells/bg<N>/meta.json` at start; the launcher
writes `result.json` beside it when the command ends. A surface in another process reads
what each command was and how it ended, and says when it cannot tell.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import re

from agent6 import path_safety
from agent6.sessions import process

SHELLS_DIR = "shells"  # under the session dir
META_NAME = "meta.json"
RESULT_NAME = "result.json"  # the launcher's outcome record (`sandbox/jail_jobs.py`)


def seq_of(name: str) -> int:
    """Return the N in a `bg<N>` directory name, 0 for any other name."""
    return int(name[2:]) if name.startswith("bg") and name[2:].isdigit() else 0


def shells_text(session_dir: pathlib.Path) -> str:
    """Return the roster as one block for a text view, or a line saying there is none."""
    return "\n".join(roster_from_dir(session_dir / SHELLS_DIR)) or "no background commands this run"


def shell_host_processes(root: pathlib.Path) -> list[process.ProcessIdentity]:
    """Return the live host processes the run's background commands recorded.

    Only historical host-process records carry these identities. Supervisor-owned
    jobs record no command pid. A stale record cannot target a recycled pid.

    Args:
        root: The run's shells dir.

    Returns:
        Each live process's pid and start time.
    """
    if not root.is_dir():
        return []
    processes: list[process.ProcessIdentity] = []
    try:
        directories = sorted(root.iterdir())
    except OSError:
        return []
    for d in directories:
        try:
            meta = json.loads((d / META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = meta.get("pid") if isinstance(meta, dict) else None
        started = meta.get("pid_start") if isinstance(meta, dict) else None
        if type(pid) is int and pid > 0 and isinstance(started, str):
            identity = (pid, started)
            if process.process_is_alive(identity):
                processes.append(identity)
    return processes


def result_record(root: pathlib.Path, shell_id: str) -> dict[str, object] | None:
    """Read a background command's last result without following links.

    Args:
        root: The run's shells directory.
        shell_id: The recorded job id.

    Returns:
        The result fields, an empty record for an unreadable ending, or None without an ending.
    """
    if not re.fullmatch(r"bg[0-9]+", shell_id):
        return None
    try:
        entry = path_safety.contain(root.parent, pathlib.Path(root.name) / shell_id / RESULT_NAME)
        raw = path_safety.read_contained(entry, errors="replace", limit_chars=65537)
    except (OSError, path_safety.ContainmentError):
        return None
    if not raw.strip():
        return None
    if len(raw) <= 65536:
        with contextlib.suppress(ValueError, IndexError):
            value = json.loads(raw.rstrip().splitlines()[-1])
            if isinstance(value, dict):
                return value
    return {}


def roster_from_dir(root: pathlib.Path) -> list[str]:
    """Return the run's background commands as lines, read off disk.

    For surfaces in another process: liveness needs the owning process, so this reports what
    each command was and how it ended, and says when it cannot tell.

    Args:
        root: The run's shells dir.

    Returns:
        One line per recorded command, in `bg<N>` order.
    """
    if not root.is_dir():
        return []
    lines: list[str] = []
    # By sequence: as text, bg10 sorts ahead of bg2.
    try:
        directories = sorted(root.iterdir(), key=lambda p: (seq_of(p.name), p.name))
    except OSError:
        return []
    for d in directories:
        if not d.is_dir():
            continue
        try:
            meta = json.loads((d / META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        command = str(meta.get("command", ""))
        record = result_record(root, d.name)
        if record is None:
            lines.append(f"[{d.name}] still running (or the run that owns it ended): {command}")
            continue
        errors = "; ".join(
            dict.fromkeys(str(record[key]) for key in ("error", "capture_error") if record.get(key))
        )
        code = record.get("returncode")
        if isinstance(code, int):
            detail = f" -- {errors}" if errors else ""
            lines.append(f"[{d.name}] exited {code}: {command}{detail}")
        elif record.get("stopped"):
            # Historical records can carry a stop without an observed exit code.
            lines.append(f"[{d.name}] stopped: {command}")
        else:
            ending = f"outcome unknown ({errors})" if errors else "ended without a result"
            lines.append(f"[{d.name}] {ending}: {command}")
    return lines
