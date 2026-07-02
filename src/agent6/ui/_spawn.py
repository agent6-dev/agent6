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
import tempfile
import time
from collections.abc import Callable
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


def _located(list_dirs: Callable[[], list[Path]], before: set[Path]) -> Path | None:
    """The newest dir from *list_dirs* not in *before* whose logs.jsonl exists."""
    for d in list_dirs():
        if d not in before and (d / "logs.jsonl").exists():
            return d
    return None


def spawn_and_locate(
    argv: list[str],
    cwd: Path,
    *,
    before: set[Path],
    list_dirs: Callable[[], list[Path]],
    env: dict[str, str] | None = None,
    timeout_s: float = 25.0,
) -> tuple[Path | None, str]:
    """Spawn *argv* detached, then poll *list_dirs* for a NEW dir (not in *before*)
    whose ``logs.jsonl`` exists, and return ``(dir, "")`` so the caller can hand it
    to the dashboard. If the child exits before producing one (no git repo, bad
    config, ...), surface its stderr tail instead of waiting out the timeout;
    return ``(None, message)`` on any failure.

    The shared launch+watch path behind both "start a run" (hub) and "create a
    machine" (machines page): spawn the same CLI a user would, then watch the new
    log dir live."""
    label = argv[1] if len(argv) > 1 else argv[0]
    err = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in finally
        mode="w+", suffix=".agent6-launch.err", delete=False
    )
    try:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            return None, f"failed to start agent6 {label}: {exc}"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            found = _located(list_dirs, before)
            if found is not None:
                return found, ""
            if proc.poll() is not None:
                # Child exited without a log dir; surface why (recheck once in case
                # the dir landed in the same instant the process exited).
                found = _located(list_dirs, before)
                if found is not None:
                    return found, ""
                err.flush()
                tail = Path(err.name).read_text(encoding="utf-8", errors="replace")[-600:]
                return None, f"agent6 {label} exited ({proc.returncode}) before starting:\n{tail}"
            time.sleep(0.2)
        return None, f"timed out waiting for `agent6 {label}` to start"
    finally:
        err.close()
        Path(err.name).unlink(missing_ok=True)
