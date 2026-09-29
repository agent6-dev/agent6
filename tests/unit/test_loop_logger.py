# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The headless loop logger flushes each line; the live one drops narration.

A run whose stdout is a pipe is block-buffered, so without the flush the whole trace lands only at
exit and the log reads as a dead run.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._live import loop_logger


def _drive(mode: str, stream: str) -> bool:
    """Run the headless logger in a child and return whether it was alive when its line arrived.

    The child logs then sleeps; a flushing logger delivers mid-sleep, a block-buffered one only at
    exit. Structural, not wall-clock: a timing threshold flaked under load.
    """
    code = (
        "import time\n"
        "from agent6.ui.cli._live import loop_logger\n"
        f"lg = loop_logger({mode!r}, None)\n"
        "lg('[agent6] LOOP: LOAD_CONTEXT')\n"
        "time.sleep(3)\n"
    )
    pipe = subprocess.PIPE
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=pipe if stream == "stdout" else subprocess.DEVNULL,
        stderr=pipe if stream == "stderr" else subprocess.DEVNULL,
        text=True,
    )
    fh = proc.stdout if stream == "stdout" else proc.stderr
    assert fh is not None
    line = fh.readline()  # blocks until a line is flushed to the pipe
    alive_at_line = proc.poll() is None
    proc.kill()
    proc.wait()
    assert "LOAD_CONTEXT" in line, line
    return alive_at_line


def test_headless_run_logger_flushes_each_line() -> None:
    # run mode logs to stdout; the line must arrive while the child still runs.
    assert _drive("run", "stdout")


def test_ask_logger_flushes_each_line() -> None:
    # ask keeps stdout for the answer and logs to stderr; that must flush too.
    assert _drive("ask", "stderr")


def test_live_console_drops_the_loop_narration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live console skips the loop's state narration and duplicate tool_error lines.

    Genuine notices pass; `AGENT6_DEBUG=1` shows everything.
    """
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    out = io.StringIO()
    log = loop_logger("run", ConsoleView(out, color=False))
    log("[agent6] LOOP: LOAD_CONTEXT")
    log("compaction: dropped 3 old tool results")
    log("compaction thresholds: drop at 471,859 chars, summarise at 983,040 [adaptive]")
    log("[agent6]   tool_error: apply_edit: old_string not found in calc.py\n<<<ON_DISK\n...")
    log("[agent6]   auto-commit: 1d44ec667018")
    log("[agent6]   final checkpoint: 1d44ec667018")
    log("[agent6] STEER: operator steering at iter 5")
    log("[agent6]   injecting steering instruction (41 chars)")
    log("[agent6]   ask answered at iter 2")
    log("[agent6] LOOP: verify adopted from verify.sh: ./verify.sh".replace("LOOP: ", ""))
    assert out.getvalue().strip() == "[agent6] verify adopted from verify.sh: ./verify.sh"
    monkeypatch.setenv("AGENT6_DEBUG", "1")
    out = io.StringIO()
    log = loop_logger("run", ConsoleView(out, color=False))
    log("compaction thresholds: drop at 1 chars, summarise at 2 [fixed]")
    assert "compaction thresholds" in out.getvalue()
