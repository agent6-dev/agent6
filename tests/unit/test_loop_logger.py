# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The headless loop logger must FLUSH each line; the live one drops narration.

A `nohup agent6 run > log` (or any run whose stdout/stderr is a pipe, not a TTY)
is block-buffered: without an explicit flush the whole LOOP trace only lands
when the process exits, so the log reads as a dead run for its entire duration.
This drives the logger in a subprocess whose stdout is a pipe and asserts the
line arrives BEFORE the process ends.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._live import loop_logger


def _drive(mode: str, stream: str) -> bool:
    """Run the *mode* headless logger in a child that logs then sleeps 3s, with
    the given std *stream* piped. Return whether the child was STILL ALIVE when
    its line arrived: a flushing logger delivers mid-sleep; a block-buffered one
    delivers only at exit. Structural, not wall-clock -- a timing threshold
    here flaked under machine load, where the child's cold import alone
    outspent the budget."""
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
    """On the live console the loop's state narration (LOOP: transitions, a
    compaction, the thresholds compaction will fire at) is noise between the
    glyphs; a tool_error line repeats the error the stream shows under its red
    glyph, an auto-commit line the sha on the ✎ item, and the STEER pair the
    operator item; genuine notices pass. `AGENT6_DEBUG=1` shows everything."""
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
    log("[agent6] LOOP: verify adopted from verify.sh: ./verify.sh".replace("LOOP: ", ""))
    assert out.getvalue().strip() == "[agent6] verify adopted from verify.sh: ./verify.sh"
    monkeypatch.setenv("AGENT6_DEBUG", "1")
    out = io.StringIO()
    log = loop_logger("run", ConsoleView(out, color=False))
    log("compaction thresholds: drop at 1 chars, summarise at 2 [fixed]")
    assert "compaction thresholds" in out.getvalue()
