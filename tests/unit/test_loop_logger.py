# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The headless loop logger must FLUSH each line.

A `nohup agent6 run > log` (or any run whose stdout/stderr is a pipe, not a TTY)
is block-buffered: without an explicit flush the whole LOOP trace only lands
when the process exits, so the log reads as a dead run for its entire duration.
This drives the logger in a subprocess whose stdout is a pipe and asserts the
line arrives BEFORE the process ends.
"""

from __future__ import annotations

import subprocess
import sys


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
