# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The test session's detach from its controlling terminal."""

from __future__ import annotations

import os
import pty

from tests import conftest


def test_the_tty_detach_spares_a_session_leader() -> None:
    """A pytest that leads its own session on a pty keeps the terminal and lives.

    `TIOCNOTTY` from the session leader hung up the terminal's foreground group, the leader
    itself.
    """
    pid, master = pty.fork()
    if pid == 0:  # the child leads a new session whose controlling terminal is the pty
        try:
            conftest.detach_controlling_tty()
        finally:
            os._exit(0)
    try:
        _, status = os.waitpid(pid, 0)
    finally:
        os.close(master)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
