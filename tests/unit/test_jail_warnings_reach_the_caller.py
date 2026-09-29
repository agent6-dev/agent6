# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A degraded jail says so; a working one stays quiet.

The launcher's own stderr diagnostics (a refused mount, a skipped grant) reach the caller on success
too; measured in rootless podman, where the fresh /proc mount is refused.
"""

from __future__ import annotations

import pathlib
import tempfile

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import jail
from agent6.tools import policy

WARNING = "[agent6-jail] warning: fresh /proc mount failed (EPERM: Operation not permitted)"


def _result(stderr: str = "") -> kinds.CommandResult:
    return kinds.CommandResult(
        argv=("/bin/true",), returncode=0, stdout="", stderr=stderr, duration_s=0.0
    )


def test_a_launcher_warning_reaches_the_caller_beside_the_child_output() -> None:
    got = jail._with_launcher_warnings(_result("cannot open shared object file"), f"{WARNING}\n")
    assert "cannot open shared object file" in got.stderr
    assert WARNING in got.stderr, "the reason was dropped, leaving only the symptom"


def test_a_quiet_launcher_adds_nothing() -> None:
    """A normal run grows no blank line or stray newline a caller would render."""
    assert jail._with_launcher_warnings(_result("boom"), "").stderr == "boom"
    assert jail._with_launcher_warnings(_result("boom"), "  \n ").stderr == "boom"
    assert jail._with_launcher_warnings(_result(), "").stderr == ""


def test_a_real_jailed_command_carries_no_launcher_noise() -> None:
    """End to end on this host, where the jail sets up cleanly, a command's stderr stays clean."""
    result = jail.run_in_jail(
        policy.jail_policy(
            pathlib.Path(tempfile.mkdtemp()),
            Config(),
            "strict",
            ("/bin/echo", "hi"),
            network="none",
        )
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "hi"
    assert result.stderr == "", f"the launcher leaked diagnostics into a clean run: {result.stderr}"
