# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A degraded jail says so; a working one stays quiet.

The launcher prints its own diagnostics on stderr -- a mount it could not
make, a grant or protect_path it had to skip -- while the child's stderr comes
back inside the result JSON. Those diagnostics were read only when the
LAUNCHER itself failed, so every one from a working run was dropped. Measured
in rootless podman, where the fresh /proc mount is refused: the caller saw
`cannot open shared object file` and nothing naming the jail.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent6.config import Config
from agent6.sandbox.jail import (
    _with_launcher_warnings,  # pyright: ignore[reportPrivateUsage]
    run_in_jail,
)
from agent6.tools.policy import jail_policy
from agent6.types import CommandResult

WARNING = "[agent6-jail] warning: fresh /proc mount failed (EPERM: Operation not permitted)"


def _result(stderr: str = "") -> CommandResult:
    return CommandResult(
        argv=("/bin/true",), returncode=0, stdout="", stderr=stderr, duration_s=0.0
    )


def test_a_launcher_warning_reaches_the_caller_beside_the_child_output() -> None:
    got = _with_launcher_warnings(_result("cannot open shared object file"), f"{WARNING}\n")
    assert "cannot open shared object file" in got.stderr
    assert WARNING in got.stderr, "the reason was dropped, leaving only the symptom"


def test_a_quiet_launcher_adds_nothing() -> None:
    """The warnings are rare; a normal run must not grow a blank line or a
    stray newline that a caller would render."""
    assert _with_launcher_warnings(_result("boom"), "").stderr == "boom"
    assert _with_launcher_warnings(_result("boom"), "  \n ").stderr == "boom"
    assert _with_launcher_warnings(_result(), "").stderr == ""


def test_a_real_jailed_command_carries_no_launcher_noise() -> None:
    """End to end on this host, where the jail sets up cleanly: the fix must
    not put internals into every command's stderr."""
    result = run_in_jail(
        jail_policy(
            Path(tempfile.mkdtemp()), Config(), "strict", ("/bin/echo", "hi"), network="none"
        )
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "hi"
    assert result.stderr == "", f"the launcher leaked diagnostics into a clean run: {result.stderr}"
