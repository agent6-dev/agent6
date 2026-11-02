# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A jailed command allocates pseudo-terminals from the jail's own devpts, never the host's."""

from __future__ import annotations

import pathlib

import pytest

from agent6.config import Config
from agent6.sandbox import jail
from agent6.tools import policy

pytestmark = pytest.mark.needs_namespaces

_OPENPTY = (
    "import os, pty\n"
    "before = sorted(os.listdir('/dev/pts'))\n"
    "master, slave = pty.openpty()\n"
    "print(before, os.ttyname(slave), os.readlink('/dev/ptmx'))\n"
)
_OPEN_PTMX = (
    "import os\n"
    "try:\n"
    "    os.open('/dev/ptmx', os.O_RDWR | os.O_NOCTTY)\n"
    "except OSError as exc:\n"
    "    print('refused', exc.errno)\n"
    "else:\n"
    "    print('opened')\n"
)


def _run(tmp_path: pathlib.Path, level: str, script: str) -> str:
    result = jail.run_in_jail(
        policy.jail_policy(
            tmp_path,
            Config(),
            level,  # pyright: ignore[reportArgumentType]
            ("python3", "-c", script),
            network="none",
        )
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_full_allocates_a_pty_from_a_private_devpts(tmp_path: pathlib.Path) -> None:
    """Under `full`, openpty works, /dev/ptmx points into the instance, no host pty is listed."""
    out = _run(tmp_path, "full", _OPENPTY)
    assert out.startswith("['ptmx'] /dev/pts/"), out
    assert out.rstrip().endswith("pts/ptmx"), out


def test_partial_keeps_the_hosts_terminals_unreachable(tmp_path: pathlib.Path) -> None:
    """Under `partial`, /dev/ptmx stays refused: the host's /dev/pts holds the operator's ttys."""
    assert _run(tmp_path, "partial", _OPEN_PTMX).startswith("refused")
