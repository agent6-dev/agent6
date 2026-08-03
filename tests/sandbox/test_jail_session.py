# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One jail process per run: its commands share the namespaces."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.sandbox.jail import JailSession
from agent6.types import JailPolicy

pytestmark = pytest.mark.needs_namespaces


def _session(cwd: Path) -> JailSession:
    return JailSession.open(JailPolicy(cwd=cwd, argv=("true",), isolation="strict", timeout_s=30.0))


def test_the_session_netns_has_loopback_up(tmp_path: Path) -> None:
    """An empty netns leaves `lo` DOWN, so nothing inside can reach even
    itself -- which is what a shared address between a run's commands needs.
    Loopback in a namespace with no other interface reaches nothing outside
    it."""
    session = _session(tmp_path)
    try:
        got = session.run(("ip", "link", "show", "lo"))
        assert got.returncode == 0, got.stderr
        assert "UP" in got.stdout, got.stdout
    finally:
        session.close()


def test_commands_in_one_session_share_a_tmp(tmp_path: Path) -> None:
    """The private /tmp is per-launcher, so per-command launchers gave every
    command a fresh one; a run's commands must see one."""
    session = _session(tmp_path)
    try:
        first = session.run(("sh", "-c", "echo shared > /tmp/marker; echo wrote"))
        assert first.returncode == 0, first.stderr
        second = session.run(("sh", "-c", "cat /tmp/marker"))
        assert second.returncode == 0, second.stderr
        assert "shared" in second.stdout
    finally:
        session.close()


def test_closing_the_session_takes_the_namespace_down(tmp_path: Path) -> None:
    """Nothing a run started outlives it: closing the request channel ends the
    PID namespace, and everything inside it goes."""
    session = _session(tmp_path)
    started = session.run(("sh", "-c", "(sleep 300 &) ; echo bg"))
    assert started.returncode == 0, started.stderr
    session.close()
    with pytest.raises(Exception):
        session.run(("true",))
