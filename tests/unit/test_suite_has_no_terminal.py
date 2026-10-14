# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The suite runs detached from any controlling terminal, as CI and the gate unit do."""

from __future__ import annotations

from agent6 import portable


def test_the_suite_has_no_controlling_terminal() -> None:
    """A test that reaches a `/dev/tty` prompt takes the headless path, never the keyboard."""
    assert not portable.has_controlling_tty()
