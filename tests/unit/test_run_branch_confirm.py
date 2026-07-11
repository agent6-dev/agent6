# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Starting a run while on another run's branch (agent6/<id>) confirms first."""

from __future__ import annotations

import sys
from unittest.mock import patch

from agent6.ui.cli._preflight import confirm_run_on_run_branch


def test_non_interactive_warns_and_proceeds() -> None:
    # A detached TUI/web run has no terminal to prompt, so it proceeds.
    with patch.object(sys.stdin, "isatty", return_value=False):
        assert confirm_run_on_run_branch("agent6/fix-bug-ABC") is True


def test_interactive_yes_proceeds() -> None:
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        assert confirm_run_on_run_branch("agent6/fix-bug-ABC") is True


def test_interactive_default_declines() -> None:
    # Blank (the [y/N] default) aborts, so a forgotten merge doesn't pile runs up.
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch("builtins.input", return_value=""),
    ):
        assert confirm_run_on_run_branch("agent6/fix-bug-ABC") is False


def test_interactive_eof_declines() -> None:
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch("builtins.input", side_effect=EOFError),
    ):
        assert confirm_run_on_run_branch("agent6/fix-bug-ABC") is False
