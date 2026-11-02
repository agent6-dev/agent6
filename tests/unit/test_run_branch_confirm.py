# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Starting a run while on another run's branch (agent6/<id>) confirms first."""

from __future__ import annotations

import sys
from unittest import mock

from agent6.ui.cli import preflight_prompts


def test_non_interactive_warns_and_proceeds() -> None:
    # A detached TUI/web run has no terminal to prompt, so it proceeds.
    with mock.patch.object(sys.stdin, "isatty", return_value=False):
        assert preflight_prompts.confirm_run_on_run_branch("agent6/fix-bug-ABC") is True


def test_interactive_yes_proceeds() -> None:
    with (
        mock.patch.object(sys.stdin, "isatty", return_value=True),
        mock.patch("builtins.input", return_value="y"),
    ):
        assert preflight_prompts.confirm_run_on_run_branch("agent6/fix-bug-ABC") is True


def test_interactive_default_declines() -> None:
    # Blank (the [y/N] default) aborts, so a forgotten merge doesn't pile runs up.
    with (
        mock.patch.object(sys.stdin, "isatty", return_value=True),
        mock.patch("builtins.input", return_value=""),
    ):
        assert preflight_prompts.confirm_run_on_run_branch("agent6/fix-bug-ABC") is False


def test_interactive_eof_declines() -> None:
    with (
        mock.patch.object(sys.stdin, "isatty", return_value=True),
        mock.patch("builtins.input", side_effect=EOFError),
    ):
        assert preflight_prompts.confirm_run_on_run_branch("agent6/fix-bug-ABC") is False
