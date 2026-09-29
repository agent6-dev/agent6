# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The first user message's operational header matches the mode's real tool surface.

A model told to edit, verify and call `finish_session` in ask mode spends a call trying to comply.
"""

from __future__ import annotations

import re

import pytest

from agent6.harness._prompt_blocks import initial_instructions
from agent6.tools.schema import mode_tools


def test_ask_gets_direct_answer_instructions() -> None:
    text = initial_instructions("ask", "ask", has_gate=True)
    assert "answer" in text.lower()
    for phantom in ("finish_session", "make edits", "run_verify_command"):
        assert phantom not in text


def test_a_no_commands_run_is_not_told_to_run_verify() -> None:
    """`run_commands = "no"` withholds the command tools and the verify gate with them."""
    assert "run_verify_command" not in initial_instructions("run", "no", has_gate=True)
    assert "finish_session" in initial_instructions("run", "no", has_gate=True)
    assert "run_verify_command" in initial_instructions("run", "ask", has_gate=True)
    assert "run_verify_command" in initial_instructions("run", "yes", has_gate=True)


def test_a_gateless_run_is_not_told_to_run_verify() -> None:
    """A gateless run's header names no verify gate, however commands are configured."""
    assert "run_verify_command" not in initial_instructions("run", "yes", has_gate=False)
    assert "finish_session" in initial_instructions("run", "yes", has_gate=False)


@pytest.mark.parametrize("mode", ["run", "plan", "ask", "agent"])
def test_no_instruction_names_a_tool_outside_the_mode_surface(mode: str) -> None:
    """Every backticked tool name in a mode's header is one that mode exposes."""
    all_tools = set().union(
        *(mode_tools(m).permitted for m in ("run", "plan", "ask", "machine", "agent"))
    )
    permitted = mode_tools(mode).permitted
    for rc in ("yes", "ask", "no"):
        named = set(re.findall(r"`([a-z_0-9.]+)`", initial_instructions(mode, rc, has_gate=True)))
        misfits = {n for n in named if n in all_tools and n not in permitted}
        assert not misfits, f"{mode} header names tools outside its surface: {sorted(misfits)}"
