# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One effective command policy, from three inputs, read the same way everywhere."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.runs.ipc import (
    effective_run_commands,
    set_away_mode,
    set_session_allow,
    set_session_deny,
)
from agent6.tools.dispatch import ToolDispatcher

_COMMAND_TOOLS = {"run_command", "run_verify_command", "run_background", "stop_background"}


@pytest.mark.parametrize("configured", ["yes", "no"])
def test_a_standing_policy_is_not_movable_in_run(tmp_path: Path, configured: str) -> None:
    """Only "ask" is a question. A configured yes or no is the operator's
    standing policy, and no in-run choice overrides it."""
    set_session_allow(tmp_path)
    set_session_deny(tmp_path)
    set_away_mode(tmp_path, "deny")
    assert effective_run_commands(configured, tmp_path) == configured


def test_ask_is_what_the_session_choice_moves(tmp_path: Path) -> None:
    assert effective_run_commands("ask", tmp_path) == "ask"
    set_session_allow(tmp_path)
    assert effective_run_commands("ask", tmp_path) == "yes"


def test_deny_for_the_session_is_the_mirror_of_allow(tmp_path: Path) -> None:
    """A single no answers one call, exactly as a single yes approves one; only
    the session choices persist, and denying withdraws rather than refuses."""
    set_session_deny(tmp_path)
    assert effective_run_commands("ask", tmp_path) == "no"


def test_an_away_mode_of_deny_withdraws_the_tools(tmp_path: Path) -> None:
    """Same wiring: "deny while away" and "deny for the session" and
    `run_commands = "no"` all mean the tools are gone, not refused per call."""
    set_away_mode(tmp_path, "deny")
    assert effective_run_commands("ask", tmp_path) == "no"


def test_waiting_is_still_a_question(tmp_path: Path) -> None:
    set_away_mode(tmp_path, "wait")
    assert effective_run_commands("ask", tmp_path) == "ask"


def test_withdrawn_tools_leave_the_model_s_surface(tmp_path: Path) -> None:
    """The point of withdrawing rather than refusing: the model never sees a
    door it cannot open, so it stops spending turns on one."""
    # A gate must be configured, or run_verify_command is hidden for its own
    # reason (a gateless run is not offered a tool that would only error).
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "workflow": {"verify_command": ["true"]}}
    )
    d = ToolDispatcher(root=tmp_path, config=cfg, run_dir=tmp_path)
    assert set(d.available_tool_names()) >= _COMMAND_TOOLS
    set_session_deny(tmp_path)
    assert _COMMAND_TOOLS.isdisjoint(d.available_tool_names())


def test_the_policy_is_re_read_not_cached(tmp_path: Path) -> None:
    """An operator who allows for the session stops being prompted from the
    next call, without restarting anything."""
    cfg = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    d = ToolDispatcher(root=tmp_path, config=cfg, run_dir=tmp_path)
    assert d.command_policy() == "ask"
    set_session_allow(tmp_path)
    assert d.command_policy() == "yes"


@pytest.mark.parametrize(
    ("commands", "refused"),
    [("ask", True), ("yes", False), ("no", False)],
)
def test_parallel_makes_the_operator_decide_once(commands: str, refused: bool) -> None:
    """ "Wait for someone to approve" is incoherent across detached lanes: it
    would mean attaching a front-end to each in turn, which is most of what
    running them in parallel was for. So `ask` refuses at launch and names the
    two coherent choices."""
    from agent6.ui.cli.parallel import (
        _parallel_approval_refusal,  # pyright: ignore[reportPrivateUsage]
    )

    cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
    err = _parallel_approval_refusal(cfg)
    assert (err is not None) is refused
    if err is not None:
        assert "--auto-approve" in err and "--no-commands" in err
