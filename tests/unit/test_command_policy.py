# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One effective command policy, from three inputs, read the same way everywhere."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sessions.ipc import (
    COMMAND_SCOPE,
    effective_run_commands,
    set_away_mode,
    set_session_allow,
    set_session_deny,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.operator_prompts import ApprovalAnswer, ApprovalRequest, OperatorPrompts

_COMMAND_TOOLS = {"run_command", "run_verify_command", "stop_background"}


@pytest.mark.parametrize("configured", ["yes", "no"])
def test_a_standing_policy_is_not_movable_in_run(tmp_path: Path, configured: str) -> None:
    """Only "ask" is a question; a configured yes or no is the operator's standing policy."""
    set_session_allow(tmp_path, COMMAND_SCOPE)
    set_session_deny(tmp_path, COMMAND_SCOPE)
    set_away_mode(tmp_path, "deny")
    assert effective_run_commands(configured, tmp_path) == configured


def test_ask_is_what_the_session_choice_moves(tmp_path: Path) -> None:
    assert effective_run_commands("ask", tmp_path) == "ask"
    set_session_allow(tmp_path, COMMAND_SCOPE)
    assert effective_run_commands("ask", tmp_path) == "yes"


def test_deny_for_the_session_is_the_mirror_of_allow(tmp_path: Path) -> None:
    """A single no answers one call as a single yes approves one.

    Only the session choices persist.
    """
    set_session_deny(tmp_path, COMMAND_SCOPE)
    assert effective_run_commands("ask", tmp_path) == "no"


def test_an_away_mode_of_deny_withdraws_the_tools(tmp_path: Path) -> None:
    """Deny while away, deny for the session and `run_commands = "no"` all withdraw the tools."""
    set_away_mode(tmp_path, "deny")
    assert effective_run_commands("ask", tmp_path) == "no"


def test_waiting_is_still_a_question(tmp_path: Path) -> None:
    set_away_mode(tmp_path, "wait")
    assert effective_run_commands("ask", tmp_path) == "ask"


def test_withdrawn_tools_leave_the_model_s_surface(tmp_path: Path) -> None:
    """Withdrawn tools leave the model's surface.

    It never spends turns on a door it cannot open.
    """
    # A gate must be configured, or run_verify_command is hidden for its own reason.
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "harness": {"verify_command": ["true"]}}
    )
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path)
    assert set(d.available_tool_names()) >= _COMMAND_TOOLS
    set_session_deny(tmp_path, COMMAND_SCOPE)
    assert _COMMAND_TOOLS.isdisjoint(d.available_tool_names())


def test_the_policy_is_re_read_not_cached(tmp_path: Path) -> None:
    """The policy is re-read on every call, so a session allow stops the prompts at once."""
    cfg = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path)
    assert d.command_policy() == "ask"
    set_session_allow(tmp_path, COMMAND_SCOPE)
    assert d.command_policy() == "yes"


@pytest.mark.parametrize(
    ("commands", "refused"),
    [("ask", True), ("yes", False), ("no", False)],
)
def test_parallel_makes_the_operator_decide_once(commands: str, refused: bool) -> None:
    """`--parallel` under `ask` refuses at launch and names the two coherent choices.

    Waiting for approval across detached lanes would mean attaching to each in turn.
    """
    from agent6.ui.cli.parallel import (
        _parallel_approval_refusal,  # pyright: ignore[reportPrivateUsage]
    )

    cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
    err = _parallel_approval_refusal(cfg)
    assert (err is not None) is refused
    if err is not None:
        assert "--auto-approve" in err and "--no-commands" in err
        # A hub relays this refusal with no flags to pass; the config remedy is what it can act on.
        assert "agent6 config set sandbox.run_commands" in err


def test_a_single_no_refuses_one_call_and_withdraws_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single "no" refuses one call and withdraws nothing."""
    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import CommandResult
    from agent6.sessions.ipc import session_deny_set

    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "harness": {"verify_command": ["true"]}}
    )
    answers = iter(["no", "no", "yes"])

    def _answer(_request: ApprovalRequest, /) -> ApprovalAnswer:
        return ApprovalAnswer(next(answers) == "yes", "stdin")

    prompts = OperatorPrompts(approver=_answer, session_dir=tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path, prompts=prompts)
    for _ in range(2):
        with pytest.raises(Exception, match="not approved"):
            d.dispatch("run_command", {"argv": ["true"]})
        assert not session_deny_set(tmp_path, COMMAND_SCOPE)
        assert d.command_policy() == "ask"
        assert set(d.available_tool_names()) >= _COMMAND_TOOLS

    # The mirror: a single "yes" runs exactly that call and widens nothing.
    def _ran(policy: JailPolicy, **_kw: object) -> CommandResult:
        return CommandResult(
            argv=tuple(policy.argv), returncode=0, stdout="", stderr="", duration_s=0.01
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", _ran)
    assert d.dispatch("run_command", {"argv": ["true"]}).to_wire()["returncode"] == 0
    assert d.command_policy() == "ask"
    assert not session_deny_set(tmp_path, COMMAND_SCOPE)


@pytest.mark.parametrize(
    ("name", "raw", "shown"),
    [
        ("run_command", {"argv": ["true"]}, "run_command: true"),
        ("run_verify_command", {}, "run_verify_command: true"),
        ("run_metric_command", {}, "run_metric_command: true"),
    ],
)
def test_every_ask_command_tool_uses_the_command_scope(
    tmp_path: Path, name: str, raw: dict[str, object], shown: str
) -> None:
    """Each command that runs something asks before acting, on the one command scope."""
    from agent6.tools.errors import ToolDeniedError

    seen: list[ApprovalRequest] = []

    def refuse(request: ApprovalRequest, /) -> ApprovalAnswer:
        seen.append(request)
        return ApprovalAnswer(False, "stdin")

    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "ask", "network": "host"},
            "harness": {
                "verify_command": ["true"],
                "metric": {"command": ["true"], "pattern": "(true)", "goal": "minimize"},
            },
        }
    )
    d = ToolDispatcher(
        root=tmp_path,
        config=cfg,
        session_dir=tmp_path,
        prompts=OperatorPrompts(approver=refuse, session_dir=tmp_path),
    )
    with pytest.raises(ToolDeniedError):
        d.dispatch(name, raw)
    assert len(seen) == 1
    assert seen[0].prompt == f"Allow {shown}"
    assert seen[0].scope == COMMAND_SCOPE


def test_a_stop_during_the_approval_wait_is_named_as_such(tmp_path: Path) -> None:
    """A stop during the approval wait is named as a stop, not as a policy refusal."""
    from agent6.sessions.ipc import request_stop

    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "harness": {"verify_command": ["true"]}}
    )

    def _wait_broken(_request: ApprovalRequest, /) -> ApprovalAnswer:
        request_stop(tmp_path)  # the stop lands while the approval waits
        return ApprovalAnswer(False, "stdin")

    prompts = OperatorPrompts(approver=_wait_broken, session_dir=tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path, prompts=prompts)
    with pytest.raises(Exception, match="asked to stop while awaiting approval"):
        d.dispatch("run_command", {"argv": ["true"]})
    with pytest.raises(Exception, match="asked to stop while awaiting approval"):
        d.dispatch("run_verify_command", {})


def test_an_interactive_start_drops_the_detach_grants_with_the_away_mode(tmp_path: Path) -> None:
    """An interactive start clears a detach's `session.allow.<scope>` markers.

    Its deny and wait siblings are cleared the same way.
    """
    from agent6.sessions.ipc import (
        clear_session_grants,
        session_allow_set,
        session_deny_set,
        set_session_allow,
        set_session_deny,
    )

    set_session_allow(tmp_path, "command")
    set_session_allow(tmp_path, "mcp.docs")
    set_session_deny(tmp_path, "mcp.web")
    clear_session_grants(tmp_path)
    assert not session_allow_set(tmp_path, "command") and not session_allow_set(
        tmp_path, "mcp.docs"
    )
    assert session_deny_set(
        tmp_path, "mcp.web"
    )  # a denial is the operator's own answer, not a detach grant
    clear_session_grants(tmp_path / "never-made")  # no approvals dir: nothing to drop
