# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The approval gate on MCP tool calls, and the scope it grants.

A server's tools are asked about on their own scope; an "allow all" for one server is never consent
for the command tools or a sibling server. The in-tree fake server makes the call really reach a
process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.ipc import COMMAND_SCOPE, set_session_allow, set_session_deny
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolDeniedError, ToolError
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.operator_prompts import ApprovalAnswer, ApprovalRequest, OperatorPrompts
from tests.unit.test_mcp_client import _fake_server_argv  # pyright: ignore[reportPrivateUsage]


def _manager() -> MCPManager:
    return MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ]
    )


def _recording(asked: list[tuple[str, str | None]]) -> OperatorPrompts:
    def approve(request: ApprovalRequest, /) -> ApprovalAnswer:
        asked.append((request.prompt, request.scope))
        return ApprovalAnswer(True, "stdin")

    return OperatorPrompts(approver=approve)


def _deny(_request: ApprovalRequest, /) -> ApprovalAnswer:
    return ApprovalAnswer(False, "stdin")


def _cli_prompts(session_dir: Path) -> OperatorPrompts:
    """The gate over the CLI's own approver, journaling into *session_dir*."""
    from agent6.ui.cli._interact import build_approver

    return OperatorPrompts(
        approver=build_approver(session_dir),
        journal=EventSink(session_dir / "logs.jsonl").emit,
        session_dir=session_dir,
    )


def _cfg(**server: object) -> Config:
    return Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"fake": {"command": ["true"], **server}}}}
    )


def test_a_tool_call_is_asked_about_before_the_server_sees_it(tmp_path: Path) -> None:
    """`approve = "ask"` is the default, so a fresh server prompts, with the arguments."""
    asked: list[tuple[str, str | None]] = []
    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            prompts=_recording(asked),
        )
        d.dispatch("mcp__fake__echo", {"text": "hello"})
    finally:
        mgr.close()
    assert len(asked) == 1
    prompt, scope = asked[0]
    assert prompt == 'Allow mcp__fake__echo: {"text": "hello"}'
    assert scope == "mcp.fake"


def test_the_prompt_carries_the_arguments_in_full(tmp_path: Path) -> None:
    """The consent line carries the arguments whole: they are the only part the model controls."""
    seen: list[str] = []

    def _capture(request: ApprovalRequest, /) -> ApprovalAnswer:
        seen.append(request.prompt)
        # Deny after the prompt is built; the server never sees it.
        return ApprovalAnswer(False, "stdin")

    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            prompts=OperatorPrompts(approver=_capture),
        )
        with pytest.raises(ToolDeniedError):
            d.dispatch("mcp__fake__echo", {"text": "x" * 500, "items": list(range(20))})
    finally:
        mgr.close()
    assert len(seen) == 1
    assert "x" * 500 in seen[0], "a clipped string is consent to an unseen payload"
    assert "16, 17, 18, 19]" in seen[0], "the list was cut at 10 items"


def test_a_denied_call_never_reaches_the_server(tmp_path: Path) -> None:
    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            prompts=OperatorPrompts(approver=_deny),
        )
        with pytest.raises(ToolDeniedError, match="approve"):
            d.dispatch("mcp__fake__echo", {"text": "hello"})
    finally:
        mgr.close()


def test_approve_yes_is_the_standing_consent(tmp_path: Path) -> None:
    """The durable way to stop being asked, visible in `agent6 config show`."""

    def _forbidden(_request: ApprovalRequest, /) -> ApprovalAnswer:
        pytest.fail("approve = 'yes' must not prompt")

    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(approve="yes"),
            mcp_manager=mgr,
            prompts=OperatorPrompts(approver=_forbidden),
        )
        assert d.dispatch("mcp__fake__echo", {"text": "hi"})
    finally:
        mgr.close()


def test_auto_approve_covers_mcp_servers() -> None:
    """A "do not prompt me this run" that still prompted would not be that."""
    cfg = _cfg().with_sandbox_overrides(auto_approve=True)
    assert cfg.mcp.servers["fake"].approve == "yes"
    assert cfg.sandbox.run_commands == "yes"


def test_allowing_every_command_does_not_allow_a_server(tmp_path: Path) -> None:
    """An "a" at a run_command prompt grants the command scope only, never MCP tools."""
    from agent6.sessions.ipc import set_away_mode

    session_dir = tmp_path / "run"
    (session_dir / "approvals").mkdir(parents=True)
    set_session_allow(session_dir, COMMAND_SCOPE)
    prompts = _cli_prompts(session_dir)

    assert prompts.approve("Allow run_command: ls", scope=COMMAND_SCOPE) is True
    # away-mode deny, so the ungranted call refuses instead of polling for a front-end.
    set_away_mode(session_dir, "deny")
    mgr = _manager()
    try:
        d = ToolDispatcher(root=tmp_path, config=_cfg(), mcp_manager=mgr, prompts=prompts)
        with pytest.raises(ToolDeniedError):
            d.dispatch("mcp__fake__echo", {"text": "hi"})
    finally:
        mgr.close()


def test_allowing_one_server_does_not_allow_its_sibling(tmp_path: Path) -> None:
    """Two servers are two threats: a grant covers the one asked about, nothing else."""
    from agent6.sessions.ipc import session_allow_set

    session_dir = tmp_path / "run"
    (session_dir / "approvals").mkdir(parents=True)
    set_session_allow(session_dir, "mcp.notes")
    approve = _cli_prompts(session_dir).approve

    assert approve("Allow mcp__notes__read: {}", scope="mcp.notes") is True
    assert not session_allow_set(session_dir, "mcp.shell")
    assert not session_allow_set(session_dir, COMMAND_SCOPE)


def test_approving_everything_while_away_covers_the_servers_too(tmp_path: Path) -> None:
    """A grant is per scope, so an away-mode "approve all" covers a server's first call too."""
    from agent6.app.frontend import apply_spawned_away_default, approval_scopes
    from agent6.sessions.ipc import session_allow_set

    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {
                    "notes": {"command": ["true"]},
                    "off": {"command": ["true"], "enabled": False},
                },
            }
        }
    )
    assert approval_scopes(cfg) == (COMMAND_SCOPE, "mcp.notes")  # a disabled server has no tools

    import os

    os.environ["AGENT6_DETACHED_AWAY"] = "approve"
    try:
        apply_spawned_away_default(tmp_path, approval_scopes(cfg))
    finally:
        del os.environ["AGENT6_DETACHED_AWAY"]
    assert session_allow_set(tmp_path, COMMAND_SCOPE)
    assert session_allow_set(tmp_path, "mcp.notes")
    assert not session_allow_set(tmp_path, "mcp.off")


def test_denying_a_server_for_the_session_withdraws_its_tools(tmp_path: Path) -> None:
    """A "deny all" withdraws that server's tools from the next turn and no other server's."""
    from agent6.sessions.ipc import set_session_deny

    session_dir = tmp_path / "run"
    (session_dir / "approvals").mkdir(parents=True)
    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            session_dir=session_dir,
            prompts=OperatorPrompts(approver=_deny),
        )
        assert "mcp__fake__echo" in d.available_tool_names()
        set_session_deny(session_dir, "mcp.other")
        assert "mcp__fake__echo" in d.available_tool_names()  # a sibling's denial is not ours
        set_session_deny(session_dir, "mcp.fake")
        assert "mcp__fake__echo" not in d.available_tool_names()
    finally:
        mgr.close()


def test_an_unconfigured_server_is_refused_before_it_is_ever_asked_about(tmp_path: Path) -> None:
    """A name that is not a configured server is not a server, and no consent is asked for it.

    The LLM chooses tool names, so a parsed server of `../../tmp/x` would become a grant's scope.
    """

    def _forbidden(request: ApprovalRequest, /) -> ApprovalAnswer:
        pytest.fail(f"the operator was asked about a server that does not exist: {request.scope}")

    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            prompts=OperatorPrompts(approver=_forbidden),
        )
        with pytest.raises(ToolError, match="unknown MCP server"):
            d.dispatch("mcp__../../tmp/x__t", {})
    finally:
        mgr.close()


def test_deny_all_for_a_server_refuses_the_next_call_not_just_the_listing(tmp_path: Path) -> None:
    """A call to a withdrawn server's tool is refused by the call gate, never run or re-prompted."""
    asked: list[tuple[str, str | None]] = []
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            prompts=_recording(asked),
            session_dir=session_dir,
        )
        d.dispatch("mcp__fake__echo", {"text": "hi"})
        assert len(asked) == 1
        set_session_deny(session_dir, "mcp.fake")
        assert "mcp__fake__echo" not in d.available_tool_names()
        with pytest.raises(ToolError, match="denied for this session"):
            d.dispatch("mcp__fake__echo", {"text": "hi"})
        assert len(asked) == 1, "the operator was asked again after denying the scope"
    finally:
        mgr.close()


def test_denying_one_server_leaves_a_sibling_alone(tmp_path: Path) -> None:
    """The deny is per scope, like the grant."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name=name, command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
            for name in ("fake", "other")
        ]
    )
    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {n: {"command": ["true"]} for n in ("fake", "other")},
            }
        }
    )
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=cfg,
            mcp_manager=mgr,
            prompts=_recording([]),
            session_dir=session_dir,
        )
        set_session_deny(session_dir, "mcp.fake")
        assert d.mcp_denied("fake") and not d.mcp_denied("other")
        assert "mcp__other__echo" in d.available_tool_names()
        d.dispatch("mcp__other__echo", {"text": "hi"})  # the sibling still answers
    finally:
        mgr.close()


def test_a_huge_payload_prompts_with_a_head_and_a_full_file(tmp_path: Path) -> None:
    """Past the bound the complete payload lands in a session-dir file and the prompt names it.

    Jailed commands cannot reach it; without a session dir the full text stays inline.
    """
    seen: list[str] = []

    def _capture(request: ApprovalRequest, /) -> ApprovalAnswer:
        seen.append(request.prompt)
        return ApprovalAnswer(False, "stdin")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    big = "y" * 10_000
    mgr = _manager()
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=_cfg(),
            mcp_manager=mgr,
            prompts=OperatorPrompts(approver=_capture),
            session_dir=session_dir,
        )
        with pytest.raises(ToolDeniedError):
            d.dispatch("mcp__fake__echo", {"text": big})
    finally:
        mgr.close()
    assert len(seen) == 1
    assert big not in seen[0], "the wall of text must not flood the prompt"
    assert "full payload:" in seen[0] and "chars total" in seen[0]
    (payload,) = session_dir.glob("approval_payload-*.json")
    assert big in payload.read_text(encoding="utf-8")
