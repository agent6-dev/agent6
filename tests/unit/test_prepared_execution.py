# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import dataclasses
import pathlib
import subprocess
from collections.abc import Callable
from unittest import mock

import pytest

from agent6 import kinds, providers
from agent6.app import execution, stamps, worktree_setup
from agent6.app import reporter as app_reporter
from agent6.app.session import runtime as session_runtime
from agent6.config import Config, RoleModel
from agent6.harness import loop_conversation, loop_state, run_chain
from agent6.harness.verify import gate as verify_gate
from agent6.sandbox import jail, jail_session, tool_paths
from agent6.sessions import event_log, layout, manifest
from agent6.tools import (
    command_tools,
    dispatch,
    operator_prompts,
    results,
    target_policy,
    tool_errors,
    verify_infer,
)


def _config(**sandbox: object) -> Config:
    return Config.model_validate(
        {
            "sandbox": {"isolation": "none", "network": "host", "run_commands": "allow", **sandbox},
            "harness": {"verify": {"command": ["sh", "-c", "echo gate"]}, "command_checkin_s": 0},
        }
    )


def _tools(
    root: pathlib.Path,
    cfg: Config,
    isolation: kinds.IsolationLevel = "none",
    *,
    approver: operator_prompts.Approver | None = None,
) -> session_runtime.SessionTools:
    state = root.parent / (root.name + "-state")
    session_layout = layout.SessionLayout(state, "prepared")
    session_layout.session_dir.mkdir(parents=True, exist_ok=True)
    return session_runtime.build_session_tools(
        cfg,
        cwd=root,
        state_dir=state,
        layout=session_layout,
        isolation=isolation,
        mode="run",
        events=event_log.EventSink(session_layout.logs_path),
        prompts=operator_prompts.OperatorPrompts(
            approver=approver
            or (
                lambda _request: operator_prompts.ApprovalAnswer(
                    approved=False, source="headless-default"
                )
            )
        ),
        loop_log=lambda _line: None,
        mcp_manager=None,
        rm_role=RoleModel(provider="test", model="test"),
    )


@pytest.mark.parametrize(
    "isolation,network,home",
    [("full", "session", "tmp"), ("partial", "host", "cache"), ("none", "host", "host")],
)
def test_effective_controls_cover_every_sandbox_leaf(
    isolation: kinds.IsolationLevel, network: str, home: str
) -> None:
    """The local record covers permissions and limits as well as confinement."""
    cfg = _config(isolation="auto", network="auto", memory_limit_mib=128)
    rows = target_policy.effective_controls(cfg, isolation)
    by_leaf = {row.leaf: row for row in rows}
    assert len(rows) == len(type(cfg.sandbox).model_fields)
    assert set(by_leaf) == {f"sandbox.{name}" for name in type(cfg.sandbox).model_fields}
    assert by_leaf["sandbox.isolation"].effective == isolation
    assert by_leaf["sandbox.network"].effective == network
    assert by_leaf["sandbox.home"].effective == home
    assert by_leaf["sandbox.memory_limit_mib"].effective == 128
    assert by_leaf["sandbox.memory_limit_mib"].owner == "agent6"
    assert by_leaf["sandbox.run_commands"].effective == "allow"
    assert by_leaf["sandbox.hide_paths"].owner == (
        "host file tools" if isolation == "none" else "agent6"
    )
    assert by_leaf["sandbox.protect_git"].effective == ("on" if isolation == "full" else "off")
    assert bool(by_leaf["sandbox.protect_git"].reason) == (isolation != "full")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rows[0].reason = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial", "none"]
)
def test_preparation_precedes_work_and_all_commands_share_it(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    jail_bin: pathlib.Path,
    isolation: kinds.IsolationLevel,
) -> None:
    """Setup, baseline, model commands and verification use the already-open session."""
    opened: list[jail_session.JailSession] = []
    real_open = jail_session.JailSession.open

    def observe(*args: object, **kwargs: object) -> jail_session.JailSession:
        session = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(session)
        return session

    monkeypatch.setattr(jail_session.JailSession, "open", observe)
    cfg = _config(isolation=isolation, network="auto", memory_limit_mib=128)
    tools = _tools(tmp_path, cfg, isolation)
    d = tools.dispatcher
    try:
        assert len(opened) == 1, "session preparation was deferred until a command"

        def no_one_shot(*args: object, **kwargs: object) -> None:
            pytest.fail("a prepared dispatcher selected one-shot execution")

        monkeypatch.setattr(jail, "run_in_jail", no_one_shot)
        target = d.execution_target
        assert target is not None and target.transport == "local"
        assert target.host_root == target.command_root == tmp_path
        assert target.protocol_version == 4
        assert target.runtime_id is None and target.container_id is None
        assert target.launcher_instance is not None and len(target.launcher_instance) == 32
        assert target.home == dict(target.environment.variables)["HOME"]
        # A shell could drop privileges before reporting them, so invoke id directly.
        for flag in ("-ru", "-u"):
            identity = d.dispatch("run_command", {"argv": ["/usr/bin/id", flag]})
            assert isinstance(identity, results.ExecResult) and identity.returncode == 0
            assert identity.stdout.strip() == target.user
        controls = {row.leaf: row for row in target.controls}
        assert len(target.controls) == len(type(cfg.sandbox).model_fields)
        assert set(controls) == {f"sandbox.{name}" for name in type(cfg.sandbox).model_fields}
        assert controls["sandbox.isolation"].effective == isolation
        assert controls["sandbox.network"].effective == (
            "session" if isolation == "full" else "host"
        )
        assert (
            controls["sandbox.home"].effective
            == {"full": "tmp", "partial": "cache", "none": "host"}[isolation]
        )
        assert controls["sandbox.memory_limit_mib"].effective == 128
        assert controls["sandbox.memory_limit_mib"].owner == "agent6"
        assert controls["sandbox.run_commands"].effective == "allow"
        assert controls["sandbox.protect_git"].effective == ("on" if isolation == "full" else "off")
        setup = d.run_setup(("sh", "-c", 'printf setup > "${TMPDIR:-/tmp}/prepared"'))
        assert setup.returncode == 0
        assert d.run_verify().returncode == 0  # baseline
        model = d.dispatch("run_command", {"argv": ["sh", "-c", 'cat "${TMPDIR:-/tmp}/prepared"']})
        assert isinstance(model, results.ExecResult) and model.stdout == "setup"
        assert d.run_verify().returncode == 0
        assert len(opened) == 1
        assert d.execution_target is target
    finally:
        d.close()


def test_preparation_failure_refuses_before_dispatch_exists(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed handshake cannot defer failure until setup or select a one-shot runner."""

    def refuse(*args: object, **kwargs: object) -> jail_session.JailSession:
        raise jail.JailUnavailableError("preparation refused")

    monkeypatch.setattr(jail_session.JailSession, "open", refuse)
    with pytest.raises(jail.JailUnavailableError, match="preparation refused"):
        _tools(tmp_path, _config())


@pytest.mark.parametrize(
    "failure,isolation",
    [
        ("prepare", "none"),
        pytest.param("gate", "full", marks=pytest.mark.needs_namespaces),
        ("gate", "none"),
        ("commands", "none"),
    ],
)
def test_run_execution_prepares_before_inference_and_reuses_the_session(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    jail_bin: pathlib.Path,
    failure: str,
    isolation: kinds.IsolationLevel,
) -> None:
    """Production inference and dispatch share preparation; failures release its resources."""
    origin = tmp_path / "origin"
    origin.mkdir()
    root = tmp_path / "repo"
    for argv in (
        ("init", "-q"),
        (
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "seed",
            "--allow-empty",
        ),
        ("worktree", "add", "-q", "--detach", str(root)),
    ):
        subprocess.run(("git", "-C", str(origin), *argv), check=True, capture_output=True)
    session_layout = layout.SessionLayout(tmp_path / "state", "prepared-execution")
    session_layout.ensure()
    stamps.write_manifest(
        session_layout.manifest_path,
        manifest.SessionManifest(
            session_id=session_layout.session_id, worktree=root, worktree_git_dir=origin / ".git"
        ),
    )
    cfg = _config(isolation=isolation, network="auto").with_setup_command(
        ("sh", "-c", "printf setup >> sequence")
    )
    prompts = operator_prompts.OperatorPrompts(session_dir=session_layout.session_dir)
    opened: list[jail_session.JailSession] = []
    accesses: list[command_tools.JailAccess] = []
    order: list[str] = []
    real_open = jail_session.JailSession.open
    real_prepare = session_runtime.prepare_local_execution
    real_worktree_prepare = worktree_setup.prepare
    providers = mock.Mock(rm_role=RoleModel(provider="test", model="test"))
    frontend = mock.MagicMock()
    frontend.stream_modes.return_value = (False, False)
    monkeypatch.setattr(
        session_runtime, "build_session_providers", mock.Mock(return_value=providers)
    )

    def observe_open(*args: object, **kwargs: object) -> jail_session.JailSession:
        order.append("prepare")
        if failure == "prepare":
            raise jail.JailUnavailableError("preparation refused")
        session = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(session)
        return session

    def observe_prepare(*args: object, **kwargs: object) -> command_tools.JailAccess:
        access = real_prepare(*args, **kwargs)  # type: ignore[arg-type]
        accesses.append(access)
        return access

    def infer(current: Config, _budget: object, available: Callable[[str], bool]) -> Config:
        assert order == ["prepare"] and len(opened) == len(accesses) == 1
        assert available == opened[0].executable_available
        assert opened[0]._proc.poll() is None
        access = accesses[0]
        assert access.session is opened[0]
        assert (access.temp is None) is (isolation == "full")
        if access.temp is not None:
            assert pathlib.Path(access.temp.name).is_dir()
        assert (access.network is not None) is (isolation == "full")
        if access.network is not None:
            assert access.network._holder is not None
            assert access.network._holder.poll() is None
            assert not access.owns_network  # run_execution owns this holder, not the dispatcher.
        order.append("infer")
        if failure == "gate":
            raise RuntimeError("inference refused")
        inferred = verify_infer.infer_verify_command(
            root, "Verify: printf gate >> sequence", available=available
        )
        assert inferred.argv == ("sh", "-c", "printf gate >> sequence")
        return current.with_verify_command(inferred.argv)

    def observe_worktree(
        current: Config,
        cwd: pathlib.Path,
        *,
        runner: worktree_setup.SetupRunner,
        reporter: app_reporter.Reporter,
    ) -> worktree_setup.Prepared:
        assert order == ["prepare", "infer"] and len(opened) == 1
        assert isinstance(runner, dispatch.ToolDispatcher)
        assert runner.execution_target is opened[0].target
        order.append("workspace")
        prepared = real_worktree_prepare(current, cwd, runner=runner, reporter=reporter)
        assert prepared.ran and prepared.problem is None
        assert runner.run_verify().returncode == 0
        model = runner.dispatch("run_command", {"argv": ["sh", "-c", "printf model >> sequence"]})
        assert isinstance(model, results.ExecResult) and model.returncode == 0
        assert runner.run_verify().returncode == 0
        assert (root / "sequence").read_text() == "setupgatemodelgate"
        assert len(opened) == 1
        # Stop before model calls; run_execution still tears down the real dispatcher.
        raise RuntimeError("commands checked")

    monkeypatch.setattr(jail_session.JailSession, "open", observe_open)
    monkeypatch.setattr(session_runtime, "prepare_local_execution", observe_prepare)
    monkeypatch.setattr(worktree_setup, "prepare", observe_worktree)
    inputs = execution.ExecutionInputs(
        explicit_leaves=frozenset(),
        session_id=session_layout.session_id,
        mode="run",
        role="worker",
        isolation=isolation,
        tui_enabled=False,
        interactive=False,
        task="do the thing",
        gate=infer,
        worktree_git_dir=origin / ".git",
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=session_layout.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=prompts,
        ask_transcript_task=None,
    )
    try:
        with pytest.raises(
            jail.JailUnavailableError if failure == "prepare" else RuntimeError,
            match={
                "prepare": "preparation refused",
                "gate": "inference refused",
                "commands": "commands checked",
            }[failure],
        ):
            execution.run_execution(
                cfg,
                session_layout,
                inputs,
                frontend=frontend,
                reporter=app_reporter.Reporter(out=lambda _line: None, err=lambda _line: None),
                events=event_log.EventSink(session_layout.logs_path),
                transcript_sink=mock.MagicMock(),
                cwd=root,
                state_dir=session_layout.state_dir,
            )
        assert (
            order
            == {
                "prepare": ["prepare"],
                "gate": ["prepare", "infer"],
                "commands": ["prepare", "infer", "workspace"],
            }[failure]
        )
        assert (root / "sequence").exists() is (failure == "commands")
        for access in accesses:
            assert access.session is not None and access.session._proc.poll() is not None
            if access.network is not None:
                assert access.network._holder is not None
                assert access.network._holder.poll() is not None
            if access.temp is not None:
                assert not pathlib.Path(access.temp.name).exists()
        providers.close.assert_called_once()
    finally:
        # Assertions precede this fallback; retained access records prevent finalizer cleanup.
        for session in opened:
            session.close()
        for access in accesses:
            if access.temp is not None:
                access.temp.cleanup()
            network = access.network
            if (
                network is not None
                and network._holder is not None
                and network._holder.poll() is None
            ):
                network.close()


@pytest.mark.parametrize("permission", ["deny", "ask"])
def test_prepared_session_does_not_grant_command_permission(
    tmp_path: pathlib.Path, permission: str, jail_bin: pathlib.Path
) -> None:
    """Owning a ready transport grants neither a denied nor an unapproved command."""
    d = _tools(tmp_path, _config(run_commands=permission)).dispatcher
    try:
        assert d.execution_target is not None
        with pytest.raises(tool_errors.ToolError, match=r"not available|not approved|denied"):
            d.dispatch("run_command", {"argv": ["sh", "-c", "touch forbidden"]})
        assert not (tmp_path / "forbidden").exists()
    finally:
        d.close()


def test_prepared_session_asks_then_runs_an_approved_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, jail_bin: pathlib.Path
) -> None:
    """An approving ask reaches the existing session after one command-scoped request."""
    opened: list[jail_session.JailSession] = []
    ran_in: list[jail_session.JailSession] = []
    requests: list[operator_prompts.ApprovalRequest] = []
    order: list[str] = []
    real_open = jail_session.JailSession.open
    real_run = jail_session.JailSession.run
    marker = tmp_path / "allowed"

    def open_session(*args: object, **kwargs: object) -> jail_session.JailSession:
        session = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(session)
        return session

    def run(
        self: jail_session.JailSession, *args: object, **kwargs: object
    ) -> kinds.CommandResult | kinds.BackgroundHandoff:
        order.append("run")
        ran_in.append(self)
        return real_run(self, *args, **kwargs)  # type: ignore[arg-type]

    def approve(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        assert len(opened) == 1 and not marker.exists()
        order.append("approve")
        requests.append(request)
        return operator_prompts.ApprovalAnswer(approved=True, source="frontend")

    monkeypatch.setattr(jail_session.JailSession, "open", open_session)
    monkeypatch.setattr(jail_session.JailSession, "run", run)
    d = _tools(tmp_path, _config(run_commands="ask"), approver=approve).dispatcher
    try:
        assert len(opened) == 1 and not order
        result = d.dispatch("run_command", {"argv": ["sh", "-c", "printf approved > allowed"]})
        assert isinstance(result, results.ExecResult) and result.returncode == 0
        assert order == ["approve", "run"]
        assert len(requests) == 1 and requests[0].scope == "command"
        assert "printf approved > allowed" in requests[0].prompt
        assert marker.read_text() == "approved"
        assert len(opened) == 1 and ran_in == opened
    finally:
        d.close()


@pytest.mark.parametrize("target_has_tool", [True, False])
def test_mid_run_gate_adoption_uses_the_prepared_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    target_has_tool: bool,
) -> None:
    """Mid-run inference and adoption use the prepared PATH, not the host's current tools."""
    root = tmp_path / "repo"
    root.mkdir()
    target_home, host_home = tmp_path / "target-home", tmp_path / "host-home"
    for home in (target_home, host_home):
        (home / ".local/bin").mkdir(parents=True)
    tool = (target_home if target_has_tool else host_home) / ".local/bin/agent6-prepared-gate"
    tool.write_text("#!/bin/sh\n[ \"$1\" = check ] || exit 2\nprintf 'target gate\\n'\n")
    tool.chmod(0o755)
    monkeypatch.setenv("HOME", str(target_home))
    d = _tools(root, _config().with_verify_command(())).dispatcher
    try:
        # The session retains its environment when the host's tool search changes.
        monkeypatch.setenv("HOME", str(host_home))
        assert d.execution_target is not None
        assert d.execution_target.home == str(target_home)
        assert d.executable_available(tool.name) is target_has_tool
        assert tool_paths.jail_resolves(tool.name) is not target_has_tool
        gate = verify_gate.VerifyGate(
            configured=(),
            when="finish",
            at_start=True,
            retries=0,
            timeout_s=10,
            infer=True,
            hide=(),
            mode="run",
            chain=run_chain.RunChain(root),
            dispatcher=d,
            log=lambda _line: None,
            emit=lambda *_args, **_kwargs: None,
        )
        state = loop_state.LoopState(original_task="build", tool_calls=0)
        turn = loop_state.TurnState(
            number=2,
            resp=providers.ProviderResponse(
                text="",
                tool_uses=(),
                stop_reason="end_turn",
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            assistant=loop_conversation.AssistantMessage((), ()),
        )
        assert not gate.present(state.verify)
        assert "run_verify_command" not in d.available_tool_names()
        gate.maybe_adopt(state, turn)
        assert not gate.present(state.verify)

        (root / "AGENTS.md").write_text("Verify: agent6-prepared-gate check\n")
        gate.maybe_adopt(state, turn)
        if target_has_tool:
            assert gate.command(state.verify) == ("agent6-prepared-gate", "check")
            assert gate.present(state.verify)
            assert "run_verify_command" in d.available_tool_names()
            assert any("verify command was adopted" in str(item) for item in turn.tool_results)
            result = d.run_verify()
            assert result.returncode == 0 and result.stdout == "target gate\n"
        else:
            assert gate.command(state.verify) == ()
            assert not gate.present(state.verify)
            assert "run_verify_command" not in d.available_tool_names()
            assert "the sandbox has no `agent6-prepared-gate`" in state.verify.declined
            assert not turn.tool_results
    finally:
        d.close()


@pytest.mark.parametrize("source", ["agents", "package", "llm"])
def test_inference_uses_injected_executable_availability(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Every inference tier consults the target, not the host tool search path."""

    def absent(_exe: str) -> bool:
        return False

    def present(_exe: str) -> bool:
        return True

    helper_calls: list[str] = []

    def helper(context: str) -> str:
        helper_calls.append(context)
        return '["target-only", "check"]'

    monkeypatch.setattr(tool_paths, "jail_resolves", absent)
    md = "Verify: target-only check" if source == "agents" else "project tests"
    if source == "package":
        (tmp_path / "package.json").write_text('{"scripts":{"test":"target-only check"}}')
    inferred = verify_infer.infer_verify_command(
        tmp_path,
        md,
        available=lambda exe: exe == "target-only",
        llm_call=helper,
    )
    assert inferred.argv == ("target-only", "check")
    assert (
        inferred.source == {"agents": "agents_md", "package": "package.json", "llm": "llm"}[source]
    )
    monkeypatch.setattr(tool_paths, "jail_resolves", present)
    helper_calls.clear()
    rejected = verify_infer.infer_verify_command(tmp_path, md, available=absent, llm_call=helper)
    assert len(helper_calls) == 1
    assert not rejected.argv
    assert (
        "the helper model",
        "names `target-only check`, and the sandbox has no `target-only`",
    ) in rejected.passed_over
