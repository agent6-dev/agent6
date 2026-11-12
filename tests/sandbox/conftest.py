from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import secrets
import shutil
import subprocess
from collections.abc import Generator
from unittest import mock

import pytest

from agent6 import kinds, paths
from agent6.app import session_runtime, stamps
from agent6.config import Config, RoleModel
from agent6.harness import loop, run_chain
from agent6.sandbox import container, jail, jail_session
from agent6.sessions import event_log, layout, manifest
from agent6.tools import command_tools, operator_prompts, policy
from agent6.viewmodel import receipt
from tests.unit import test_harness


@pytest.fixture
def static_launcher(monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    if not os.environ.get("AGENT6_JAIL_BIN"):
        binary = jail.locate_jail_binary()
        if binary is None:
            pytest.skip("set AGENT6_JAIL_BIN to the rebuilt static musl launcher")
        # Classify an unprovisioned editable build independently of the validator under test.
        file_tool = shutil.which("file")
        if file_tool is not None:
            description = subprocess.check_output(
                (file_tool, "-b", str(binary.resolve())), text=True
            )
            if "dynamically linked" in description or "interpreter " in description:
                pytest.skip(
                    "the default launcher is dynamic; set AGENT6_JAIL_BIN to a static build"
                )
    binary = jail.require_jail_binary().resolve()
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(binary))
    return binary


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    discovery: container.ContainerDiscovery
    home: pathlib.Path
    tool_dir: pathlib.Path


@pytest.fixture
def target(tmp_path: pathlib.Path, static_launcher: pathlib.Path) -> Target:
    root = tmp_path / "host workspace;not-a-command"
    root.mkdir()
    home = tmp_path / "target-home"
    home.mkdir()
    tools = tmp_path / "target-tools"
    tools.mkdir()
    tool = tools / "inside-only"
    tool.write_text("#!/bin/sh\nprintf 'inside\\n'\n")
    tool.chmod(0o755)
    prefix = ("/usr/bin/env", "-i", f"HOME={home}", f"PATH={tools}:/usr/bin:/bin")
    return Target(container.discover(root, container.ContainerOptions(exec=prefix)), home, tools)


def assert_selected_user(session: jail_session.JailSession, uid: int, gid: int) -> None:
    """Require the requested non-root identity for commands and the private installation."""
    assert uid != 0
    observed = session.probe()
    assert (observed.uid, observed.gid) == (uid, gid)
    assert session.target is not None and session.target.user == f"{uid}:{gid}"
    result = session.run(
        (
            "/bin/sh",
            "-ec",
            'id -u; id -g; stat -c "%u:%g:%a" -- "$TMPDIR"',
        )
    )
    assert isinstance(result, kinds.CommandResult) and result.returncode == 0
    assert result.stdout.splitlines() == [str(uid), str(gid), f"{uid}:{gid}:700"]


def assert_target_only_executable(session: jail_session.JailSession, directory: str) -> None:
    """Require bare-name lookup and execution of a tool absent from the host filesystem."""
    name = "a6-inside-" + secrets.token_hex(8)
    inside = directory + "/" + name
    assert not pathlib.Path(inside).exists()
    result = session.run(
        (
            "/bin/sh",
            "-c",
            'printf "#!/bin/sh\\necho inside-only\\n" > "$1"; chmod 500 "$1"',
            "make-tool",
            inside,
        )
    )
    assert isinstance(result, kinds.CommandResult) and result.returncode == 0
    try:
        assert not pathlib.Path(inside).exists()
        assert session.executable_available(name)
        result = session.run((name,))
        assert isinstance(result, kinds.CommandResult) and result.stdout == "inside-only\n"
        assert not pathlib.Path(inside).exists()
    finally:
        result = session.run(("/bin/rm", "--", inside))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0


def assert_execution_receipt(session: jail_session.JailSession) -> str:
    """Run an injected session and replay its journal-derived receipt after launcher closure."""
    target = session.target
    assert target is not None and target.transport == "container"
    assert target.user and target.home and target.launcher_instance and target.launcher_digest
    observe = (
        "/bin/sh",
        "-c",
        'printf "%s\\n" "$1"; pwd -P; id -u; id -g; printf "%s\\n" "$HOME" "$TMPDIR"',
        "observe",
    )
    cfg = Config.model_validate(
        {
            "sandbox": {"isolation": "none", "network": "host", "run_commands": "allow"},
            "workspace": {"setup_command": (*observe, "setup")},
            "harness": {
                "verify": {"command": (*observe, "verify"), "at_start": True, "when": "finish"},
                "command_checkin_s": 0,
                "max_turns": 3,
                "prompt": {"decompose": "off"},
            },
            "review": {"when": "never"},
        }
    )
    saved = layout.SessionLayout(paths.state_dir(target.host_root), "container-receipt")
    saved.ensure()
    assert not saved.session_dir.is_relative_to(target.host_root)
    stamps.write_manifest(
        saved.manifest_path, manifest.SessionManifest(session_id=saved.session_id)
    )
    events = event_log.EventSink(saved.logs_path)
    tools = session_runtime.build_session_tools(
        cfg,
        cwd=target.host_root,
        state_dir=saved.state_dir,
        layout=saved,
        isolation="none",
        mode="run",
        events=events,
        prompts=operator_prompts.OperatorPrompts(session_dir=saved.session_dir),
        loop_log=lambda _line: None,
        mcp_manager=None,
        rm_role=RoleModel(provider="test", model="test"),
        command_access=command_tools.JailAccess(
            policy=kinds.JailPolicy(cwd=target.command_root, argv=(), isolation="none"),
            session=session,
            grants=policy.resolve_grants(cfg, target.host_root),
        ),
    )
    dispatcher = tools.dispatcher
    try:
        assert dispatcher.execution_target is target
        assert dispatcher.run_setup(cfg.workspace.setup_command).returncode == 0
        provider = mock.Mock()
        provider.call.side_effect = [
            test_harness._tool_resp("run_command", {"argv": [*observe, "model"]}, tool_id="model"),
            test_harness._tool_resp("run_verify_command", tool_id="verify"),
            test_harness._tool_resp("finish_session", {"summary": "checked"}, tool_id="finish"),
        ]
        harness = loop.Harness(
            chain=run_chain.RunChain(
                target.host_root,
                per_turn=False,
                base_sha=subprocess.check_output(
                    ("git", "-C", str(target.host_root), "rev-parse", "HEAD"), text=True
                ).strip(),
            ),
            config=tools.cfg,
            provider=provider,
            dispatcher=dispatcher,
            events=events,
            max_turns=3,
        )
        result = harness.run("Observe the prepared execution environment")
        assert result.completed and result.reason == "finish_session"
    finally:
        dispatcher.close()
    assert not session.close()
    assert session._proc.poll() == 0  # pyright: ignore[reportPrivateUsage]
    view = receipt.read_receipt(saved.session_dir, target.host_root)
    assert not view.evidence_notes
    assert view.execution_id == dispatcher.execution_id
    assert len(view.targets) == 1
    stamp = view.targets[0]
    assert stamp.target_id == dispatcher.target_id
    assert stamp.execution_id == view.execution_id
    for field in (
        "command_root",
        "container_id",
        "home",
        "host_root",
        "launcher_instance",
        "launcher_digest",
        "lifecycle_ref",
        "protocol_version",
        "transport",
        "user",
    ):
        assert getattr(stamp.target, field) == getattr(target, field)
    assert len(view.setups) == 1
    assert len(view.verifications) == 2 and view.verifications[0].baseline
    expected = [
        str(target.command_root),
        *target.user.split(":"),
        target.home,
        dict(target.environment.variables)["TMPDIR"],
    ]
    for record in (*view.setups, *view.verifications):
        assert record.execution_id == view.execution_id
        assert record.target_id == stamp.target_id
        assert record.exit_code == 0
        assert record.stdout_tail.splitlines() == [record.command[-1], *expected]
    assert view.setups[0].command == cfg.workspace.setup_command
    assert all(record.command == cfg.harness.verify.command for record in view.verifications)
    assert all(record.certifies(view.end_tree) for record in view.verifications)
    assert view.final_verify_id == view.verifications[-1].verify_id
    assert len(view.jobs) == len(view.environments) == 1
    assert view.jobs[0].outcome == "settled"
    assert view.environments[0].outcome == "unknown"
    for observation in (*view.jobs, *view.environments):
        assert observation.execution_id == view.execution_id
        assert observation.target_id == stamp.target_id
    rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
    model = next(
        row for row in rows if row["type"] == "tool.result" and row["name"] == "run_command"
    )
    assert model["ok"]
    assert model["stdout_tail"].splitlines() == ["model", *expected]
    rendered = receipt.render(view)
    assert target.launcher_instance in rendered and stamp.target_id in rendered
    assert str(target.host_root) in rendered and str(target.command_root) in rendered
    assert "Run jobs settled" in rendered and "Environment lifecycle unknown" in rendered
    # The records alone carry the receipt: a second read with no runtime access renders the same.
    assert receipt.render(receipt.read_receipt(saved.session_dir, target.host_root)) == rendered
    return rendered


@contextlib.contextmanager
def opened(target: Target) -> Generator[jail_session.JailSession]:
    session = container.prepare(target.discovery)
    try:
        yield session
    finally:
        session.close()
