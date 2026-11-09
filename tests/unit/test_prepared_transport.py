# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import pytest

from agent6 import kinds, paths
from agent6.app import session_runtime
from agent6.config import Config, RoleModel
from agent6.sandbox import jail, jail_session
from agent6.sessions import event_log, layout
from agent6.tools import command_tools, operator_prompts, policy, results, target_policy

_FORWARDER = """\
from __future__ import annotations
import subprocess
import sys
import threading

with subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
    def send() -> None:
        with open(sys.argv[2], "wb", buffering=0) as wire:
            for frame in sys.stdin.buffer:
                wire.write(frame)
                child.stdin.write(frame)
                child.stdin.flush()
        child.stdin.close()

    sender = threading.Thread(target=send)
    sender.start()
    for frame in child.stdout:
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()
    sender.join()
    sys.exit(child.wait())
"""


def test_injected_transport_serves_setup_commands_and_verification(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup, ordinary commands and verification use an independently prepared transport."""
    root = tmp_path / "workspace"
    root.mkdir()
    home = tmp_path / "target-home"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    tool = bin_dir / "target-command"
    tool.write_text('#!/bin/sh\nprintf "%s:%s\\n" "$1" "$HOME"\nprintf "%s\\n" "$1" >> sequence\n')
    tool.chmod(0o755)
    setup = (tool.name, "setup")
    gate = (tool.name, "verify")
    model = (tool.name, "model")
    cfg = Config.model_validate(
        {
            "sandbox": {"isolation": "none", "network": "host", "run_commands": "allow"},
            "harness": {"verify_command": gate, "command_checkin_s": 0},
            "workspace": {"setup_command": setup},
        }
    )
    env = (("HOME", str(home)), ("PATH", str(bin_dir)))
    command_policy = kinds.JailPolicy(cwd=root, argv=(), isolation="none", env=env)
    target = kinds.ExecutionTarget(
        transport="test-forwarder",
        host_root=root,
        command_root=root,
        user=str(os.getuid()),
        home=str(home),
        environment=kinds.ExecutionEnvironment(env),
        controls=target_policy.effective_controls(cfg, "none"),
    )
    session_layout = layout.SessionLayout(paths.state_dir(root), "forwarded")
    session_layout.ensure()
    wire = tmp_path / "wire.jsonl"
    launcher = jail.require_jail_binary()
    grants = policy.resolve_grants(cfg, root)

    def no_local(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an injected transport selected a local launcher or one-shot runner")

    monkeypatch.setattr(jail_session.JailSession, "open", no_local)
    monkeypatch.setattr(jail, "spawn_launcher", no_local)
    monkeypatch.setattr(jail, "run_in_jail", no_local)
    with contextlib.ExitStack() as cleanup:
        transport = cleanup.enter_context(
            subprocess.Popen(
                (sys.executable, "-u", "-c", _FORWARDER, str(launcher), str(wire)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        )
        session = jail_session.JailSession.connect(transport, command_policy, target=target)
        cleanup.callback(session.close)
        access = command_tools.JailAccess(policy=command_policy, session=session, grants=grants)
        dispatcher = session_runtime.build_session_tools(
            cfg,
            cwd=root,
            state_dir=session_layout.state_dir,
            layout=session_layout,
            isolation="none",
            mode="run",
            events=event_log.EventSink(session_layout.logs_path),
            prompts=operator_prompts.OperatorPrompts(session_dir=session_layout.session_dir),
            loop_log=lambda _line: None,
            mcp_manager=None,
            rm_role=RoleModel(provider="test", model="test"),
            command_access=access,
        ).dispatcher
        cleanup.callback(dispatcher.close)
        assert dispatcher.execution_target is session.target
        assert session.target == dataclasses.replace(target, protocol_version=3)
        assert session._proc is transport
        setup_result = dispatcher.run_setup(cfg.workspace.setup_command)
        baseline = dispatcher.run_verify()
        ordinary = dispatcher.dispatch("run_command", {"argv": list(model)})
        verification = dispatcher.dispatch("run_verify_command", {})
        assert isinstance(ordinary, results.ExecResult)
        assert isinstance(verification, results.ExecResult)
        for result, stage in (
            (setup_result, "setup"),
            (baseline, "verify"),
            (ordinary, "model"),
            (verification, "verify"),
        ):
            assert result.returncode == 0
            assert result.stdout == f"{stage}:{home}\n"
        assert (root / "sequence").read_text().splitlines() == [
            "setup",
            "verify",
            "model",
            "verify",
        ]
        assert verification.verification is not None
        assert verification.verification.outcome == "passed"
        dispatcher.close()
        assert transport.poll() == 0
        assert session.close() == ""

    frames = [json.loads(line) for line in wire.read_text().splitlines()]
    assert frames[0]["mode"] == "serve"
    assert frames[0]["cwd"] == str(root)
    assert frames[0]["env"] == [list(pair) for pair in env]
    assert [frame["kind"] for frame in frames[1:]] == ["run"] * 4
    assert [frame["id"] for frame in frames[1:]] == [1, 2, 3, 4]
    assert [frame["argv"] for frame in frames[1:]] == [
        list(argv) for argv in (setup, gate, model, gate)
    ]
