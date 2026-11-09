# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import pathlib
import subprocess
import time
from unittest import mock

import pytest

from agent6 import paths
from agent6.app import reporter as app_reporter
from agent6.app import run as run_lifecycle
from agent6.app import session_runtime, session_setup
from agent6.config import Config, RoleModel
from agent6.harness import loop, loop_snapshot
from agent6.sandbox import tool_paths
from agent6.sessions import layout, manifest
from agent6.tools import command_tools


@pytest.mark.parametrize("target_has_tool", [True, False])
def test_initial_inference_uses_the_prepared_target_through_run_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, target_has_tool: bool
) -> None:
    """The run lifecycle forwards target availability through preflight, not host lookup."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("Verify: agent6-preflight-gate check\n")
    fallback = root / "verify.sh"
    fallback.write_text("#!/bin/sh\nprintf 'fallback gate\\n'\n")
    fallback.chmod(0o755)
    for argv in (("init", "-q"), ("add", "AGENTS.md", "verify.sh"), ("commit", "-qm", "seed")):
        subprocess.run(("git", "-C", str(root), *argv), check=True, capture_output=True)
    target_home, host_home = tmp_path / "target-home", tmp_path / "host-home"
    for home in (target_home, host_home):
        (home / ".local/bin").mkdir(parents=True)
    tool = (target_home if target_has_tool else host_home) / ".local/bin/agent6-preflight-gate"
    tool.write_text("#!/bin/sh\n[ \"$1\" = check ] || exit 2\nprintf 'target gate\\n'\n")
    tool.chmod(0o755)
    monkeypatch.setenv("HOME", str(target_home))
    monkeypatch.chdir(root)
    cfg = Config.model_validate(
        {
            "sandbox": {"isolation": "none", "network": "host", "run_commands": "allow"},
            "harness": {"command_checkin_s": 0},
        }
    )
    assert not cfg.harness.verify_command
    providers = session_runtime.SessionProviders(
        budget=session_setup.budget_tracker(cfg),
        rm_role=RoleModel(provider="test", model="test"),
        provider=mock.Mock(),
        summariser_provider=None,
        review_seats=[],
    )
    monkeypatch.setattr(
        session_runtime, "build_session_providers", mock.Mock(return_value=providers)
    )
    accesses: list[command_tools.JailAccess] = []
    real_prepare = session_runtime.prepare_local_execution

    def prepare(*args: object, **kwargs: object) -> command_tools.JailAccess:
        access = real_prepare(*args, **kwargs)  # type: ignore[arg-type]
        accesses.append(access)
        # The target retains its prepared environment when the host's lookup changes.
        monkeypatch.setenv("HOME", str(host_home))
        assert access.session is not None and access.session.target is not None
        assert access.session.target.home == str(target_home)
        assert access.session.executable_available(tool.name) is target_has_tool
        assert tool_paths.jail_resolves(tool.name) is not target_has_tool
        return access

    expected = ("agent6-preflight-gate", "check") if target_has_tool else ("./verify.sh",)
    observed: list[tuple[str, ...]] = []

    def check_gate(
        harness: loop.Harness, _task: str, *, operator_task: str | None = None
    ) -> loop_snapshot.SessionResult:
        assert harness.config.harness.verify_command == expected
        assert len(accesses) == 1 and accesses[0].session is not None
        assert harness.dispatcher.execution_target is accesses[0].session.target
        result = harness.dispatcher.run_verify()
        assert result.returncode == 0
        assert result.stdout == ("target gate\n" if target_has_tool else "fallback gate\n")
        observed.append(harness.config.harness.verify_command)
        raise RuntimeError("initial gate checked")

    monkeypatch.setattr(session_runtime, "prepare_local_execution", prepare)
    monkeypatch.setattr(loop.Harness, "run", check_gate)
    frontend = mock.MagicMock()
    frontend.should_spawn_tui.return_value = False
    frontend.stream_modes.return_value = (False, False)
    frontend.confirm_unconfined_autorun.return_value = True
    frontend.capabilities.can_ask = False
    session_layout = layout.SessionLayout(paths.state_dir(root), "prepared-preflight")
    try:
        with pytest.raises(RuntimeError, match="initial gate checked"):
            run_lifecycle.run_task(
                cfg,
                "check the workspace",
                cwd=root,
                started_at=time.time(),
                session_id=session_layout.session_id,
                frontend=frontend,
                reporter=app_reporter.Reporter(out=lambda _line: None, err=lambda _line: None),
            )
        assert observed == [expected]
        saved = manifest.read_manifest(session_layout.session_dir)
        assert saved.harness.gate == manifest.GateStamp(
            command=expected, source="inferred", availability="ready"
        )
        assert all(
            access.session is not None and access.session._proc.poll() is not None
            for access in accesses
        )
    finally:
        for access in accesses:
            access.close()
