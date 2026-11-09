# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import pathlib
import shlex

import pytest

from agent6.app import session_runtime
from agent6.config import Config
from agent6.sandbox import detect, jail
from agent6.tools import dispatch, tool_errors


@pytest.fixture
def refused_holder(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Fail the holder after the host's normal confinement probe has completed."""
    detect.probed_environment()
    calls = tmp_path / "holder-calls"
    launcher = tmp_path / "refused-launcher"
    launcher.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" >> {shlex.quote(str(calls))}\n'
        'printf "network holder refused\\n" >&2\nexit 71\n'
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(launcher))
    return calls


def _config() -> Config:
    return Config.model_validate(
        {
            "sandbox": {"isolation": "full", "network": "session", "run_commands": "allow"},
            "harness": {
                "verify_command": ["/bin/sh", "-c", "printf ran > work"],
                "command_checkin_s": 0,
            },
        }
    )


@pytest.mark.parametrize(
    ("surface", "label"),
    [
        ("run_command", "run_command"),
        ("run_verify_command", "verify_command"),
        ("setup", "setup_command"),
        ("verify", "verify_command"),
    ],
)
def test_oneshot_network_failure_is_a_tool_error(
    tmp_path: pathlib.Path, refused_holder: pathlib.Path, surface: str, label: str
) -> None:
    """One-shot network failure keeps its tool error and unavailable verification evidence."""
    cfg = _config()
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg, isolation="full")
    try:
        with pytest.raises(
            tool_errors.ToolError, match=rf"^{label}: jail unavailable:.*network holder refused"
        ) as error:
            if surface == "setup":
                d.run_setup(cfg.harness.verify_command)
            elif surface == "verify":
                d.run_verify()
            else:
                d.dispatch(
                    surface,
                    {"argv": list(cfg.harness.verify_command)} if surface == "run_command" else {},
                )
        assert isinstance(error.value.__cause__, jail.JailUnavailableError)
        assert refused_holder.read_text().splitlines() == ["--hold-netns"]
        assert not (tmp_path / "work").exists()
        if surface in {"verify", "run_verify_command"}:
            assert d.last_verification is not None
            assert d.last_verification.outcome == "unavailable"
            assert d.last_verification.reason == str(error.value)
    finally:
        d.close()


def test_prepared_network_failure_propagates_before_work(
    tmp_path: pathlib.Path, refused_holder: pathlib.Path
) -> None:
    """Prepared startup exposes its sandbox failure instead of selecting one-shot execution."""
    with pytest.raises(jail.JailUnavailableError, match="network holder refused"):
        session_runtime.prepare_local_execution(_config(), tmp_path, "full")
    assert refused_holder.read_text().splitlines() == ["--hold-netns"]
    assert not (tmp_path / "work").exists()
