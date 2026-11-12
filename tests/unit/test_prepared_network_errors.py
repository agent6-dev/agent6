# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import pathlib
import shlex

import pytest

from agent6.app import session_runtime
from agent6.config import Config
from agent6.sandbox import detect, jail


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


def test_prepared_network_failure_propagates_before_work(
    tmp_path: pathlib.Path, refused_holder: pathlib.Path
) -> None:
    """Preparing a session propagates the network holder's failure before any command runs."""
    with pytest.raises(jail.JailUnavailableError, match="network holder refused"):
        session_runtime.prepare_local_execution(_config(), tmp_path, "full")
    assert refused_holder.read_text().splitlines() == ["--hold-netns"]
    assert not (tmp_path / "work").exists()
