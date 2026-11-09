from __future__ import annotations

import dataclasses
import secrets
import shutil
import time

import pytest

from agent6 import kinds
from agent6.sandbox import container
from tests.sandbox import conftest


@pytest.mark.parametrize(
    "value",
    [
        "${",
        "${command:evil}",
        "${containerEnv:BAD-NAME}",
        "${containerEnv:PATH",
        "${containerEnv:PATH}${",
        "${containerEnv:PATH}${localEnv:SECRET}",
        "${localEnv:SECRET}",
        "${unsupported:PATH}",
        "${}",
    ],
)
def test_remote_env_refuses_unsupported_templates(target: conftest.Target, value: str) -> None:
    discovery = dataclasses.replace(target.discovery, remote_env=(("PATH", value),))
    with pytest.raises(container.ContainerError, match="substitution"):
        container.prepare(discovery)


@pytest.mark.parametrize(
    "value",
    ["${containerEnv:PATH}", "${localEnv:SECRET}", "${unterminated", "${}", "echo ${HOME}"],
)
def test_remote_env_substituted_values_stay_literal(target: conftest.Target, value: str) -> None:
    """Target environment values are data, not recursive templates or shell expressions."""
    discovery = dataclasses.replace(
        target.discovery,
        exec=(*target.discovery.exec, "SCRIPT=" + value),
        remote_env=(("COPY", "before${containerEnv:SCRIPT}after"),),
    )
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        result = session.run(("/usr/bin/printenv", "COPY"))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
        assert result.stdout == "before" + value + "after\n"


def test_container_env_expansion_and_unsetting(target: conftest.Target) -> None:
    discovery = dataclasses.replace(
        target.discovery,
        exec=(*target.discovery.exec, "REMOVED=target-canary"),
        remote_env=(("PATH", "${containerEnv:PATH}:/extra"), ("REMOVED", None)),
    )
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        assert session.probe().environment["REMOVED"] == "target-canary"
        result = session.run(("/usr/bin/env",))
        assert isinstance(result, kinds.CommandResult)
        assert f"PATH={target.tool_dir}:/usr/bin:/bin:/extra\n" in result.stdout
        assert "${" not in result.stdout
        assert "REMOVED=" not in result.stdout
        assert "target-canary" not in result.stdout


def test_background_uses_the_prepared_environment(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Background commands need the target environment and remoteEnv removals independently."""
    name = "a6-background-env-" + secrets.token_hex(8)
    assert shutil.which(name) is None
    executable = target.tool_dir / name
    executable.write_text("#!/bin/sh\nexec /usr/bin/env\n")
    executable.chmod(0o755)
    discovery = dataclasses.replace(
        target.discovery,
        exec=(*target.discovery.exec, "REMOVED=target-canary"),
        remote_env=(("PATH", "${containerEnv:PATH}:/extra"), ("REMOVED", None)),
    )
    monkeypatch.setenv("HOME", "/host-home-must-not-cross")
    monkeypatch.setenv("PATH", "/host-path-must-not-cross")
    monkeypatch.setenv("TMPDIR", "/host-tmp-must-not-cross")
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        probe = session.probe()
        assert probe.environment["REMOVED"] == "target-canary"
        handle = session.start_background((name,))
        deadline = time.monotonic() + 5
        while True:
            status = session.status_background(handle)
            if not status.running or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert not status.running and status.returncode == 0 and not status.error, status
        stopped = session.stop_background(handle)
        assert stopped.returncode == 0 and not stopped.error and not stopped.capture_error, stopped
        capture = session.read_background(handle, offset=0)
        assert not capture.capture_error
        observed = dict(line.split("=", 1) for line in capture.data.decode().splitlines())
        assert observed["HOME"] == str(target.home)
        assert observed["PATH"] == f"{target.tool_dir}:/usr/bin:/bin:/extra"
        assert observed["TMPDIR"] == probe.environment["TMPDIR"]
        assert observed["TMPDIR"].startswith("/tmp/a6.") and len(observed["TMPDIR"]) < 40
        assert "REMOVED" not in observed
