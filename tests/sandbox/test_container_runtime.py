from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import re
import secrets
import signal
import subprocess

import pydantic
import pytest

from agent6 import kinds
from agent6.sandbox import container, jail
from agent6.tools import background
from tests.sandbox import conftest


@pytest.fixture
def runtime_discovery() -> container.ContainerDiscovery:
    """Require explicit runtime argv and attach to an existing labelled dev container first."""
    configured = os.environ.get("AGENT6_CONTAINER_TEST_RUNTIME")
    if configured is None:
        pytest.skip(
            "set AGENT6_CONTAINER_TEST_RUNTIME to an approved Docker/Podman argv JSON array"
        )
    runtime = pydantic.TypeAdapter(tuple[str, ...]).validate_json(configured)
    assert runtime, "the runtime prefix must be explicit"
    root = pathlib.Path(
        os.environ.get("AGENT6_CONTAINER_TEST_ROOT", pathlib.Path(__file__).resolve().parents[2])
    ).resolve()
    return container.discover(root, container.ContainerOptions(runtime=runtime))


def _runtime(runtime: tuple[str, ...], *argv: str) -> str:
    result = subprocess.run(
        (*runtime, *argv), check=True, capture_output=True, text=True, timeout=60
    )
    return result.stdout.strip()


def _kill_client_tree(runtime: tuple[str, ...], pid: int) -> None:
    """Kill the runtime client and its descendants, through sudo when the prefix uses it.

    Under a sudo prefix the host client runs as root, so the test's own signals reach sudo
    alone and the orphaned client keeps the channel open.
    """
    pids = [pid]
    for parent in pids:
        with contextlib.suppress(OSError):
            children = pathlib.Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
            pids.extend(int(child) for child in children)
    if pathlib.Path(runtime[0]).name == "sudo":
        subprocess.run((runtime[0], "-n", "kill", "-9", *map(str, pids)), check=True, timeout=30)
        return
    for target in reversed(pids):
        with contextlib.suppress(ProcessLookupError):
            os.kill(target, signal.SIGKILL)


def test_existing_dev_container_then_owned_restart(
    runtime_discovery: container.ContainerDiscovery,
    static_launcher: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Inspect attached evidence before testing client loss and a test-owned restart."""
    discovery = runtime_discovery
    assert discovery.target.container_id is not None
    assert discovery.target.host_root != discovery.target.command_root
    runtime = discovery.runtime
    remote_env = dict(discovery.remote_env)
    remote_path = remote_env.get("PATH", "${containerEnv:PATH}")
    assert remote_path is not None, "the runtime smoke requires a target PATH"
    remote_env["PATH"] = "${containerEnv:TMPDIR}:" + remote_path
    session = container.prepare(
        dataclasses.replace(discovery, remote_env=tuple(sorted(remote_env.items())))
    )
    shells = background.BackgroundShells(tmp_path / "logs")
    try:
        target = session.target
        assert target is not None and target.lifecycle_ref is None
        assert target.container_id is not None and target.user is not None
        selected_uid, selected_gid = map(int, target.user.split(":"))
        assert selected_uid != 0
        variables = dict(target.environment.variables)
        result = session.run(("/bin/sh", "-c", 'printf "%s\\n" "$HOME" "$PATH" "$TMPDIR"'))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0
        home, path, temp = result.stdout.splitlines()
        assert home == target.home and path == variables["PATH"]
        assert temp.startswith("/tmp/a6.") and len(temp) < 40
        assert path.split(":")[0] == temp
        conftest.assert_target_only_executable(session, temp)

        try:
            view = shells.start(
                ("/bin/sh", "-c", "printf container-log"),
                lambda argv, rw: kinds.JailPolicy(
                    cwd=target.command_root, argv=argv, isolation="none"
                ),
                session=session,
            )
            view, output = shells.read(view.id, tail_lines=10, wait_s=10)
            assert view.returncode == 0 and "container-log" in output
            assert "container-log" in (shells.log_root / view.id / "out.log").read_text()
        finally:
            shells.stop_all()
        rendered = conftest.assert_execution_receipt(session)
        assert discovery.target.container_id in rendered
        print(rendered)
    finally:
        assert not session.close()
    assert (
        _runtime(runtime, "inspect", "--format", "{{.State.Running}}", target.container_id)
        == "true"
    )

    image = _runtime(runtime, "inspect", "--format", "{{.Image}}", target.container_id)
    token = secrets.token_hex(16)
    label = "agent6.container-session-test"
    command_root = pathlib.Path("/agent6-workspace")
    owned_id = _runtime(
        runtime,
        "create",
        "--pull=never",
        "--name",
        "agent6-test-" + token,
        "--label",
        label + "=" + token,
        "--network",
        "none",
        "--user",
        "0:0",
        "--env",
        "HOME=" + home,
        "--env",
        "PATH=" + path.removeprefix(temp + ":"),
        "--mount",
        f"type=bind,source={target.host_root},target={command_root}",
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        "while :; do sleep 1; done",
    )
    assert re.fullmatch(r"[0-9a-f]{64}", owned_id)
    assert owned_id != target.container_id

    def owned_operation(operation: str, *args: str) -> None:
        observed = _runtime(
            runtime,
            "inspect",
            "--format",
            '{{index .Config.Labels "agent6.container-session-test"}}',
            owned_id,
        )
        assert observed == token, "refuse lifecycle operations without the test's ownership label"
        _runtime(runtime, operation, *args, owned_id)

    try:
        owned_operation("start")
        default = _runtime(runtime, "exec", owned_id, "/bin/sh", "-ec", "id -u; id -g")
        assert default.splitlines() == ["0", "0"]
        selected = container.discover(
            target.host_root,
            container.ContainerOptions(
                exec=(*runtime, "exec", "-i", "--user", target.user, owned_id),
                workspace=command_root,
            ),
        )
        identity_session = container.prepare(selected)
        try:
            conftest.assert_selected_user(identity_session, selected_uid, selected_gid)
        finally:
            assert not identity_session.close()
        first = container.prepare(selected)
        try:
            handle = first.start_background(("/bin/sleep", "60"))
            client = first._proc  # pyright: ignore[reportPrivateUsage]
            _kill_client_tree(runtime, client.pid)
            client.wait(timeout=10)
            with pytest.raises(jail.JailUnavailableError, match="protocol failed"):
                first.status_background(handle)
        finally:
            assert "unknown" in first.close()
        assert _runtime(runtime, "inspect", "--format", "{{.State.Running}}", owned_id) == "true"

        before_restart = container.prepare(selected)
        try:
            assert before_restart.target is not None
            old_instance = before_restart.target.launcher_instance
            handle = before_restart.start_background(("/bin/sleep", "60"))
            owned_operation("restart")
            with pytest.raises(jail.JailUnavailableError, match="protocol failed"):
                before_restart.status_background(handle)
            replacement = container.prepare(selected)
            try:
                assert replacement.target is not None
                assert replacement.target.launcher_instance != old_instance
                status = replacement.status_background(handle)
                assert status.returncode is None and status.error == "unknown job handle"
            finally:
                assert not replacement.close()
        finally:
            assert "unknown" in before_restart.close()
    finally:
        owned_operation("rm", "--force")
