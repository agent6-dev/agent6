from __future__ import annotations

import os
import pathlib
import subprocess
import time

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_session


@pytest.mark.parametrize("constructor", ["open", "connect"])
@pytest.mark.parametrize(
    "isolation", ["none", "partial", pytest.param("full", marks=pytest.mark.needs_namespaces)]
)
def test_local_capture_ignores_inherited_tmpdir(
    tmp_path: pathlib.Path,
    host_tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    constructor: str,
    isolation: kinds.IsolationLevel,
) -> None:
    """Inherited TMPDIR, unwritable or hidden, cannot select capture storage."""
    assert os.geteuid() != 0
    inherited = host_tmp_path / "inherited-temp"
    inherited.mkdir(mode=0o500)
    assert not os.access(inherited, os.W_OK)
    monkeypatch.setenv("TMPDIR", str(inherited))
    policy = kinds.JailPolicy(cwd=tmp_path, argv=(), isolation=isolation, network="host")
    if constructor == "open":
        session = jail_session.JailSession.open(policy)
    else:
        proc = subprocess.Popen(
            (str(jail.require_jail_binary()),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"TMPDIR": str(inherited)},
        )
        target = kinds.ExecutionTarget(
            transport="local",
            host_root=tmp_path,
            command_root=tmp_path,
            user=None,
            home=None,
            environment=kinds.ExecutionEnvironment(()),
        )
        session = jail_session.JailSession.connect(proc, policy, target=target)
    try:
        expected = None if constructor == "open" else str(inherited)
        assert session.probe().environment.get("TMPDIR") == expected
        result = session.run(
            (
                "/bin/sh",
                "-ec",
                'if test -e "$1"; then echo visible; else echo hidden; fi; echo stderr >&2',
                "capture",
                str(inherited),
            )
        )
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
        assert result.stdout == ("hidden\n" if isolation == "full" else "visible\n")
        assert result.stderr == "stderr\n"
        handle = session.start_background(("/bin/sh", "-c", "printf captured; sleep 30"))
        deadline = time.monotonic() + 5
        while True:
            capture = session.read_background(handle, offset=0)
            if capture.data == b"captured" or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert capture.data == b"captured" and not capture.capture_error
        assert not session.stop_background(handle).error
    finally:
        cleanup = session.close()
    assert not cleanup
    assert not list(inherited.iterdir())
