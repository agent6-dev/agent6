from __future__ import annotations

import array
import concurrent.futures
import dataclasses
import fcntl
import os
import subprocess
import sys
import termios
import time
from typing import Any, cast

import pytest

from agent6 import kinds
from agent6.sandbox import container
from tests.sandbox import conftest


@pytest.mark.parametrize("name", ["architecture", "directory", "target_umask", "utility"])
def test_bootstrap_preserves_inherited_scratch_names(target: conftest.Target, name: str) -> None:
    """Bootstrap assignments must not overwrite exported target values or their substitutions."""
    value = "target-" + name + " ${HOME};$(false)"
    discovery = dataclasses.replace(
        target.discovery,
        exec=(*target.discovery.exec, name + "=" + value),
        remote_env=(("COPY", "${containerEnv:" + name + "}"),),
    )
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        assert session.probe().environment[name] == value
        result = session.run(("/usr/bin/printenv", name, "COPY"))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
        assert result.stdout == f"{value}\n{value}\n"


def test_stalled_bootstrap_transfer_times_out_and_closes_transport(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live peer holding a full unread pipe must not defeat startup's transfer deadline."""
    processes: list[subprocess.Popen[bytes]] = []
    start = subprocess.Popen

    def record(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        proc = cast("subprocess.Popen[bytes]", start(*args, **kwargs))
        assert proc.stdin is not None
        fcntl.fcntl(proc.stdin, fcntl.F_SETPIPE_SZ, 4096)
        processes.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", record)
    discovery = dataclasses.replace(
        target.discovery,
        exec=(sys.executable, "-c", "import signal; signal.pause()"),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        started = time.monotonic()
        pending = pool.submit(container.prepare, discovery)
        try:
            deadline = started + 3
            full = False
            while time.monotonic() < deadline and not full:
                if processes:
                    proc = processes[0]
                    assert proc.stdin is not None and not proc.stdin.closed
                    unread = array.array("i", [0])
                    fcntl.ioctl(proc.stdin, termios.FIONREAD, unread, True)
                    full = unread[0] == fcntl.fcntl(proc.stdin, fcntl.F_GETPIPE_SZ)
                if not full:
                    time.sleep(0.01)
            assert full, "the launcher transfer must fill the peer's unread stdin pipe"
            (proc,) = processes
            assert proc.poll() is None and not pending.done()
            with pytest.raises(container.ContainerError, match="bootstrap transfer timed out"):
                pending.result(timeout=12)
            assert time.monotonic() - started < 15
            assert proc.returncode is not None
            with pytest.raises(ChildProcessError):
                os.waitpid(proc.pid, os.WNOHANG)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                assert stream is not None and stream.closed
            assert not list(discovery.target.host_root.iterdir())
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=3)
            pending.exception(timeout=3)
            for proc in processes:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
