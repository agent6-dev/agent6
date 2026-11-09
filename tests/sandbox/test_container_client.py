from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import pathlib
import secrets
import select
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Generator

import pytest

from agent6.sandbox import jail
from tests.sandbox import conftest


@pytest.fixture
def runtime_target(
    target: conftest.Target,
) -> Generator[tuple[conftest.Target, list[subprocess.Popen[bytes]]]]:
    """Keep the supervisor's stdin and stdout alive independently of the runtime client."""
    release = threading.Event()
    supervisors: list[subprocess.Popen[bytes]] = []
    address = "a6-test-" + secrets.token_hex(16)
    with socket.socket(socket.AF_UNIX) as listener, concurrent.futures.ThreadPoolExecutor() as pool:
        listener.bind("\0" + address)
        listener.listen(1)
        listener.settimeout(10)

        def serve() -> None:
            with listener.accept()[0] as connection:
                message, descriptors, flags, _ = socket.recv_fds(connection, 1, 1)
                assert message == b"F" and not flags
                (stdout,) = descriptors
                with (
                    os.fdopen(stdout, "wb", buffering=0) as output,
                    connection.makefile("rb", buffering=0) as wire,
                ):
                    argv = json.loads(wire.readline())
                    with subprocess.Popen(
                        argv,
                        stdin=subprocess.PIPE,
                        stdout=output,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    ) as supervisor:
                        supervisors.append(supervisor)
                        assert supervisor.stdin is not None
                        try:
                            while chunk := wire.read(65536):
                                supervisor.stdin.write(chunk)
                                supervisor.stdin.flush()
                            release.wait()
                        finally:
                            supervisor.stdin.close()
                            try:
                                supervisor.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                supervisor.kill()
                                supervisor.wait(timeout=5)
                                raise

        server = pool.submit(serve)
        client = (
            "import json,os,socket,sys,threading\n"
            "connection=socket.socket(socket.AF_UNIX)\n"
            "connection.connect('\\0'+sys.argv[1])\n"
            "socket.send_fds(connection,[b'F'],[1])\n"
            "connection.sendall((json.dumps(sys.argv[2:])+'\\n').encode())\n"
            "def feed():\n"
            "    while data:=os.read(0,65536): connection.sendall(data)\n"
            "    connection.shutdown(socket.SHUT_WR)\n"
            "threading.Thread(target=feed,daemon=True).start()\n"
            "assert not connection.recv(1)\n"
        )
        prefix = (sys.executable, "-u", "-c", client, address, *target.discovery.exec)
        discovery = dataclasses.replace(target.discovery, exec=prefix)
        try:
            yield dataclasses.replace(target, discovery=discovery), supervisors
        finally:
            release.set()
            server.result(timeout=15)


def test_dead_runtime_client_never_falls_back(
    runtime_target: tuple[conftest.Target, list[subprocess.Popen[bytes]]],
) -> None:
    """A lost client cannot confirm job completion even while the remote job keeps running."""
    target, supervisors = runtime_target
    trigger = target.home / "continue"
    survived = target.home / "survived"
    with conftest.opened(target) as session:
        (supervisor,) = supervisors
        handle = session.start_background(
            (
                "/bin/sh",
                "-c",
                'while ! test -f "$1"; do sleep 0.02; done; : > "$2"',
                "job",
                str(trigger),
                str(survived),
            )
        )
        proc = session._proc  # pyright: ignore[reportPrivateUsage]
        assert proc.pid != supervisor.pid
        assert pathlib.Path(f"/proc/{proc.pid}/comm").read_text().strip() != "launcher"
        proc.kill()
        proc.wait(timeout=5)
        assert supervisor.poll() is None
        trigger.touch()
        deadline = time.monotonic() + 5
        while not survived.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert survived.exists()
        with pytest.raises(jail.JailUnavailableError, match="protocol failed"):
            session.status_background(handle)
        assert "unknown" in session.close()
        assert supervisor.poll() is None


def test_inflight_client_death_without_stdout_eof(
    runtime_target: tuple[conftest.Target, list[subprocess.Popen[bytes]]],
) -> None:
    """Client death fails an in-flight request even when the supervisor keeps stdout open."""
    target, supervisors = runtime_target
    started = target.home / "started"
    finish = target.home / "finish"
    with conftest.opened(target) as session:
        (supervisor,) = supervisors
        proc = session._proc  # pyright: ignore[reportPrivateUsage]
        assert proc.stdout is not None and proc.pid != supervisor.pid
        with (
            os.fdopen(os.dup(proc.stdout.fileno()), "rb", buffering=0) as stdout,
            concurrent.futures.ThreadPoolExecutor() as pool,
        ):
            pending = pool.submit(
                session.run,
                (
                    "/bin/sh",
                    "-ec",
                    ': > "$1"; while ! test -f "$2"; do sleep 0.02; done',
                    "job",
                    str(started),
                    str(finish),
                ),
                timeout_s=10,
            )
            try:
                deadline = time.monotonic() + 5
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert started.exists() and not pending.done()
                proc.kill()
                proc.wait(timeout=5)
                assert not select.select([stdout], [], [], 0)[0]
                with pytest.raises(
                    jail.JailUnavailableError, match=r"protocol failed.*transport exited"
                ):
                    pending.result(timeout=3)
                assert supervisor.poll() is None
                assert not select.select([stdout], [], [], 0)[0]
                assert "unknown" in session.close()
            finally:
                finish.touch()
