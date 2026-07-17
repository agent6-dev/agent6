# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The pre-forked host spawner behind the detach path.

The unit tests need no namespaces: the helper's contract (narrow argv, ack
protocol, idempotent close) is the same everywhere. The isolation test proves
the point of the helper: after ``enter_network_isolation`` the process cannot
reach loopback, but a spawn through the helper can. It is gated on
unprivileged user namespaces and runs in a forked subprocess because
isolation mutates the calling process irreversibly.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agent6.sandbox.host_spawn import fork_host_spawner

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)

# Stands in for the agent6 executable: writes its argv tail + the marker env
# var into cwd so the test can assert what the helper launched, and where.
_FAKE_AGENT6 = (
    "import os, sys, pathlib; "
    "pathlib.Path('spawned.txt').write_text("
    "' '.join(sys.argv[1:]) + '|' + os.environ.get('AGENT6_STREAM_TO_LOG', ''))"
)

# Same, but also records AGENT6_SUBRUN so a lane spawn can assert its env extras
# were merged over the helper's isolation-free base env.
_FAKE_LANE = (
    "import os, sys, pathlib; "
    "pathlib.Path('lane.txt').write_text("
    "' '.join(sys.argv[1:]) + '|' + os.environ.get('AGENT6_SUBRUN', '')"
    " + '|' + os.environ.get('AGENT6_STREAM_TO_LOG', ''))"
)


def _wait_for(path: Path, timeout_s: float = 10.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_spawn_resume_round_trip(tmp_path: Path) -> None:
    spawner = fork_host_spawner([sys.executable, "-c", _FAKE_AGENT6])
    try:
        assert spawner.spawn_resume(tmp_path, "RID-1") == ""
        content = _wait_for(tmp_path / "spawned.txt")
        argv_tail, stream_to_log = content.split("|")
        assert argv_tail == "resume RID-1"
        assert stream_to_log == "1"
    finally:
        spawner.close()


def test_spawn_lane_round_trip(tmp_path: Path) -> None:
    """A coordinator lane spawn: the helper prepends its trusted exe prefix, passes
    the exe-less lane argv verbatim, merges the request's env extras, AND keeps its
    isolation-free base env (AGENT6_STREAM_TO_LOG, set at fork)."""
    spawner = fork_host_spawner([sys.executable, "-c", _FAKE_LANE])
    try:
        err = spawner.spawn_lane(
            tmp_path, ["run", "--run-id", "L1", "--", "do the task"], {"AGENT6_SUBRUN": "1"}
        )
        assert err == ""
        argv_tail, subrun, stream = _wait_for(tmp_path / "lane.txt").split("|")
        assert argv_tail == "run --run-id L1 -- do the task"  # exe-less argv, `--` intact
        assert subrun == "1"  # request env extra merged
        assert stream == "1"  # helper's base env preserved
    finally:
        spawner.close()


def test_spawn_lane_error_is_reported(tmp_path: Path) -> None:
    spawner = fork_host_spawner(["/nonexistent/agent6"])
    try:
        err = spawner.spawn_lane(tmp_path, ["run", "--", "x"], {})
        assert "background lane failed to start" in err
    finally:
        spawner.close()


def test_spawn_error_is_reported(tmp_path: Path) -> None:
    spawner = fork_host_spawner(["/nonexistent/agent6"])
    try:
        err = spawner.spawn_resume(tmp_path, "RID-2")
        assert "background resume failed to start" in err
    finally:
        spawner.close()


def test_close_is_idempotent_and_spawn_after_close_errors(tmp_path: Path) -> None:
    spawner = fork_host_spawner([sys.executable, "-c", _FAKE_AGENT6])
    spawner.close()
    spawner.close()
    assert "detach helper" in spawner.spawn_resume(tmp_path, "RID-3")


def _userns_available() -> bool:
    res = subprocess.run(["unshare", "-U", "-r", "true"], capture_output=True, check=False)
    return res.returncode == 0


@pytest.mark.skipif(not _userns_available(), reason="unprivileged user namespaces unavailable")
def test_spawner_escapes_network_isolation(tmp_path: Path) -> None:
    """From inside the empty namespace loopback is unreachable, but a spawn
    through the pre-forked helper runs with host networking (the detach fix)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(20.0)
    port = listener.getsockname()[1]

    connect_snippet = (
        "import socket, sys; "
        f"s = socket.create_connection(('127.0.0.1', {port}), timeout=10); "
        "s.sendall(' '.join(sys.argv[1:]).encode()); s.close()"
    )
    child_script = textwrap.dedent(
        f"""
        import socket, sys
        from agent6.sandbox.broker import enter_network_isolation
        from agent6.sandbox.host_spawn import fork_host_spawner

        spawner = fork_host_spawner([{sys.executable!r}, "-c", {connect_snippet!r}])
        enter_network_isolation()
        try:
            socket.create_connection(("127.0.0.1", {port}), timeout=2)
        except OSError:
            pass  # expected: the namespace has no route to the host loopback
        else:
            sys.exit(3)  # isolation did not isolate; the test is meaningless
        err = spawner.spawn_resume({str(tmp_path)!r}, "RID-ISO")
        spawner.close()
        sys.exit(0 if err == "" else 4)
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", child_script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        )
        assert proc.returncode == 0, f"child failed rc={proc.returncode}: {proc.stderr[-800:]}"
        conn, _ = listener.accept()
        with conn:
            assert conn.recv(4096) == b"resume RID-ISO"
    finally:
        listener.close()
