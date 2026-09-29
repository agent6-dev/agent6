# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec` and `agent6 forward`: the operator's way into a run's network.

A run's session network has no way in from outside, which is the point and also
the problem: the dev server the agent started is invisible to the person who
asked for it. These two commands are the door, and they are the operator's --
the model reaches none of this.
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

from agent6.config import Config
from agent6.sandbox import jail
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import dispatch
from agent6.ui.cli import net_cmds

pytestmark = pytest.mark.needs_namespaces

_PORT = 28411  # below the ephemeral range, like test_session_network's


def _serving(
    tmp_path: pathlib.Path, port: int, session_id: str
) -> tuple[dispatch.ToolDispatcher, jail.SessionNetwork, sessions_layout.SessionLayout]:
    """A live session holding a dev server on the port, laid out as a run so the CLI finds it."""
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id=session_id, subdir="runs")
    session_dir = layout.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    ipc.write_worker_pid(session_dir, os.getpid())  # exec joins a live run only
    net = jail.SessionNetwork.open()
    ipc.write_session_netns_pid(session_dir, net.holder_pid)
    dispatcher = dispatch.ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"run_commands": "yes"}}),
        isolation="strict",
        session_dir=session_dir,
        use_jail_session=True,
        session_net=net,
    )
    serve = (
        "import http.server,socketserver;"
        f"socketserver.TCPServer(('127.0.0.1',{port}),"
        "http.server.SimpleHTTPRequestHandler).serve_forever()"
    )
    dispatcher.dispatch(
        "run_command", {"argv": ["/usr/bin/python3", "-c", serve], "background": True}
    )
    time.sleep(2.0)
    return dispatcher, net, layout


def test_the_ports_a_run_serves_are_visible_only_from_inside(tmp_path: pathlib.Path) -> None:
    """Listing the run's ports happens in the run's network, and stops working when the run ends.

    Nothing on this machine can see the dev server.
    """
    dispatcher, net, layout = _serving(tmp_path, _PORT, "serving-1")
    try:
        assert ipc.listening_ports(layout.session_dir) == [_PORT]
        with pytest.raises(OSError):  # not reachable from this machine
            socket.create_connection(("127.0.0.1", _PORT), timeout=2).close()
    finally:
        dispatcher.close()
        net.close()
    assert ipc.listening_ports(layout.session_dir) == [], "the network outlived the run"


def test_exec_runs_where_the_agent_runs(tmp_path: pathlib.Path) -> None:
    """The whole claim of `agent6 exec`: what you see is what the agent sees."""
    dispatcher, net, layout = _serving(tmp_path, _PORT + 1, "exec-1")
    try:
        code = net_cmds.exec_in_session(
            layout,
            Config.model_validate({"sandbox": {"run_commands": "yes"}}),
            tmp_path,
            (
                "/usr/bin/python3",
                "-c",
                "import urllib.request;urllib.request.urlopen("
                f"'http://127.0.0.1:{_PORT + 1}/', timeout=4)",
            ),
        )
        assert code == 0, "exec could not reach the run's dev server"
    finally:
        dispatcher.close()
        net.close()


def test_joining_a_session_without_a_network_says_why(tmp_path: pathlib.Path) -> None:
    """A run on the host network has nothing to join; the refusal names the setting, no errno."""
    with pytest.raises(net_cmds.SessionNetworkUnavailableError, match=r"sandbox\.network"):
        net_cmds.join_session_network(tmp_path)


def test_forward_bridges_a_port_to_this_machine(tmp_path: pathlib.Path) -> None:
    """The dev-server ergonomic: a plain client on this machine reaches a server inside the run."""
    dispatcher, net, layout = _serving(tmp_path, _PORT + 2, "fwd-1")
    local = _PORT + 100
    bridge = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys;sys.path.insert(0,'src');"
            "from pathlib import Path;"
            "from agent6.sessions.layout import SessionLayout;"
            "from agent6.ui.cli.net_cmds import forward;"
            f"lay=SessionLayout(state_dir=Path({str(tmp_path)!r}),"
            f" session_id={layout.session_id!r}, subdir='runs');"
            f"forward(lay, {_PORT + 2}, {local})",
        ],
        cwd=pathlib.Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 20
        body = b""
        while time.time() < deadline and not body:
            try:
                with socket.create_connection(("127.0.0.1", local), timeout=2) as sock:
                    sock.sendall(b"GET / HTTP/1.0\r\n\r\n")
                    body = sock.recv(64)
            except OSError:
                time.sleep(0.5)
        assert b"HTTP/1.0 200" in body, f"the bridge served nothing: {body!r}"
    finally:
        bridge.kill()
        bridge.wait(timeout=10)
        dispatcher.close()
        net.close()


def test_forward_refuses_a_run_with_no_network_instead_of_waiting(tmp_path: pathlib.Path) -> None:
    """A bridge to nowhere says so before it looks like a bridge.

    The join happens in the per-connection child, so binding the local port and blocking in
    accept() would drop each connection in silence until someone tried to use it.
    """
    import io

    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="no-net", subdir="runs")
    layout.session_dir.mkdir(parents=True)
    out = io.StringIO()
    assert net_cmds.forward(layout, 3000, 3000, out=out) == 2
    assert "network" in out.getvalue()  # the refusal names the missing network


def test_forward_stops_when_its_session_ends(tmp_path: pathlib.Path) -> None:
    """A bridge does not outlive its run.

    Left accepting connections and dropping them, it reads as a broken server rather than a
    finished session (`ss` still shows the listener, curl gets nothing).
    """
    import io
    import threading

    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path, session_id="ends-mid-forward", subdir="runs"
    )
    layout.session_dir.mkdir(parents=True)
    net = jail.SessionNetwork.open()
    ipc.write_session_netns_pid(layout.session_dir, net.holder_pid)
    out = io.StringIO()
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(net_cmds.forward(layout, _PORT + 6, _PORT + 106, out=out)),
        daemon=True,
    )
    thread.start()
    time.sleep(1.0)
    assert thread.is_alive(), "the bridge should still be waiting while the run lives"
    net.close()
    ipc.clear_session_netns_pid(layout.session_dir)  # the run's teardown
    thread.join(timeout=15)
    assert not thread.is_alive(), "the bridge outlived its session"
    assert result == [0] and "ended" in out.getvalue(), out.getvalue()
