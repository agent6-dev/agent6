# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Integration tests for the rootless egress broker.

The splice test needs no namespaces: the broker child stays in the host
network namespace and proxies a localhost TCP server, so it runs
anywhere. The isolation test is gated on unprivileged user namespaces and
runs in a forked subprocess because ``enter_network_isolation`` mutates
the calling process irreversibly.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from agent6.sandbox.broker import (
    BrokerHandle,
    EgressBrokerError,
    Endpoint,
    start_egress_broker,
)


def _userns_available() -> bool:
    res = subprocess.run(
        ["unshare", "-U", "-r", "true"],
        capture_output=True,
        check=False,
    )
    return res.returncode == 0


def _start_echo_server() -> tuple[socket.socket, int, threading.Thread]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]

    def serve() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    conn.sendall(data)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return srv, port, thread


def test_start_egress_broker_requires_endpoints(tmp_path: Path) -> None:
    with pytest.raises(EgressBrokerError):
        start_egress_broker([], sock_dir=tmp_path)


def test_broker_splices_to_fixed_upstream(tmp_path: Path) -> None:
    srv, port, _ = _start_echo_server()
    handle: BrokerHandle | None = None
    try:
        handle = start_egress_broker([Endpoint(host="127.0.0.1", port=port)], sock_dir=tmp_path)
        uds = handle.uds_for("127.0.0.1", port)
        assert uds is not None

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect(uds)
        client.sendall(b"hello broker")
        echoed = client.recv(4096)
        client.close()
        assert echoed == b"hello broker"
    finally:
        if handle is not None:
            handle.close()
        srv.close()


def test_broker_socket_is_private(tmp_path: Path) -> None:
    srv, port, _ = _start_echo_server()
    handle: BrokerHandle | None = None
    try:
        handle = start_egress_broker([Endpoint(host="127.0.0.1", port=port)], sock_dir=tmp_path)
        uds = handle.uds_for("127.0.0.1", port)
        assert uds is not None
        mode = Path(uds).stat().st_mode & 0o777
        assert mode == 0o600
    finally:
        if handle is not None:
            handle.close()
        srv.close()


def test_broker_handle_close_is_idempotent(tmp_path: Path) -> None:
    srv, port, _ = _start_echo_server()
    handle = start_egress_broker([Endpoint(host="127.0.0.1", port=port)], sock_dir=tmp_path)
    handle.close()
    handle.close()  # second close must not raise
    srv.close()


@pytest.mark.needs_namespaces
def test_network_isolation_blocks_egress(tmp_path: Path) -> None:
    if not _userns_available():
        pytest.skip("unprivileged user namespaces not available")

    # Run in a forked child: enter_network_isolation mutates the process
    # irreversibly, so it must not touch the pytest process.
    script = (
        "import socket, sys\n"
        "from agent6.sandbox.broker import enter_network_isolation\n"
        "enter_network_isolation()\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    print('CONNECTED')\n"
        "except OSError as exc:\n"
        "    print('BLOCKED')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "BLOCKED" in res.stdout, res.stderr
