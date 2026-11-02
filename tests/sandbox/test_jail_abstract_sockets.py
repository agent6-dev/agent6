# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A jailed command cannot reach a host daemon's abstract unix socket."""

from __future__ import annotations

import pathlib
import shutil
import socket
import threading
import uuid
from collections.abc import Iterator

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import detect, jail
from agent6.tools import policy

_CONNECT = (
    "import socket, sys\n"
    "s = socket.socket(socket.AF_UNIX)\n"
    "s.connect(b'\\0' + sys.argv[1].encode())\n"
    "print('CONNECTED')\n"
)


@pytest.fixture
def listener() -> Iterator[str]:
    """An abstract unix socket this process serves, under a name no other test holds."""
    name = f"agent6-test-{uuid.uuid4().hex[:12]}"
    server = socket.socket(socket.AF_UNIX)
    server.bind(b"\0" + name.encode())
    server.listen(1)

    def accept() -> None:
        try:
            while True:
                conn, _ = server.accept()
                conn.close()
        except OSError:
            return

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield name
    server.shutdown(socket.SHUT_RDWR)
    server.close()


def _connect(
    root: pathlib.Path, level: kinds.IsolationLevel, network: kinds.NetworkMode, name: str
):
    python = shutil.which("python3", path="/usr/bin:/bin")
    if python is None:
        pytest.skip("no /usr/bin/python3 for the jailed client")
    argv = (python, "-c", _CONNECT, name)
    built = policy.jail_policy(root, Config(), level, argv, network=network)
    return jail.run_in_jail(built)


def _needs_scope() -> None:
    if detect.probe_landlock_abi() < 6:
        pytest.skip("Landlock ABI 6 scopes abstract sockets; this kernel is older")


def test_partial_cannot_connect_to_a_host_abstract_socket(
    tmp_path: pathlib.Path, listener: str
) -> None:
    """`partial` shares the host's network namespace; the Landlock scope refuses the connect."""
    _needs_scope()
    res = _connect(tmp_path, "partial", "host", listener)
    assert res.returncode != 0 and "CONNECTED" not in res.stdout, res
    assert "Operation not permitted" in res.stderr, res.stderr


@pytest.mark.needs_namespaces
def test_full_on_the_host_network_cannot_connect_to_a_host_abstract_socket(
    tmp_path: pathlib.Path, listener: str
) -> None:
    """On the host network the scope is what keeps the daemon out of reach."""
    _needs_scope()
    res = _connect(tmp_path, "full", "host", listener)
    assert res.returncode != 0 and "CONNECTED" not in res.stdout, res
    assert "Operation not permitted" in res.stderr, res.stderr


@pytest.mark.needs_namespaces
def test_full_off_the_host_network_has_no_such_socket(
    tmp_path: pathlib.Path, listener: str
) -> None:
    """Off the host network the socket is in another namespace: refused, not scoped."""
    res = _connect(tmp_path, "full", "none", listener)
    assert res.returncode != 0 and "Connection refused" in res.stderr, res
