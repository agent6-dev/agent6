# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rootless provider-only egress broker.

Implements the host-level egress allow-list promised by
``[sandbox] agent_network = "providers"`` without root, nftables, or any
external package.

Design (strict profile only — requires unprivileged user namespaces):

1. While still in the host network namespace, the agent process binds
   one listening ``AF_UNIX`` socket per allow-listed provider endpoint,
   then ``fork()``s. The child becomes the **broker**: it stays in the
   host netns and, for each connection accepted on a given socket,
   dials the single fixed ``host:port`` that socket represents and
   blind-splices bytes in both directions. The target is bound to the
   socket, never chosen by the (untrusted) agent at connect time, so the
   agent can reach *only* the pre-approved endpoints — the allow-list is
   enforced structurally, not by parsing what the agent sends.

2. The parent (agent) then ``unshare(CLONE_NEWUSER | CLONE_NEWNET)``s
   into a fresh, empty network namespace. It now has no IP route of its
   own; its only egress is the broker's unix sockets, which keep working
   because ``AF_UNIX`` pathname sockets are not namespaced by the network
   namespace.

TLS stays end-to-end: the broker only ever sees ciphertext, so it cannot
intercept or tamper with provider traffic. DNS resolution happens in the
broker per-connect, so the allow-list is robust to CDN IP rotation.

Security review note: this is a network-confinement boundary. The broker
process is trusted (it is our code, forked from the agent before the
agent drops its network access) and can only connect to the fixed
endpoints captured at startup. The kernel network namespace is the
enforcement mechanism; the unix sockets are merely the sanctioned
channel out. A bug that failed to register a route fails closed (the
agent cannot connect at all) rather than open.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import selectors
import signal
import socket
import struct
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# The egress broker is Linux-only (user/network namespaces + SIOCSIFFLAGS
# ioctl on the loopback device). `fcntl` does not exist on Windows; guard the
# import so the module merely *imports* everywhere (sandbox/__init__ re-exports
# it). Every function that touches `fcntl` is gated behind os.unshare, which is
# itself Linux-only, so the name is always bound where it is actually used.
if sys.platform == "linux":
    import fcntl

_RECV_CHUNK = 65536
_UPSTREAM_CONNECT_TIMEOUT_S = 30.0


class EgressBrokerError(RuntimeError):
    """The provider-only egress broker could not be established."""


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A single ``host:port`` the agent is permitted to reach."""

    host: str
    port: int


@dataclass(frozen=True, slots=True)
class BrokerHandle:
    """Handle to a running broker child process.

    ``routes`` maps each allow-listed endpoint to the unix-socket path the
    agent must dial to reach it.
    """

    pid: int
    routes: tuple[tuple[Endpoint, str], ...]

    def uds_for(self, host: str, port: int) -> str | None:
        for ep, path in self.routes:
            if ep.host == host and ep.port == port:
                return path
        return None

    def close(self) -> None:
        """Terminate the broker and reap it. Idempotent."""
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGTERM)
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(self.pid, 0)


def start_egress_broker(endpoints: Iterable[Endpoint], *, sock_dir: Path) -> BrokerHandle:
    """Fork a broker process serving one unix socket per endpoint.

    Must be called while the process is still in the host network
    namespace (i.e. *before* :func:`enter_network_isolation`) and while
    the process is single-threaded. ``sock_dir`` must already exist and
    be private to this run.
    """
    ordered = sorted(set(endpoints), key=lambda e: (e.host, e.port))
    if not ordered:
        raise EgressBrokerError("no provider endpoints to broker")

    listeners: list[tuple[Endpoint, socket.socket]] = []
    routes: list[tuple[Endpoint, str]] = []
    for index, ep in enumerate(ordered):
        path = sock_dir / f"egress-{index}.sock"
        ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ls.bind(str(path))
        path.chmod(0o600)
        ls.listen(64)
        listeners.append((ep, ls))
        routes.append((ep, str(path)))

    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - exercised in a child process
        try:
            _run_broker_child(listeners)
        finally:
            os._exit(0)

    # Parent: the broker owns the listening sockets now.
    for _, ls in listeners:
        ls.close()
    return BrokerHandle(pid=pid, routes=tuple(routes))


def enter_network_isolation() -> None:
    """Move the *current* process into a fresh, empty network namespace.

    Uses an unprivileged user namespace to obtain the capability to
    create the network namespace, then writes identity uid/gid maps so
    file ownership and the effective uid are unchanged. After this call
    the process (and any children it spawns) has no IP connectivity
    except via already-open ``AF_UNIX`` sockets.

    The process must be single-threaded (a kernel requirement for
    ``unshare(CLONE_NEWUSER)``). Raises :class:`EgressBrokerError` on any
    failure so the caller fails closed rather than running unconfined.
    """
    unshare = getattr(os, "unshare", None)
    clone_newuser = getattr(os, "CLONE_NEWUSER", None)
    clone_newnet = getattr(os, "CLONE_NEWNET", None)
    if unshare is None or clone_newuser is None or clone_newnet is None:
        raise EgressBrokerError(
            "network isolation requires os.unshare + CLONE_NEW* (Python 3.12+ on Linux)"
        )
    uid = os.getuid()
    gid = os.getgid()
    try:
        unshare(clone_newuser | clone_newnet)
    except OSError as exc:
        raise EgressBrokerError(f"failed to create user+network namespace: {exc}") from exc
    try:
        Path("/proc/self/uid_map").write_text(f"{uid} {uid} 1\n", encoding="ascii")
        Path("/proc/self/setgroups").write_text("deny\n", encoding="ascii")
        Path("/proc/self/gid_map").write_text(f"{gid} {gid} 1\n", encoding="ascii")
    except OSError as exc:
        raise EgressBrokerError(f"failed to write namespace id maps: {exc}") from exc
    _bring_up_loopback()


# --------------------------------------------------------------------------
# Broker child internals
# --------------------------------------------------------------------------


def _run_broker_child(listeners: list[tuple[Endpoint, socket.socket]]) -> None:  # pragma: no cover
    _set_parent_death_signal()

    def _exit(_signum: int, _frame: object) -> None:
        os._exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    keep = {ls.fileno() for _, ls in listeners}
    _close_inherited_fds(keep)

    sel = selectors.DefaultSelector()
    for ep, ls in listeners:
        ls.setblocking(False)
        sel.register(ls, selectors.EVENT_READ, ep)
    while True:
        for key, _ in sel.select():
            listen_sock = key.fileobj
            endpoint = key.data
            assert isinstance(listen_sock, socket.socket)
            assert isinstance(endpoint, Endpoint)
            try:
                conn, _addr = listen_sock.accept()
            except OSError:
                continue
            threading.Thread(
                target=_handle_connection,
                args=(conn, endpoint),
                daemon=True,
            ).start()


def _handle_connection(client: socket.socket, endpoint: Endpoint) -> None:  # pragma: no cover
    try:
        upstream = socket.create_connection(
            (endpoint.host, endpoint.port), timeout=_UPSTREAM_CONNECT_TIMEOUT_S
        )
    except OSError:
        client.close()
        return
    try:
        _splice_bidirectional(client, upstream)
    finally:
        with contextlib.suppress(OSError):
            client.close()
        with contextlib.suppress(OSError):
            upstream.close()


def _splice_bidirectional(a: socket.socket, b: socket.socket) -> None:  # pragma: no cover
    t1 = threading.Thread(target=_pump_one_way, args=(a, b), daemon=True)
    t2 = threading.Thread(target=_pump_one_way, args=(b, a), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def _pump_one_way(src: socket.socket, dst: socket.socket) -> None:  # pragma: no cover
    try:
        while True:
            data = src.recv(_RECV_CHUNK)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


def _set_parent_death_signal() -> None:  # pragma: no cover
    """Ask the kernel to SIGTERM us if the agent process dies."""
    pr_set_pdeathsig = 1
    with contextlib.suppress(OSError):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)


def _close_inherited_fds(keep: set[int]) -> None:  # pragma: no cover
    """Close fds inherited from the agent except the listening sockets."""
    try:
        entries = list(Path("/proc/self/fd").iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            fd = int(entry.name)
        except ValueError:
            continue
        if fd <= 2 or fd in keep:
            continue
        with contextlib.suppress(OSError):
            os.close(fd)


def _bring_up_loopback() -> None:
    """Best-effort ``ip link set lo up`` for the new netns.

    Provider egress does not need loopback (it goes through the broker),
    but bringing ``lo`` up keeps any localhost-based IPC working. We hold
    CAP_NET_ADMIN in the freshly created namespace, so this normally
    succeeds; failures are non-fatal.
    """
    siocsifflags = 0x8914
    iff_up = 0x1
    iff_running = 0x40
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return
    try:
        # struct ifreq: 16-byte name + short flags, padded to 40 bytes.
        ifr = struct.pack("16sh22x", b"lo", iff_up | iff_running)
        fcntl.ioctl(sock, siocsifflags, ifr)
    except OSError:
        pass
    finally:
        sock.close()
