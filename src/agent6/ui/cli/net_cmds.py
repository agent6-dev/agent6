# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec` and `agent6 forward`: reach into a live run's session network.

A run's commands share one network with no route off the box, so nothing outside the
run can reach a server the agent started, the operator included. These commands are the
way in, and they are the operator's, never the model's: `exec` runs a command the way
the agent would, `forward` bridges one of the run's ports to a port on this machine.

A join goes through the holder pid the run publishes (`netns.pid`). `forward` always
joins; `exec` joins only when the run's own commands took the session network, so a
`host` stamp keeps it on this machine's network even while the run holds a netns for a
scoped MCP server. Entering a network namespace needs capabilities in the user namespace
that owns it, so each join takes that one first, as the launcher does.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import selectors
import socket
import sys
from typing import TextIO

from agent6 import kinds
from agent6.app import _setup
from agent6.config import Config
from agent6.sandbox import detect, jail
from agent6.sessions import ipc, manifest
from agent6.sessions import layout as sessions_layout
from agent6.tools import policy as tools_policy
from agent6.ui.cli import _common
from agent6.viewmodel import session_is_live, summarize_session_dir

# The namespaces to enter; the `os` flags are looked up at join time so the module imports anywhere.
_JOIN_ORDER = (("user", "CLONE_NEWUSER"), ("net", "CLONE_NEWNET"))


class SessionNetworkUnavailableError(Exception):
    """The run has no session network to join, and why."""


def join_session_network(session_dir: pathlib.Path) -> None:
    """Put this process in the run's session network.

    Irreversible: nothing here ever leaves a namespace it entered.

    Args:
        session_dir: The run's session dir.

    Raises:
        SessionNetworkUnavailableError: The run publishes no holder, the host is not Linux, or
            the namespaces could not be entered.
    """
    pid = ipc.read_session_netns_pid(session_dir)
    if pid is None:
        raise SessionNetworkUnavailableError(
            "this session has no network of its own to join. A run only makes one"
            " under the strict isolation with sandbox.network = auto|session;"
            " with network = host its commands are already on this machine's."
        )
    setns = getattr(os, "setns", None)
    if setns is None:
        raise SessionNetworkUnavailableError("joining a session's network needs Linux")
    for kind, flag_name in _JOIN_ORDER:
        flag: int = getattr(os, flag_name)
        try:
            fd = os.open(f"/proc/{pid}/ns/{kind}", os.O_RDONLY)
        except OSError as exc:
            raise SessionNetworkUnavailableError(f"the session's network is gone: {exc}") from exc
        try:
            setns(fd, flag)
        except OSError as exc:
            raise SessionNetworkUnavailableError(
                f"could not join the session's {kind} namespace: {exc}"
            ) from exc
        finally:
            os.close(fd)


def _pump(a: socket.socket, b: socket.socket) -> None:
    """Shuttle bytes both ways until either side hangs up."""
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ, b)
    sel.register(b, selectors.EVENT_READ, a)
    try:
        while True:
            for key, _ in sel.select():
                src, dst = key.fileobj, key.data
                assert isinstance(src, socket.socket)
                chunk = src.recv(65536)
                if not chunk:
                    return
                dst.sendall(chunk)
    except OSError:
        return
    finally:
        sel.close()


def no_session_network_reason(layout: sessions_layout.SessionLayout) -> str:
    """Return why `forward` finds no session network to reach into.

    The run is not live (its network lives only while it does), or a live run made none
    (host network, or an isolation short of strict).
    """
    if not session_is_live(layout.session_dir):
        word = summarize_session_dir(layout.session_dir).status
        return f"{layout.session_id} is {word}; a session network exists only while its run does."
    return (
        f"{layout.session_id} has no network of its own to reach into. A run only makes one"
        " under strict isolation, for commands with sandbox.network other than host or for"
        " an MCP server scoped to the session; otherwise its commands are already on this"
        " machine's."
    )


def forward(
    layout: sessions_layout.SessionLayout,
    remote_port: int,
    local_port: int | None,
    out: TextIO = sys.stderr,
) -> int:
    """Bridge a port inside the run to a port on this machine.

    One forked child per connection: it joins the run's network and connects there, then
    shuttles bytes over the socket it inherited. A child cannot come back out of a namespace,
    and the parent must stay outside to keep accepting, so the fork is the bridge.

    Args:
        layout: The run.
        remote_port: The port inside the run.
        local_port: The port on this machine; None takes the same number, 0 asks for a free one.
        out: Where the status lines go.

    Returns:
        The exit code; 2 when there is no network to join or the bind fails.
    """
    # Refuse before binding: the join happens per connection, in the child.
    if ipc.read_session_netns_pid(layout.session_dir) is None:
        print(f"REFUSING: {no_session_network_reason(layout)}", file=out)
        return 2
    # The same number on both sides unless told otherwise, as `kubectl port-forward 3000` means.
    if local_port is None:
        local_port = remote_port
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", local_port))
        except (OSError, OverflowError) as exc:
            print(
                f"ERROR: cannot listen on 127.0.0.1:{local_port}: {exc}."
                " Pick another with --local-port.",
                file=out,
            )
            return 2
        listener.listen(16)
        # Wake between connections: a bridge outliving its run reads as a broken server.
        listener.settimeout(2.0)
        bound = listener.getsockname()[1]
        print(
            f"[agent6] forwarding http://127.0.0.1:{bound} -> port {remote_port} inside"
            f" {layout.session_id}. Ctrl-C to stop.",
            file=out,
        )
        try:
            while True:
                with contextlib.suppress(ChildProcessError):  # reap finished bridges
                    while os.waitpid(-1, os.WNOHANG)[0]:
                        pass
                try:
                    conn, _ = listener.accept()
                except TimeoutError:
                    if ipc.read_session_netns_pid(layout.session_dir) is None:
                        print(
                            f"[agent6] {layout.session_id} ended; nothing left to reach.",
                            file=out,
                        )
                        return 0
                    continue
                try:
                    child = os.fork()
                except OSError as exc:
                    # A process cap on this machine, not a broken bridge: drop the one connection.
                    print(f"[agent6] could not fork a bridge for a connection: {exc}", file=out)
                    conn.close()
                    continue
                if child == 0:  # pragma: no cover - one process per connection
                    listener.close()
                    code = 0
                    try:
                        join_session_network(layout.session_dir)
                        inside = socket.create_connection(("127.0.0.1", remote_port), timeout=10)
                        inside.settimeout(None)  # the 10 s bounds the connect, not the bridge
                        _pump(conn, inside)
                    except (SessionNetworkUnavailableError, OSError):
                        code = 1
                    finally:
                        conn.close()
                    os._exit(code)
                conn.close()
        except KeyboardInterrupt:
            return 0


def _stamped_policy(
    layout: sessions_layout.SessionLayout,
) -> tuple[str, kinds.NetworkMode | None] | None:
    """Return the run's recorded `(isolation, network)`.

    Returns:
        The stamp, or None when unreadable or unstamped; an unknown network word reads as
        unset rather than guessed.
    """
    try:
        stamp = manifest.read_manifest(layout.session_dir).policy
    except manifest.ManifestError:
        return None
    if stamp.isolation not in ("strict", "hardened", "none"):
        return None
    # "auto" and "" (unstamped) read as None: jail_policy applies its own auto semantics.
    network: kinds.NetworkMode | None = (
        stamp.network if stamp.network in ("host", "session", "none") else None
    )
    return stamp.isolation, network


def exec_in_session(
    layout: sessions_layout.SessionLayout, cfg: Config, cwd: pathlib.Path, argv: tuple[str, ...]
) -> int:
    """Run a command the way the run's own commands run: same jail, same network.

    The operator's command, not the model's, so it is neither approved nor logged as a tool
    call; confined identically, so what you see is what the agent sees. The output prints
    when the command ends, and a Ctrl-C ends it with none. Unbounded (`timeout_s=0.0`), so
    the policy's default timeout cannot cut a slow probe. A server is the agent's to start
    and the operator's to reach with `agent6 forward`.

    Args:
        layout: The run.
        cfg: The effective config, the fallback when the run is unstamped.
        cwd: The workspace.
        argv: The command.

    Returns:
        The command's exit code; 2 when the run is not live or its network is gone.
    """
    # A live run only: a finished run's jail is gone with it, and a fresh one is a different place.
    if not session_is_live(layout.session_dir):
        _common.refuse(f"{no_session_network_reason(layout)}")
        return 2
    pid = ipc.read_session_netns_pid(layout.session_dir)
    # The run's recorded policy, not today's config; unstamped falls back with a warning.
    stamped = _stamped_policy(layout)
    if stamped is not None:
        isolation_word, network_word = stamped
        isolation = detect.resolve_isolation(isolation_word, _setup.detect_env())
        # The recorded word, not the holder: a host-network run can still hold a netns for MCP.
        network = network_word if network_word is not None else ("session" if pid else None)
    else:
        print(
            "[agent6] WARNING: this run recorded no launch policy; using the"
            " current config, which may differ from what the run's commands got.",
            file=sys.stderr,
        )
        isolation = detect.resolve_isolation(cfg.sandbox.isolation, _setup.detect_env())
        network = "session" if pid else None
    try:
        policy = tools_policy.jail_policy(cwd, cfg, isolation, argv, network=network, timeout_s=0.0)
    except jail.JailUnavailableError as exc:
        _common.error(f"{exc}")
        return 2
    if policy.network == "session" and pid is None:
        # Recorded as session but holding none (an isolation short of strict): refuse.
        _common.refuse(f"{no_session_network_reason(layout)}")
        return 2
    if policy.network == "session":
        # Borrowed through the holder, which can exit between the read above and this open.
        userns_fd = -1
        try:
            userns_fd = os.open(f"/proc/{pid}/ns/user", os.O_RDONLY)
            borrowed = jail.SessionNetwork(
                userns_fd=userns_fd,
                netns_fd=os.open(f"/proc/{pid}/ns/net", os.O_RDONLY),
                holder_pid=int(pid or 0),
            )
        except OSError as exc:
            if userns_fd >= 0:
                os.close(userns_fd)
            _common.refuse(f"the session's network is gone: {exc}")
            return 2
    else:
        borrowed = None
    try:
        result = jail.run_in_jail(policy, session_net=borrowed)
    except jail.JailUnavailableError as exc:
        _common.error(f"{exc}")
        return 2
    finally:
        if borrowed is not None:
            borrowed.close()
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode
