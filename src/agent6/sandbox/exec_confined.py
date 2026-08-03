# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Confine this process, then become the given command.

    python -m agent6.sandbox.exec_confined --read A --write B [--no-network] -- argv...

For a long-lived child agent6 spawns but does not drive: an MCP server the
operator configured. The jail launcher is the wrong tool there -- it captures
stdio and owns the process to completion, while an MCP server needs a live
bidirectional pipe for the whole session.

Both mechanisms are restrict-self-then-exec, so confining ourselves and then
becoming the server confines the server and everything it spawns:

- Landlock's domain is irrevocable and inherited across `execve`.
- A network namespace is entered by `unshare` and inherited the same way. The
  stdio pipes are file descriptors, which an unshare does not touch, so the
  live JSON-RPC pipe survives it.

`preexec_fn` would be the obvious place for both and is unsafe in a process
with threads, which the MCP client has.

Order matters: the namespace work writes `/proc/self/{setgroups,uid_map,
gid_map}` and opens a socket, none of which the Landlock domain grants. It
therefore runs FIRST, and Landlock closes over it.

Nothing here is reachable by the model: the paths come from the operator's
config, the argv from the same, and this module is only ever spawned by
agent6 itself.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import struct
import sys
from pathlib import Path

from agent6.sandbox.landlock import LandlockNotSupportedError, apply_agent_landlock

# ioctl to set an interface's flags, and the one flag we set (linux/if.h).
_SIOCSIFFLAGS = 0x8914
_IFF_UP = 0x1


class NoNetworkError(Exception):
    """The server asked for `network = "none"` and this host cannot give it."""


def _drop_network() -> None:
    """Move into a private network namespace, keeping our own uid and loopback.

    An unprivileged process gets CAP_NET_ADMIN only inside a user namespace of
    its own, so the two are unshared together. That remaps every uid to
    `nobody` until `uid_map` is written, which a server would meet as its own
    files suddenly being unreadable, so the identity is mapped straight through.
    """
    # Read BEFORE the unshare: inside the new namespace we are already the
    # overflow uid (65534), and a map naming THAT is a map of an id we do not
    # own on the outside -- which the kernel refuses.
    uid, gid = os.getuid(), os.getgid()
    try:
        os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
    except (AttributeError, OSError) as exc:
        raise NoNetworkError(
            f"this host cannot give a server its own network namespace ({exc});"
            " unprivileged user namespaces are disabled or unavailable"
        ) from exc
    try:
        # setgroups must be denied before gid_map is writable by an
        # unprivileged process (kernel rule, not a policy of ours).
        Path("/proc/self/setgroups").write_text("deny", encoding="ascii")
        Path("/proc/self/uid_map").write_text(f"{uid} {uid} 1", encoding="ascii")
        Path("/proc/self/gid_map").write_text(f"{gid} {gid} 1", encoding="ascii")
    except OSError as exc:
        raise NoNetworkError(
            f"could not keep the operator's identity in the namespace: {exc}"
        ) from exc
    try:
        _bring_up_loopback()
    except OSError as exc:
        raise NoNetworkError(f"could not bring up the namespace's loopback: {exc}") from exc


def _bring_up_loopback() -> None:
    """A fresh netns has `lo` DOWN. Leaving it there breaks any server that
    binds localhost while confining nothing extra: this loopback reaches only
    this namespace."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # struct ifreq: 16-byte name, then the flags in the union.
        ifreq = struct.pack("16sh", b"lo", _IFF_UP)
        fcntl.ioctl(sock.fileno(), _SIOCSIFFLAGS, ifreq)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent6.sandbox.exec_confined",
        description="Landlock this process, then exec the command after `--`.",
    )
    parser.add_argument("--read", action="append", default=[], metavar="PATH")
    parser.add_argument("--write", action="append", default=[], metavar="PATH")
    parser.add_argument(
        "--require",
        action="store_true",
        help="Fail rather than exec unconfined when the kernel has no Landlock.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Run in a network namespace of its own (loopback only, reaching nothing else).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- ARGV")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after `--`")

    if args.no_network:
        try:
            _drop_network()
        except NoNetworkError as exc:
            # `none` is an explicit enforce value, so it refuses. Degrading here
            # would leave a server the operator believes is offline with the
            # whole network.
            print(f"agent6: refusing to run this server with the network: {exc}", file=sys.stderr)
            return 2

    try:
        apply_agent_landlock(
            read_paths=tuple(Path(p).expanduser() for p in args.read),
            write_paths=tuple(Path(p).expanduser() for p in args.write),
        )
    except LandlockNotSupportedError as exc:
        if args.require:
            print(f"agent6: refusing to run unconfined: {exc}", file=sys.stderr)
            return 2
        # Degrade, never silently: the caller asked for confinement this kernel
        # cannot give, and the server is still worth running.
        print(f"agent6: WARNING: running this server unconfined: {exc}", file=sys.stderr)

    try:
        # The operator's argv, from their own config; no shell, and no LLM
        # input reaches here.
        os.execvp(command[0], command)  # noqa: S606
    except OSError as exc:
        print(f"agent6: could not exec {command[0]}: {exc}", file=sys.stderr)
        return 127
    return 0  # pragma: no cover -- execvp does not return


if __name__ == "__main__":  # pragma: no cover -- the module entry point
    raise SystemExit(main())
