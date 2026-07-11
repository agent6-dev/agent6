# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-forked host-namespace spawner for the detach path.

``enter_network_isolation`` moves the agent process, and every child it forks
afterwards, into an empty network namespace. A detached ``agent6 resume``
spawned from inside it inherits the cage, its egress broker dials from nowhere,
and the run dies as ``provider_error``. This helper is forked while the process
is still in the host namespaces (right before isolation, like the egress broker
child) and sleeps on a pipe; at detach the isolated parent asks it to spawn the
background resume, which then runs with host networking.

Narrow contract: a request carries only the run id and working directory. The
helper builds argv itself from the executable prefix captured at fork time and
never executes request-provided argv. The request pipe's write end lives only
in the trusted agent6 operator process (close-on-exec, so no tool child can
inherit it).
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.sandbox.broker import close_inherited_fds, set_parent_death_signal

_ACK_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class HostSpawner:
    """Handle to the pre-forked helper.

    ``spawn_resume`` asks it to launch a detached ``agent6 resume <run_id>`` in
    the host namespaces and waits for its confirmation; ``close`` shuts the
    helper down (pipe EOF) and reaps it.
    """

    pid: int
    req_wfd: int
    ack_rfd: int

    def spawn_resume(self, cwd: Path, run_id: str) -> str:
        """Spawn the background resume; return "" on a confirmed start, else an
        error message."""
        req = json.dumps({"cwd": str(cwd), "run_id": run_id}) + "\n"
        try:
            os.write(self.req_wfd, req.encode("utf-8"))
        except OSError as exc:
            return f"detach helper is gone ({exc}); could not spawn the background resume"
        ready, _, _ = select.select([self.ack_rfd], [], [], _ACK_TIMEOUT_S)
        if not ready:
            return "detach helper did not confirm the background resume"
        try:
            ack = os.read(self.ack_rfd, 4096).decode("utf-8", "replace").strip()
        except OSError as exc:
            return f"detach helper confirmation failed: {exc}"
        if ack == "ok":
            return ""
        return f"background resume failed to start: {ack.removeprefix('err ')}"

    def close(self) -> None:
        """EOF the request pipe so the helper exits, and reap it. Idempotent."""
        for fd in (self.req_wfd, self.ack_rfd):
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(self.pid, 0)


def fork_host_spawner(argv_prefix: list[str]) -> HostSpawner:
    """Fork the helper child. Must be called while the process is still in the
    host namespaces (before ``enter_network_isolation``) and single-threaded,
    the same window as ``start_egress_broker``. *argv_prefix* is the agent6
    executable invocation the helper prepends to ``["resume", <run_id>]``.
    """
    prefix = list(argv_prefix)
    env = {**os.environ, "AGENT6_STREAM_TO_LOG": "1"}
    req_r, req_w = os.pipe()
    ack_r, ack_w = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - exercised in a child process
        try:
            _run_helper(req_r, ack_w, prefix, env)
        finally:
            os._exit(0)
    os.close(req_r)
    os.close(ack_w)
    return HostSpawner(pid=pid, req_wfd=req_w, ack_rfd=ack_r)


def _run_helper(
    req_rfd: int, ack_wfd: int, prefix: list[str], env: dict[str, str]
) -> None:  # pragma: no cover - runs in the forked child
    set_parent_death_signal()

    def _exit(_signum: int, _frame: object) -> None:
        os._exit(0)

    signal.signal(signal.SIGTERM, _exit)
    # A terminal Ctrl-C signals the whole foreground process group; the helper
    # must survive it, because detach happens right after a Ctrl-C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    close_inherited_fds({req_rfd, ack_wfd})
    with os.fdopen(req_rfd, "r", encoding="utf-8", errors="replace") as requests:
        for line in requests:
            try:
                req = json.loads(line)
                subprocess.Popen(
                    [*prefix, "resume", str(req["run_id"])],
                    cwd=str(req["cwd"]),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=env,
                )
            except (OSError, ValueError, KeyError) as exc:
                msg = str(exc).replace("\n", " ")[:300]
                with contextlib.suppress(OSError):
                    os.write(ack_wfd, f"err {msg}\n".encode())
                continue
            with contextlib.suppress(OSError):
                os.write(ack_wfd, b"ok\n")
