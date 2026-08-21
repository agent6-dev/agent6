# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A SIGKILLed agent6 cannot strand a jailed command.

The launcher had no parent-death tie: killing the agent left the command
running in its namespaces until it exited on its own. The chain is now
python (preexec PDEATHSIG on the launcher) -> launcher -> pid-ns init
(PDEATHSIG in the fork child) -> command (pre_exec PDEATHSIG), so the
kernel tears the whole tree down on any link's death. This drives the real
stack: a child process runs a jailed sleep, dies by SIGKILL, and the sleep
must be gone from the host within a bound.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from tests.jail_env import require_userns_jail


def _pgrep(token: str) -> list[str]:
    out = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, check=False)
    return [ln for ln in out.stdout.split() if ln.strip()]


def test_sigkilled_agent_takes_its_jailed_command_down(tmp_path: Path) -> None:
    require_userns_jail()
    token = f"60.{os.getpid()}"  # a unique sleep duration doubling as the pgrep handle
    script = tmp_path / "agent.py"
    script.write_text(
        textwrap.dedent(f"""
        from pathlib import Path

        from agent6.config import Config
        from agent6.sandbox.jail import run_in_jail
        from agent6.tools.dispatch import jail_policy

        run_in_jail(
            jail_policy(
                Path({str(tmp_path)!r}),
                Config(),
                "strict",
                ("sleep", {token!r}),
                network="none",
            )
        )
        """),
        encoding="utf-8",
    )
    agent = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not _pgrep(token):
            if agent.poll() is not None:
                raise AssertionError(f"agent exited early with {agent.returncode}")
            time.sleep(0.1)
        assert _pgrep(token), "jailed sleep never appeared on the host"

        os.kill(agent.pid, signal.SIGKILL)
        agent.wait(timeout=10)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _pgrep(token):
            time.sleep(0.1)
        leftovers = _pgrep(token)
        assert not leftovers, f"jailed command survived the agent's SIGKILL: pids {leftovers}"
    finally:
        if agent.poll() is None:
            agent.kill()
        for pid in _pgrep(token):
            with __import__("contextlib").suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGKILL)
