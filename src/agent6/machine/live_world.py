# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The production `World`: tool states run in the jail, agent states through a runner, waits sleep.

The agent runner, the tool policy and the jail runner are injected, so the engine stays a
pure reducer and this module imports neither the harness nor config nor git.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import shutil
import time
from collections.abc import Callable
from typing import Any

from agent6 import kinds, paths, portable
from agent6.machine import _contracts
from agent6.machine import journal as machine_journal
from agent6.sandbox import jail
from agent6.sessions import layout

__all__ = ["LiveWorld", "ToolPolicyFactory"]


# (state, argv, timeout_s, pass_env) -> the jail policy the shared builder produced for that state.
ToolPolicyFactory = Callable[[str, tuple[str, ...], float, tuple[str, ...]], kinds.JailPolicy]


def _state_log_seq(p: pathlib.Path) -> int:
    """Return the seq of a `<seq>-<state>` log dir name, so the sort is numeric."""
    prefix = p.name.split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else -1


def _prune_state_logs(root: pathlib.Path, *, keep: int) -> None:
    """Keep the newest per-state log dirs, leaving room for the one about to be written.

    Best effort; the journal keeps the full history regardless.

    Args:
        root: The per-state log root.
        keep: The number to keep, including the next; 0 prunes nothing.
    """
    if keep == 0:
        return
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=_state_log_seq)
    except OSError:
        return
    for stale in dirs[: max(0, len(dirs) - keep + 1)]:
        shutil.rmtree(stale, ignore_errors=True)


@dataclasses.dataclass(frozen=True, slots=True)
class LiveWorld:
    """The production world: tools go through the jail, waits sleep.

    The agent runner, the tool policy and the jail runner are injected by the CLI, so this
    module imports neither the harness nor config nor git.

    Attributes:
        cwd: The workspace.
        journal: The instance's journal.
        agent_runner: Runs an agent state given its request and its own logs.jsonl path;
            reaching an agent state without one fails loudly.
        poll_interval_s: How often a blocking wait checks for a poke or a stop.
        tool_policy: Builds each tool jail's policy from the shared builder with the state's
            own config and the machine deltas (bundle protect paths, the data dir grant), so
            a tool is confined like a run command.
        state_log_root: Where each agent state writes `<seq>-<state>/logs.jsonl`; None
            disables per-state logs.
        state_log_keep: How many per-state log dirs to keep (`[machine].state_log_keep`);
            0 keeps all.
        notify_hook: The operator's `[machine.notify].on_event` argv, run on the host; None
            for no hook (the front-ends still render the journaled events).
        data_dir: The machine's persistent scratch dir, `$AGENT6_MACHINE_DATA_DIR` to a tool,
            outside the workspace; where a tool keeps durable state.
        jail_runner: Executes one tool policy; the CLI overrides it for `mode = "run"`
            machines so tools run in the machine's own tree. None is the plain jail.
        on_wait: Fired once as a wait starts to block, so the foreground run names where it
            parked; it cannot affect the sleep.
        on_tool_failure: Fired with the state and the result of a tool command that exited
            nonzero or timed out, so the foreground run names it; it cannot affect the step.
    """

    cwd: pathlib.Path
    journal: machine_journal.MachineJournal
    agent_runner: (
        Callable[[_contracts.AgentRequest, pathlib.Path | None], _contracts.AgentExecResult] | None
    ) = None
    poll_interval_s: float = 0.5
    tool_policy: ToolPolicyFactory | None = None
    state_log_root: pathlib.Path | None = None
    state_log_keep: int = 50
    notify_hook: Callable[[str, str, str, str], None] | None = None
    data_dir: pathlib.Path | None = None
    jail_runner: (
        Callable[[str, kinds.JailPolicy, pathlib.Path | None], kinds.CommandResult] | None
    ) = None
    on_wait: Callable[[], None] | None = None
    on_tool_failure: Callable[[str, _contracts.ToolExecResult], None] | None = None

    def run_tool(
        self,
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        state: str = "",
        pass_env: tuple[str, ...] = (),
        step_seq: int = 0,
    ) -> _contracts.ToolExecResult:
        """Run one tool command in its state's jail.

        Startup bounded the state's sandbox by the operator's and gated `pass_env`.

        Args:
            argv: The rendered command.
            timeout_s: The wall-clock cap.
            state: The state, which selects the config its policy is built from.
            pass_env: The operator environment variables the command receives.
            step_seq: The transition number for the per-state log directory.

        Returns:
            The exit code, output and whether the command timed out.

        Raises:
            _contracts.EngineError: No tool policy is wired, or the jail is unavailable.
        """
        if self.tool_policy is None:
            raise _contracts.EngineError("LiveWorld has no tool_policy factory wired")
        try:
            policy = self.tool_policy(state, tuple(argv), float(timeout_s), pass_env)
            result = (
                self.jail_runner(state, policy, self._state_log(state, step_seq))
                if self.jail_runner
                else jail.run_in_jail(policy)
            )
        except jail.JailUnavailableError as exc:
            raise _contracts.EngineError(f"jail unavailable: {exc}") from exc
        # run_in_jail returns rc 124 on a timeout and never raises TimeoutExpired.
        outcome = _contracts.ToolExecResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            timed_out=result.returncode == 124,
            stderr=result.stderr,
        )
        if outcome.exit_code != 0 and self.on_tool_failure is not None:
            with contextlib.suppress(OSError):  # a dead terminal never stops a machine
                self.on_tool_failure(state, outcome)
        return outcome

    def run_agent(self, request: _contracts.AgentRequest) -> _contracts.AgentExecResult:
        """Run one agent loop through the injected runner.

        Returns:
            The loop's result.

        Raises:
            _contracts.EngineError: No agent runner is configured.
        """
        if self.agent_runner is None:
            raise _contracts.EngineError(
                "machine reached an `agent` state but no agent runner is configured"
            )
        return self.agent_runner(request, self._state_log(request.state_name, request.step_seq))

    def _state_log(self, state: str, step_seq: int) -> pathlib.Path | None:
        """Return the state's own logs.jsonl path, pruning old ones first, or None."""
        if self.state_log_root is None or not state:
            return None
        _prune_state_logs(self.state_log_root, keep=self.state_log_keep)
        return self.state_log_root / f"{step_seq:04d}-{state}" / layout.LOGS_NAME

    def now(self) -> float:
        """Return the wall clock as an epoch."""
        return time.time()

    def sleep_until(self, wake_epoch: float | None) -> _contracts.WaitWake:
        """Block until the wake instant, a poke or a stop request, whichever comes first.

        Args:
            wake_epoch: The instant; None is a timerless wait, until a poke.

        Returns:
            How the wait woke.
        """
        if self.on_wait is not None:
            with contextlib.suppress(OSError):  # a dead terminal never stops a machine
                self.on_wait()
        while True:
            signaled, payload = self.journal.take_signal()
            if signaled:
                return _contracts.WaitWake("signal", payload)
            if machine_journal.stop_requested(self.journal.root):
                return _contracts.WaitWake("stop")  # the pending wait stays armed for the next run
            if wake_epoch is None:
                time.sleep(self.poll_interval_s)
                continue
            remaining = wake_epoch - time.time()
            if remaining <= 0:
                return _contracts.WaitWake("tick")
            time.sleep(min(remaining, self.poll_interval_s))

    def materialize_poke(self, payload: Any) -> None:
        """Write a poke's payload to `$AGENT6_MACHINE_DATA_DIR/poke.json` for the next tool.

        Atomic, and called before the StepEvent is appended, so a durable step implies a
        durable file. A no-op without a data dir.
        """
        if self.data_dir is None:
            return
        paths.mkdir_for_real_user(self.data_dir)
        portable.atomic_write(self.data_dir / "poke.json", json.dumps(payload, sort_keys=True))

    def notify(self, kind: str, state: str, message: str, level: str) -> None:
        """Fire the operator hook, when one is wired."""
        if self.notify_hook is not None:
            self.notify_hook(kind, state, message, level)
