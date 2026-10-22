# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prepare a tool state's clone under its jail policy."""

from __future__ import annotations

import dataclasses
import pathlib

from agent6 import event_log, kinds
from agent6.app import reporter as app_reporter
from agent6.app import workspace_setup
from agent6.config import Config
from agent6.machine import EngineError
from agent6.sandbox import jail


@dataclasses.dataclass(frozen=True, slots=True)
class _Runner:
    config: Config
    policy: kinds.JailPolicy
    events_log: pathlib.Path | None

    @property
    def isolation(self) -> kinds.IsolationLevel:
        return self.policy.isolation

    def command_policy(self) -> str:
        return self.config.sandbox.run_commands

    def run_setup(self, argv: tuple[str, ...]) -> kinds.CommandResult:
        result = jail.run_in_jail(dataclasses.replace(self.policy, argv=argv))
        if self.events_log is not None:
            event_log.EventSink(self.events_log).emit(
                "workspace.setup", **kinds.workspace_setup_fields(result)
            )
        return result


def tool_setup_failure(
    cfg: Config,
    policy: kinds.JailPolicy,
    events_log: pathlib.Path | None,
    reporter: app_reporter.Reporter,
) -> kinds.CommandResult | None:
    """Return a setup failure, or None when the tool can run.

    Host failures propagate to LiveWorld, which aborts the machine with EngineError.

    Args:
        cfg: The tool state's config, whose `[workspace].setup_command` runs.
        policy: The tool's jail policy, which the setup command runs under.
        events_log: The log the `workspace.setup` event goes to; None writes none.
        reporter: Receives the setup's notes.

    Returns:
        The failed setup command's result, or None when the tool can run.

    Raises:
        EngineError: Setup failed without a command result.
    """
    prepared = workspace_setup.setup(
        cfg, runner=_Runner(cfg, policy, events_log), reporter=reporter
    )
    if prepared.problem is None:
        return None
    if prepared.result is not None:
        return prepared.result
    raise EngineError(prepared.problem)
