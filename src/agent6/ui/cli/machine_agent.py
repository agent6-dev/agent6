# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run one machine `agent` state as a subprocess.

Invoked as `python -m agent6.ui.cli.machine_agent <request.json> <result.json>`: validates
the `MachineAgentRequest`, runs `agent6.app.machine_agent.run_one` with the live-view console
attached, and writes the `AgentExecResult`. The engine enforces the timeout by killing this
process, which gives true mid-call cancellation.
"""

from __future__ import annotations

import os
import pathlib
import sys

from agent6 import events as agent6_events
from agent6.app import machine_agent
from agent6.ui.cli import _common, _console_view


def _attach_console(events: agent6_events.EventSink) -> None:
    """Render the live conversation to stderr at a TTY or under AGENT6_FORCE_STREAM=1.

    The console consumes the same events the per-state sink records.

    Args:
        events: The state's event sink.
    """
    if sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1":
        events.subscribe(_console_view.ConsoleView(sys.stderr))


def main() -> int:
    """Run the request in argv[1] and write the result to argv[2].

    Returns:
        The exit code; 1 when the per-state journal could not be written.
    """
    req = machine_agent.MachineAgentRequest.model_validate_json(
        pathlib.Path(sys.argv[1]).read_bytes()
    )
    try:
        out = machine_agent.run_one(req, attach_console=_attach_console)
    except agent6_events.EventWriteError as exc:
        # Off the CLI dispatch backstop, a raw traceback would leave the engine a bare "error".
        _common.error(f"{exc}")
        return 1
    pathlib.Path(sys.argv[2]).write_text(out.model_dump_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
