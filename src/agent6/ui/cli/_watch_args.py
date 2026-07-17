# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for the three ways to observe/drive a run: `attach` (raw
tail or full-screen TUI on one run/machine), `tui` (the run/plan/ask hub),
and `web` (the browser UI)."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import _sub
from agent6.ui.cli.completers import _complete_watch_targets


def _add_attach_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    watch_p = _sub(
        sub,
        "attach",
        help=(
            "Attach to a run or machine live and drive it: follow the conversation"
            " (the same render as `agent6 run`) and, on a terminal, answer its"
            " run_command approvals and ask_user questions right here -- as if you"
            " never detached. --raw is the no-deps event-line tail, --tui the"
            " full-screen TUI, --json a one-shot snapshot of the folded state."
            " Omit the target for the most recent run."
        ),
    )
    watch_target = watch_p.add_argument(
        "target",
        nargs="?",
        default="",
        help="Run id (exact or prefix) or machine id. Omit for the most recent run.",
    )
    watch_target.completer = _complete_watch_targets  # type: ignore[attr-defined]
    watch_p.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the default plain line tail.",
    )
    watch_p.add_argument(
        "--json",
        action="store_true",
        help="Print a one-shot JSON snapshot of the folded state and exit (the web wire form).",
    )
    watch_p.add_argument(
        "--raw",
        action="store_true",
        help="Follow the no-deps event-line tail (type + key fields) instead of the conversation.",
    )
    watch_p.add_argument(
        "--since",
        type=int,
        default=0,
        metavar="N",
        help="--raw only: replay the last N events before following (0 = from end).",
    )


def _add_tui_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    _sub(
        sub,
        "tui",
        help="Open the TUI hub: browse runs and start a new run/plan/ask.",
    )


def _add_web_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    web_p = _sub(
        sub,
        "web",
        help=(
            "Serve the browser UI (loopback by default): watch and drive runs and"
            " machines from a desktop or phone. Put `tailscale serve` in front for"
            " remote access."
        ),
    )
    web_target = web_p.add_argument(
        "target",
        nargs="?",
        default="",
        help="Run id (exact or prefix) or machine id to open on load. Omit for the hub.",
    )
    web_target.completer = _complete_watch_targets  # type: ignore[attr-defined]
    web_p.add_argument(
        "--host",
        default=None,
        metavar="ADDR",
        help="Bind address (default 127.0.0.1). A non-loopback bind widens the network surface.",
    )
    web_p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Listen port (default 7658).",
    )
    web_p.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Opt in to bind a non-loopback --host (else a non-loopback bind is refused).",
    )
