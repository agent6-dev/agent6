# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the parsers that observe and drive a run: `attach`, `tui`, `web`, `steer` and kin."""

from __future__ import annotations

import argparse

from agent6.ui.cli import common, completers, sessions_findings


def _add_attach_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `attach`, a raw tail or the TUI on one run or machine."""
    watch_p = common._sub(
        sub,
        "attach",
        help=(
            "Attach to a session or machine and drive it live: the conversation as `agent6 run`"
            " prints it and, on a terminal, its approvals and questions to answer. For a run, --raw prints one line"
            " per event, --tui opens the full-screen view and --json prints one snapshot of its"
            " state. The plain and raw followers exit non-zero if the worker dies before the run"
            " ends. Omit the target for the newest session."
        ),
    )
    watch_target = watch_p.add_argument(
        "target",
        nargs="?",
        default="",
        help=f"{common.SESSION_ID} or machine id; omit for the newest.",
    )
    watch_target.completer = completers._complete_watch_targets  # type: ignore[attr-defined]
    # One presentation at a time; argparse refuses the combination instead of picking one.
    watch_mode = watch_p.add_mutually_exclusive_group()
    watch_mode.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the default conversation follow.",
    )
    watch_mode.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON snapshot of the session's state and exit (what the web UI reads).",
    )
    watch_mode.add_argument(
        "--raw",
        action="store_true",
        help=(
            "For a run, follow the event log instead of the conversation, one line per"
            " event (its type and key fields); a machine has no raw mode."
        ),
    )
    watch_p.add_argument(
        "--since",
        type=int,
        default=None,
        metavar="N",
        help="--raw only: replay the last N events before following (0 = from end).",
    )


def _add_tui_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `tui`, the run, plan and ask hub."""
    tui_p = common._sub(
        sub,
        "tui",
        help="Open the TUI hub: browse runs and start a new run/plan/ask.",
    )
    tui_target = tui_p.add_argument(
        "target",
        nargs="?",
        default="",
        help=f"{common.SESSION_ID} or machine id to open (what `attach --tui` opens); "
        "omit for the hub.",
    )
    tui_target.completer = completers._complete_watch_targets  # type: ignore[attr-defined]


def _add_web_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `web`, the browser UI."""
    web_p = common._sub(
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
        help=(
            f"{common.SESSION_ID}, machine id or `machine create` draft to open on load;"
            " omit for the hub."
        ),
    )
    web_target.completer = completers._complete_watch_targets  # type: ignore[attr-defined]
    web_p.add_argument(
        "--host",
        default=None,
        metavar="ADDR",
        help=(
            "Bind address (default: [web].host, 127.0.0.1 unless configured)."
            " A non-loopback bind widens the network surface."
        ),
    )
    web_p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Listen port (default: [web].port, 7658 unless configured).",
    )
    web_p.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help=(
            "Opt in to a non-loopback `--host` for this invocation; a non-loopback `[web].host`"
            " needs `[web].allow_non_loopback = true`."
        ),
    )
    web_p.add_argument(
        "--secret",
        default=None,
        metavar="VALUE",
        help=(
            "The secret a browser signs in with, at least 12 characters. Default:"
            " AGENT6_WEB_SECRET, else a six-word secret minted per launch and printed. A flag"
            " value shows in the process list; the variable keeps it out."
        ),
    )


def _add_steer_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `steer`, which queues an instruction for a live run."""
    steer_p = common._sub(
        sub,
        "steer",
        help=(
            "Send an instruction to a live run; it takes it at its next step. The composer"
            " directives work too: `/task <text>` queues work and `/btw <question>` asks beside"
            " the run. `/undo` needs a composer: the TUI, the web, a `run -i` prompt or a live"
            " run's pause menu. For a stopped session, `agent6 resume ID --steer TEXT` queues the"
            " instruction for its next execution."
        ),
    )
    steer_target = steer_p.add_argument("target", help=f"{common.SESSION_ID}.")
    steer_target.completer = completers.complete_steer_ids  # type: ignore[attr-defined]
    instruction = steer_p.add_mutually_exclusive_group(required=True)
    instruction.add_argument(
        "text",
        nargs="?",
        help=(
            "The instruction, or a composer directive (/pin, /compact, /task, /standing,"
            " /retire, /btw, /now)."
        ),
    )
    finding = instruction.add_argument(
        "--finding",
        action="append",
        help="Submit Fix for a stored finding ID; repeat it to fix several in one steer.",
    )
    finding.completer = completers.complete_finding_ids  # type: ignore[attr-defined]
    instruction.add_argument(
        "--all-findings",
        action="store_true",
        help="Submit Fix for every open finding without a queued Fix, in one steer.",
    )
    steer_p.add_argument(
        "--resume", action="store_true", help="Confirm Resume to fix for a stopped run's findings."
    )
    sessions_findings.add_action_options(steer_p)
    steer_p.add_argument(
        "--now",
        action="store_true",
        help=(
            "Interrupt the in-flight model call to take the steer immediately"
            " (the default waits for the next turn boundary; an approval or"
            " question wait cannot be interrupted either way)."
        ),
    )


def _add_stop_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `stop`."""
    stop_p = common._sub(
        sub,
        "stop",
        help=(
            "Stop a live session now: cut its model call and end its running commands; its lanes"
            " stop with it. The run ends as stopped. A worker still live after 5 s (30 s while it"
            " drains lanes) is killed with its commands. A run `agent6 resume` accepts stays"
            " resumable. Default: the newest live session."
        ),
    )
    common._add_session_id(stop_p, completers._complete_live_session_ids)
    stop_p.add_argument(
        "--after-turn",
        action="store_true",
        help=(
            "Let the current turn finish, then stop: a running command runs to its end or its"
            " check-in, and the turn's tool results and auto-commit land."
        ),
    )
    stop_p.add_argument(
        "--all",
        action="store_true",
        help="Stop every live session of this repository (a fan-out's lanes with it).",
    )


def _add_answer_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `answer`."""
    answer_p = common._sub(
        sub,
        "answer",
        help=(
            "Answer a run's question or approval without a terminal, through the answer file the"
            " TUI, the web and attach write. With no TEXT, list what is pending with its options;"
            " else give one TEXT per question, in order. An approval takes yes, no, session or"
            " session-deny, and is answered first unless --question is given. A queued question's"
            " answer reaches the run at its next turn; an ended run's at its resume."
        ),
    )
    answer_target = answer_p.add_argument("target", help=f"{common.SESSION_ID}.")
    answer_target.completer = completers._complete_answerable_ids  # type: ignore[attr-defined]
    answer_p.add_argument(
        "--question",
        metavar="ID",
        default=None,
        help="The question set to answer when several are pending; a bare call lists them.",
    )
    answer_p.add_argument(
        "answers",
        nargs="*",
        metavar="TEXT",
        help="One TEXT per question, or an approval option: yes, no, session, session-deny.",
    )


def _add_net_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `exec` and `forward`, which reach into a live run's session network.

    Top-level verbs, like `attach`, because they act on a running session; `exec --new`
    alone needs none.

    Args:
        sub: The top-level subparsers.
    """
    exec_p = common._sub(
        sub,
        "exec",
        help=(
            "Run a command of yours inside a live session's sandbox, to see what the agent sees:"
            " the run's recorded isolation and network, with mounts from your current config. It"
            " is never approved or recorded as a tool call; what it writes in the workspace lands"
            " in the run's next commit. On a terminal it gets a terminal of its own; otherwise"
            " stdin, stdout and stderr pass through. `agent6 exec CMD...` runs in the newest"
            " session; `agent6 exec SESSION -- CMD...` names one, and only the first `--`"
            " separates. `agent6 exec --new [CMD...]` runs the command, or your shell, in a fresh"
            " sandbox built from the current config."
        ),
    )
    exec_p.add_argument(
        "--new",
        action="store_true",
        help=(
            "Run in a fresh sandbox built from the current config, with the mounts, hidden paths,"
            " network and isolation a run's commands get. Without CMD, opens $SHELL (/bin/sh when"
            " the sandbox's PATH lacks it)."
        ),
    )
    common._add_sandbox_flags(exec_p, approvals=False)
    exec_rest = exec_p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[SESSION --] CMD... The command rides verbatim.",
    )
    exec_rest.completer = completers._complete_exec_rest  # type: ignore[attr-defined]

    fwd_p = common._sub(
        sub,
        "forward",
        help=(
            "Bridge a port inside a live session's network to one on this"
            " machine, so a browser can open the dev server the agent started."
            " Without a port, lists what the session is listening on."
        ),
    )
    fwd_target = fwd_p.add_argument(
        "target",
        nargs="?",
        default="",
        help=(
            f"{common.SESSION_ID_HELP} A bare number"
            " here is read as the PORT of the newest session (name a numeric"
            " session by giving both arguments)."
        ),
    )
    fwd_target.completer = completers._complete_live_session_ids  # type: ignore[attr-defined]
    fwd_port = fwd_p.add_argument(
        "port", nargs="?", type=int, help="The port inside the session. Omit to list them."
    )
    fwd_port.completer = completers._complete_session_ports  # type: ignore[attr-defined]
    fwd_p.add_argument(
        "--local-port",
        type=int,
        default=None,
        help=(
            "The port on this machine (default: the same number, as kubectl/docker/ssh"
            " mean it; 0 picks a free one, named when the bridge starts)."
        ),
    )
