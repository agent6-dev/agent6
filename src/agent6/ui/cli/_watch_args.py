# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the parsers that observe and drive a run: `attach`, `tui`, `web`, `steer` and kin."""

from __future__ import annotations

import argparse

from agent6.ui.cli import _common, completers


def _add_attach_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `attach`, a raw tail or the TUI on one run or machine."""
    watch_p = _common._sub(
        sub,
        "attach",
        help=(
            "Attach to a session or machine and drive it live: the conversation as"
            " `agent6 run` prints it, and, on a terminal, its command approvals and"
            " questions to answer. For a run, --raw follows the event log one line per"
            " event, --tui opens the full-screen view, --json prints one snapshot of its"
            " state. The plain and raw run followers exit non-zero if the worker dies"
            " before the run ends. Omit the target for the newest session."
        ),
    )
    watch_target = watch_p.add_argument(
        "target",
        nargs="?",
        default="",
        help=f"{_common.SESSION_ID} or machine id; omit for the newest.",
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
    tui_p = _common._sub(
        sub,
        "tui",
        help="Open the TUI hub: browse runs and start a new run/plan/ask.",
    )
    tui_target = tui_p.add_argument(
        "target",
        nargs="?",
        default="",
        help=f"{_common.SESSION_ID} or machine id to open (what `attach --tui` opens); "
        "omit for the hub.",
    )
    tui_target.completer = completers._complete_watch_targets  # type: ignore[attr-defined]


def _add_web_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `web`, the browser UI."""
    web_p = _common._sub(
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
            f"{_common.SESSION_ID}, machine id or `machine create` draft to open on load;"
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
            "Opt in to a non-loopback bind for this invocation; otherwise"
            " [web].allow_non_loopback must be true."
        ),
    )


def _add_steer_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `steer`, which queues an instruction for a live run."""
    steer_p = _common._sub(
        sub,
        "steer",
        help=(
            "Send an instruction to a live run, as the TUI and web composers do; the"
            " run takes it at its next step. It takes the composer directives too, so"
            " `/task <text>` queues work and `/btw <question>` asks beside the run"
            " (pause-menu directives such as abort and /undo travel the same way)."
            " Live runs only: for a stopped session, `agent6 resume ID --steer TEXT`"
            " queues one for its next execution."
        ),
    )
    steer_target = steer_p.add_argument("target", help=f"{_common.SESSION_ID}.")
    steer_target.completer = completers._complete_live_session_ids  # type: ignore[attr-defined]
    steer_p.add_argument(
        "text",
        help=(
            "The instruction, or a composer directive (/pin, /compact, /task, /standing,"
            " /retire, /btw, /now)."
        ),
    )
    steer_p.add_argument(
        "--now",
        action="store_true",
        help=(
            "Interrupt the in-flight model call to take the steer immediately"
            " (the default waits for the next step boundary; an approval or"
            " question wait cannot be interrupted either way)."
        ),
    )


def _add_stop_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `stop`."""
    stop_p = _common._sub(
        sub,
        "stop",
        help=(
            "Stop a live session now: its model call is cut, a running command is handed"
            " back, the run ends as stopped and every command it started is ended; a"
            " worker that has not ended after 5 s is killed with them. The session stays"
            " resumable (`agent6 resume ID`). Default: the newest session."
        ),
    )
    _common._add_session_id(stop_p, completers._complete_live_session_ids)
    stop_p.add_argument(
        "--after-step",
        action="store_true",
        help="Let the current step finish (its tool results and auto-commit land), then stop.",
    )
    stop_p.add_argument(
        "--all",
        action="store_true",
        help="Stop every live session of this repository (a fan-out's lanes with it).",
    )


def _add_answer_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `answer`."""
    answer_p = _common._sub(
        sub,
        "answer",
        help=(
            "Answer a live run's ask_user question without a terminal: the same"
            " answer file the TUI, web and attach write. With no TEXT it prints"
            " the open question and its options; one TEXT per question, in order."
        ),
    )
    answer_target = answer_p.add_argument("target", help=f"{_common.SESSION_ID}.")
    answer_target.completer = completers._complete_live_session_ids  # type: ignore[attr-defined]
    answer_p.add_argument(
        "answers",
        nargs="*",
        metavar="TEXT",
        help="One answer per question, in the order asked; an option's text, or free text.",
    )


def _add_net_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `exec` and `forward`, which reach into a live run's session network.

    Top-level verbs, like `attach`, because they act on a running session.
    """
    exec_p = _common._sub(
        sub,
        "exec",
        help=(
            "Run a command inside a live session's sandbox: the run's recorded"
            " isolation and network (mounts derive from your current config), so"
            " you see what the agent sees. The command is yours, not"
            " the model's, so it is never approved or recorded as a tool call;"
            " its output prints when it ends (reach a server the agent runs"
            " with `agent6 forward`). `agent6 exec CMD...` runs in the newest session;"
            " `agent6 exec SESSION -- CMD...` names one. Only the first `--`"
            " separates; later ones belong to the command."
        ),
    )
    exec_rest = exec_p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[SESSION --] CMD... The command rides verbatim.",
    )
    exec_rest.completer = completers._complete_live_session_ids  # type: ignore[attr-defined]

    fwd_p = _common._sub(
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
            f"{_common.SESSION_ID_HELP} A bare number"
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
