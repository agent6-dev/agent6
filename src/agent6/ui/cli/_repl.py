# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-run REPL hook `agent6 run -i` fires after each auto-commit."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent6.ui.cli._console_view import ConsoleView

from agent6.budget import BudgetTracker
from agent6.init import init_workspace
from agent6.kinds import AutoCommitDirective
from agent6.paths import repo_config_path, state_dir
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.tools.mcp_client import MCPManager
from agent6.ui.cli._interact import _pause
from agent6.ui.cli._steer import idle_prompt_sigint
from agent6.ui.cli.plan_watch import (
    event_epoch,
    format_plain_event,
)
from agent6.ui.cli.sessions_cmds import _cmd_diff
from agent6.ui.steer import SteerState
from agent6.viewmodel.state import LOG_NOISE_EVENTS, STREAM_DELTA_EVENTS

REPL_HELP = (
    "  /continue  (empty enter) - let the agent take another iteration\n"
    "  /cost                    - print the running token + USD summary\n"
    "  /diff                    - git diff: base_sha -> the run branch's tip\n"
    "                              (read-only; `agent6 sessions diff`, no pager)\n"
    "  /watch                   - print the last 20 audit events from this run\n"
    "                              (snapshot, no streaming deltas; not a live tail)\n"
    "  /mcp                     - list MCP servers + tools currently wired\n"
    "                              into the agent's tool surface\n"
    "  /init                    - run the `agent6 init` setup wizard in the\n"
    "                              current cwd (prompts; never overwrites files)\n"
    "  /undo                    - take back the last message (the same /undo\n"
    "                              as steering, the TUI and web): this run\n"
    "                              ends, the tree as it stands is committed on\n"
    "                              this run's chain ref, the tree goes back to\n"
    "                              the turn before, and a fork holds that\n"
    "                              state; resume it with the message edited.\n"
    "  /help                    - show this help\n"
    "  /quit                    - stop the agent cleanly after this commit\n"
    "  /exit                    - stop and leave (no follow-up prompt; resume later)\n"
)


def build_repl_hook(
    root: Path,
    budget: BudgetTracker,
    *,
    session_id: str = "",
    mcp_manager: MCPManager | None = None,
    console_view: ConsoleView | None = None,
    steer_cell: Sequence[SteerState | None] = (),
) -> Callable[[int, str], AutoCommitDirective]:
    """Build the after-auto-commit hook for `agent6 run -i`.

    The CLI's extra state stays in the closure, so the harness knows nothing of it.
    `/undo` returns the harness's own undo directive.

    Args:
        root: The repo root, for `/diff` and `/init`.
        budget: The tracker `/cost` prints.
        session_id: The run, for `/diff` and `/watch`.
        mcp_manager: The live MCP manager `/mcp` lists.
        console_view: The live view, paused for the prompt.
        steer_cell: Holds the steer state once it exists; a Ctrl-C pause armed during
            the commit opens its menu only after this prompt, so the banner says so.

    Returns:
        The hook, taking the iteration and the commit sha.
    """

    def hook(iteration: int, sha: str) -> AutoCommitDirective:
        # Paused, the heartbeat cannot erase the prompt; idle, Ctrl-C stands aside for all of it.
        with _pause(console_view), idle_prompt_sigint():
            return _prompt_loop(iteration, sha)

    def _prompt_loop(iteration: int, sha: str) -> AutoCommitDirective:
        print(
            f"\n[agent6] iter {iteration} committed {sha[:12]}. "
            f"REPL: /continue /cost /diff /watch /mcp /init /undo /help /quit /exit",
            file=sys.stderr,
        )
        steer = steer_cell[0] if steer_cell else None
        if steer is not None and steer.armed():
            print(
                "[agent6] Ctrl-C pause armed: the steer menu opens after /continue.",
                file=sys.stderr,
            )
        while True:
            try:
                raw = input("agent6> ").strip()
            except EOFError:
                print("[agent6] EOF - stopping interactively.", file=sys.stderr)
                return "stop"
            except KeyboardInterrupt:
                print("\n[agent6] Ctrl-C - stopping interactively.", file=sys.stderr)
                return "stop"
            cmd = raw.lower()
            if cmd in {"", "/continue", "/c"}:
                return "continue"
            if cmd in {"/quit", "/q", "/stop"}:
                return "stop"
            if cmd == "/exit":
                return "exit"
            if cmd == "/undo":
                return "undo"
            # One Ctrl-C cancels the command it lands in; escaping the hook would end the run.
            try:
                _run_command(cmd, raw)
            except KeyboardInterrupt:
                print(f"\n[agent6] {cmd} cancelled.", file=sys.stderr)

    def _run_command(cmd: str, raw: str) -> None:
        if cmd in {"/help", "/h", "?"}:
            print(REPL_HELP, file=sys.stderr)
        elif cmd == "/cost":
            print(budget.format_summary(), file=sys.stderr)
        elif cmd == "/diff":
            repl_run_diff(session_id)
        elif cmd == "/watch":
            repl_show_recent_events(root, session_id, n=20)
        elif cmd == "/mcp":
            repl_list_mcp(mcp_manager)
        elif cmd == "/init":
            repl_run_init(root)
        else:
            print(f"[agent6] unknown command {raw!r}; try /help", file=sys.stderr)

    return hook


def repl_run_diff(session_id: str) -> None:
    """Print the run's diff for `/diff`, with no pager to take over the prompt's terminal."""
    try:
        _cmd_diff(session_id=session_id, stat=False, paths=(), paginate=False)
    except Exception as exc:
        print(f"[agent6] /diff failed: {exc}", file=sys.stderr)


def repl_show_recent_events(root: Path, session_id: str, *, n: int) -> None:
    """Print the last n events of the run's log for `/watch`.

    A snapshot, not a tail: the REPL sits between turns, and a tail would block the
    next one.

    Args:
        root: The repo root.
        session_id: The run.
        n: How many events.
    """
    if not session_id:
        print("[agent6] /watch: no run id available", file=sys.stderr)
        return
    # Across buckets: the REPL also runs inside an ask.
    try:
        layout = resolve_session(state_dir(root), session_id)
    except SessionIdError as exc:
        print(f"[agent6] /watch: {exc}", file=sys.stderr)
        return
    events_path = layout.logs_path
    if not events_path.is_file():
        print(f"[agent6] /watch: no logs.jsonl at {events_path}", file=sys.stderr)
        return
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[agent6] /watch failed: {exc}", file=sys.stderr)
        return
    session_start_ts: float | None = None
    if lines:
        try:
            obj0 = json.loads(lines[0])
            if isinstance(obj0, dict):
                session_start_ts = event_epoch(obj0.get("ts"))
        except json.JSONDecodeError:
            session_start_ts = None

    def _audit_line(raw: str) -> bool:
        """Return whether the line is one every log view shows, not a delta or a mirror."""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return False
        etype = obj.get("type") if isinstance(obj, dict) else None
        return etype not in STREAM_DELTA_EVENTS and etype not in LOG_NOISE_EVENTS

    tail = [raw for raw in lines if _audit_line(raw)][-n:]
    print(f"[agent6] /watch: last {len(tail)} events from {session_id}", file=sys.stderr)
    for raw in tail:
        print(format_plain_event(raw, session_start_ts=session_start_ts))


def repl_list_mcp(mcp_manager: MCPManager | None) -> None:
    """Print the configured MCP servers and their tools for `/mcp`."""
    if mcp_manager is None:
        print(
            "[agent6] /mcp: no MCP servers configured (set [mcp] in your config)",
            file=sys.stderr,
        )
        return
    descriptors = mcp_manager.descriptors()
    by_server: dict[str, list[str]] = {server: [] for server in mcp_manager.networks}
    for d in descriptors:
        by_server.setdefault(d.server_name, []).append(d.tool_name)
    print(f"[agent6] /mcp: {len(descriptors)} tools across {len(by_server)} server(s)")
    for server, tools in sorted(by_server.items()):
        print(f"  {server}: {len(tools)} tool(s)")
        for t in sorted(tools):
            print(f"    - {t}")
    for failure in mcp_manager.failures:
        print(f"  {failure.name}: failed to start ({failure.error})")


def repl_run_init(root: Path) -> None:
    """Run the setup wizard for `/init`; it prompts and never overwrites a file."""
    try:
        rc = init_workspace(
            root,
            repo_config_target=repo_config_path(root),
            interactive=sys.stdin.isatty(),
        )
    except Exception as exc:
        print(f"[agent6] /init failed: {exc}", file=sys.stderr)
        return
    print("[agent6] /init: ok" if rc == 0 else f"[agent6] /init: exit {rc}", file=sys.stderr)
