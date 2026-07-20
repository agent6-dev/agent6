# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive in-run REPL hook fired after each auto-commit: show diffs,
recent events, MCP tools, (re)init the workspace, or steer the next step.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent6.ui.cli._console_view import ConsoleView

from agent6.budget import BudgetTracker
from agent6.config.layer import repo_config_path_for
from agent6.git_ops import (
    GitError,
    revert_head,
)
from agent6.init import init_workspace
from agent6.tools.mcp_client import MCPManager
from agent6.ui.cli._common import _runs_dir
from agent6.ui.cli._interact import _pause
from agent6.ui.cli._steer import repl_prompt_sigint
from agent6.ui.cli.plan_watch import (
    event_epoch,
    format_plain_event,
)
from agent6.ui.cli.runs_cmds import _cmd_diff

REPL_HELP = (
    "  /continue  (empty enter) - let the agent take another iteration\n"
    "  /cost                    - print the running token + USD summary\n"
    "  /diff                    - git diff: base_sha -> this run's HEAD\n"
    "                              (read-only; same as `agent6 runs diff`)\n"
    "  /watch                   - print the last 20 events from this run\n"
    "                              (snapshot; not a live tail)\n"
    "  /mcp                     - list MCP servers + tools currently wired\n"
    "                              into the agent's tool surface\n"
    "  /init                    - run the `agent6 init` setup wizard in the\n"
    "                              current cwd (prompts; never overwrites files)\n"
    "  /undo                    - git revert HEAD (forward revert of the\n"
    "                              last auto-commit; safe under git policy).\n"
    "                              History is preserved: a NEW commit is\n"
    "                              added that inverts the last one. Nothing\n"
    "                              is destroyed; ``git reset --hard`` is\n"
    "                              forbidden by agent6's git policy.\n"
    "  /help                    - show this help\n"
    "  /quit                    - stop the agent cleanly after this commit\n"
)


def build_repl_hook(
    root: Path,
    budget: BudgetTracker,
    *,
    run_id: str = "",
    mcp_manager: MCPManager | None = None,
    console_view: ConsoleView | None = None,
) -> Callable[[int, str], Literal["continue", "stop"]]:
    """Build the after_auto_commit hook for ``agent6 run -i``.

    Captures the budget tracker (for ``/cost``), the repo root (for
    ``/undo`` and ``/diff``), the current run id (for ``/diff`` and
    ``/watch``), and the live MCP manager (for ``/mcp``) in a closure
    so Workflow stays agnostic of the CLI's extra state.
    extends with /diff, /watch, /mcp, /init.
    """

    def hook(iteration: int, sha: str) -> Literal["continue", "stop"]:
        # The whole prompt session sits inside the console-view pause: the run
        # is waiting on the OPERATOR, and the heartbeat's per-tick line-erase
        # would otherwise wipe the "agent6> " prompt and the typed characters,
        # replacing them with a lying "working…" spinner (same wiring as the
        # approval/question prompts).
        with _pause(console_view):
            return _prompt_loop(iteration, sha)

    def _prompt_loop(iteration: int, sha: str) -> Literal["continue", "stop"]:
        print(
            f"\n[agent6] iter {iteration} committed {sha[:12]}. "
            f"REPL: /continue /cost /diff /watch /mcp /init /undo /help /quit",
            file=sys.stderr,
        )
        while True:
            try:
                with repl_prompt_sigint():
                    raw = input("agent6> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[agent6] EOF - stopping interactively.", file=sys.stderr)
                return "stop"
            cmd = raw.lower()
            if cmd in {"", "/continue", "/c"}:
                return "continue"
            if cmd in {"/quit", "/q", "/stop"}:
                return "stop"
            if cmd in {"/help", "/h", "?"}:
                print(REPL_HELP, file=sys.stderr)
                continue
            if cmd == "/cost":
                print(budget.format_summary(), file=sys.stderr)
                continue
            if cmd == "/diff":
                repl_run_diff(run_id)
                continue
            if cmd == "/watch":
                repl_show_recent_events(root, run_id, n=20)
                continue
            if cmd == "/mcp":
                repl_list_mcp(mcp_manager)
                continue
            if cmd == "/init":
                repl_run_init(root)
                continue
            if cmd == "/undo":
                try:
                    revert_sha = revert_head(root)
                except GitError as exc:
                    print(f"[agent6] /undo failed: {exc}", file=sys.stderr)
                    continue
                print(
                    f"[agent6] /undo: reverted {sha[:12]} via new commit {revert_sha[:12]}",
                    file=sys.stderr,
                )
                continue
            print(
                f"[agent6] unknown command {raw!r}; try /help",
                file=sys.stderr,
            )

    return hook


def repl_run_diff(run_id: str) -> None:
    """REPL /diff: print `git diff base_sha..HEAD` for the live run."""
    try:
        _cmd_diff(run_id=run_id, stat=False, paths=())
    except Exception as exc:
        print(f"[agent6] /diff failed: {exc}", file=sys.stderr)


def repl_show_recent_events(root: Path, run_id: str, *, n: int) -> None:
    """REPL /watch: snapshot the last n events from this run's logs.jsonl.

    Intentionally NOT a live tail - the REPL is between turns of the
    agent loop; a tail would block the next iteration. Operators who
    want continuous tail use ``agent6 attach`` in another shell.
    """
    if not run_id:
        print("[agent6] /watch: no run id available", file=sys.stderr)
        return
    events_path = _runs_dir(root) / run_id / "logs.jsonl"
    if not events_path.is_file():
        print(f"[agent6] /watch: no logs.jsonl at {events_path}", file=sys.stderr)
        return
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[agent6] /watch failed: {exc}", file=sys.stderr)
        return
    run_start_ts: float | None = None
    if lines:
        try:
            obj0 = json.loads(lines[0])
            if isinstance(obj0, dict):
                run_start_ts = event_epoch(obj0.get("ts"))
        except json.JSONDecodeError:
            run_start_ts = None
    tail = lines[-n:]
    print(f"[agent6] /watch: last {len(tail)} events from {run_id}", file=sys.stderr)
    for raw in tail:
        print(format_plain_event(raw, run_start_ts=run_start_ts))


def repl_list_mcp(mcp_manager: MCPManager | None) -> None:
    """REPL /mcp: print configured MCP servers + their tool surface."""
    if mcp_manager is None:
        print(
            "[agent6] /mcp: no MCP servers configured (set [mcp] in your config)",
            file=sys.stderr,
        )
        return
    descriptors = mcp_manager.descriptors()
    if not descriptors:
        print("[agent6] /mcp: 0 tools (servers started but exposed nothing)", file=sys.stderr)
        return
    by_server: dict[str, list[str]] = {}
    for d in descriptors:
        by_server.setdefault(d.server_name, []).append(d.tool_name)
    print(f"[agent6] /mcp: {len(descriptors)} tools across {len(by_server)} server(s)")
    for server, tools in sorted(by_server.items()):
        print(f"  {server}: {len(tools)} tool(s)")
        for t in sorted(tools):
            print(f"    - {t}")


def repl_run_init(root: Path) -> None:
    """REPL /init: run the setup wizard. Prompts on a TTY (the REPL is
    interactive) and never overwrites existing files; the ecosystem is
    auto-detected (no hard-coded profile)."""
    try:
        rc = init_workspace(
            root,
            repo_config_target=repo_config_path_for(root),
            interactive=sys.stdin.isatty(),
        )
    except Exception as exc:
        print(f"[agent6] /init failed: {exc}", file=sys.stderr)
        return
    print("[agent6] /init: ok" if rc == 0 else f"[agent6] /init: exit {rc}", file=sys.stderr)
