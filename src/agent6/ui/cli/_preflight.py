# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive pre-run confirm prompts `agent6 run`/`resume` inject into
the lifecycle: run-on-run-branch and unconfined autorun. The non-interactive
guards live in `agent6.app.preflight`."""

from __future__ import annotations

import sys

from agent6.config import Config
from agent6.types import IsolationLevel
from agent6.ui.cli._steer import tty_prompt


def confirm_run_on_run_branch(base_branch: str) -> bool:
    """The checkout is on another run's branch (agent6/<id>); a new run would branch
    off it. Confirm before proceeding. A non-interactive caller (a detached TUI/web
    run) has no terminal to prompt, so it warns and proceeds."""
    warning = (
        f"[agent6] You are on run branch '{base_branch}', not a base branch. A new run\n"
        "  branches off it -- you may have meant to merge it (agent6 sessions merge) or\n"
        "  switch back (git switch <base>) first."
    )
    if not sys.stdin.isatty():
        print(warning + " Proceeding (non-interactive).", file=sys.stderr)
        return True
    print(warning, file=sys.stderr)
    try:
        ans = input("  Start a new run here anyway? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in {"y", "yes"}


def confirm_replay_after_crash(iteration: int, tools: tuple[str, ...]) -> bool:
    """Resume found a mid-turn-crash marker for the turn about to re-run: its
    tools may have partially applied, and replaying can repeat a
    non-idempotent effect. Interactive: ask, default NO (abort and inspect).
    Headless: warn loudly and proceed -- the at-least-once recovery a detached
    resume always had, now with the risk named."""
    named = ", ".join(tools) if tools else "unknown tools"
    warning = (
        f"[agent6] The previous run died mid-turn (iteration {iteration}; {named}).\n"
        "  Its tools may have PARTIALLY APPLIED; replaying the turn can repeat a\n"
        "  non-idempotent effect (an appending command, a migration, an MCP call)."
    )
    if not sys.stdin.isatty():
        print(warning + " Proceeding (non-interactive).", file=sys.stderr)
        return True
    print(warning, file=sys.stderr)
    try:
        ans = input("  Re-run the turn anyway? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in {"y", "yes"}


def confirm_unconfined_autorun(isolation: IsolationLevel, cfg: Config) -> bool:
    """The one genuinely dangerous combination: the sandbox is OFF and
    run_command is auto-approved, so the agent can run any command on the host
    with no confinement and no prompt. Get one explicit consent at startup when
    interactive; proceed with a loud warning when not (the explicit opt-outs
    are already the consent, and machines/CI must not block). Not a per-command
    guard -- once unconfined, guarding individual commands would be theatre.

    Returns True to proceed, False to abort.
    """
    if isolation != "none" or cfg.sandbox.run_commands != "yes":
        return True
    print(
        "[agent6] DANGER: the sandbox is DISABLED and run_command is"
        " AUTO-APPROVED. The agent can run ANY command on this host with no"
        " confinement and no prompt.",
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        print("[agent6] proceeding (non-interactive).", file=sys.stderr)
        return True
    answer = tty_prompt("Continue? [y/N]: ")
    return (answer or "").strip().lower() in {"y", "yes"}
