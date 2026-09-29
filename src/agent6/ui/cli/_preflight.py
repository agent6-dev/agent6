# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive confirm prompts `run` and `resume` inject into the lifecycle.

The non-interactive guards live in `agent6.app.preflight`.
"""

from __future__ import annotations

import sys

from agent6 import kinds
from agent6.config import Config
from agent6.ui.cli import _steer


def confirm_run_on_run_branch(base_branch: str) -> bool:
    """Confirm a new run on another run's branch, which it would branch off.

    Args:
        base_branch: The checked-out run branch.

    Returns:
        Whether to proceed; a non-interactive caller warns and proceeds.
    """
    warning = (
        f"[agent6] You are on run branch '{base_branch}', so a new run branches off\n"
        "  it. Merge it (agent6 sessions merge) or switch back (git switch <base>)\n"
        "  first."
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
    """Confirm replaying a turn whose tools may have partially applied before a crash.

    Args:
        iteration: The turn about to re-run.
        tools: The tools the crashed turn had dispatched.

    Returns:
        Whether to proceed; interactive defaults to no, headless warns and proceeds.
    """
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


def confirm_unconfined_autorun(isolation: kinds.IsolationLevel, cfg: Config) -> bool:
    """Confirm once, at startup, a run with the sandbox off and run_command auto-approved.

    One consent when interactive, a loud warning when not: the explicit opt-outs are
    the consent, and a machine must not block. Never per command: once unconfined,
    guarding single commands would be theatre.

    Args:
        isolation: The resolved isolation level.
        cfg: The run's config.

    Returns:
        Whether to proceed.
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
    answer = _steer.tty_prompt("Continue? [y/N]: ")
    return (answer or "").strip().lower() in {"y", "yes"}
