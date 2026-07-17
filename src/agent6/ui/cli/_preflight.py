# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive pre-run confirm prompts `agent6 run`/`resume` inject into
the lifecycle: run-on-run-branch, unconfined autorun, and the
``git.branch_from`` start-point choice. The non-interactive guards live in
`agent6.app.preflight`."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.app.preflight import BranchChoice, resolve_base_branch
from agent6.config import Config
from agent6.types import SandboxProfile
from agent6.ui.cli._steer import tty_prompt


def confirm_run_on_run_branch(base_branch: str) -> bool:
    """The checkout is on another run's branch (agent6/<id>); a new run would branch
    off it. Confirm before proceeding. A non-interactive caller (a detached TUI/web
    run) has no terminal to prompt, so it warns and proceeds."""
    warning = (
        f"[agent6] You are on run branch '{base_branch}', not a base branch. A new run\n"
        "  branches off it -- you may have meant to merge it (agent6 runs merge) or\n"
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


def confirm_unconfined_autorun(selected_profile: SandboxProfile, cfg: Config) -> bool:
    """The one genuinely dangerous combination: the sandbox is OFF and
    run_command is auto-approved, so the agent can run any command on the host
    with no confinement and no prompt. Get one explicit consent at startup when
    interactive; proceed with a loud warning when not (the explicit opt-outs
    are already the consent, and machines/CI must not block). Not a per-command
    guard -- once unconfined, guarding individual commands would be theatre.

    Returns True to proceed, False to abort.
    """
    if selected_profile != "none" or cfg.sandbox.run_commands != "yes":
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


def _ask_branch_start_point(current_branch: str, base: str) -> BranchChoice:
    """The ``branch_from = "ask"`` prompt: on a terminal, choose base / stack /
    abort; headless falls back to the clean base (the un-surprising choice)."""
    if not sys.stdin.isatty():
        return BranchChoice(start_point=base)
    print(
        f"[agent6] You are on {current_branch!r}, not the base branch {base!r}.",
        file=sys.stderr,
    )
    ans = tty_prompt(
        f"  Cut this run from: [b]ase {base!r} (clean start) /"
        f" [s]tack on {current_branch!r} / [a]bort? [b]: ",
        fall_back_to_stdin=False,
    )
    choice = (ans or "").strip().lower()
    if choice in {"s", "stack"}:
        return BranchChoice(start_point=None)
    if choice in {"a", "abort"}:
        return BranchChoice(start_point=None, abort=True)
    return BranchChoice(start_point=base)


def choose_branch_start_point(cfg: Config, state_dir: Path, current_branch: str) -> BranchChoice:
    """Decide where the run branch is cut from, per ``git.branch_from``:
    ``current`` stacks on HEAD; ``base`` cuts from the resolved base line;
    ``ask`` prompts (base / stack / abort) when you are not already on the base.
    No decision to make when the current branch IS the base."""
    if cfg.git.branch_from == "current":
        return BranchChoice(start_point=None)
    base = resolve_base_branch(state_dir, current_branch)
    if current_branch == base:
        return BranchChoice(start_point=None)  # already on the base; nothing to stack on
    if cfg.git.branch_from == "base":
        return BranchChoice(start_point=base)
    return _ask_branch_start_point(current_branch, base)
