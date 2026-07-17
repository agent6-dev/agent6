# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-cutting CLI helpers: run dirs, budget flags, key/root checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent6.config.layer import (
    resolved_state_dir,
)
from agent6.paths import (
    effective_user,
    is_root,
    root_optin_enabled,
)
from agent6.runs.id import RunIdError, list_run_ids
from agent6.runs.layout import RunLayout


def _add_budget_flags(parser: argparse.ArgumentParser) -> None:
    """Add per-run budget override flags (override ``[budget]`` config)."""
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Override [budget].best_effort_usd_limit for this run (0 disables)."
            " Passing the flag explicitly refuses to start when the worker model"
            " has no price data, since the limit could not be enforced."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override [budget].max_input_tokens for this run.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override [budget].max_output_tokens for this run.",
    )


def _add_sandbox_flags(parser: argparse.ArgumentParser, *, auto_approve: bool = True) -> None:
    """Add the per-invocation sandbox/approval override flags.

    ``--dangerously-disable-sandbox`` runs the agent's commands UNCONFINED on
    the host (equivalent to a one-off ``sandbox.profile = "none"``); the env
    ``AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`` does the same. ``--auto-approve``
    auto-approves ``run_command`` for this run (``sandbox.run_commands = yes``),
    which stays safe because the command is still jailed; pass ``auto_approve``
    False on commands (like ``machine run``) where approvals are config-driven.
    """
    parser.add_argument(
        "--dangerously-disable-sandbox",
        action="store_true",
        help=(
            "Run the agent's commands UNCONFINED on the host (no Landlock/"
            "seccomp/namespaces). Only for a disposable or already-isolated"
            " machine; the host becomes the only boundary."
        ),
    )
    if auto_approve:
        parser.add_argument(
            "--auto-approve",
            action="store_true",
            help=(
                "Auto-approve run_command for this run instead of prompting"
                " (same as sandbox.run_commands = yes). Safe while sandboxed;"
                " combined with --dangerously-disable-sandbox it hands the agent"
                " unprompted host access."
            ),
        )


def sgr(text: str, code: str) -> str:
    """Wrap *text* in an ANSI style, tty only, so piped output stays plain.
    The one place the CLI's faded/bold hints are styled (was duplicated in
    skills_cmds / memory_cmds)."""
    return f"\x1b[{code}m{text}\x1b[0m" if sys.stdout.isatty() else text


def _state_dir(repo_root: Path) -> Path:
    """The per-repo agent6 state dir (config + run state), out of the workspace.

    Resolved from the global ``[agent6].state_dir`` base (default
    ``$XDG_STATE_HOME/agent6``) plus a per-repo id, so this is cheap and works
    for read-only commands (``runs``/``history``/...) without a full config
    merge.
    """
    return resolved_state_dir(repo_root)


def _runs_dir(repo_root: Path) -> Path:
    """The ``runs/`` directory under the per-repo state dir."""
    return _state_dir(repo_root) / "runs"


# Every run-style bucket a listing spans: `run`/`plan` under runs/, `ask` under
# asks/, `machine create` authoring logs under machine-drafts/. Kept in one place
# so "most recent" and "search all history" match what `agent6 runs` lists.
RUN_BUCKETS: tuple[str, ...] = ("runs", "asks", "machine-drafts")


def all_run_dirs(repo_root: Path) -> list[Path]:
    """Every run directory across all RUN_BUCKETS. So latest-run resolution and
    history search cover asks and machine-drafts, not just runs/ (a bare `attach`
    or `history search` right after an `ask` must find that ask)."""
    state = _state_dir(repo_root)
    dirs: list[Path] = []
    for subdir in RUN_BUCKETS:
        bucket = state / subdir
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir())
    return dirs


def _machines_dir(repo_root: Path) -> Path:
    """The ``machines/`` directory under the per-repo state dir."""
    return _state_dir(repo_root) / "machines"


def resolve_run_layout(repo_root: Path, query: str) -> RunLayout:
    """Resolve a run id (or unique prefix) across every run-style bucket --
    ``runs/``, ``asks/``, and ``machine-drafts/`` -- returning a ``RunLayout``
    with the matching subdir.

    `agent6 run`/`plan` live under ``runs/``, `agent6 ask` under ``asks/``, and
    `machine create` authoring logs under ``machine-drafts/``; read-only
    commands (``runs show``/``watch``/``history search``) use this so anything
    a listing shows is also inspectable by id. Raises ``RunIdError`` if no run
    matches in any bucket.
    """
    if not query:
        raise RunIdError("empty run id")
    state = _state_dir(repo_root)
    exact: list[tuple[str, str]] = []
    prefix: list[tuple[str, str]] = []
    for subdir in ("runs", "asks", "machine-drafts"):
        d = state / subdir
        if not d.is_dir():
            continue
        for rid in list_run_ids(d):
            if rid == query:
                exact.append((subdir, rid))
            elif rid.startswith(query):
                prefix.append((subdir, rid))
    if len(exact) == 1:
        subdir, rid = exact[0]
        return RunLayout(state_dir=state, run_id=rid, subdir=subdir)
    if len(exact) > 1:
        preview = ", ".join(f"{subdir}/{rid}" for subdir, rid in sorted(exact)[:5])
        raise RunIdError(
            f"run id {query!r} is ambiguous ({len(exact)} exact matches): {preview}",
            ambiguous=True,
        )
    if len(prefix) == 1:
        subdir, rid = prefix[0]
        return RunLayout(state_dir=state, run_id=rid, subdir=subdir)
    if len(prefix) > 1:
        preview = ", ".join(f"{subdir}/{rid}" for subdir, rid in sorted(prefix)[:5])
        raise RunIdError(
            f"run id {query!r} is ambiguous ({len(prefix)} matches): {preview}",
            ambiguous=True,
        )
    raise RunIdError(f"no run matches {query!r} under {state}/(runs|asks|machine-drafts)")


def _enforce_root_policy(allow_root: bool) -> int | None:
    """Gate running as root behind an explicit opt-in.

    Returns a non-zero exit code (to refuse) when running as root without
    ``--allow-root`` / ``AGENT6_ALLOW_ROOT=1``; returns None to proceed. When
    proceeding as root it prints a loud banner. We deliberately do NOT drop
    privileges: under sudo the LLM's verify/run commands need to run as root
    inside the jail, so the jail, not the process uid, is the boundary.
    """
    if not is_root():
        return None
    if not root_optin_enabled(allow_root):
        print(
            "[agent6] REFUSING to run as root. Running an LLM-driven agent as root"
            " is dangerous. If a task genuinely needs it, re-run with --allow-root"
            " (or set AGENT6_ALLOW_ROOT=1).",
            file=sys.stderr,
        )
        return 2
    user = effective_user()
    who = f" on behalf of {user.name} (uid {user.uid})" if user.via_sudo else ""
    print(
        f"[agent6] WARNING: running as root{who}. The LLM's commands execute as"
        " root inside the jail; files agent6 writes under the repo are chowned"
        " back to you when invoked via sudo. Proceed with care.",
        file=sys.stderr,
    )
    return None
