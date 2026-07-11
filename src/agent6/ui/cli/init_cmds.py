# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init` command (scaffold a workspace + offer git setup)."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.config import ConfigError
from agent6.config_layer import repo_config_path_for
from agent6.git_ops import (
    GitError,
    commit_paths,
    init_repo,
    is_git_repo,
    paths_dirty,
    unignored,
)
from agent6.init import _ask, init_workspace
from agent6.paths import chown_to_real_user

_SCAFFOLD_COMMIT_MESSAGE = "chore: scaffold agent6 config"


def _scaffold_rel_paths(root: Path, created: tuple[Path, ...]) -> tuple[str, ...]:
    """The repo-relative scaffold files git would track. The per-repo config
    lives out of the workspace under the state dir, so it is never a candidate
    here; filter to paths under root defensively, then unignored() drops
    anything the just-written .gitignore covers so we never `git add -f`."""
    return unignored(
        root,
        tuple(
            str(p.relative_to(root)) for p in created if p.exists() and root in p.resolve().parents
        ),
    )


def _offer_git_setup(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """Leave the repo ready for the advertised `agent6 run`: in a non-repo,
    offer to `git init` + commit the scaffold (non-interactively just print a
    note); in an existing repo, offer to commit the uncommitted scaffold, which
    would otherwise make `agent6 run` refuse on a dirty tree."""
    if is_git_repo(root):
        _offer_scaffold_commit(root, created, interactive=interactive)
        return
    print()
    if not interactive:
        print(f"Note: {root} is not a git repository; `agent6 run`/`plan` need one.")
        print('  Run: git init && git add -A && git commit -m "initial commit"')
        return
    if not _ask("This directory is not a git repository. Initialise one now?", default=True):
        print("  Skipped. `agent6 run` needs a repo; run `git init` here first.")
        return
    try:
        init_repo(root)
    except GitError as exc:
        print(f"  git init failed: {exc}")
        return
    print("  created: .git/  (git init)")
    rel = _scaffold_rel_paths(root, created)
    if not rel:
        print("  (nothing to commit; the created files are all gitignored)")
        return
    if not _ask("Commit the files agent6 just created?", default=True):
        print(f"  Not committed. When ready: git add {' '.join(rel)} && git commit")
        return
    try:
        commit_paths(root, _SCAFFOLD_COMMIT_MESSAGE, rel)
        print(f"  committed the agent6 scaffold ({', '.join(rel)})")
    except GitError as exc:
        # Most likely a missing git identity, actionable, not fatal.
        print(f"  commit skipped: {exc}")
        print("  Set git user.name / user.email, then: git add -A && git commit")


def _offer_scaffold_commit(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """*root* is already a git repo, so the scaffold init wrote sits uncommitted
    and `agent6 run` refuses a dirty tree. Offer to commit it (auto-yes when
    non-interactive, i.e. --yes); when declined or the commit fails, print the
    exact command so the advertised next step works."""
    rel = _scaffold_rel_paths(root, created)
    if not rel:
        return
    try:
        if not paths_dirty(root, rel):
            # Scaffold already committed; nothing to commit for these paths.
            # (Whole-tree is_clean would false-trigger on unrelated WIP and then
            # fail the path-limited commit with "nothing to commit".)
            return
    except GitError:
        return
    manual = f"git add {' '.join(rel)} && git commit -m '{_SCAFFOLD_COMMIT_MESSAGE}'"
    print()
    if interactive and not _ask(
        "Commit the agent6 scaffold now (`agent6 run` needs a clean tree)?", default=True
    ):
        print(f"  Not committed. Before `agent6 run`: {manual}")
        return
    try:
        commit_paths(root, _SCAFFOLD_COMMIT_MESSAGE, rel)
    except GitError as exc:
        print(f"  commit failed: {exc}")
        print(f"  Commit it yourself before `agent6 run`: {manual}")
        return
    print(f"  committed the agent6 scaffold ({', '.join(rel)})")


def _print_next_steps() -> None:
    print()
    print("Next:")
    print("  agent6 connect                 # add a provider + API key (global), if not done")
    print("  agent6 model worker <provider> <model>   # pick your worker model")
    print("  agent6 config show             # audit the effective config")
    print('  agent6 run "<task>"            # verify is inferred if you skipped it above')


def _cmd_init(*, profile: str, assume_yes: bool = False) -> int:
    cwd = Path.cwd()
    target = repo_config_path_for(cwd)
    if not assume_yes and not sys.stdin.isatty():
        # Refuse rather than silently take every default and write files
        # (matching `agent6 connect`); consent comes from a TTY or --yes.
        print(
            "ERROR: no input. stdin is not a TTY; re-run with --yes to accept every default.",
            file=sys.stderr,
        )
        return 2
    interactive = not assume_yes
    try:
        rc = init_workspace(
            cwd,
            profile=profile,
            repo_config_target=target,
            interactive=interactive,
        )
    except ConfigError as exc:
        # init loads the effective config to infer a verify command. A
        # pre-existing invalid config is the user's to fix, not an agent6 crash;
        # surface it the way every other config-loading command does (a clean
        # CONFIG ERROR, not the generic "unexpected" traceback handler) -- doubly
        # so here, since init is the command a user runs to repair their setup.
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        print(
            f"\nFix or delete the per-repo config at {target}, then re-run `agent6 init`.",
            file=sys.stderr,
        )
        return 2
    if rc == 0:
        # Only the repo-tracked scaffold; the per-repo config is out of the
        # workspace (under the state dir) and never committed.
        _offer_git_setup(
            cwd,
            (cwd / "AGENTS.md", cwd / ".gitignore"),
            interactive=interactive,
        )
        _print_next_steps()
    # Don't leave root-owned scaffolding in the user's repo (sudo case).
    chown_to_real_user(target.parent)
    return rc
