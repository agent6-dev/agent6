# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init` command (scaffold a workspace + offer git setup)."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.config_layer import repo_config_path_for
from agent6.git_ops import (
    GitError,
    commit_paths,
    init_repo,
    is_git_repo,
    unignored,
)
from agent6.init import _ask, init_workspace
from agent6.paths import chown_to_real_user


def _offer_git_setup(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """When *root* is not a git repo, offer to `git init` + commit the scaffold
    interactively; non-interactively just print a note. agent6 run/plan need a
    repo, so a fresh `init` in a bare directory shouldn't leave the user one
    confusing error away from their first run."""
    if is_git_repo(root):
        return
    print()
    if not interactive:
        print(f"Note: {root} is not a git repository — `agent6 run`/`plan` need one.")
        print('  Run: git init && git add -A && git commit -m "initial commit"')
        return
    if not _ask("This directory is not a git repository — initialise one now?", default=True):
        print("  Skipped. `agent6 run` needs a repo; run `git init` here first.")
        return
    try:
        init_repo(root)
    except GitError as exc:
        print(f"  git init failed: {exc}")
        return
    print("  created: .git/  (git init)")
    # Commit only the scaffold git tracks (AGENTS.md, .gitignore). The per-repo
    # config lives out of the workspace under the state dir, so it is never a
    # candidate here; filter to paths under root defensively, then unignored()
    # drops anything the just-written .gitignore covers so we never `add -f`.
    rel = unignored(
        root,
        tuple(
            str(p.relative_to(root)) for p in created if p.exists() and root in p.resolve().parents
        ),
    )
    if not rel:
        print("  (nothing to commit — the created files are all gitignored)")
        return
    if not _ask("Commit the files agent6 just created?", default=True):
        print(f"  Not committed. When ready: git add {' '.join(rel)} && git commit")
        return
    try:
        commit_paths(root, "chore: scaffold agent6 config", rel)
        print(f"  committed the agent6 scaffold ({', '.join(rel)})")
    except GitError as exc:
        # Most likely a missing git identity, actionable, not fatal.
        print(f"  commit skipped: {exc}")
        print("  Set git user.name / user.email, then: git add -A && git commit")


def _cmd_init(*, profile: str, assume_yes: bool = False) -> int:
    cwd = Path.cwd()
    target = repo_config_path_for(cwd)
    interactive = not assume_yes and sys.stdin.isatty()
    rc = init_workspace(
        cwd,
        profile=profile,
        repo_config_target=target,
        interactive=interactive,
    )
    if rc == 0:
        # Only the repo-tracked scaffold; the per-repo config is out of the
        # workspace (under the state dir) and never committed.
        _offer_git_setup(
            cwd,
            (cwd / "AGENTS.md", cwd / ".gitignore"),
            interactive=interactive,
        )
    # Don't leave root-owned scaffolding in the user's repo (sudo case).
    chown_to_real_user(target.parent)
    return rc
