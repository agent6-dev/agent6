# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init`: scaffold a workspace and offer the git setup."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective
from agent6.errors import OperatorError
from agent6.git_ops import (
    GitError,
    commit_paths,
    init_repo,
    is_git_repo,
    paths_dirty,
    unignored,
)
from agent6.init import _ask, init_workspace
from agent6.paths import chown_to_real_user, repo_config_path
from agent6.ui.cli._common import error

_SCAFFOLD_COMMIT_MESSAGE = "chore: scaffold agent6 config"


def _workspace_rel_paths(root: Path, created: tuple[Path, ...]) -> tuple[str, ...]:
    """Return the created paths that exist under the root, relative to it."""
    return tuple(
        str(p.relative_to(root)) for p in created if p.exists() and root in p.resolve().parents
    )


def _scaffold_rel_paths(root: Path, created: tuple[Path, ...]) -> tuple[str, ...]:
    """Return the repo-relative scaffold files git would record.

    The per-repo config lives under the state dir, never here. `unignored` drops what the
    just-written .gitignore covers, so nothing is ever `git add -f`; a path with nothing
    pending is dropped too, so the commit line names what the commit holds.

    Args:
        root: The workspace.
        created: The paths init wrote.
    """
    candidates = unignored(root, _workspace_rel_paths(root, created))
    return tuple(rel for rel in candidates if paths_dirty(root, (rel,)))


def _offer_git_setup(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """Leave the repo ready for `agent6 run`.

    In a non-repo, offer to `git init` and commit the scaffold (non-interactively, print the
    commands); in a repo, offer to commit the uncommitted scaffold, which would otherwise
    make `agent6 run` refuse on a dirty tree.

    Args:
        root: The workspace.
        created: The paths init wrote.
        interactive: Ask before acting.
    """
    if is_git_repo(root):
        _offer_scaffold_commit(root, created, interactive=interactive)
        return
    print()
    if not interactive:
        print(f"Note: {root} is not a git repository; `agent6 run`/`plan` need one.")
        rel = _workspace_rel_paths(root, created)
        if rel:
            print(f'  Run: git init && git add {" ".join(rel)} && git commit -m "initial commit"')
        else:
            print("  Run: git init, then review and commit the files you want tracked.")
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
        print(f"  Set git user.name / user.email, then: git add {' '.join(rel)} && git commit")


def _offer_scaffold_commit(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """Offer to commit the scaffold in an existing repo.

    Auto-yes when non-interactive; when declined or the commit fails, print the exact
    command so the advertised next step works.

    Args:
        root: The repo.
        created: The paths init wrote.
        interactive: Ask before committing.
    """
    rel = _scaffold_rel_paths(root, created)
    if not rel:
        return
    try:
        if not paths_dirty(root, rel):
            # Already committed; a whole-tree check would false-trigger on unrelated work.
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


def _print_next_steps(cwd: Path, config_path: Path | None) -> None:
    """Print the commands still between this repo and a first run.

    `connect` and `model` appear only while the effective config lacks a provider or a
    worker model.

    Args:
        cwd: The repo.
        config_path: The `--config` file, if any.
    """
    try:
        cfg: Config | None = load_effective(cwd, config_path).config
    except ConfigError:
        cfg = None
    print()
    print("Next:")
    if cfg is None or not cfg.providers:
        print("  agent6 connect                 # add a provider + API key (global)")
    if cfg is None or cfg.models.resolve("worker") is None:
        print("  agent6 model worker <provider> <model>   # pick your worker model")
    print("  agent6 config show             # audit the effective config")
    gated = cfg is not None and bool(cfg.harness.verify_command)
    print('  agent6 run "<task>"' + ("" if gated else "            # verify is inferred per run"))


def _cmd_init(*, ecosystem: str, assume_yes: bool = False, config_path: Path | None = None) -> int:
    """Scaffold the workspace, offer the git setup, and print the next steps.

    Args:
        ecosystem: The language ecosystem the scaffold targets.
        assume_yes: Take every default without a TTY.
        config_path: The `--config` file, if any.

    Returns:
        The exit code; 2 without a TTY or `--yes`.

    Raises:
        OperatorError: The effective config is invalid; the message names the way out.
    """
    cwd = Path.cwd()
    target = repo_config_path(cwd)
    if not assume_yes and not sys.stdin.isatty():
        # Consent to write files comes from a TTY or --yes.
        error("no input. stdin is not a TTY; re-run with --yes to accept every default.")
        return 2
    interactive = not assume_yes
    # A scaffold path init leaves untouched is the operator's: excluded from the commit, reported.
    scaffold_all = (cwd / "AGENTS.md", cwd / ".gitignore")
    missing_before = tuple(p for p in scaffold_all if not p.exists())
    if is_git_repo(cwd):
        theirs = tuple(p for p in scaffold_all if paths_dirty(cwd, (str(p.relative_to(cwd)),)))
    else:
        theirs = tuple(p for p in scaffold_all if p.exists())
    try:
        try:
            rc = init_workspace(
                cwd,
                ecosystem=ecosystem,
                repo_config_target=target,
                interactive=interactive,
                config_path=config_path,
            )
        except ConfigError as exc:
            # init is also the command that repairs a setup, so the refusal carries the way out.
            raise OperatorError(
                f"{exc}\nFix or delete the invalid config, following the error above,"
                " then re-run `agent6 init`."
            ) from exc
        if rc == 0:
            # Only the repo-tracked scaffold; the per-repo config lives under the state dir.
            _offer_git_setup(
                cwd, tuple(p for p in scaffold_all if p not in theirs), interactive=interactive
            )
            # Only where a scaffold commit was on the table: outside a repo nothing was left out.
            dirty = (
                tuple(p for p in theirs if paths_dirty(cwd, (str(p.relative_to(cwd)),)))
                if is_git_repo(cwd)
                else ()
            )
            if dirty:
                names = ", ".join(sorted(p.name for p in dirty))
                print(f"  left uncommitted (already edited): {names}")
            _print_next_steps(cwd, config_path)
        return rc
    finally:
        # Under sudo, nothing root-owned stays in the real user's trees, even after a failed step.
        chown_to_real_user(target.parent)
        for path in missing_before:
            if path.exists():
                chown_to_real_user(path)
