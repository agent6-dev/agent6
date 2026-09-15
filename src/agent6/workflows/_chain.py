# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's detached commit chain: where the loop's commits go, and what the
worktree holds beyond them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent6.config import GitCommitConfig
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_commit,
    chain_dirty,
    chain_dirty_paths,
    chain_tip,
    diff_since,
    worktree_name_status,
    worktree_tree,
)
from agent6.git_ops import status as git_status

_DIRTY_NOTE_CAP = 500  # paths a dirty-worktree note counts before it says "+"


def commit_identity(commit: GitCommitConfig, trailer: str | None) -> CommitIdentity | None:
    """The author and provenance trailer of the run's commits; None leaves the
    identity to git's own resolution. `[git.commit].name` and `.email` are the
    only identity on a machine whose git has none: preflight accepts them, so
    dropping them would fail every chain commit with "Author identity
    unknown"."""
    if not (commit.name or commit.email or trailer):
        return None
    return CommitIdentity(name=commit.name or None, email=commit.email or None, trailer=trailer)


@dataclass(frozen=True, slots=True)
class RunChain:
    """The run's commit line in the repository at `root`.

    Per-step commits land on `ref` (`refs/agent6/<session>/head`, the gc
    anchor) through a temp index: HEAD, the operator's index and the checkout
    are never touched, so operator or model git activity mid-run cannot
    collide with the run's own record. `ref` None (a plan, an ask, a unit-test
    embedder) means the loop never commits. `branch` is the visible
    `refs/heads/<name>` advanced to the same tip (`[git].branch_per_run`; None
    keeps the hidden ref only). `fallback_parent` parents the chain's first
    commit while `ref` does not exist: HEAD at run start, None in an unborn
    repo. `untracked_at_start` holds the files that were untracked when the
    run started, repo-root-relative: the operator's, so no commit records
    them and no dirty check counts them, while a file the model creates is
    committed. `identity` is the author and trailer of every commit (None:
    git's own). `per_step` is `[git].commit_per_step`: False means the chain
    never advances, the work stays only in the worktree, and resume-from-git,
    `sessions diff`/`merge` and `/parallel` dispatch from a changed tree
    degrade. `base_sha` is where the run started, the base of the review diff
    and the gate's scope ("" when unknown)."""

    root: Path
    ref: str | None = None
    branch: str | None = None
    fallback_parent: str | None = None
    untracked_at_start: frozenset[str] = frozenset()
    identity: CommitIdentity | None = None
    per_step: bool = True
    base_sha: str = ""

    def commit(self, subject: str) -> str:
        """One commit of the worktree onto the chain; "" when nothing changed
        since the tip or there is no chain."""
        if self.ref is None:
            return ""
        return (
            chain_commit(
                self.root,
                subject,
                ref=self.ref,
                fallback_parent=self.fallback_parent,
                identity=self.identity,
                also_branch=self.branch,
                exclude=self.untracked_at_start,
            )
            or ""
        )

    def tip(self) -> str:
        """Tip of the chain; "" when it has no commits and no fallback (an
        unborn repo) or there is no chain."""
        if self.ref is None:
            return ""
        try:
            return chain_tip(self.root, self.ref) or self.fallback_parent or ""
        except (GitError, OSError):
            return ""

    def is_dirty(self) -> bool:
        """Whether the worktree holds content the chain does not (an edit a
        worker made via run_command that no auto-commit captured yet); raises
        GitError or OSError when git cannot say. Plain `git status` would be
        wrong: the operator's HEAD never moves during a run, so everything
        long since committed to the chain still reads as dirty against it.
        Without a chain, status decides."""
        if self.ref is not None:
            return chain_dirty(
                self.root, self.ref, self.fallback_parent, exclude=self.untracked_at_start
            )
        return not git_status(self.root, exclude=self.untracked_at_start).is_clean

    def dirty(self) -> bool:
        """`is_dirty`, reading clean when git cannot say, so a hiccup cannot
        wedge a detector."""
        try:
            return self.is_dirty()
        except (GitError, OSError):
            return False

    def dirty_note(self) -> str:
        """Summary suffix naming an uncommitted worktree, or "" when it is
        clean or there is no chain. An operator stop skips the checkpoint
        (committing over someone taking over would remove their choice to
        discard), and `sessions diff` and `merge` read git history, so the
        state is stated rather than left silent."""
        if self.ref is None:
            return ""
        try:
            paths = chain_dirty_paths(
                self.root,
                self.ref,
                self.fallback_parent,
                _DIRTY_NOTE_CAP,
                exclude=self.untracked_at_start,
            )
        except (GitError, OSError):
            return ""
        if not paths:
            return ""
        more = "+" if len(paths) == _DIRTY_NOTE_CAP else ""
        noun = "file" if len(paths) == 1 and not more else "files"
        return f"; worktree left dirty ({len(paths)}{more} {noun} uncommitted, not checkpointed)"

    def tree_sha(self) -> str:
        """Tree sha of the worktree's content (minus `untracked_at_start`),
        seeded on the chain tip like a chain commit; "" when git cannot say."""
        try:
            return worktree_tree(
                self.root, self.tip() or self.fallback_parent, self.untracked_at_start
            )
        except (GitError, OSError):
            return ""

    def checkpoint_head_sha(self) -> str:
        """The chain tip for the per-turn checkpoint (fork cuts its chain here;
        resume compares it to the live chain to warn about divergence); HEAD
        without a chain; "" when unreadable, since a checkpoint is best-effort
        recovery state and a missing sha must not crash the snapshot."""
        if self.ref is not None:
            return self.tip()
        try:
            return git_status(self.root).head_sha
        except (GitError, OSError):
            return ""

    def name_status(self) -> tuple[tuple[str, str], ...]:
        """`(status, path)` for every pending change, the operator's untracked
        files excluded."""
        return worktree_name_status(self.root, exclude=self.untracked_at_start)

    def diff_since_base(self) -> str:
        """The run's cumulative change: the base commit against the working
        tree, committed and uncommitted edits alike, the run's new untracked
        files as additions and the operator's excluded; "" with no base or on
        a git failure. Routed through git_ops so the repo-controlled
        fsmonitor, diff.external and hooks keys stay neutralised (a raw
        `git diff` would run a poisoned `.git/config` payload on the host)."""
        if not self.base_sha:
            return ""
        return diff_since(self.root, self.base_sha, exclude=self.untracked_at_start)
