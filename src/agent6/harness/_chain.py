# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's detached commit chain.

Where the loop's commits go, and what the worktree holds beyond them.
"""

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
    """Return the author and provenance trailer of the run's commits.

    `[git.commit].name` and `.email` are the only identity on a machine whose git
    has none, and preflight accepts them.

    Args:
        commit: The `[git.commit]` config.
        trailer: The provenance trailer line, or None for none.

    Returns:
        The identity, or None to leave it to git's own resolution.
    """
    if not (commit.name or commit.email or trailer):
        return None
    return CommitIdentity(name=commit.name or None, email=commit.email or None, trailer=trailer)


@dataclass(frozen=True, slots=True)
class RunChain:
    """The run's commit line in the repository at `root`.

    Per-step commits land on `ref` through a temp index: HEAD, the operator's
    index and the checkout are never touched, so git activity mid-run cannot
    collide with the run's own record.

    Attributes:
        root: The repository root.
        ref: The hidden ref the commits land on (`refs/agent6/<session>/head`, the
            gc anchor); None (a plan, an ask, an embedder) means the loop never commits.
        branch: The visible `refs/heads/<name>` advanced to the same tip
            (`[git].branch_per_run`); None keeps the hidden ref only.
        fallback_parent: The parent of the chain's first commit while `ref` does not
            exist: HEAD at run start, None in an unborn repo.
        untracked_at_start: The files untracked when the run started, repo-root
            relative: the operator's, so no commit records them and no dirty check
            counts them; a file the model creates is committed.
        identity: The author and trailer of every commit; None leaves it to git.
        per_step: `[git].commit_per_step`; False keeps the work in the worktree only,
            and resume-from-git, `sessions diff`, `merge` and a `/parallel` dispatch
            from a changed tree degrade.
        base_sha: Where the run started, the base of the review diff and the gate's
            scope; "" when unknown.
    """

    root: Path
    ref: str | None = None
    branch: str | None = None
    fallback_parent: str | None = None
    untracked_at_start: frozenset[str] = frozenset()
    identity: CommitIdentity | None = None
    per_step: bool = True
    base_sha: str = ""

    def commit(self, subject: str) -> str:
        """Commit the worktree onto the chain.

        Args:
            subject: The commit subject.

        Returns:
            The commit's sha; "" when nothing changed since the tip or there is no chain.
        """
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
        """Return the chain's tip.

        Returns:
            The tip's sha; "" when the chain has no commits and no fallback, or there
            is no chain.
        """
        if self.ref is None:
            return ""
        try:
            return chain_tip(self.root, self.ref) or self.fallback_parent or ""
        except (GitError, OSError):
            return ""

    def is_dirty(self) -> bool:
        """Return whether the worktree holds content the chain does not.

        The operator's HEAD never moves during a run, so plain `git status` would
        read everything committed to the chain as dirty; without a chain, status
        decides.

        Returns:
            True when an edit no commit captured yet is in the worktree.

        Raises:
            GitError: git could not read the worktree.
            OSError: The repository could not be read.
        """
        if self.ref is not None:
            return chain_dirty(
                self.root, self.ref, self.fallback_parent, exclude=self.untracked_at_start
            )
        return not git_status(self.root, exclude=self.untracked_at_start).is_clean

    def dirty(self) -> bool:
        """Return `is_dirty`, reading clean when git cannot say.

        Returns:
            True when the worktree holds uncommitted content; False on a git fault.
        """
        try:
            return self.is_dirty()
        except (GitError, OSError):
            return False

    def dirty_note(self) -> str:
        """Return a summary suffix naming an uncommitted worktree.

        An operator stop skips the checkpoint, and `sessions diff` and `merge` read
        git history, so the receipt states the worktree's state.

        Returns:
            The suffix, or "" when the worktree is clean or there is no chain.
        """
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
        """Return the tree sha of the worktree's content, minus `untracked_at_start`.

        Returns:
            The sha, seeded on the chain tip like a chain commit; "" when git cannot say.
        """
        try:
            return worktree_tree(
                self.root, self.tip() or self.fallback_parent, self.untracked_at_start
            )
        except (GitError, OSError):
            return ""

    def checkpoint_head_sha(self) -> str:
        """Return the sha the per-turn checkpoint records.

        A fork cuts its chain here; a resume compares it to the live chain to warn
        about divergence.

        Returns:
            The chain tip; HEAD without a chain; "" when unreadable, since a missing
            sha must not crash the snapshot.
        """
        if self.ref is not None:
            return self.tip()
        try:
            return git_status(self.root).head_sha
        except (GitError, OSError):
            return ""

    def name_status(self) -> tuple[tuple[str, str], ...]:
        """Return (status, path) for every pending change, the operator's untracked files excluded.

        Returns:
            The pairs in `git status` order.
        """
        return worktree_name_status(self.root, exclude=self.untracked_at_start)

    def diff_since_base(self) -> str:
        """Return the run's cumulative change: the base commit against the working tree.

        Committed and uncommitted edits alike, the run's new untracked files as
        additions, the operator's excluded. Routed through git_ops so a poisoned
        `.git/config` (fsmonitor, diff.external, hooks) runs nothing on the host.

        Returns:
            The unified diff; "" with no base or on a git failure.
        """
        if not self.base_sha:
            return ""
        return diff_since(self.root, self.base_sha, exclude=self.untracked_at_start)
