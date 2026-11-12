# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's detached commit chain.

Where the loop's commits go, and what the worktree holds beyond them.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6 import errors, review_records
from agent6.config import GitCommitConfig
from agent6.vcs import git_ops, git_read, git_snapshots

_DIRTY_NOTE_CAP = 500  # paths a dirty-worktree note counts before it says "+"


def commit_identity(commit: GitCommitConfig, trailer: str | None) -> git_read.CommitIdentity | None:
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
    return git_read.CommitIdentity(
        name=commit.name or None, email=commit.email or None, trailer=trailer
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RunChain:
    """The run's commit line in the repository at `root`.

    Per-turn commits land on `ref` through a temp index: HEAD, the operator's
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
        hidden: `[sandbox].hide_paths` inside the repository, repo-root relative: the run
            never sees them, so no tree it writes or compares holds them.
        identity: The author and trailer of every commit; None leaves it to git.
        per_turn: `[git].commit_per_turn`; False keeps the work in the worktree only,
            and resume-from-git, `sessions diff`, `merge` and a `/parallel` dispatch
            from a changed tree degrade.
        base_sha: Where the run's un-landed work starts, the base of the review diff and
            the gate's scope; "" when unknown.
    """

    root: pathlib.Path
    ref: str | None = None
    branch: str | None = None
    fallback_parent: str | None = None
    untracked_at_start: frozenset[str] = frozenset()
    hidden: frozenset[str] = frozenset()
    identity: git_read.CommitIdentity | None = None
    per_turn: bool = True
    base_sha: str = ""

    @property
    def excluded(self) -> frozenset[str]:
        """Return the paths every tree the chain writes or compares leaves out."""
        return self.untracked_at_start | self.hidden

    def commit(self, subject: str, *, task_id: str | None = None) -> str:
        """Commit the worktree onto the chain.

        Args:
            subject: The commit subject.
            task_id: The task graph subtask the commit worked on, carried as the
                `Agent6-Task` trailer `sessions merge` groups by; None for none.

        Returns:
            The commit's sha; "" when nothing changed since the tip or there is no chain.
        """
        if self.ref is None:
            return ""
        return (
            git_ops.chain_commit(
                self.root,
                subject,
                ref=self.ref,
                fallback_parent=self.fallback_parent,
                trailers={git_read.TASK_TRAILER: task_id} if task_id else None,
                identity=self.identity,
                also_branch=self.branch,
                exclude=self.excluded,
            )
            or ""
        )

    def left_out(self, paths: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the given paths the chain's commits leave out, by reason.

        Args:
            paths: Repo-relative paths the run wrote.

        Returns:
            The paths git ignores, then the paths untracked when the run started; both
            empty when the run never commits, and the first empty when git cannot say.
        """
        if self.ref is None or not self.per_turn:
            return (), ()
        untracked = tuple(p for p in paths if p in self.untracked_at_start)
        rest = tuple(p for p in paths if p not in self.untracked_at_start)
        try:
            kept = git_read.unignored(self.root, rest)
        except (errors.GitError, OSError):
            return (), untracked
        return tuple(p for p in rest if p not in kept), untracked

    def tip(self) -> str:
        """Return the chain's tip.

        Returns:
            The tip's sha; "" when the chain has no commits and no fallback, or there
            is no chain.
        """
        if self.ref is None:
            return ""
        try:
            return git_read.chain_tip(self.root, self.ref) or self.fallback_parent or ""
        except (errors.GitError, OSError):
            return ""

    def head_sha(self) -> str:
        """Return the checkout's HEAD commit, or "" when it has none."""
        try:
            return git_read.chain_tip(self.root, "HEAD") or ""
        except (errors.GitError, OSError):
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
            return git_read.chain_dirty(
                self.root, self.ref, self.fallback_parent, exclude=self.excluded
            )
        return not git_read.status(self.root, exclude=self.excluded).is_clean

    def dirty(self) -> bool:
        """Return `is_dirty`, reading clean when git cannot say.

        Returns:
            True when the worktree holds uncommitted content; False on a git fault.
        """
        try:
            return self.is_dirty()
        except (errors.GitError, OSError):
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
            paths = git_read.chain_dirty_paths(
                self.root,
                self.ref,
                self.fallback_parent,
                _DIRTY_NOTE_CAP,
                exclude=self.excluded,
            )
        except (errors.GitError, OSError):
            return ""
        if not paths:
            return ""
        more = "+" if len(paths) == _DIRTY_NOTE_CAP else ""
        noun = "file" if len(paths) == 1 and not more else "files"
        return f"; worktree left dirty ({len(paths)}{more} {noun} uncommitted, not checkpointed)"

    def tree_sha(self) -> str:
        """Return the tree sha of the worktree's content, minus the excluded paths.

        Returns:
            The sha, seeded on the record's head like a chain commit; "" when git cannot say.
        """
        try:
            return git_read.worktree_tree(
                self.root, self.checkpoint_head_sha() or self.fallback_parent, self.excluded
            )
        except (errors.GitError, OSError):
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
            return git_read.status(self.root).head_sha
        except (errors.GitError, OSError):
            return ""

    def name_status(self) -> tuple[tuple[str, str], ...]:
        """Return the changes the next checkpoint records against the chain tip.

        Returns:
            Status and destination path pairs, with rename scores; empty without a chain.
        """
        if self.ref is None:
            return ()
        parent = git_read.chain_tip(self.root, self.ref) or self.fallback_parent
        old_tree = git_read.tree_sha(self.root, parent) if parent else git_read.EMPTY_TREE
        new_tree = git_read.worktree_tree(self.root, parent, self.excluded)
        return git_read.tree_name_status(self.root, old_tree, new_tree)

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
        return git_read.diff_since(self.root, self.base_sha, exclude=self.excluded)

    def review_target(self) -> review_records.ReviewTarget | None:
        """Return the base and head a panel reviews.

        The head is the record's head (the chain tip; the checkout's HEAD under model git
        control, where the model's commits are the record), or the worktree's tree while
        edits sit uncommitted (a turn the gate failed is not checkpointed), so the panel
        reads what the model wrote. Panel preparation retains that exact tree on a separate
        review ref; reading the target never checkpoints the run.

        Returns:
            The target; None for an empty unborn repository.

        Raises:
            GitError: The recorded head or worktree cannot be read.
            OSError: The repository cannot be read.
        """
        tip = (
            git_read.chain_tip(self.root, self.ref) or self.fallback_parent
            if self.ref is not None
            else git_read.chain_tip(self.root, "HEAD")
        )
        committed = git_read.tree_sha(self.root, tip) if tip else git_read.EMPTY_TREE
        worktree = git_read.worktree_tree(self.root, tip, self.excluded)
        if not tip and worktree == git_read.EMPTY_TREE:
            return None
        uncommitted = worktree != committed
        return review_records.ReviewTarget(
            base_commit=self.base_sha or None,
            base_tree=(
                git_read.tree_sha(self.root, self.base_sha)
                if self.base_sha
                else git_read.EMPTY_TREE
            ),
            head_commit=None if uncommitted else tip,
            head_tree=worktree if uncommitted else committed,
            snapshot_commit="" if uncommitted else tip or "",
            paths=(),
            source="worktree" if uncommitted else "revisions",
            hidden_paths=tuple(sorted(self.hidden)),
        )

    def retain_review_target(
        self, target: review_records.ReviewTarget, session_id: str
    ) -> review_records.ReviewTarget:
        """Anchor both trees under the owning session without advancing its run.

        Args:
            target: The already captured comparison, including path coverage.
            session_id: The record that owns the snapshot's retention.

        Returns:
            The same comparison with a durable cloneable snapshot binding.

        Raises:
            GitError: Either tree or its recorded commit is unreadable or inconsistent.
            OSError: The repository cannot be read.
        """
        base = target.base_commit
        if base and git_read.tree_sha(self.root, base) != target.base_tree:
            raise errors.GitError("review base commit does not match its tree")
        if (
            target.head_commit
            and git_read.tree_sha(self.root, target.head_commit) != target.head_tree
        ):
            raise errors.GitError("review head commit does not match its tree")
        if not base and target.base_tree != git_read.EMPTY_TREE:
            base = git_snapshots.snapshot_commit(
                self.root,
                target.base_tree,
                parents=(),
                ref=git_read.review_snapshot_ref_for(
                    session_id, target.base_tree, target.base_tree
                ),
                identity=self.identity,
            )
        parent = target.head_commit or self.checkpoint_head_sha()
        ref = git_read.review_snapshot_ref_for(session_id, target.base_tree, target.head_tree)
        snapshot = git_snapshots.snapshot_commit(
            self.root,
            target.head_tree,
            parents=tuple(p for p in (parent, base) if p),
            ref=ref,
            identity=self.identity,
        )
        return dataclasses.replace(target, snapshot_commit=snapshot, snapshot_ref=ref)
