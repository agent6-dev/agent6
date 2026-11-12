"""Retain immutable review trees without advancing ordinary git history."""

from __future__ import annotations

import pathlib

from agent6 import errors
from agent6.vcs import git_command, git_read


def snapshot_commit(
    path: pathlib.Path,
    tree: str,
    *,
    parents: tuple[str, ...],
    ref: str,
    identity: git_read.CommitIdentity | None = None,
) -> str:
    """Retain an exact tree and its evidence parents without changing a checkout or run chain.

    Args:
        path: The repository.
        tree: The immutable tree, never re-read from the worktree.
        parents: Commits retaining the comparison's base and head history.
        ref: A hidden review ref, created once and never rewritten.
        identity: The snapshot identity; None uses agent6's evidence identity.

    Returns:
        The retained commit, reusing a matching parent when it contains the other parents.

    Raises:
        GitError: A tree is unreadable, the ref differs, or retention fails.
    """
    if not ref.startswith(f"{git_read.CHAIN_NS}/") or "/review/" not in ref:
        raise errors.GitError("snapshot requires a hidden review ref")
    if git_read.tree_sha(path, tree) != tree:
        raise errors.GitError("snapshot requires an exact tree id")
    existing = git_read.chain_tip(path, ref)
    if existing and git_read.tree_sha(path, existing) != tree:
        raise errors.GitError("snapshot ref already retains a different tree")
    parents = tuple(dict.fromkeys(parents))
    checked = git_command.run(
        path,
        "rev-list",
        "--objects",
        "--verify-objects",
        "--no-object-names",
        tree,
        *parents,
        *((existing,) if existing else ()),
    )
    # rev-list reports corrupt blobs on stderr even with exit status zero.
    if checked.stderr:
        raise errors.GitError("review snapshot object verification reported an error")
    if existing:
        return existing
    sha = next(
        (
            parent
            for parent in parents
            if git_read.tree_sha(path, parent) == tree
            and all(
                other == parent or git_read.is_ancestor(path, other, parent) for other in parents
            )
        ),
        "",
    )
    if not sha:
        author = identity or git_read.CommitIdentity()
        env = {
            f"GIT_{role}_{field}": value
            for role in ("AUTHOR", "COMMITTER")
            for field, value in (
                ("NAME", author.name or "agent6"),
                ("EMAIL", author.email or "agent6@localhost"),
            )
        }
        sha = git_command.run(
            path,
            "commit-tree",
            tree,
            *(arg for parent in parents for arg in ("-p", parent)),
            "-m",
            "agent6 review snapshot",
            env_extra=env,
        ).stdout.strip()
    # A competing preparation may win creation; it must never replace an existing anchor.
    created = git_command.run(path, "update-ref", ref, sha, "0" * len(sha), check=False)
    if not created.ok:
        existing = git_read.chain_tip(path, ref)
        if existing and git_read.tree_sha(path, existing) == tree:
            return existing
        raise errors.GitError("could not retain review snapshot ref")
    return sha
