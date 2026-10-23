# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run git for agent6 with hard safety invariants.

No code path here spells a push, a force flag or a history rewrite, so there is
nothing to refuse at runtime (pinned by `test_the_vcs_package_never_spells_a_destructive_verb`);
the one sanctioned exception is `force_delete_squash_merged_branch`. Config can
loosen benign options (auto-stash, branch-per-run) and never these.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Collection, Sequence

from agent6 import errors
from agent6.vcs import _git_command, git_policy, git_read


def stash_tracked_changes(path: pathlib.Path, message: str, paths: Collection[str] = ()) -> None:
    """Stash the tracked files' uncommitted changes; untracked files stay in place.

    Args:
        path: The repository.
        message: The stash message, which the finalizer finds it by.
        paths: The repo-relative paths to stash; empty stashes every tracked change.
    """
    if not paths:
        _git_command.run(path, "stash", "push", "--message", message)
        return
    tracked = git_read.tree_paths(path, "HEAD") | set(git_read.tracked_files(path))
    # Select the root and exclude other tracked paths: stash's add rejects a staged deletion.
    specs = [":/", *(f":(top,exclude,literal){rel}" for rel in sorted(tracked - set(paths)))]
    _git_command.run(
        path,
        "stash",
        "push",
        "--message",
        message,
        "--pathspec-from-file=-",
        "--pathspec-file-nul",
        stdin_text="\0".join(specs),
    )


def auto_stash_message(session_id: str) -> str:
    """Return the auto-stash message the finalizer finds the stash by, never by position."""
    return f"agent6 auto-stash before run {session_id}"


# `git stash drop` prints "Dropped stash@{0} (<sha>)".
_DROPPED_SHA_RE = re.compile(r"^Dropped .*\(([0-9a-f]{7,64})\)", re.MULTILINE)


def restore_stash(path: pathlib.Path, stash: git_read.StashEntry) -> bool:
    """Apply the stash back onto the working tree by sha, dropping it on a clean apply.

    A conflicted apply leaves the markers and the stash in place: nothing undoes it.

    Args:
        path: The repository.
        stash: The entry to restore.

    Returns:
        True on a clean apply; False when the apply failed and everything stays put.

    Raises:
        GitError: A raced drop took a concurrent stash and putting it back failed; the
            apply has landed and the message carries the recovery command.
    """
    if not _git_command.run(path, "stash", "apply", stash.sha, check=False).ok:
        return False
    _drop_by_sha(path, stash.sha)
    return True


def _drop_by_sha(path: pathlib.Path, sha: str) -> None:
    """Drop the stash entry with this commit, putting back a bystander taken by mistake.

    `git stash drop` takes a position, and a stash pushed between the lookup and the
    drop shifts every position. git names the commit it dropped: a stranger's is stored
    straight back under its own subject, and agent6's own then stays listed, since a
    second attempt would race the same way.

    Args:
        path: The repository.
        sha: The stash commit; nothing is dropped when no entry has it.

    Raises:
        GitError: The bystander could not be stored back.
    """
    listed = _git_command.run(path, "stash", "list", "--format=%gd%x09%H", check=False)
    ref = ""
    for line in listed.stdout.splitlines():
        entry_ref, _, entry_sha = line.partition("\t")
        if entry_sha == sha:
            ref = entry_ref
            break
    if not ref:
        return
    dropped = _DROPPED_SHA_RE.search(
        _git_command.run(path, "stash", "drop", ref, check=False).stdout
    )
    if dropped is None:
        return
    taken = dropped.group(1)
    # Prefix either way: an abbreviated oid must not read as a stranger's stash.
    if sha.startswith(taken) or taken.startswith(sha):
        return
    # The stash commit's subject is the reflog's "On <branch>: <message>".
    subject = _git_command.run(path, "log", "-1", "--format=%s", taken, check=False).stdout.strip()
    subject = subject or "restored by agent6"
    store = _git_command.run(path, "stash", "store", "-m", subject, taken, check=False)
    if not store.ok:
        # The commit still exists under `taken`, so the message carries the recovery.
        detail = store.stderr.strip() or store.stdout.strip() or f"exit {store.returncode}"
        raise errors.GitError(
            f"a stash pushed concurrently ({subject!r}) was taken by a raced drop and "
            f"putting it back failed ({detail}); restore it with:\n"
            f"    git stash store -m {subject!r} {taken}"
        )


def delete_branch_if_merged(path: pathlib.Path, branch: str) -> bool:
    """Delete the branch with the safe delete, which git refuses unless it is reachable-merged.

    Args:
        path: The repository.
        branch: The branch.

    Returns:
        True when deleted; False when git refused, a squash-merged or unmerged branch.
    """
    return _git_command.run(path, "branch", "-d", branch, check=False).ok


def force_delete_squash_merged_branch(path: pathlib.Path, branch: str) -> bool:
    """Force-delete a squash-merged run branch, the one sanctioned force-delete.

    The safe delete refuses a squash-merged branch, since its commits are not reachable
    from the base even though its content is. Only `sessions prune --delete-squashed`
    calls this, for a branch the manifest confirms was squash-merged.

    Args:
        path: The repository.
        branch: The branch.

    Returns:
        True when deleted.
    """
    return _git_command.run(path, "branch", "-D", branch, check=False).ok


def create_branch(path: pathlib.Path, name: str, *, start_point: str | None = None) -> None:
    """Create the branch and check it out, or only check it out when it exists.

    An existing branch is never moved, so a resumed run reuses its branch.

    Args:
        path: The repository.
        name: The branch.
        start_point: Where a new branch is cut from; None is HEAD.
    """
    existing = _git_command.run(path, "branch", "--list", name, check=False)
    if existing.ok and existing.stdout.strip():
        _git_command.run(path, "checkout", name)
    elif start_point:
        _git_command.run(path, "checkout", "-b", name, start_point)
    else:
        _git_command.run(path, "checkout", "-b", name)


def set_ref(path: pathlib.Path, ref: str, sha: str) -> None:
    """Point one of agent6's own refs at a sha with no checkout; branches use `create_branch_at`."""
    _git_command.run(path, "update-ref", ref, sha)


def delete_ref(path: pathlib.Path, ref: str) -> None:
    """Delete the ref; a missing one is a no-op."""
    _git_command.run(path, "update-ref", "-d", ref, check=False)


def checkout_detached(path: pathlib.Path, rev: str) -> None:
    """Check out the rev detached, in an agent6-owned clone, never the operator's checkout."""
    _git_command.run(path, "checkout", "-q", "--detach", rev)


def add_worktree(path: pathlib.Path, dest: pathlib.Path, sha: str) -> None:
    """Add a linked worktree detached at the sha, a fork's own checkout.

    It shares the repository's refs and objects, so a chain commit made there is
    visible from every checkout; the operator's checkout, HEAD and index are untouched.

    Args:
        path: The repository.
        dest: Where the worktree goes.
        sha: The commit to check out.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git_command.run(path, "worktree", "add", "--detach", str(dest.absolute()), sha)


def remove_worktree(path: pathlib.Path, dest: pathlib.Path) -> bool:
    """Delete a linked worktree of the repository and git's record of it.

    Only that entry goes: a repo-wide `worktree prune` would also drop the record of an
    operator's worktree whose directory is missing for the moment.

    Args:
        path: The repository.
        dest: The worktree.

    Returns:
        True when deleted; False when the directory is not a worktree of this repository
        or could not be removed.
    """
    pointer = dest / ".git"
    if not pointer.is_file():
        return False
    text = pointer.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return False
    admin = pathlib.Path(text[len("gitdir:") :].strip())
    if not admin.is_absolute():
        admin = dest / admin
    admin = admin.resolve()
    if admin.parent != git_read.git_common_dir(path) / "worktrees":
        return False
    try:
        shutil.rmtree(dest)
    except OSError:
        return False  # what stayed keeps its record, so git and the caller still see it
    shutil.rmtree(admin, ignore_errors=True)
    return True


def create_branch_at(path: pathlib.Path, name: str, sha: str) -> None:
    """Create the branch at the sha without checking it out.

    Additive only: HEAD and the working tree are untouched, so a fork cuts its branch
    at a historical sha while the operator's checkout stays put.

    Args:
        path: The repository.
        name: The branch.
        sha: The commit it points at.

    Raises:
        GitError: The branch exists pointing elsewhere; a branch is never moved.
    """
    existing = _git_command.run(
        path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False
    )
    if existing.ok and existing.stdout.strip():
        if existing.stdout.strip() == sha:
            return
        raise errors.GitError(
            f"branch {name!r} already exists at {existing.stdout.strip()[:12]}, not {sha[:12]}; "
            "refusing to move it"
        )
    _git_command.run(path, "branch", name, sha)


def init_repo(path: pathlib.Path) -> None:
    """Initialize a repository at the path."""
    _git_command.run(path, "init")


def clone_repo(origin: pathlib.Path, dest: pathlib.Path) -> None:
    """Clone one local repository into a destination.

    Both are plain paths, so git hardlinks on the same filesystem and copies across
    devices. Run from "/", so a missing origin fails as a GitError from git itself.

    Args:
        origin: The repository to clone.
        dest: Where the clone goes.
    """
    _git_command.run(pathlib.Path("/"), "clone", str(origin.absolute()), str(dest.absolute()))


def commit_all(
    path: pathlib.Path,
    message: str,
    *,
    trailers: dict[str, str] | None = None,
    identity: git_read.CommitIdentity | None = None,
) -> str:
    """Stage everything and commit.

    Args:
        path: The repository.
        message: The commit message.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides; None uses the repo's config,
            validated at startup by `git_read.verify_git_identity`.

    Returns:
        The new HEAD sha.
    """
    _git_command.run(path, "add", "-A")
    return _commit(path, message, trailers=trailers, identity=identity)


def commit_paths(
    path: pathlib.Path,
    message: str,
    paths: tuple[str, ...],
    *,
    trailers: dict[str, str] | None = None,
    identity: git_read.CommitIdentity | None = None,
) -> str:
    """Stage and commit only these repo-relative paths.

    Unrelated staged changes stay staged and uncommitted, so `agent6 init`'s scaffold
    commit never folds the operator's work in.

    Args:
        path: The repository.
        message: The commit message.
        paths: The repo-relative paths to record.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides; None uses the repo's config.

    Returns:
        The new HEAD sha.

    Raises:
        GitError: No paths were given.
    """
    if not paths:
        raise errors.GitError("commit_paths requires at least one path")
    _git_command.run(path, "add", "--", *paths, env_extra=git_read.LITERAL_PATHSPECS)
    return _commit(path, message, trailers=trailers, identity=identity, only_paths=paths)


def _identity_env(identity: git_read.CommitIdentity | None) -> dict[str, str] | None:
    """Return the author and committer env for a commit, or None for the repo's own identity."""
    if identity is None:
        return None
    env: dict[str, str] = {}
    if identity.name:
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = identity.name
    if identity.email:
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = identity.email
    return env or None


def _full_message(
    message: str, trailers: dict[str, str] | None, identity: git_read.CommitIdentity | None
) -> str:
    """Return the message with the identity trailer and any extra trailers appended once."""
    merged = dict(trailers or {})
    if identity is not None and identity.trailer and identity.trailer not in message:
        key, _, value = identity.trailer.partition(": ")
        merged[key] = value
    if not merged:
        return message
    trailer_lines = "\n".join(f"{k}: {v}" for k, v in merged.items())
    return f"{message}\n\n{trailer_lines}"


def chain_commit(
    path: pathlib.Path,
    message: str,
    *,
    ref: str,
    fallback_parent: str | None,
    trailers: dict[str, str] | None = None,
    identity: git_read.CommitIdentity | None = None,
    also_branch: str | None = None,
    exclude: Collection[str] = (),
) -> str | None:
    """Record the worktree's content on the run's commit chain, touching no checkout.

    The tree is staged into a temp index and committed parented on the ref's current
    value: the ref is the chain state, so resume and concurrent runs compose without
    bookkeeping.

    Args:
        path: The repository.
        message: The commit message.
        ref: The chain ref, advanced to the new commit.
        fallback_parent: The parent when the ref does not exist yet, HEAD at run start;
            None makes a root commit in an unborn repo.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides.
        also_branch: A visible branch moved with the ref; a checked-out one has its
            index brought forward.
        exclude: Repo-relative paths kept out, the run's `untracked_at_start`.

    Returns:
        The new sha, or None when the tree equals the parent's.
    """
    parent = git_read.chain_tip(path, ref) or fallback_parent
    tree = git_read.worktree_tree(path, parent, exclude)
    parent_args: list[str] = []
    if parent is not None:
        if git_read.tree_sha(path, parent) == tree:
            return None
        parent_args = ["-p", parent]
    sha = _git_command.run(
        path,
        "commit-tree",
        tree,
        *parent_args,
        "-m",
        _full_message(message, trailers, identity),
        env_extra=_identity_env(identity),
    ).stdout.strip()
    _git_command.run(path, "update-ref", ref, sha)
    _advance_run_branch(path, also_branch, sha, expected=parent)
    return sha


def _advance_run_branch(
    path: pathlib.Path, branch: str | None, sha: str, *, expected: str | None
) -> None:
    """Move the run's visible branch to the sha, only from the tip the commit was built on.

    A compare-and-swap: a bare `update-ref` would rewind the operator's own commit on
    the branch. A branch that moved keeps its tip; the chain ref is the record either
    way. A checked-out branch has its index and worktree brought forward.

    Args:
        path: The repository.
        branch: The branch; None does nothing.
        sha: The new tip.
        expected: The tip the branch must still be at; None only creates a missing branch.
    """
    if not branch:
        return
    ref = f"refs/heads/{branch}"
    current = git_read.chain_tip(path, ref)
    if current is None:
        _git_command.run(path, "update-ref", ref, sha)  # the run's first commit creates it
        return
    if expected is not None and current == expected:
        _git_command.run(path, "update-ref", ref, sha, expected)
        _bring_index_forward(path, branch, expected, sha)


def chain_merge(
    path: pathlib.Path,
    merge_rev: str,
    message: str,
    *,
    ref: str,
    fallback_parent: str | None = None,
    identity: git_read.CommitIdentity | None = None,
    also_branch: str | None = None,
) -> str | None:
    """Merge a rev into the chain without touching HEAD, and sync the worktree to it.

    A rev descending from the tip fast-forwards instead of stacking an empty merge.

    Args:
        path: The repository.
        merge_rev: The commit to merge in.
        message: The merge commit's message.
        ref: The chain ref, advanced to the result.
        fallback_parent: The base when the ref does not exist yet.
        identity: The author and committer overrides.
        also_branch: A visible branch moved with the ref.

    Returns:
        The new tip, the old one when it already contained the rev, or None on a
        textual conflict, which leaves the chain and worktree untouched.

    Raises:
        GitError: The ref does not exist and no fallback parent was given.
    """
    ours = git_read.chain_tip(path, ref) or fallback_parent
    if ours is None:
        raise errors.GitError(f"chain ref {ref} does not exist and no fallback parent was given")
    theirs = _git_command.run(
        path, "rev-parse", "--verify", f"{merge_rev}^{{commit}}"
    ).stdout.strip()
    if git_read.is_ancestor(path, theirs, ours):
        return ours
    if git_read.is_ancestor(path, ours, theirs):
        sha = theirs
    else:
        res = _git_command.run(path, "merge-tree", "--write-tree", ours, theirs, check=False)
        if res.returncode != 0:
            return None
        tree = res.stdout.strip().splitlines()[0]
        sha = _git_command.run(
            path,
            "commit-tree",
            tree,
            "-p",
            ours,
            "-p",
            theirs,
            "-m",
            _full_message(message, None, identity),
            env_extra=_identity_env(identity),
        ).stdout.strip()
    _git_command.run(path, "update-ref", ref, sha)
    sync_worktree(path, ours, sha)
    _advance_run_branch(path, also_branch, sha, expected=ours)
    return sha


def chain_continue_from(
    path: pathlib.Path,
    head: str,
    message: str,
    *,
    ref: str,
    tip: str,
    identity: git_read.CommitIdentity | None = None,
    also_branch: str | None = None,
) -> str:
    """Make the chain continue from a commit that already holds its landed work.

    Refs only move forward: a tip the commit descends from fast-forwards to it, and any other
    tip gets a commit with the head's tree and both as parents, so the run's earlier steps
    stay reachable. No checkout and no working tree is touched.

    Args:
        path: The repository.
        head: The commit to continue from.
        message: The joining commit's message.
        ref: The chain ref, moved to the result.
        tip: The run's tip that landed.
        identity: The author and committer overrides.
        also_branch: A visible branch moved with the ref.

    Returns:
        The commit the chain continues from; the tip itself when it already holds the head.
    """
    if git_read.is_ancestor(path, head, tip):
        sha = tip
    elif git_read.is_ancestor(path, tip, head):
        sha = head
    else:
        sha = _git_command.run(
            path,
            "commit-tree",
            f"{head}^{{tree}}",
            "-p",
            tip,
            "-p",
            head,
            "-m",
            _full_message(message, None, identity),
            env_extra=_identity_env(identity),
        ).stdout.strip()
    _git_command.run(path, "update-ref", ref, sha)
    _advance_run_branch(path, also_branch, sha, expected=tip)
    return sha


def sync_worktree(path: pathlib.Path, from_rev: str, to_rev: str) -> None:
    """Move the worktree's files from one tree to another through a temp index.

    HEAD and the shared index stay untouched. The worktree must match the first tree,
    the chain invariant after a chain commit. The temp index is refreshed before the
    merge: `read-tree` records no stat data, and the merge touches only entries it can
    prove up to date.

    Args:
        path: The repository.
        from_rev: The tree the worktree holds.
        to_rev: The tree to move it to.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="agent6-chain-"))
    env = {"GIT_INDEX_FILE": str(tmp / "index")}
    try:
        _git_command.run(path, "read-tree", from_rev, env_extra=env)
        _git_command.run(path, "update-index", "--refresh", env_extra=env, check=False)
        _git_command.run(path, "read-tree", "-m", "-u", from_rev, to_rev, env_extra=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tree_with_paths_from(path: pathlib.Path, tree: str, rev: str, paths: Sequence[str]) -> str:
    """Return a tree equal to `tree` with the given paths taken from a commit.

    A path the commit lacks is dropped. Built in a temp index: HEAD and the shared index
    stay untouched.

    Args:
        path: The repository.
        tree: The tree to start from.
        rev: The commit the paths are taken from.
        paths: The repo-relative paths.

    Returns:
        The tree sha.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="agent6-tree-"))
    env = {"GIT_INDEX_FILE": str(tmp / "index")}
    try:
        _git_command.run(path, "read-tree", tree, env_extra=env)
        if paths:
            specs = (f":(top,literal){p}" for p in paths)
            _git_command.run(
                path, "restore", "--staged", f"--source={rev}", "--", *specs, env_extra=env
            )
        return _git_command.run(path, "write-tree", env_extra=env).stdout.strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _commit(
    path: pathlib.Path,
    message: str,
    *,
    trailers: dict[str, str] | None,
    identity: git_read.CommitIdentity | None,
    only_paths: tuple[str, ...] | None = None,
) -> str:
    """Commit the index, or only the given paths from the worktree.

    Args:
        path: The repository.
        message: The commit message.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides; None uses the repo's config.
        only_paths: The repo-relative paths; None commits the index.

    Returns:
        The new HEAD sha.
    """
    env_extra = {**(_identity_env(identity) or {}), **git_read.LITERAL_PATHSPECS}
    full_message = _full_message(message, trailers, identity)
    argv = ["commit", "-m", full_message]
    if only_paths is not None:
        argv.extend(["--", *only_paths])
    _git_command.run(path, *argv, env_extra=env_extra)
    return _git_command.run(path, "rev-parse", "HEAD").stdout.strip()


@dataclasses.dataclass(frozen=True, slots=True)
class MergeResult:
    """The outcome of landing a run branch on a target.

    Attributes:
        merged_sha: The target's new tip; "" on a conflict, where nothing moved.
        conflicted: Whether the merge conflicted.
        conflicts: The conflicted paths.
        left_behind: Paths the checkout keeps its own version of, so it does not hold
            what was merged.
    """

    merged_sha: str
    conflicted: bool
    conflicts: tuple[str, ...]
    left_behind: tuple[str, ...] = ()


def fetch_branch(path: pathlib.Path, remote_path: pathlib.Path, refspec: str) -> None:
    """Fetch a refspec from another repository's path, without adding a remote."""
    _git_command.run(path, "fetch", str(remote_path), refspec)


def _merge_tree(
    path: pathlib.Path, ours: str, theirs: str, merge_base: str | None
) -> tuple[str, tuple[str, ...]]:
    """Merge two commits into a tree without touching the worktree.

    Args:
        path: The repository.
        ours: The first parent.
        theirs: The commit merged in.
        merge_base: The base to use instead of the one git would find.

    Returns:
        The merged tree's oid and no paths, or "" and the conflicted paths.

    Raises:
        GitError: git predates `--merge-base` (2.40), or the merge failed outright.
    """
    base = [f"--merge-base={merge_base}"] if merge_base else []
    res = _git_command.run(
        path, "merge-tree", "--write-tree", "--name-only", *base, ours, theirs, check=False
    )
    lines = res.stdout.splitlines()
    if res.returncode == 1:
        # The tree oid, the conflicted paths, a blank line, then git's messages.
        paths: list[str] = []
        for line in lines[1:]:
            if not line.strip():
                break
            paths.append(line)
        return "", tuple(paths)
    if res.returncode == 129:
        # git's usage exit: `--merge-base` arrived in 2.40, the floor docs/installation.md states.
        version = _git_command.run(path, "--version", check=False).stdout.strip() or "this git"
        raise errors.GitError(
            f"merge-tree rejected its arguments ({version}); agent6 needs git 2.40 or newer"
        )
    if res.returncode != 0 or not lines:
        raise errors.GitError(f"merge-tree failed: {res.stderr.strip() or 'exit'}")
    return lines[0].strip(), ()


def plumb_merge(
    path: pathlib.Path,
    target: str,
    merge_rev: str,
    *,
    method: str,
    message: str | None = None,
    identity: git_read.CommitIdentity | None = None,
    merge_base: str | None = None,
) -> MergeResult:
    """Land a rev on a branch with plumbing only: no checkout, no clean-tree requirement.

    The operator's worktree is never the medium, so a worktree carrying the run's own
    work does not block the landing. When the target is checked out, the index entries
    the merge changed and the operator did not are brought forward.

    Args:
        path: The repository.
        target: The branch to land on.
        merge_rev: The commit to land.
        method: "merge" (a two-parent commit), "squash" (a single-parent commit) or
            "ff_only" (the ref moves to the rev).
        message: The merge commit's message.
        identity: The author and committer overrides.
        merge_base: The base instead of the one git would find: a run squash-merged
            earlier is content the target holds under a commit unrelated to the chain.

    Returns:
        The result; a rev the target already contains returns the unchanged tip, and a
        conflict moves nothing and lists the paths.

    Raises:
        GitError: The "ff_only" method cannot fast-forward, or the target moved concurrently.
    """
    ref = f"refs/heads/{target}"
    ours = _git_command.run(path, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    theirs = _git_command.run(
        path, "rev-parse", "--verify", f"{merge_rev}^{{commit}}"
    ).stdout.strip()
    if git_read.is_ancestor(path, theirs, ours):
        return MergeResult(ours, False, ())
    ff_able = git_read.is_ancestor(path, ours, theirs)
    if method == "ff_only":
        if not ff_able:
            raise errors.GitError(
                f"{target!r} has moved; a fast-forward to {merge_rev!r} is impossible"
            )
        sha = theirs
    else:
        if ff_able:
            tree = git_read.tree_sha(path, theirs)
        else:
            tree, conflicts = _merge_tree(path, ours, theirs, merge_base)
            if conflicts:
                return MergeResult("", True, conflicts)
        if tree == git_read.tree_sha(path, ours):
            return MergeResult(ours, False, ())  # content-identical: nothing to land
        text = message or f"Merge {merge_rev}"
        trailer = identity.trailer if identity else None
        if trailer and trailer not in text:
            text = f"{text}\n\n{trailer}"
        parent_args = ["-p", ours] if method == "squash" else ["-p", ours, "-p", theirs]
        sha = _git_command.run(
            path,
            "commit-tree",
            tree,
            *parent_args,
            "-m",
            text,
            env_extra=_identity_env(identity),
        ).stdout.strip()
    # Compare-and-swap: refuses when the target moved concurrently.
    _git_command.run(path, "update-ref", ref, sha, ours)
    return MergeResult(sha, False, (), _bring_index_forward(path, target, ours, sha))


def _bring_index_forward(
    path: pathlib.Path, target: str, old_tip: str, new_tip: str
) -> tuple[str, ...]:
    """Bring a checked-out branch's index and worktree forward after its ref moved.

    Each path the move changed is updated only where it still matches the old tip, so
    the operator's own staged or edited content stays as it is. Without the index half
    `git git_read.status` shows a phantom staged reversal; without the worktree half a merge onto
    a reverted checkout leaves the files behind.

    Args:
        path: The repository.
        target: The branch; the checkout is whichever worktree has it checked out.
        old_tip: The tip before the move.
        new_tip: The tip after it.

    Returns:
        The paths the checkout keeps a third version of, which the move did not reach;
        empty when no live checkout has the branch.
    """
    checkout = git_read.worktree_of_branch(path, target)
    if checkout is None or not checkout.is_dir():
        return ()
    path = checkout
    records = _git_command.nul_fields(
        path, "diff-tree", "-r", "--no-renames", "-z", old_tip, new_tip
    )
    staged = {
        entry.split("\t", 1)[1]: entry.split("\t", 1)[0].split()
        for entry in _git_command.nul_fields(path, "ls-files", "--stage", "-z")
        if "\t" in entry
    }  # path -> [mode, sha, stage]
    updates: list[str] = []
    left: list[str] = []
    i = 0
    while i + 1 < len(records):
        meta, rel = records[i], records[i + 1]
        i += 2
        if not meta.startswith(":"):
            continue
        old_mode, new_mode, old_sha, new_sha, _status = meta[1:].split(" ")[:5]
        entry = staged.get(rel)
        if (entry is None and old_mode == "000000") or (
            entry is not None and entry[0] == old_mode and entry[1] == old_sha
        ):
            # mode 000000 removes the entry (a merge-side deletion).
            updates.append(f"{new_mode} {new_sha}\t{rel}")
        if not _bring_worktree_file_forward(path, rel, old_mode, old_sha, new_mode, new_sha):
            left.append(rel)
    if updates:
        _git_command.run(
            path, "update-index", "-z", "--index-info", stdin_text="\x00".join(updates) + "\x00"
        )
    return tuple(left)


def _bring_worktree_file_forward(
    path: pathlib.Path, rel: str, old_mode: str, old_sha: str, new_mode: str, new_sha: str
) -> bool:
    """Move one worktree file from the old tip's content to the new tip's.

    Only a file still at the old tip moves (absent matches a deletion or a not-yet-added
    path); symlinks and submodule pointers are left to the operator.

    Args:
        path: The checkout.
        rel: The file, repo-relative.
        old_mode: The mode at the old tip; "000000" when absent.
        old_sha: The blob at the old tip.
        new_mode: The mode at the new tip; "000000" when deleted.
        new_sha: The blob at the new tip.

    Returns:
        False when the checkout keeps a third version, which the caller names.
    """
    file = path / rel
    if new_mode in ("120000", "160000") or old_mode in ("120000", "160000"):
        return True  # left to the operator, and never named as left behind
    try:
        current = (
            ""
            if not file.is_file() or file.is_symlink()
            else _git_command.run(path, "hash-object", "--", rel, check=False).stdout.strip()
        )
        matches_old = not file.exists() if old_mode == "000000" else current == old_sha
        if not matches_old:
            # Already at the new content is landed, not left behind.
            return current == new_sha or (new_mode == "000000" and not file.exists())
        if new_mode == "000000":
            file.unlink(missing_ok=True)
            return True
        file.parent.mkdir(parents=True, exist_ok=True)
        # Bytes straight from git: lossy decoding would corrupt a binary blob.
        with file.open("wb") as out:
            subprocess.run(
                [
                    git_policy.git(),
                    *git_policy.git_hardening_flags(path),
                    "cat-file",
                    "blob",
                    new_sha,
                ],
                cwd=path,
                stdout=out,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=_git_command.GIT_TIMEOUT_S,
            )
        if new_mode == "100755":
            file.chmod(file.stat().st_mode | 0o111)
    except (errors.GitError, OSError, subprocess.SubprocessError):
        return False  # an unwritable path leaves truthful dirt, never a crash
    return True
