"""Read a file's diff with the bytes that ground its attribution."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib

from agent6 import errors
from agent6.vcs.git import files as git_files
from agent6.vcs.git import history as git_history


@dataclasses.dataclass(frozen=True, slots=True)
class FileDiff:
    """A patch and its regular-file inputs; None means absent or unavailable."""

    patch: str
    truncated: bool
    before: bytes | None
    after: bytes | None

    @property
    def base_digest(self) -> str | None:
        """Return the digest of the base bytes, when available."""
        return hashlib.sha256(self.before).hexdigest() if self.before is not None else None

    @property
    def head_digest(self) -> str | None:
        """Return the digest of the head bytes, when available."""
        return hashlib.sha256(self.after).hexdigest() if self.after is not None else None


# How many times the head bytes and the patch are read before a changing file is given up.
WORKTREE_READS = 3


def read(
    repo: pathlib.Path,
    base: str,
    head: str,
    name: str,
    old_name: str = "",
    *,
    max_bytes: int | None,
    blob_max_bytes: int,
    full_context: bool = False,
) -> FileDiff:
    """Return a patch and the bytes that ground its attribution.

    A working-tree patch is git's own, run in the repository, so its attributes (line-ending
    conversion, diff drivers, filters) apply; the head bytes, and the file's identity (inode,
    size, change and modification times), are read before and after it and must agree, or
    the pair is read again: a write that is undone before the second read still moves the
    change time.

    Args:
        repo: The repository.
        base: The resolved base commit or tree.
        head: The resolved head commit, tree or WORKTREE.
        name: The checked head path, or the base path for a deletion.
        old_name: The checked base path of a rename.
        max_bytes: The patch cap, or None for all bytes.
        blob_max_bytes: The regular-file read cap.
        full_context: Include all unchanged lines in the patch.

    Returns:
        The patch, its truncation flag and its regular-file inputs.
        Unsupported or oversized worktree inputs have no head bytes.

    Raises:
        GitError: git cannot diff the file, or the file kept changing while it was read.
    """
    old_path = old_name or name
    old = git_files.blob_at(repo, base, old_path, max_bytes=blob_max_bytes)
    before = old.data if old else None
    if head != git_history.WORKTREE:
        new = git_files.blob_at(repo, head, name, max_bytes=blob_max_bytes)
        patch, truncated = git_history.file_patch(
            repo, base, head, name, old_name, max_bytes=max_bytes, full_context=full_context
        )
        return FileDiff(patch, truncated, before, new.data if new else None)
    untracked = git_history.worktree_entry(repo, name) == "untracked"
    for _attempt in range(WORKTREE_READS):
        stat_before = _stat_identity(repo / name)
        first = git_files.worktree_blob(repo, name, max_bytes=blob_max_bytes)
        patch, truncated = git_history.worktree_patch(
            repo,
            base,
            name,
            old_name,
            untracked=untracked,
            max_bytes=max_bytes,
            full_context=full_context,
        )
        again = git_files.worktree_blob(repo, name, max_bytes=blob_max_bytes)
        if _identity(first) == _identity(again) and stat_before == _stat_identity(repo / name):
            return FileDiff(patch, truncated, before, first.data if first else None)
    raise errors.GitError(f"{name} kept changing while its diff was read")


def _identity(blob: git_files.Blob | None) -> tuple[int, bytes | None] | None:
    return (blob.size, blob.data) if blob else None


def _stat_identity(path: pathlib.Path) -> tuple[int, int, int, int] | None:
    """Return what every write to the file moves: inode, size, change and modification times."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_ctime_ns, st.st_mtime_ns)
