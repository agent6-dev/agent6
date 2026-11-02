"""Read diff inputs once and keep working-tree patches tied to those bytes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import re
import tempfile

from agent6 import errors
from agent6.vcs import git_command, git_files, git_history, git_read


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


def _quoted(name: str) -> str:
    return (
        json.dumps(name, ensure_ascii=False)
        if any(c.isspace() or c in '\\"' for c in name)
        else name
    )


def _snapshot_patch(
    repo: pathlib.Path,
    name: str,
    old_name: str,
    before: bytes | None,
    after: bytes | None,
    old_mode: int,
    *,
    full_context: bool,
) -> bytes:
    """Diff private copies without reopening the working-tree file.

    Args:
        repo: The repository whose Git policy applies.
        name: The head path.
        old_name: The base path of a rename, or empty.
        before: The checked base bytes, or None for an addition.
        after: The checked head bytes, or None for a deletion.
        old_mode: The base file's permission bits.
        full_context: Include every unchanged line.

    Returns:
        The patch bytes with repository-relative paths.

    Raises:
        GitError: A snapshot cannot be created or Git cannot diff it.
    """
    try:
        mode = (repo / name).lstat().st_mode & 0o777 if after is not None else 0o644
        with tempfile.TemporaryDirectory(prefix="agent6-diff-") as tmp:
            root = pathlib.Path(tmp)
            names: list[str] = []
            for side, data, permissions in (("a", before, old_mode), ("b", after, mode)):
                names.append(f"{side}/file" if data is not None else "/dev/null")
                if data is not None:
                    dest = root / side / "file"
                    dest.parent.mkdir()
                    dest.write_bytes(data)
                    dest.chmod(permissions)
            args = (
                "diff",
                "--no-index",
                "--no-color",
                "--src-prefix=",
                "--dst-prefix=",
                "--unified=2147483647" if full_context else "--unified=3",
                "--",
                *names,
            )
            _, code, out, err = git_command.spawn(root, args, None, None, repo=repo)
            if code not in (0, 1):
                error = git_command.failure(args, code, os.fsdecode(out), os.fsdecode(err))
                raise error
    except OSError as exc:
        raise errors.GitError(f"cannot snapshot {name!r}: {exc}") from exc
    # Only the preamble names the private copies; hunk text is never rewritten.
    preamble, marker, hunks = out.partition(b"\n@@ ")
    labels = {
        b"a/file": os.fsencode(_quoted("a/" + (old_name or name))),
        b"b/file": os.fsencode(_quoted("b/" + name)),
    }
    preamble = re.sub(rb"[ab]/file", lambda match: labels[match[0]], preamble)
    if old_name:
        if not preamble:
            preamble = os.fsencode(
                f"diff --git {_quoted('a/' + old_name)} {_quoted('b/' + name)}\n"
            )
        first, newline, rest = preamble.partition(b"\n")
        rename = os.fsencode(f"rename from {_quoted(old_name)}\nrename to {_quoted(name)}\n")
        preamble = first + newline + rename + rest
    return preamble + marker + hunks


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
    """
    old_path = old_name or name
    old_entry = git_read.tree_entry(repo, base, old_path)
    old = git_files.blob_at(repo, base, old_path, max_bytes=blob_max_bytes)
    before = old.data if old else None
    if head != git_history.WORKTREE:
        new = git_files.blob_at(repo, head, name, max_bytes=blob_max_bytes)
        patch, truncated = git_history.file_patch(
            repo, base, head, name, old_name, max_bytes=max_bytes, full_context=full_context
        )
        return FileDiff(patch, truncated, before, new.data if new else None)
    entry = git_history.worktree_entry(repo, name)
    new = git_files.worktree_blob(repo, name, max_bytes=blob_max_bytes)
    after = new.data if new else None
    old_ready = not old_entry or before is not None
    new_ready = after is not None or not entry or not os.path.lexists(repo / name)
    if old_ready and new_ready:
        mode = 0o755 if old_entry.startswith("100755 ") else 0o644
        out = _snapshot_patch(repo, name, old_name, before, after, mode, full_context=full_context)
        truncated = max_bytes is not None and len(out) > max_bytes
        if truncated:
            out = out[: out.rfind(b"\n", 0, max_bytes) + 1]
        return FileDiff(out.decode(errors="replace"), truncated, before, after)
    patch, truncated = git_history.worktree_patch(
        repo,
        base,
        name,
        old_name,
        untracked=entry == "untracked",
        max_bytes=max_bytes,
        full_context=full_context,
    )
    return FileDiff(patch, truncated, before, None)
