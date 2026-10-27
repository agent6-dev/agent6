"""Read complete review rules through bounded links in an immutable tree."""

from __future__ import annotations

import pathlib
import posixpath

from agent6 import errors, review_records
from agent6.tools import _path_safety
from agent6.tools import errors as tools_errors
from agent6.vcs import git_files, git_read

_LINK_LIMIT = 8
_LINK_BYTES = 64 * 1024
_NOT_REGULAR = "not a regular file in the base tree"


def _rule_entry(workspace: _path_safety.Workspace, tree: str, name: str) -> tuple[str, str] | None:
    """Resolve a tree path without following any checkout link.

    Args:
        workspace: The worker's root and hidden-path boundary.
        tree: The immutable reviewed base tree.
        name: The repository-relative rule path.

    Returns:
        The resolved name and metadata, or None when an optional path is absent.

    Raises:
        GitError: A link escapes, exceeds the bound or cannot supply a visible regular file.
    """
    root = workspace.root
    pending = list(pathlib.PurePosixPath(name).parts)
    resolved = pathlib.PurePosixPath()
    links = 0
    entry = ""
    while pending:
        resolved = pathlib.PurePosixPath(posixpath.normpath((resolved / pending.pop(0)).as_posix()))
        try:
            path = _path_safety.contain(root, resolved.as_posix())
        except tools_errors.ToolError as exc:
            raise errors.GitError(_NOT_REGULAR) from exc
        if resolved == pathlib.PurePosixPath():
            if pending:
                continue
            raise errors.GitError(_NOT_REGULAR)
        entry = git_read.tree_entry(root, tree, resolved.as_posix())
        if not entry:
            if links:
                raise errors.GitError(_NOT_REGULAR)
            return None
        # Tree metadata can establish absence without reading a hidden blob.
        if workspace.is_denied(path.abs_path) and not entry.startswith("040000 tree "):
            raise errors.GitError("path is hidden")
        if entry.startswith("120000 blob "):
            links += 1
            if links > _LINK_LIMIT:
                raise errors.GitError(_NOT_REGULAR)
            blob = git_files.read_blob(root, entry.split()[2], max_bytes=_LINK_BYTES)
            if not blob.data or b"\x00" in blob.data:
                raise errors.GitError(_NOT_REGULAR)
            target = blob.data.decode("utf-8")
            pending = [*pathlib.PurePosixPath(target).parts, *pending]
            resolved = resolved.parent
        elif pending and not entry.startswith("040000 tree "):
            raise errors.GitError(f"{resolved} is not a directory in the base tree")
    if not entry.startswith(("100644 blob ", "100755 blob ")):
        raise errors.GitError(_NOT_REGULAR)
    return resolved.as_posix(), entry


def rule_blob(
    workspace: _path_safety.Workspace, tree: str, name: str, *, max_bytes: int
) -> tuple[review_records.RuleSource, str] | None:
    """Read an optional rule blob through at most eight in-tree links.

    Args:
        workspace: The worker's root and hidden-path boundary.
        tree: The immutable reviewed base tree.
        name: The repository-relative rule path.
        max_bytes: The remaining byte allowance for the combined rule set.

    Returns:
        The source and text, or None when absent.

    Raises:
        GitError: A present path cannot supply a whole regular rule file within the boundary.
        UnicodeDecodeError: A link or rule blob is not UTF-8 text.
    """
    found = _rule_entry(workspace, tree, name)
    if found is None:
        return None
    resolved, entry = found
    blob = git_files.blob_at(workspace.root, tree, resolved, max_bytes=max_bytes)
    if blob is None:
        raise errors.GitError("rule blob is unavailable")
    if blob.data is None:
        raise errors.GitError("combined rules exceed 64 KiB; reduce the base-tree rule files")
    text = blob.data.decode("utf-8")
    if "\x00" in text:
        raise errors.GitError("rule blob contains NUL bytes")
    return review_records.RuleSource(name, entry.split()[2], len(blob.data)), text
