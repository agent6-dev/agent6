"""Keep a child's recorded worktree and index in its parent's Git object store."""

from __future__ import annotations

import pathlib
import re

from agent6 import errors
from agent6.config import Config
from agent6.sessions import child_dispatch
from agent6.sessions import layout as sessions_layout
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel.session import log as session_log


def pin(session_id: str, tree: str) -> str:
    """Return the session-owned ref for this exact tree, not a moving execution tip.

    Args:
        session_id: The child whose refs hold the tree.
        tree: The recorded tree id.

    Returns:
        The full ref name.

    Raises:
        GitError: The recorded id is not a full Git object id.
    """
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", tree):
        raise errors.GitError(f"invalid recorded tree id: {tree!r}")
    return f"refs/agent6/{session_id}/trees/{tree}"


def verify_pin(repo: pathlib.Path, session_id: str, kind: str, tree: str) -> None:
    """Refuse a recorded tree whose ref is missing or holds different content.

    Args:
        repo: The parent repository.
        session_id: The child's session id.
        kind: The tree named in a refusal.
        tree: The recorded tree id.

    Raises:
        GitError: The parent does not have the recorded tree on its session ref.
    """
    ref = pin(session_id, tree)
    if not git_read.object_exists(repo, ref) or git_read.tree_sha(repo, ref) != tree:
        raise errors.GitError(f"the child's recorded {kind} tree is missing from {ref}")


def awaits_steer(record: pathlib.Path, group: str) -> bool:
    """Return whether the parent still owns a pending or stopped child in this checkout."""
    parent_id, _, _ = group.rpartition("-p")
    if not parent_id or not sessions_layout.is_safe_session_id(parent_id):
        return False
    path = child_dispatch.record_path(record.with_name(parent_id), group)
    if not path.is_file():
        return False
    dispatch = child_dispatch.read_record(path)
    return any(
        c.session_id == record.name and (c.merge is None or c.merge.status == "stopped")
        for c in dispatch.children
    )


def preserve(origin: pathlib.Path, clone: pathlib.Path, record: pathlib.Path, cfg: Config) -> str:
    """Pin both end trees and return what still needs the clone, or an empty reason.

    Args:
        origin: The parent's repository.
        clone: The child's checkout, with no live worker.
        record: Its session directory.
        cfg: Supplies the same hidden-path exclusions as the end measurement.

    Returns:
        Why cleanup cannot proceed. A lost measurement does not prove the clone clean.
    """
    scan = session_log.scan_session_log(record / sessions_layout.LOGS_NAME)
    try:
        for kind, tree in (("end", scan.end_tree), ("index", scan.index_tree)):
            if tree:
                ref = pin(record.name, tree)
                git_ops.fetch_branch(origin, clone, f"{tree}:{ref}")
                verify_pin(origin, record.name, kind, tree)
        head = git_read.tree_sha(clone, "HEAD")
        index = git_read.index_tree(clone)
        if index != (scan.index_tree or head):
            return "uncommitted staged files outside the recorded index tree"
        seed = (
            scan.index_tree
            or git_read.chain_tip(clone, git_read.chain_ref_for(record.name))
            or index
        )
        excluded = (
            sessions_layout.read_untracked_at_start(record)
            | git_read.hidden_in_repo(clone, cfg.sandbox.hide_paths)
            | {".agent6"}
        )
        if git_read.worktree_tree(clone, seed, excluded) != (
            scan.end_tree or git_read.tree_sha(clone, seed)
        ):
            return "uncommitted files outside the recorded end tree"
    except (errors.GitError, OSError) as exc:
        return f"could not preserve the child's files: {exc}"
    return ""
