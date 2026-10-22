# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the Git tab's JSON payloads: refs, the commit graph, a commit, a compare, a diff.

Every revision and path arrives from the operator's browser and is still checked before
git sees it: a value starting with `-` is refused, a revision resolves through
`git rev-parse --verify --end-of-options <rev>^{commit}` before any other call, a ref name
must be one `git for-each-ref` listed, and a file must be in one of the two trees. Nothing
here writes. Each list is capped and says when it was cut.
"""

from __future__ import annotations

import pathlib
from typing import Any

from agent6.vcs import git_history, git_ops
from agent6.viewmodel import format

# Commits per graph page, and the most a request may ask for.
GRAPH_PAGE = 200
GRAPH_PAGE_MAX = 500
# Files listed for one commit or compare before the list is cut.
FILES_MAX = 2000
# Where one file's patch is cut.
DIFF_MAX_BYTES = 256 * 1024
# Refs and commits a search answers, each.
SEARCH_MAX = 30


class GitRequestError(Exception):
    """A Git tab request the server refuses: a malformed or unknown revision, ref or path."""


def _repo(repo: pathlib.Path) -> None:
    if not git_ops.is_git_repo(repo):
        raise GitRequestError(f"not a git repository: {repo}")


def _check_shape(value: str, what: str) -> str:
    """Return the value when it is a plausible revision or path.

    Args:
        value: The revision or path as the browser sent it.
        what: Its name in the refusal, "revision" or "path".

    Returns:
        The value unchanged.

    Raises:
        GitRequestError: The value is empty, starts with `-`, or holds a control character.
    """
    if not value:
        raise GitRequestError(f"a {what} is required")
    if value.startswith("-"):
        raise GitRequestError(f"a {what} cannot start with '-': {value!r}")
    if len(value) > 4096 or any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        raise GitRequestError(f"not a {what}: {value!r}")
    return value


def _resolve(repo: pathlib.Path, rev: str, *, tree_ok: bool = False) -> str:
    """Return the full sha a revision names.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it.
        tree_ok: Accept `git_ops.EMPTY_TREE`, the base a root commit diffs against.

    Returns:
        The full sha, or the empty tree's own.

    Raises:
        GitRequestError: The revision is malformed, unknown, or not a commit.
    """
    _check_shape(rev, "revision")
    if rev == git_ops.EMPTY_TREE:
        if tree_ok:
            return rev
        raise GitRequestError(f"not a commit: {rev}")
    sha = git_history.resolve_commit(repo, rev)
    if not sha:
        raise GitRequestError(f"unknown revision {rev!r}")
    return sha


def _decorations(refs: tuple[git_history.RefRow, ...]) -> dict[str, list[dict[str, Any]]]:
    """Return the refs at each sha, in listing order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in refs:
        out.setdefault(r.sha, []).append({"name": r.name, "kind": r.kind, "head": r.head})
    return out


def _ref_row(r: git_history.RefRow) -> dict[str, Any]:
    return {"name": r.name, "full": r.full, "kind": r.kind, "sha": r.sha, "head": r.head}


def _file_row(f: git_history.ChangedFile) -> dict[str, Any]:
    return {
        "status": f.status,
        "path": f.path,
        "old_path": f.old_path,
        "additions": f.additions,
        "deletions": f.deletions,
        "binary": f.binary,
    }


def _files(repo: pathlib.Path, base: str, head: str) -> tuple[list[dict[str, Any]], bool]:
    """Return the changed files between two trees, cut at `FILES_MAX`, and whether they were."""
    rows = git_history.changed_files(repo, base, head)
    return [_file_row(f) for f in rows[:FILES_MAX]], len(rows) > FILES_MAX


def _log_row(c: git_history.LogCommit) -> dict[str, Any]:
    return {
        "sha": c.sha,
        "short": c.sha[:7],
        "parents": list(c.parents),
        "author": c.author,
        "date": c.date,
        "when": format.format_when(c.epoch),
        "subject": c.subject,
        "is_merge": len(c.parents) > 1,
    }


def refs_payload(repo: pathlib.Path) -> dict[str, Any]:
    """Return every ref with its kind and tip, and HEAD's sha and branch."""
    _repo(repo)
    sha, branch = git_history.head_commit(repo)
    return {
        "head": {"sha": sha, "branch": branch},
        "refs": [_ref_row(r) for r in git_history.list_refs(repo)],
    }


def _whole_number(raw: str, name: str, default: int) -> int:
    if raw == "":
        return default
    if not (raw.isascii() and raw.isdigit()):
        raise GitRequestError(f"{name} must be a whole number")
    return int(raw)


def graph_payload(
    repo: pathlib.Path, refs: list[str] | None, *, limit: str = "", skip: str = ""
) -> dict[str, Any]:
    """Return one page of the commit graph.

    Args:
        repo: The repository.
        refs: The full ref names (or `HEAD`) to walk from; None walks every branch, remote
            and tag plus HEAD, leaving out the `refs/agent6/*` chain refs.
        limit: Commits in the page, as the query gave it; "" is `GRAPH_PAGE`, more than
            `GRAPH_PAGE_MAX` is clamped.
        skip: Commits to pass over first, as the query gave it.

    Returns:
        `commits`, each with its refs, then `has_more`, `skip` and `limit`.

    Raises:
        GitRequestError: A name is not a listed ref, or a number is not one.
    """
    _repo(repo)
    page = min(_whole_number(limit, "limit", GRAPH_PAGE), GRAPH_PAGE_MAX)
    start = _whole_number(skip, "skip", 0)
    listed = git_history.list_refs(repo)
    head_sha, _branch = git_history.head_commit(repo)
    known = {r.full for r in listed}
    if refs is None:
        revs = [r.full for r in listed if r.kind != "agent6"] + (["HEAD"] if head_sha else [])
    else:
        for name in refs:
            if name != "HEAD" and name not in known:
                raise GitRequestError(f"no such ref {name!r}")
        revs = [name for name in refs if name != "HEAD" or head_sha]
    rows = git_history.log_commits(repo, revs, limit=page + 1, skip=start) if revs else ()
    decorated = _decorations(listed)
    commits = [{**_log_row(c), "refs": decorated.get(c.sha, [])} for c in rows[:page]]
    return {"commits": commits, "has_more": len(rows) > page, "skip": start, "limit": page}


def commit_payload(repo: pathlib.Path, rev: str) -> dict[str, Any]:
    """Return a commit's header, message, refs and the files it changed from its first parent.

    `base` is that parent, or the empty tree for a root commit, so a file's diff is read
    with `/api/git/diff?base=<base>&head=<sha>`.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it.

    Returns:
        The commit's header, message, refs, `base`, `files` and `truncated`.

    Raises:
        GitRequestError: The revision is malformed or unknown.
    """
    _repo(repo)
    sha = _resolve(repo, rev)
    info = git_history.commit_info(repo, sha)
    base = info.parents[0] if info.parents else git_ops.EMPTY_TREE
    files, truncated = _files(repo, base, sha)
    return {
        "sha": sha,
        "short": sha[:7],
        "parents": list(info.parents),
        "author": {"name": info.author, "email": info.author_email, "date": info.author_date},
        "committer": {
            "name": info.committer,
            "email": info.committer_email,
            "date": info.committer_date,
        },
        "subject": info.subject,
        "body": info.body,
        "refs": _decorations(git_history.list_refs(repo)).get(sha, []),
        "is_merge": len(info.parents) > 1,
        "base": base,
        "files": files,
        "truncated": truncated,
    }


def _endpoint(repo: pathlib.Path, rev: str, sha: str) -> dict[str, Any]:
    """Return one side of a compare: the rev as sent, the sha it names, and its subject."""
    subject = (
        "(empty tree)" if sha == git_ops.EMPTY_TREE else git_history.commit_info(repo, sha).subject
    )
    return {"rev": rev, "sha": sha, "short": sha[:7], "subject": subject}


def compare_payload(repo: pathlib.Path, base: str, head: str) -> dict[str, Any]:
    """Return the files that differ between two revisions, as `git diff base head` lists them.

    Args:
        repo: The repository.
        base: The older revision as the browser sent it; the empty tree is accepted.
        head: The newer revision as the browser sent it.

    Returns:
        `base` and `head` as sent and resolved, then `files` and `truncated`.

    Raises:
        GitRequestError: A revision is malformed, unknown, or (for `head`) not a commit.
    """
    _repo(repo)
    base_sha = _resolve(repo, base, tree_ok=True)
    head_sha = _resolve(repo, head)
    files, truncated = _files(repo, base_sha, head_sha)
    return {
        "base": _endpoint(repo, base, base_sha),
        "head": _endpoint(repo, head, head_sha),
        "files": files,
        "truncated": truncated,
    }


def diff_payload(
    repo: pathlib.Path,
    base: str,
    head: str,
    path: str,
    old_path: str = "",
    *,
    whole: bool = False,
) -> dict[str, Any]:
    """Return one file's unified diff between two revisions.

    Args:
        repo: The repository.
        base: The older revision; the empty tree is accepted.
        head: The newer revision.
        path: The file's path in `head` (in `base` for a deletion).
        old_path: Its path in `base` when renamed, so the patch reads as a rename.
        whole: Keep every byte instead of cutting at `DIFF_MAX_BYTES`: the page is the one
            place that shows a whole patch.

    Returns:
        The resolved shas, the paths, the patch, and the `binary`, `truncated` and
        `max_bytes` facts.

    Raises:
        GitRequestError: A revision or path is malformed, or the path is in neither tree.
    """
    _repo(repo)
    base_sha = _resolve(repo, base, tree_ok=True)
    head_sha = _resolve(repo, head)
    _check_shape(path, "path")
    if old_path:
        _check_shape(old_path, "path")
    if not (
        git_history.tree_has_blob(repo, head_sha, path)
        or git_history.tree_has_blob(repo, base_sha, path)
    ):
        raise GitRequestError(f"no file {path!r} in either revision")
    if old_path and not git_history.tree_has_blob(repo, base_sha, old_path):
        raise GitRequestError(f"no file {old_path!r} in the base revision")
    cap = None if whole else DIFF_MAX_BYTES
    patch, truncated = git_history.file_patch(
        repo, base_sha, head_sha, path, old_path, max_bytes=cap
    )
    return {
        "base": base_sha,
        "head": head_sha,
        "path": path,
        "old_path": old_path,
        "patch": patch,
        "binary": git_history.is_binary_patch(patch),
        "truncated": truncated,
        "max_bytes": cap,
    }


def search_payload(repo: pathlib.Path, query: str) -> dict[str, Any]:
    """Return the refs whose name contains the text and the commits whose message or sha does.

    An empty query lists every ref and the newest commits, the picker's opening list.
    Each list stops at `SEARCH_MAX`; `truncated` says when one did.

    Args:
        repo: The repository.
        query: The text typed into a picker.

    Returns:
        `refs`, `commits` and `truncated`.
    """
    _repo(repo)
    q = query.strip()
    if q.startswith("-"):
        return {"refs": [], "commits": [], "truncated": False}
    needle = q.lower()
    refs = [r for r in git_history.list_refs(repo) if needle in r.name.lower()]
    commits = list(git_history.search_log(repo, q, limit=SEARCH_MAX + 1))
    if q and all(c in "0123456789abcdef" for c in needle) and len(needle) >= 4:
        sha = git_history.resolve_commit(repo, needle)
        if sha and all(c.sha != sha for c in commits):
            commits[:0] = git_history.log_commits(repo, [sha], limit=1)
    truncated = len(refs) > SEARCH_MAX or len(commits) > SEARCH_MAX
    return {
        "refs": [_ref_row(r) for r in refs[:SEARCH_MAX]],
        "commits": [_log_row(c) for c in commits[:SEARCH_MAX]],
        "truncated": truncated,
    }
