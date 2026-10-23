# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build read-only Git and Files payloads with verified revisions and marked caps.

Revisions resolve through `rev-parse --verify --end-of-options`; paths follow `--`.
WORKTREE names disk content for files, or the head of a diff or comparison.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Any

from agent6.ui.web import highlight
from agent6.vcs import git_files, git_history, git_read
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
# The largest file `/api/git/blob` and `/api/git/text` serve.
BLOB_MAX_BYTES = 20 * 1024 * 1024
# Where a file's text is cut.
TEXT_MAX_BYTES = 1024 * 1024
# Files listed for one tree before the list is cut.
TREE_MAX = 20000
# The content types a blob is served under, by extension; every other file is an untyped
# download, so the browser never renders a repository file as a document.
IMAGE_TYPES = {
    "avif": "image/avif",
    "bmp": "image/bmp",
    "gif": "image/gif",
    "ico": "image/x-icon",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "svg": "image/svg+xml",
    "webp": "image/webp",
}


class GitRequestError(Exception):
    """A Git tab request the server refuses: a malformed or unknown revision, ref or path."""


class BlobTooLargeError(GitRequestError):
    """A file over `BLOB_MAX_BYTES`, which the server answers 413."""


@dataclasses.dataclass(frozen=True, slots=True)
class BlobResponse:
    """A file's bytes with the type the browser may read them as.

    Attributes:
        content_type: An image type by extension, else `application/octet-stream`.
        attachment: Whether the response downloads: true for everything but an image.
        data: The bytes.
    """

    content_type: str
    attachment: bool
    data: bytes


def _repo(repo: pathlib.Path) -> None:
    if not git_read.is_git_repo(repo):
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


def _check_path(value: str) -> str:
    """Return the value when it is shaped like a repository-relative path.

    Args:
        value: The path as the browser sent it.

    Returns:
        The value unchanged.

    Raises:
        GitRequestError: The value fails `_check_shape`, is absolute, or has a `..` part.
    """
    _check_shape(value, "path")
    if value.startswith("/") or ".." in value.split("/"):
        raise GitRequestError(f"not a repository path: {value!r}")
    return value


def _resolve(
    repo: pathlib.Path, rev: str, *, tree_ok: bool = False, worktree_ok: bool = False
) -> str:
    """Return the full sha a revision names.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it.
        tree_ok: Accept `git_read.EMPTY_TREE`, the base a root commit diffs against.
        worktree_ok: Accept `git_history.WORKTREE`, the working tree as a head.

    Returns:
        The full sha, or the empty tree's own, or `WORKTREE`.

    Raises:
        GitRequestError: The revision is malformed, unknown, or not a commit.
    """
    _check_shape(rev, "revision")
    if rev == git_history.WORKTREE:
        if worktree_ok:
            return rev
        raise GitRequestError(f"{rev} names the working tree: it is only a diff or compare head")
    if rev == git_read.EMPTY_TREE:
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
    """Return the files that differ from the base, cut at `FILES_MAX`, and whether they were."""
    if head == git_history.WORKTREE:
        rows = git_history.worktree_files(repo, base)
    else:
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
        refs: Full ref names, resolved shas or `HEAD`; None walks branches, remotes and tags
            plus HEAD, leaving out the `refs/agent6/*` chain refs.
        limit: Commits in the page, as the query gave it; "" is `GRAPH_PAGE`, more than
            `GRAPH_PAGE_MAX` is clamped.
        skip: Commits to pass over first, as the query gave it.

    Returns:
        `commits`, each with its refs, then `tips`, `has_more`, `skip` and `limit`.

    Raises:
        GitRequestError: A name is not a listed ref, or a number is not one.
    """
    _repo(repo)
    page = min(_whole_number(limit, "limit", GRAPH_PAGE), GRAPH_PAGE_MAX)
    start = _whole_number(skip, "skip", 0)
    listed = git_history.list_refs(repo)
    head_sha, _branch = git_history.head_commit(repo)
    known = {r.full: r.sha for r in listed}
    known["HEAD"] = head_sha
    if refs is None:
        revs = [r.sha for r in listed if r.kind != "agent6"] + ([head_sha] if head_sha else [])
    else:
        revs: list[str] = []
        for name in refs:
            if name in known:
                if known[name]:
                    revs.append(known[name])
            elif (
                len(name) in (40, 64)
                and all(c in "0123456789abcdef" for c in name)
                and git_history.resolve_commit(repo, name) == name
            ):
                revs.append(name)
            else:
                raise GitRequestError(f"no such ref {name!r}")
    revs = list(dict.fromkeys(_resolve(repo, rev) for rev in revs))
    rows = git_history.log_commits(repo, revs, limit=page + 1, skip=start) if revs else ()
    decorated = _decorations(listed)
    commits = [{**_log_row(c), "refs": decorated.get(c.sha, [])} for c in rows[:page]]
    return {
        "commits": commits,
        "tips": revs,
        "has_more": len(rows) > page,
        "skip": start,
        "limit": page,
    }


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
    base = info.parents[0] if info.parents else git_read.EMPTY_TREE
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


def worktree_payload(repo: pathlib.Path) -> dict[str, Any]:
    """Return the working tree's uncommitted files, tracked changes first.

    Args:
        repo: The repository.

    Returns:
        `base` (HEAD's sha, the empty tree on an unborn branch), `files` and `truncated`.
    """
    _repo(repo)
    sha, _branch = git_history.head_commit(repo)
    base = sha or git_read.EMPTY_TREE
    files, truncated = _files(repo, base, git_history.WORKTREE)
    return {"base": base, "files": files, "truncated": truncated}


def _endpoint(repo: pathlib.Path, rev: str, sha: str) -> dict[str, Any]:
    """Return one side of a compare: the rev as sent, the sha it names, and its subject."""
    if sha == git_history.WORKTREE:
        subject = "(working tree)"
    elif sha == git_read.EMPTY_TREE:
        subject = "(empty tree)"
    else:
        subject = git_history.commit_info(repo, sha).subject
    short = sha if sha == git_history.WORKTREE else sha[:7]
    return {"rev": rev, "sha": sha, "short": short, "subject": subject}


def compare_payload(repo: pathlib.Path, base: str, head: str) -> dict[str, Any]:
    """Return the files that differ between two revisions, as `git diff base head` lists them.

    Args:
        repo: The repository.
        base: The older revision as the browser sent it; the empty tree is accepted.
        head: The newer revision as the browser sent it; `WORKTREE` is the working tree.

    Returns:
        `base` and `head` as sent and resolved, then `files` and `truncated`.

    Raises:
        GitRequestError: A revision is malformed, unknown, or (for `head`) not a commit.
    """
    _repo(repo)
    base_sha = _resolve(repo, base, tree_ok=True)
    head_sha = _resolve(repo, head, worktree_ok=True)
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
        head: The newer revision; `WORKTREE` is the working tree.
        path: The file's path in `head` (in `base` for a deletion).
        old_path: Its path in `base` when renamed, so the patch reads as a rename.
        whole: Keep every byte instead of cutting at `DIFF_MAX_BYTES`: the page is the one
            place that shows a whole patch.

    Returns:
        The resolved shas, the paths, the patch, and the `binary`, `truncated` and
        `max_bytes` facts.

    Raises:
        GitRequestError: A revision or path is malformed, the path is absolute or climbs out
            of the repository, or it is in neither tree.
    """
    _repo(repo)
    base_sha = _resolve(repo, base, tree_ok=True)
    head_sha = _resolve(repo, head, worktree_ok=True)
    _check_path(path)
    if old_path:
        _check_path(old_path)
    worktree = head_sha == git_history.WORKTREE
    entry = git_history.worktree_entry(repo, path) if worktree else ""
    in_head = bool(entry) if worktree else git_history.tree_has_entry(repo, head_sha, path)
    if not (in_head or git_history.tree_has_entry(repo, base_sha, path)):
        raise GitRequestError(f"no file {path!r} in either revision")
    if old_path and not git_history.tree_has_entry(repo, base_sha, old_path):
        raise GitRequestError(f"no file {old_path!r} in the base revision")
    cap = None if whole else DIFF_MAX_BYTES
    if worktree:
        patch, truncated = git_history.worktree_patch(
            repo, base_sha, path, old_path, untracked=entry == "untracked", max_bytes=cap
        )
    else:
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


def _read(repo: pathlib.Path, rev: str, path: str) -> tuple[str, bytes]:
    """Return a regular file's bytes at a revision or in the working tree.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it; `WORKTREE` reads the file from disk.
        path: The file's repository-relative path.

    Returns:
        The resolved revision (`WORKTREE` for the working tree) and the bytes.

    Raises:
        GitRequestError: The revision or path is malformed, or no regular file is there.
        BlobTooLargeError: The file is over `BLOB_MAX_BYTES`.
    """
    sha = _resolve(repo, rev, worktree_ok=True)
    _check_path(path)
    if sha == git_history.WORKTREE:
        blob = git_files.worktree_blob(repo, path, max_bytes=BLOB_MAX_BYTES)
        where = "the working tree"
    else:
        blob = git_files.blob_at(repo, sha, path, max_bytes=BLOB_MAX_BYTES)
        where = rev
    if blob is None:
        raise GitRequestError(f"no file {path!r} in {where}")
    if blob.data is None:
        raise BlobTooLargeError(
            f"{path!r} is {blob.size} bytes; the most served is {BLOB_MAX_BYTES} bytes"
            f" ({BLOB_MAX_BYTES >> 20} MiB)"
        )
    return sha, blob.data


def blob_payload(repo: pathlib.Path, rev: str, path: str) -> BlobResponse:
    """Return a file's bytes at a revision or in the working tree.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it; `WORKTREE` reads the file from disk.
        path: The file's repository-relative path.

    Returns:
        The bytes, typed by extension for an image, an attachment otherwise.

    Raises:
        GitRequestError: The revision or path is malformed, or no regular file is there.
        BlobTooLargeError: The file is over `BLOB_MAX_BYTES`.
    """
    _repo(repo)
    _sha, data = _read(repo, rev, path)
    ext = path.rpartition("/")[2].rpartition(".")[2].lower()
    ctype = IMAGE_TYPES.get(ext)
    return BlobResponse(ctype or "application/octet-stream", ctype is None, data)


def _line_count(text: str) -> int:
    """Return the lines in a text, as the page splits it: a final newline starts none."""
    return text.count("\n") + (0 if not text or text.endswith("\n") else 1)


def _display_text(data: bytes, cap: int | None) -> tuple[str, bool, bool]:
    """Return text, binary and truncated with the shared text and blame cap."""
    binary = git_history.looks_binary(data)
    truncated = not binary and cap is not None and len(data) > cap
    if truncated:
        data = data[: data.rfind(b"\n", 0, cap) + 1]
    return "" if binary else data.decode("utf-8", errors="replace"), binary, truncated


def blame_payload(
    repo: pathlib.Path, rev: str, path: str, *, whole: bool = False
) -> dict[str, Any]:
    """Return per-line attribution with the text route's caps, refusals and digest."""
    _repo(repo)
    sha, data = _read(repo, rev, path)
    cap = None if whole else TEXT_MAX_BYTES
    text, binary, truncated = _display_text(data, cap)
    rows = git_files.blame_lines(repo, sha, path, data, lines=_line_count(text))
    return {
        "rev": sha,
        "path": path,
        "digest": hashlib.sha256(data).hexdigest(),
        "lines": [dataclasses.asdict(row) for row in rows],
        "binary": binary,
        "truncated": truncated,
        "max_bytes": cap,
    }


def text_payload(repo: pathlib.Path, rev: str, path: str, *, whole: bool = False) -> dict[str, Any]:
    """Return a file's text at a revision or in the working tree, with its highlight spans.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it; `WORKTREE` reads the file from disk.
        path: The file's repository-relative path.
        whole: Keep every byte instead of cutting at `TEXT_MAX_BYTES`.

    Returns:
        `rev` resolved, `path`, `digest` (sha256 of the bytes read; the blame route sends
        the same, so rows attach only to the text shown), `text` ("" for a binary file),
        `lines`, `binary`, `truncated`, `max_bytes`, `language` (the pack's name, "" when
        none) and `spans` (per line `[start, end, class]`, empty when nothing is highlighted).

    Raises:
        GitRequestError: The revision or path is malformed, or no regular file is there.
        BlobTooLargeError: The file is over `BLOB_MAX_BYTES`.
    """
    _repo(repo)
    sha, data = _read(repo, rev, path)
    cap = None if whole else TEXT_MAX_BYTES
    text, binary, truncated = _display_text(data, cap)
    language = highlight.language_of(path)
    return {
        "rev": sha,
        "path": path,
        "digest": hashlib.sha256(data).hexdigest(),
        "text": text,
        "lines": _line_count(text),
        "binary": binary,
        "truncated": truncated,
        "max_bytes": cap,
        "language": language,
        "spans": highlight.spans(text, language) if text else [],
    }


def tree_payload(repo: pathlib.Path, rev: str) -> dict[str, Any]:
    """Return the files at a revision or in the working tree.

    Args:
        repo: The repository.
        rev: The revision as the browser sent it; `WORKTREE` lists the index plus the
            untracked files `.gitignore` keeps.

    Returns:
        `rev` resolved, `files` as `{path, size, kind, state}` cut at `TREE_MAX`, and `truncated`.

    Raises:
        GitRequestError: The revision is malformed or unknown.
    """
    _repo(repo)
    sha = _resolve(repo, rev, worktree_ok=True)
    if sha == git_history.WORKTREE:
        rows = git_files.worktree_tree_files(repo)
    else:
        rows = git_files.tree_files(repo, sha)
    files = [dataclasses.asdict(f) for f in rows[:TREE_MAX]]
    return {"rev": sha, "files": files, "truncated": len(rows) > TREE_MAX}
