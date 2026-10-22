# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read the repository's history for display: refs, the log, a commit, its files, a patch.

The web Git tab's reads. Every rev these take comes from the operator; `--end-of-options`
keeps a flag-shaped one from reaching git as an option, and a path follows `--`. Nothing
here writes.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Sequence

from agent6 import errors
from agent6.vcs import _git_command, git_ops


def tree_has_blob(path: pathlib.Path, rev: str, name: str) -> bool:
    """Return whether the tree at the rev holds the name as a file or symlink, not a directory."""
    return " blob " in git_ops.tree_entry(path, rev, name)


# Every rev these take comes from the operator; `--end-of-options` keeps a flag-shaped one
# from reaching git as an option, and a path follows `--`.


@dataclasses.dataclass(frozen=True, slots=True)
class RefRow:
    """One ref of the repository.

    Attributes:
        name: The short name: `main`, `origin/main`, `v1.0`, `agent6/<id>/head`.
        full: The full ref name.
        kind: "branch", "remote", "tag" or "agent6" (a run's chain ref).
        sha: The commit the ref points at; an annotated tag's is the commit it peels to.
        head: Whether HEAD is this branch.
    """

    name: str
    full: str
    kind: str
    sha: str
    head: bool


_REF_KINDS = (
    ("refs/heads/", "branch"),
    ("refs/remotes/", "remote"),
    ("refs/tags/", "tag"),
    ("refs/agent6/", "agent6"),
)


def list_refs(path: pathlib.Path) -> tuple[RefRow, ...]:
    """Return every branch, remote-tracking ref, tag and agent6 chain ref, in git's order.

    Args:
        path: The repository.

    Returns:
        The refs, each with its kind, its tip and whether HEAD is on it.

    Raises:
        GitError: The path is not a repository.
    """
    out = _git_command.run(
        path,
        "for-each-ref",
        "--format=%(refname)%00%(objectname)%00%(*objectname)%00%(HEAD)",
        *(prefix.rstrip("/") for prefix, _ in _REF_KINDS),
    ).stdout
    rows: list[RefRow] = []
    for line in out.splitlines():
        fields = line.split("\x00")
        if len(fields) != 4:
            continue
        full, sha, peeled, head = fields
        for prefix, kind in _REF_KINDS:
            if full.startswith(prefix):
                name = full.removeprefix("refs/" if kind == "agent6" else prefix)
                rows.append(RefRow(name, full, kind, peeled or sha, head == "*"))
                break
    return tuple(rows)


def head_commit(path: pathlib.Path) -> tuple[str, str]:
    """Return HEAD's sha and branch; the sha is "" when unborn, the branch "" when detached."""
    sha = _git_command.run(
        path, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False
    ).stdout.strip()
    branch = _git_command.run(
        path, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
    ).stdout.strip()
    return sha, branch


def resolve_commit(path: pathlib.Path, rev: str) -> str:
    """Return the full sha of the commit a revision names, or "" when it names none.

    `^{commit}` peels a tag and rejects a tree or blob; a flag-shaped rev resolves to "".

    Args:
        path: The repository.
        rev: A branch, tag, ref name, sha prefix or revision expression.

    Returns:
        The full sha, or "" when the revision names no commit.
    """
    res = _git_command.run(
        path,
        "rev-parse",
        "--verify",
        "--quiet",
        "--end-of-options",
        f"{rev}^{{commit}}",
        check=False,
    )
    return res.stdout.strip() if res.ok else ""


@dataclasses.dataclass(frozen=True, slots=True)
class LogCommit:
    """One commit of a log walk.

    Attributes:
        sha: The full sha.
        parents: The parent shas, first parent first.
        author: The author's name.
        date: The author date, ISO 8601.
        epoch: The author date as epoch seconds.
        subject: The first line of the message.
    """

    sha: str
    parents: tuple[str, ...]
    author: str
    date: str
    epoch: int
    subject: str


_LOG_FORMAT = "%H%x1f%P%x1f%an%x1f%aI%x1f%at%x1f%s"


def _parse_log(stdout: str) -> tuple[LogCommit, ...]:
    """Return the commits of a `-z` log in `_LOG_FORMAT`."""
    rows: list[LogCommit] = []
    for rec in stdout.split("\x00"):
        fields = rec.split("\x1f")
        if len(fields) != 6:
            continue
        sha, parents, author, date, epoch, subject = fields
        rows.append(LogCommit(sha, tuple(parents.split()), author, date, int(epoch), subject))
    return tuple(rows)


def log_commits(
    path: pathlib.Path, revs: Sequence[str], *, limit: int, skip: int = 0
) -> tuple[LogCommit, ...]:
    """Return the commits reachable from the revs, newest first in topological order.

    Args:
        path: The repository.
        revs: The starting points, ref names or shas.
        limit: The most commits returned.
        skip: Commits passed over first; the walk is stable for the same revs, so this pages.

    Returns:
        The commits; empty when a rev does not resolve or the branch is unborn.
    """
    res = _git_command.run(
        path,
        "log",
        "-z",
        "--topo-order",
        f"--max-count={limit}",
        f"--skip={skip}",
        f"--format={_LOG_FORMAT}",
        "--end-of-options",
        *revs,
        check=False,
    )
    return _parse_log(res.stdout) if res.ok else ()


def search_log(path: pathlib.Path, text: str, *, limit: int) -> tuple[LogCommit, ...]:
    """Return the newest commits on any ref whose message contains the text.

    Args:
        path: The repository.
        text: A case-insensitive literal, not a pattern; "" matches every commit.
        limit: The most commits returned.

    Returns:
        The matching commits, newest first.
    """
    grep = ("-i", "--fixed-strings", f"--grep={text}") if text else ()
    res = _git_command.run(
        path,
        "log",
        "-z",
        "--all",
        f"--max-count={limit}",
        f"--format={_LOG_FORMAT}",
        *grep,
        check=False,
    )
    return _parse_log(res.stdout) if res.ok else ()


@dataclasses.dataclass(frozen=True, slots=True)
class CommitInfo:
    """One commit's header and message.

    Attributes:
        sha: The full sha.
        parents: The parent shas, first parent first.
        author: The author's name.
        author_email: The author's email.
        author_date: The author date, ISO 8601.
        committer: The committer's name.
        committer_email: The committer's email.
        committer_date: The commit date, ISO 8601.
        subject: The first line of the message.
        body: The message after the subject, without the trailing newline.
    """

    sha: str
    parents: tuple[str, ...]
    author: str
    author_email: str
    author_date: str
    committer: str
    committer_email: str
    committer_date: str
    subject: str
    body: str


def commit_info(path: pathlib.Path, sha: str) -> CommitInfo:
    """Return a commit's header and message.

    Args:
        path: The repository.
        sha: A commit, as `resolve_commit` returned it.

    Returns:
        The header fields and the message.

    Raises:
        GitError: The sha is not a commit.
    """
    fmt = "%H%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI%x1f%s%x1f%b"
    out = _git_command.run(
        path, "log", "--max-count=1", f"--format={fmt}", "--end-of-options", sha
    ).stdout
    fields = out.split("\x1f", 9)
    if len(fields) != 10:
        raise errors.GitError(f"git log printed no commit for {sha}")
    sha, parents, an, ae, ad, cn, ce, cd, subject, body = fields
    return CommitInfo(
        sha, tuple(parents.split()), an, ae, ad, cn, ce, cd, subject, body.rstrip("\n")
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ChangedFile:
    """One file that differs between two trees.

    Attributes:
        status: The letter git gives: A, M, D, R (renamed) or T (type changed).
        path: The path in the newer tree; the older tree's for a deletion.
        old_path: The older tree's path when renamed, else "".
        additions: Lines added; None for a binary file.
        deletions: Lines deleted; None for a binary file.
        binary: Whether git counts the file as binary.
    """

    status: str
    path: str
    old_path: str
    additions: int | None
    deletions: int | None
    binary: bool


def _numstat(path: pathlib.Path, base: str, head: str) -> dict[str, tuple[int, int] | None]:
    """Return each changed path's line counts, None for a binary file."""
    fields = _git_command.nul_fields(
        path, "diff-tree", "-r", "-M", "-z", "--numstat", "--end-of-options", base, head
    )
    counts: dict[str, tuple[int, int] | None] = {}
    i = 0
    while i < len(fields) and fields[i]:
        added, deleted, name = fields[i].split("\t", 2)
        if name:
            i += 1
        else:
            # A rename's record is `added<TAB>deleted<TAB>` and then the two paths.
            name = fields[i + 2]
            i += 3
        counts[name] = None if added == "-" else (int(added), int(deleted))
    return counts


def changed_files(path: pathlib.Path, base: str, head: str) -> tuple[ChangedFile, ...]:
    """Return the files that differ between two trees, renames detected.

    Args:
        path: The repository.
        base: The older commit or tree; `EMPTY_TREE` lists a root commit's files.
        head: The newer commit or tree.

    Returns:
        One row per changed path; a rename is one row carrying `old_path`.

    Raises:
        GitError: A rev cannot be read.
    """
    fields = _git_command.nul_fields(
        path, "diff-tree", "-r", "-M", "-z", "--name-status", "--end-of-options", base, head
    )
    counts = _numstat(path, base, head)
    rows: list[ChangedFile] = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i][:1]
        if status == "R":
            old_path, name = fields[i + 1], fields[i + 2]
            i += 3
        else:
            old_path, name = "", fields[i + 1]
            i += 2
        lines = counts.get(name)
        rows.append(
            ChangedFile(
                status,
                name,
                old_path,
                None if lines is None else lines[0],
                None if lines is None else lines[1],
                binary=lines is None,
            )
        )
    return tuple(rows)


_BINARY_DIFF_RE = re.compile(r"^Binary files .* differ$", re.MULTILINE)


def file_patch(
    path: pathlib.Path, base: str, head: str, name: str, old_name: str = "", *, max_bytes: int
) -> tuple[str, bool]:
    """Return one file's unified diff between two trees, cut at a cap.

    Args:
        path: The repository.
        base: The older commit or tree.
        head: The newer commit or tree.
        name: The file's path in the newer tree (the older one's for a deletion).
        old_name: The older tree's path when renamed, so the rename shows as one.
        max_bytes: Where the patch is cut, at the last line boundary before it.

    Returns:
        The patch text (a binary file's says `Binary files ... differ`) and whether it was cut.

    Raises:
        GitError: A rev cannot be read.
    """
    specs = (old_name, name) if old_name else (name,)
    args = ("diff", "-M", "--no-color", "--end-of-options", base, head, "--", *specs)
    _argv, rc, out, err = _git_command.spawn(path, args, git_ops.LITERAL_PATHSPECS, None)
    if rc != 0:
        raise errors.GitError(
            _git_command.failure(
                args, rc, out.decode(errors="replace"), err.decode(errors="replace")
            )
        )
    if len(out) <= max_bytes:
        return out.decode(errors="replace"), False
    cut = out.rfind(b"\n", 0, max_bytes) + 1
    return out[:cut].decode(errors="replace"), True


def is_binary_patch(patch: str) -> bool:
    """Return whether a `file_patch` says the file is binary."""
    return _BINARY_DIFF_RE.search(patch) is not None
