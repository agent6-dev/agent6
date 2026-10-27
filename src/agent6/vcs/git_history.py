# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read the repository for display: refs, the log, a commit, its files, a patch, the working tree.

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
from agent6.vcs import _git_command, git_read

# The revision that names the working tree as the head of a diff or compare.
WORKTREE = "WORKTREE"


def tree_has_entry(path: pathlib.Path, rev: str, name: str) -> bool:
    """Return whether the tree at the rev holds the name as an entry a diff can show.

    A file, a symlink or a submodule pointer (whose patch is the pointer move), not a
    directory.

    Args:
        path: The repository.
        rev: The commit or tree.
        name: The entry's path in the tree.

    Returns:
        True for a blob or a gitlink entry.
    """
    metadata = git_read.tree_entry(path, rev, name)
    return " blob " in metadata or " commit " in metadata


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


# The diff command for two trees, and for the working tree against one tree.
_TREES = ("diff-tree", "-r")
_WORKTREE = ("diff",)


def _listing(
    path: pathlib.Path, cmd: tuple[str, ...], what: str, revs: tuple[str, ...]
) -> list[str]:
    """Return the NUL-separated fields of a `--name-status` or `--numstat` diff listing."""
    return _git_command.nul_fields(path, *cmd, "-M", "-z", what, "--end-of-options", *revs)


def _numstat(
    path: pathlib.Path, cmd: tuple[str, ...], revs: tuple[str, ...]
) -> dict[str, tuple[int, int] | None]:
    """Return each changed path's line counts, None for a binary file."""
    fields = _listing(path, cmd, "--numstat", revs)
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


def _changed(
    path: pathlib.Path, cmd: tuple[str, ...], revs: tuple[str, ...]
) -> tuple[ChangedFile, ...]:
    """Return the files a diff listing names, renames detected, with their counts."""
    fields = _listing(path, cmd, "--name-status", revs)
    counts = _numstat(path, cmd, revs)
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
    return _changed(path, _TREES, (base, head))


# git calls a file binary on a NUL in its first 8000 bytes.
_BINARY_PROBE = 8000


def looks_binary(data: bytes) -> bool:
    """Return whether git would call the bytes binary: a NUL in the first 8000."""
    return b"\0" in data[:_BINARY_PROBE]


def _added_counts(full: pathlib.Path) -> tuple[int | None, bool]:
    """Return an untracked file's line count and whether git calls it binary.

    A symlink is its one-line target, as `git diff` shows it; a file that is neither a
    symlink nor a regular file gets no counts.

    Args:
        full: The file's path on disk.

    Returns:
        The lines `git diff --no-index` would add, or None with True for a binary file.

    Raises:
        OSError: The file cannot be read.
    """
    if full.is_symlink():
        return 1, False
    if not full.is_file():
        return None, True
    with full.open("rb") as fh:
        chunk = fh.read(_BINARY_PROBE)
        if looks_binary(chunk):
            return None, True
        lines, tail = 0, b""
        while chunk:
            lines += chunk.count(b"\n")
            tail = chunk[-1:]
            chunk = fh.read(1 << 16)
    return lines + (1 if tail and tail != b"\n" else 0), False


def worktree_files(path: pathlib.Path, base: str) -> tuple[ChangedFile, ...]:
    """Return the files the working tree changes against a tree, then its untracked files.

    Tracked changes, staged or not, come from `git diff <base>` with git's letters and
    counts. Untracked files honour `.gitignore` (`ls-files --others --exclude-standard`)
    and read `A`, with their line count or `binary`.

    Args:
        path: The repository.
        base: The commit or tree to diff against; `EMPTY_TREE` on an unborn branch.

    Returns:
        One row per changed or untracked path; a file gone before it was read is left out.

    Raises:
        GitError: The base cannot be read.
    """
    rows = list(_changed(path, _WORKTREE, (base,)))
    names = _git_command.nul_fields(path, "ls-files", "-z", "--others", "--exclude-standard")
    for name in names:
        if not name:
            continue
        try:
            added, binary = _added_counts(path / name)
        except OSError:
            continue
        rows.append(ChangedFile("A", name, "", added, None if binary else 0, binary))
    return tuple(rows)


def worktree_entry(path: pathlib.Path, name: str) -> str:
    """Return "tracked" or "untracked" when the working tree holds the path as a file, else "".

    `ls-files -t` tags an index entry `H` (`S` skip-worktree, `M` unmerged) and an
    untracked, unignored file `?`; an ignored file, a directory, and a path outside the
    repository or through a symlink get no entry.

    Args:
        path: The repository.
        name: The repository-relative path.

    Returns:
        "tracked", "untracked" or "".
    """
    fields = _git_command.nul_fields(
        path,
        "ls-files",
        "-z",
        "-t",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        name,
        check=False,
        env_extra=git_read.LITERAL_PATHSPECS,
    )
    for entry in fields:
        tag, _, found = entry.partition(" ")
        if found == name:
            return "untracked" if tag == "?" else "tracked"
    return ""


_BINARY_DIFF_RE = re.compile(r"^Binary files .* differ$", re.MULTILINE)


def file_patch(
    path: pathlib.Path,
    base: str,
    head: str,
    name: str,
    old_name: str = "",
    *,
    max_bytes: int | None,
) -> tuple[str, bool]:
    """Return one file's unified diff between two trees, cut at a cap.

    Args:
        path: The repository.
        base: The older commit or tree.
        head: The newer commit or tree.
        name: The file's path in the newer tree (the older one's for a deletion).
        old_name: The older tree's path when renamed, so the rename shows as one.
        max_bytes: Where the patch is cut, at the last line boundary before it; None keeps
            every byte.

    Returns:
        The patch text (a binary file's says `Binary files ... differ`) and whether it was cut.

    Raises:
        GitError: A rev cannot be read.
    """
    specs = (old_name, name) if old_name else (name,)
    args = ("diff", "-M", "--no-color", "--end-of-options", base, head, "--", *specs)
    return _patch(path, args, ok=(0,), max_bytes=max_bytes)


def worktree_patch(
    path: pathlib.Path,
    base: str,
    name: str,
    old_name: str = "",
    *,
    untracked: bool = False,
    max_bytes: int | None,
) -> tuple[str, bool]:
    """Return one working-tree file's unified diff against a tree, cut at a cap.

    Args:
        path: The repository.
        base: The commit or tree to diff against.
        name: The file's path in the working tree (the tree's for a deletion).
        old_name: The tree's path when renamed, so the rename shows as one.
        untracked: The file is not in the index: it diffs against `/dev/null`.
        max_bytes: Where the patch is cut, at the last line boundary before it; None keeps
            every byte.

    Returns:
        The patch text and whether it was cut.

    Raises:
        GitError: The base cannot be read or the file cannot be diffed.
    """
    if untracked:
        # `--no-index` exits 1 when the two sides differ, as a new file always does.
        args = ("diff", "--no-color", "--no-index", "--", "/dev/null", name)
        return _patch(path, args, ok=(0, 1), max_bytes=max_bytes)
    specs = (old_name, name) if old_name else (name,)
    args = ("diff", "-M", "--no-color", "--end-of-options", base, "--", *specs)
    return _patch(path, args, ok=(0,), max_bytes=max_bytes)


def _patch(
    path: pathlib.Path, args: tuple[str, ...], *, ok: tuple[int, ...], max_bytes: int | None
) -> tuple[str, bool]:
    """Run a diff and return its text, cut at a cap.

    Args:
        path: The repository.
        args: The diff command, `diff` first.
        ok: The exit codes that are not failures.
        max_bytes: Where the text is cut, at the last line boundary before it; None keeps
            every byte.

    Returns:
        The text and whether it was cut.

    Raises:
        GitError: git exited with another code.
    """
    _argv, rc, out, err = _git_command.spawn(path, args, git_read.LITERAL_PATHSPECS, None)
    if rc not in ok:
        error = _git_command.failure(
            args, rc, out.decode(errors="replace"), err.decode(errors="replace")
        )
        raise error
    if max_bytes is None or len(out) <= max_bytes:
        return out.decode(errors="replace"), False
    cut = out.rfind(b"\n", 0, max_bytes) + 1
    return out[:cut].decode(errors="replace"), True


def is_binary_patch(patch: str) -> bool:
    """Return whether a `file_patch` says the file is binary."""
    return _BINARY_DIFF_RE.search(patch) is not None
