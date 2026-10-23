"""Read file trees, regular file bytes and blame without following working-tree symlinks."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import os
import pathlib
import stat

from agent6 import errors
from agent6.vcs import _git_command, git_history, git_read


@dataclasses.dataclass(frozen=True, slots=True)
class Blob:
    """A file's bytes, or only its size when it is over the reader's cap.

    Attributes:
        size: The file's size in bytes.
        data: The bytes; None when the size is over the cap.
    """

    size: int
    data: bytes | None


def blob_at(path: pathlib.Path, rev: str, name: str, *, max_bytes: int) -> Blob | None:
    """Return a regular file's bytes at a commit or tree.

    Args:
        path: The repository.
        rev: The commit or tree.
        name: The file's path in the tree.
        max_bytes: The most bytes read; a larger file comes back as its size only.

    Returns:
        The blob, or None when the tree holds no regular file there (a symlink or a
        directory is none).

    Raises:
        GitError: The rev cannot be read.
    """
    metadata = git_read.tree_entry(path, rev, name)
    if not metadata.startswith(("100644 blob ", "100755 blob ")):
        return None
    sha = metadata.split()[2]
    size = int(_git_command.run(path, "cat-file", "-s", sha).stdout)
    if size > max_bytes:
        return Blob(size, None)
    args = ("cat-file", "blob", sha)
    _argv, rc, out, err = _git_command.spawn(path, args, None, None)
    if rc != 0:
        raise errors.GitError(
            _git_command.failure(
                args, rc, out.decode(errors="replace"), err.decode(errors="replace")
            )
        )
    return Blob(size, out)


def worktree_blob(path: pathlib.Path, name: str, *, max_bytes: int) -> Blob | None:
    """Return a working-tree file's bytes, read from disk.

    Only a path `worktree_entry` names is opened; no ancestor or leaf symlink is followed,
    and nonblocking opens plus fstat keep special files from blocking or being read.

    Args:
        path: The repository.
        name: The repository-relative path.
        max_bytes: The most bytes read; a larger file comes back as its size only.

    Returns:
        The blob, or None when git lists no file there or it is not a regular file.
    """
    try:
        root = path.resolve(strict=True)
        full = root / name
        parts = pathlib.PurePosixPath(name).parts
        if (
            not parts
            or ".." in parts
            or pathlib.PurePosixPath(name).is_absolute()
            or not full.resolve(strict=True).is_relative_to(root)
            or not git_history.worktree_entry(path, name)
        ):
            return None
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        # Descriptor-relative opens keep an ancestor swap from escaping the checked root.
        with contextlib.ExitStack() as stack:
            parent = os.open(root, flags | os.O_DIRECTORY)
            stack.callback(os.close, parent)
            for part in parts[:-1]:
                parent = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
                stack.callback(os.close, parent)
            if not stat.S_ISREG(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False).st_mode):
                return None
            fd = os.open(parts[-1], flags, dir_fd=parent)
            with os.fdopen(fd, "rb") as fh:
                st = os.fstat(fh.fileno())
                if not stat.S_ISREG(st.st_mode):
                    return None
                if st.st_size > max_bytes:
                    return Blob(st.st_size, None)
                data = fh.read(max_bytes + 1)
    except (OSError, RuntimeError):
        return None
    return Blob(len(data), None if len(data) > max_bytes else data)


@dataclasses.dataclass(frozen=True, slots=True)
class BlameLine:
    """One line's last commit; an empty sha marks uncommitted content."""

    line: int
    sha: str
    author: str
    date: str


def _ignore_revs_refusal(path: pathlib.Path) -> str:
    """Return why blame must not run, or "" when every ignore-revs file is a repository file.

    Git opens each `blame.ignoreRevsFile` on the host, quotes its first unparsable line in
    its error and blocks on a FIFO; a blank `-c` override clears the list only after the
    files are read, so the paths are checked here instead.

    Args:
        path: The repository.

    Returns:
        The refusal naming the offending value, or "".
    """
    res = _git_command.run(
        path, "config", "--get-all", "--type=path", "blame.ignoreRevsFile", check=False
    )
    root = path.resolve()
    for value in filter(None, res.stdout.splitlines()):
        real = (root / value).resolve()
        if not (real.is_relative_to(root) and real.is_file()):
            return (
                f"blame refused: blame.ignoreRevsFile names {value}, which is not a regular "
                "file inside the repository"
            )
    return ""


def blame_lines(
    path: pathlib.Path, rev: str, name: str, data: bytes, *, lines: int
) -> tuple[BlameLine, ...]:
    """Return attribution from one porcelain read, using checked bytes for the working tree.

    Args:
        path: The repository.
        rev: A resolved commit or WORKTREE.
        name: The checked repository path.
        data: The bytes the regular-file reader returned, never reopened by blame.
        lines: The displayed lines, after the text cap.

    Returns:
        Attribution in displayed line order, with UTC dates.

    Raises:
        GitError: Git cannot attribute the file, or a `blame.ignoreRevsFile` leaves the repository.
    """
    if not lines:
        return ()
    working = rev == git_history.WORKTREE
    if working and (
        git_history.worktree_entry(path, name) == "untracked"
        or not git_history.head_commit(path)[0]
    ):
        return tuple(BlameLine(n, "", "not committed", "") for n in range(1, lines + 1))
    if reason := _ignore_revs_refusal(path):
        raise errors.GitError(reason)
    args = ("--contents", "-") if working else ()
    revs = () if working else (rev,)
    # Git 2.53 blame needs the verified sha before --end-of-options when a path follows --.
    out = _git_command.run(
        path,
        "blame",
        "--porcelain",
        "-L",
        f"1,{lines}",
        *args,
        *revs,
        "--end-of-options",
        "--",
        name,
        stdin_text=os.fsdecode(data) if working else None,
    ).stdout
    metadata: dict[str, dict[str, str]] = {}
    rows: list[BlameLine] = []
    sha, number = "", 0
    for line in out.split("\n"):
        if line.startswith("\t"):
            meta = metadata[sha]
            committed = bool(sha.strip("0"))
            date = (
                datetime.datetime.fromtimestamp(int(meta["author-time"]), datetime.UTC).isoformat()
                if committed
                else ""
            )
            rows.append(
                BlameLine(
                    number,
                    sha if committed else "",
                    meta["author"] if committed else "not committed",
                    date,
                )
            )
        elif len(line.split()) in (3, 4) and all(c in "0123456789abcdef" for c in line.split()[0]):
            fields = line.split()
            sha, number = fields[0], int(fields[2])
            metadata.setdefault(sha, {})
        else:
            key, _, value = line.partition(" ")
            if key in ("author", "author-time"):
                metadata[sha][key] = value
    return tuple(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class TreeFile:
    """One entry of a tree listing.

    Attributes:
        path: The repository-relative path.
        size: The size in bytes; a symlink's is its target's length, a submodule's 0.
        kind: "file", "symlink" or "submodule".
        state: "tracked", "untracked" or "modified" in the working tree, else "".
    """

    path: str
    size: int
    kind: str
    state: str = ""


def tree_files(path: pathlib.Path, rev: str) -> tuple[TreeFile, ...]:
    """Return every file of a commit or tree, in git's order.

    Args:
        path: The repository.
        rev: The commit or tree.

    Returns:
        The entries with their size and kind.

    Raises:
        GitError: The rev cannot be read.
    """
    fields = _git_command.nul_fields(path, "ls-tree", "-r", "-z", "-l", "--end-of-options", rev)
    rows: list[TreeFile] = []
    for entry in fields:
        meta, _, name = entry.partition("\t")
        parts = meta.split()
        if len(parts) != 4:
            continue
        mode, kind, _sha, size = parts
        if mode == "160000" and kind == "commit":
            rows.append(TreeFile(name, 0, "submodule"))
        elif mode == "120000":
            rows.append(TreeFile(name, int(size), "symlink"))
        else:
            rows.append(TreeFile(name, int(size), "file"))
    return tuple(rows)


def worktree_tree_files(path: pathlib.Path) -> tuple[TreeFile, ...]:
    """Return the working tree's files: the index plus untracked files `.gitignore` keeps.

    Args:
        path: The repository.

    Returns:
        The entries on disk with their size and kind, sorted by path; a tracked file gone
        from disk is left out.
    """
    entries = _git_command.nul_fields(
        path, "ls-files", "-z", "-t", "--stage", "--cached", "--others", "--exclude-standard"
    )
    changed: set[str] = set()
    status = iter(
        _git_command.nul_fields(path, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    )
    for entry in status:
        if entry:
            changed.add(entry[3:])
            if "R" in entry[:2] or "C" in entry[:2]:
                next(status, "")
    rows: list[TreeFile] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            continue
        tag, _, rest = entry.partition(" ")
        metadata, _, name = rest.partition("\t") if tag != "?" else ("", "", rest)
        if name in seen:
            continue
        seen.add(name)
        state = "untracked" if tag == "?" else "modified" if name in changed else "tracked"
        if metadata.startswith("160000 "):
            rows.append(TreeFile(name, 0, "submodule", state))
            continue
        try:
            st = os.lstat(path / name)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            rows.append(TreeFile(name, st.st_size, "file", state))
        elif stat.S_ISLNK(st.st_mode):
            rows.append(TreeFile(name, st.st_size, "symlink", state))

    return tuple(sorted(rows, key=lambda f: f.path))
