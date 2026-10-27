"""Read file trees, regular file bytes and blame without following working-tree symlinks."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import os
import pathlib
import stat
import tempfile

from agent6 import errors
from agent6.tools import _path_safety
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
    return read_blob(path, metadata.split()[2], max_bytes=max_bytes)


def read_blob(path: pathlib.Path, sha: str, *, max_bytes: int) -> Blob:
    """Read a blob by object ID under a byte cap.

    Args:
        path: The repository.
        sha: The blob's object ID.
        max_bytes: The most bytes read; a larger blob comes back as its size only.

    Returns:
        The blob's size and, when within the cap, its bytes.

    Raises:
        GitError: The blob cannot be read.
    """
    size = int(_git_command.run(path, "cat-file", "-s", sha).stdout)
    if size > max_bytes:
        return Blob(size, None)
    args = ("cat-file", "blob", sha)
    _argv, rc, out, err = _git_command.spawn(path, args, None, None)
    if rc != 0:
        error = _git_command.failure(
            args, rc, out.decode(errors="replace"), err.decode(errors="replace")
        )
        raise error
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
        with os.fdopen(_open_regular(root, parts), "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size > max_bytes:
                return Blob(size, None)
            data = fh.read(max_bytes + 1)
    except (OSError, RuntimeError):
        return None
    return Blob(len(data), None if len(data) > max_bytes else data)


def _open_regular(root: pathlib.Path, parts: tuple[str, ...]) -> int:
    """Open a regular file under the root through descriptor-relative opens that follow no link.

    Args:
        root: The resolved repository root.
        parts: The file's path components under the root.

    Returns:
        The descriptor, fstat-checked as a regular file after the open.

    Raises:
        FileNotFoundError: A component is missing or a link, or the leaf is not a regular file.
        OSError: The open itself fails.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    # Descriptor-relative opens keep an ancestor swap from escaping the checked root.
    with contextlib.ExitStack() as stack:
        parent = os.open(root, flags | os.O_DIRECTORY)
        stack.callback(os.close, parent)
        for part in parts[:-1]:
            parent = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            stack.callback(os.close, parent)
        if not stat.S_ISREG(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False).st_mode):
            raise FileNotFoundError("/".join(parts))
        fd = os.open(parts[-1], flags, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise FileNotFoundError("/".join(parts))
        return fd


@dataclasses.dataclass(frozen=True, slots=True)
class BlameLine:
    """One line's last commit; an empty sha marks uncommitted content."""

    line: int
    sha: str
    author: str
    date: str


# An ignore-revs file lists shas, one per line; a larger one is not that.
_IGNORE_REVS_MAX_BYTES = 1 << 20


def _ignore_revs_copies(path: pathlib.Path, into: pathlib.Path, hide: tuple[str, ...]) -> str:
    """Copy each `blame.ignoreRevsFile` into blame's work tree, or return why blame must not run.

    Git opens each configured file by its path when blame starts, after any check made here,
    so a jailed command could swap the real file for a link in between: blame runs with a
    private work tree holding copies read through descriptors that follow no link, and an
    absolute or outside path is refused.

    Args:
        path: The repository.
        into: The private work tree blame runs in.
        hide: `[sandbox].hide_paths`, refused before opening a configured file.

    Returns:
        The refusal naming the setting and failed constraint, or "".
    """
    res = _git_command.run(
        path, "config", "--get-all", "-z", "--type=path", "blame.ignoreRevsFile", check=False
    )
    root = path.resolve()
    workspace = _path_safety.host_workspace(root, hide)
    lexical_hidden = tuple(pathlib.Path(os.path.abspath(p)) for p in hide)  # noqa: PTH100
    workspace = dataclasses.replace(workspace, denied=(*workspace.denied, *lexical_hidden))
    # NUL-delimited: a value holding a newline is one path to git, and must be one here.
    for value in filter(None, res.stdout.split("\0")):
        parts = pathlib.PurePosixPath(value).parts
        refusal = (
            "blame refused: blame.ignoreRevsFile must name a readable, "
            "repository-relative regular file"
        )
        if not parts or pathlib.PurePosixPath(value).is_absolute() or ".." in parts:
            return refusal
        if workspace.is_denied(root.joinpath(*parts)):
            return "blame refused: blame.ignoreRevsFile is hidden by sandbox.hide_paths"
        try:
            with os.fdopen(_open_regular(root, parts), "rb") as fh:
                data = fh.read(_IGNORE_REVS_MAX_BYTES + 1)
        except OSError:
            return refusal
        if len(data) > _IGNORE_REVS_MAX_BYTES:
            return "blame refused: blame.ignoreRevsFile exceeds 1 MiB"
        copy = into.joinpath(*parts)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(data)
    return ""


def _author_date(seconds: str) -> str:
    """Return an ISO author time, or the raw `@seconds` when Python cannot represent it."""
    try:
        return datetime.datetime.fromtimestamp(int(seconds), datetime.UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return f"@{seconds}"


def blame_lines(
    path: pathlib.Path,
    rev: str,
    name: str,
    data: bytes,
    *,
    lines: int,
    start_line: int = 1,
    hide: tuple[str, ...] = (),
) -> tuple[BlameLine, ...]:
    """Return attribution from one porcelain read, using checked bytes for the working tree.

    Args:
        path: The repository.
        rev: A resolved commit or WORKTREE.
        name: The checked repository path.
        data: The bytes the regular-file reader returned, never reopened by blame.
        lines: The number of lines to attribute.
        start_line: The first line to attribute, counted from one.
        hide: `[sandbox].hide_paths`, refused for ignore-revs inputs.

    Returns:
        Attribution in displayed line order, with UTC dates.

    Raises:
        GitError: The range is invalid, Git cannot attribute the file, or a
            `blame.ignoreRevsFile` is hidden or not a repository-relative regular file.
    """
    if start_line < 1 or lines < 0:
        raise errors.GitError("blame requires a positive start line and a nonnegative line count")
    if not lines:
        return ()
    end_line = start_line + lines - 1
    working = rev == git_history.WORKTREE
    if working and (
        git_history.worktree_entry(path, name) == "untracked"
        or not git_history.head_commit(path)[0]
    ):
        return tuple(BlameLine(n, "", "not committed", "") for n in range(start_line, end_line + 1))
    args = ("--contents", "-") if working else ()
    revs = () if working else (rev,)
    with tempfile.TemporaryDirectory(prefix="agent6-blame-") as tmp:
        if reason := _ignore_revs_copies(path, pathlib.Path(tmp), hide):
            raise errors.GitError(reason)
        gitdir = _git_command.run(path, "rev-parse", "--absolute-git-dir").stdout.strip()
        # Git 2.53 blame needs the verified sha before --end-of-options when a path follows --.
        out = _git_command.run(
            pathlib.Path(tmp),
            "blame",
            "--porcelain",
            "-L",
            f"{start_line},{end_line}",
            *args,
            *revs,
            "--end-of-options",
            "--",
            name,
            env_extra={"GIT_DIR": gitdir, "GIT_WORK_TREE": tmp},
            stdin_text=os.fsdecode(data) if working else None,
            repo=path,
        ).stdout
    metadata: dict[str, dict[str, str]] = {}
    rows: list[BlameLine] = []
    sha, number = "", 0
    for line in out.split("\n"):
        if line.startswith("\t"):
            meta = metadata[sha]
            committed = bool(sha.strip("0"))
            date = _author_date(meta["author-time"]) if committed else ""
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
