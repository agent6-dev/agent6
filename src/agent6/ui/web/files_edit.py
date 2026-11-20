"""Read and save operator-authored UTF-8 text in the working tree."""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import stat
import threading
from collections.abc import Generator
from typing import Any

import pydantic

from agent6 import path_safety
from agent6.ui.web import git_model
from agent6.vcs.git import read as git_read

MAX_BODY_BYTES = 6 * git_model.BLOB_MAX_BYTES + 8192
_WRITE_LOCK = threading.Lock()


class FileTextBody(pydantic.BaseModel):
    """A working-tree save, conditional on the bytes the operator loaded.

    Attributes:
        path: The repository-relative file path.
        text: Complete UTF-8 text to save, at most 20 MiB after encoding.
        digest: SHA-256 of the loaded bytes; null creates a file only if absent.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    path: str
    text: str
    digest: str | None = pydantic.Field(pattern=r"^[0-9a-f]{64}$")


class FileEditError(Exception):
    """A file edit refused with an HTTP status and a reason."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def _parts(path: str) -> list[str]:
    parts = path.split("/")
    if len(path) > 4096 or any(
        not part
        or part.startswith(".")
        or "\\" in part
        or any(ord(c) < 32 or ord(c) == 127 for c in part)
        for part in parts
    ):
        raise FileEditError("use a repository path with no hidden components or '..'")
    return parts


@contextlib.contextmanager
def _parent(repo: pathlib.Path, path: str) -> Generator[tuple[int, str]]:
    if not git_read.is_git_repo(repo):
        raise FileEditError("file edits require a Git working tree")
    parts = _parts(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        with contextlib.ExitStack() as stack:
            parent = os.open(repo, flags)
            stack.callback(os.close, parent)
            for part in parts[:-1]:
                parent = os.open(part, flags, dir_fd=parent)
                stack.callback(os.close, parent)
            yield parent, parts[-1]
    except OSError as exc:
        raise FileEditError(
            f"cannot open {path!r} without following links: {exc.strerror}"
        ) from exc


def _read(fd: int) -> bytes:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise FileEditError("only regular files with one hard link can be edited")
    if info.st_size > git_model.BLOB_MAX_BYTES:
        raise FileEditError("the file exceeds the 20 MiB limit", 413)
    with os.fdopen(os.dup(fd), "rb") as stream:
        data = stream.read(git_model.BLOB_MAX_BYTES + 1)
    _text(data)
    return data


def _text(data: bytes) -> str:
    if len(data) > git_model.BLOB_MAX_BYTES:
        raise FileEditError("the file exceeds the 20 MiB limit", 413)
    if b"\x00" in data:
        raise FileEditError("a binary file cannot be edited as text")
    try:
        return data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise FileEditError("the file is not UTF-8 text") from exc


def _snapshot(path: str, data: bytes | None) -> dict[str, Any]:
    """Return the complete edit snapshot.

    Attributes:
        path: The repository-relative file path.
        text: The complete text with CRLF normalized to LF, or null if absent.
        digest: SHA-256 of the exact bytes on disk, or null if absent.

    Args:
        path: The repository-relative file path.
        data: The file's bytes, or None when the file is absent.

    Returns:
        The edit snapshot.
    """
    return {
        "path": path,
        "text": _text(data) if data is not None else None,
        "digest": hashlib.sha256(data).hexdigest() if data is not None else None,
    }


def read(repo: pathlib.Path, path: str) -> dict[str, Any]:
    """Return a complete working-tree snapshot through no-follow descriptor-relative opens."""
    with _parent(repo, path) as (parent, name):
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        except FileNotFoundError:
            return _snapshot(path, None)
        try:
            return _snapshot(path, _read(fd))
        finally:
            os.close(fd)


def save(repo: pathlib.Path, body: FileTextBody) -> dict[str, Any]:
    """Save text only when the opened file matches the loaded digest.

    Web saves serialize; other working-tree writers do not share this lock.

    Args:
        repo: The working tree served by the web UI.
        body: The path, text and expected digest.

    Returns:
        The saved text snapshot.

    Raises:
        FileEditError: The path, content or digest is refused.
    """
    try:
        encoded = body.text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FileEditError("the new text is not valid UTF-8") from exc
    _text(encoded)
    with _WRITE_LOCK, _parent(repo, body.path) as (parent, name):
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        if body.digest is None:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(name, flags, 0o644, dir_fd=parent)
        except (FileNotFoundError, FileExistsError) as exc:
            raise FileEditError(
                "the file changed on disk; load it again before saving", 409
            ) from exc
        with os.fdopen(fd, "r+b") as stream:
            before = _read(fd)
            if body.digest is not None and hashlib.sha256(before).hexdigest() != body.digest:
                raise FileEditError("the file changed on disk; load it again before saving", 409)
            # Text areas supply LF. A markdown snapshot supplies its original line endings.
            data = encoded if b"\r\n" in encoded else path_safety.disk_bytes(body.text, like=before)
            _text(data)
            stream.seek(0)
            stream.write(data)
            stream.truncate()
            stream.flush()
            os.fsync(fd)
        return _snapshot(body.path, data)
