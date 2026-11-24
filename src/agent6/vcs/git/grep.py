"""Search one revision with fixed strings and bounded git output."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import selectors
import subprocess
import time

from agent6 import errors
from agent6.vcs.git import command as git_command
from agent6.vcs.git import history as git_history

MAX_HITS = 100
MAX_BYTES = 256 * 1024
TIMEOUT_S = 3
# With -z -n, only the end of the matching text uses a newline. Paths can contain one.
_ROW = re.compile(rb"([^\0]+)\0([0-9]+)\0([^\n]*)\n")


@dataclasses.dataclass(frozen=True, slots=True)
class Hit:
    """One matching source line.

    Attributes:
        path: The path relative to the checkout directory.
        line: The one-based source line number.
        text: The matching line without its line ending.
    """

    path: str
    line: int
    text: str


def _collect(proc: subprocess.Popen[bytes], prefix: bytes) -> tuple[list[Hit], bool]:
    """Read both pipes under one deadline.

    Args:
        proc: The running grep process.
        prefix: Git's commit prefix on each path, empty for the working tree.

    Returns:
        Complete matching lines and whether the search was cut.

    Raises:
        GitError: Git reported an error or returned an incomplete row.
        TimeoutExpired: The reader reached its deadline.
    """
    assert proc.stdout is not None and proc.stderr is not None
    hits: list[Hit] = []
    out, err = bytearray(), bytearray()
    pos = 0
    deadline = time.monotonic() + TIMEOUT_S
    with selectors.DefaultSelector() as pipes:
        for pipe in (proc.stdout, proc.stderr):
            pipes.register(pipe, selectors.EVENT_READ)
        while pipes.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, TIMEOUT_S)
            for key, _ in pipes.select(remaining):
                buf = out if key.fileobj is proc.stdout else err
                chunk = os.read(key.fd, min(65536, MAX_BYTES + 1 - len(buf)))
                if not chunk:
                    pipes.unregister(key.fd)
                    continue
                buf.extend(chunk)
                if buf is err:
                    if len(err) > MAX_BYTES:
                        raise errors.GitError("git grep exceeded its diagnostic byte limit")
                    continue
                while row := _ROW.match(out, pos, MAX_BYTES):
                    if len(hits) == MAX_HITS:
                        return hits, True
                    path = os.fsdecode(row[1].removeprefix(prefix))
                    text = row[3].removesuffix(b"\r").decode(errors="replace")
                    hits.append(Hit(path, int(row[2]), text))
                    pos = row.end()
                if len(out) > MAX_BYTES:
                    return hits, True
    code = proc.wait(timeout=max(0, deadline - time.monotonic()))
    if code not in (0, 1) or err:
        raise errors.GitError(f"git grep: {err.decode(errors='replace').strip() or f'exit {code}'}")
    if pos != len(out):
        raise errors.GitError("git grep returned an incomplete match")
    return hits, False


def search(repo: pathlib.Path, rev: str, pattern: str) -> tuple[list[Hit], bool]:
    """Search nonbinary files without running a text converter.

    Args:
        repo: The checkout directory.
        rev: A verified commit SHA or WORKTREE.
        pattern: The operator's single line of literal text.

    Returns:
        Complete hits and whether the row or byte limit cut the search.

    Raises:
        GitError: Git failed or the search timed out.
    """
    worktree = rev == git_history.WORKTREE
    args = (
        "grep",
        "-F",
        "-n",
        "-H",
        "-z",
        "-I",
        "--no-color",
        "--no-column",
        "--no-full-name",
        "--no-textconv",
        "--threads=1",
        *(("--untracked", "--exclude-standard") if worktree else ()),
        "-e",
        pattern,
        "--end-of-options",
        *(() if worktree else (rev,)),
        "--",
    )
    argv, env = git_command.invocation(repo, args)
    with subprocess.Popen(
        argv,
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        try:
            return _collect(proc, b"" if worktree else f"{rev}:".encode())
        except subprocess.TimeoutExpired as exc:
            raise errors.GitError(f"git grep timed out after {TIMEOUT_S}s") from exc
        finally:
            if proc.poll() is None:
                proc.kill()
