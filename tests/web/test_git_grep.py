"""Pin the Files search command and its limits."""

from __future__ import annotations

import json
import pathlib
import signal
import subprocess
import sys
from typing import Any

import pytest

from agent6 import errors
from agent6.ui.web import git_model
from tests.unit import test_git_history
from tests.web import test_web_git

repo = test_web_git.repo
port = test_web_git.port


@pytest.mark.parametrize("revision", ["WORKTREE", "main"])
def test_git_grep_uses_one_literal_pattern_and_a_verified_revision(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    """Even a flag-shaped pattern stays one argument after -e on the hardened git argv."""
    checkout = tmp_path / "repo"
    checkout.mkdir()
    shas = test_git_history.build_history(checkout)
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
    pattern = "-- [a.*]"
    worktree = revision == "WORKTREE"
    name = "docs: draft.txt"
    (checkout / name).write_text(f"first\n{pattern}\n")
    payload = git_model.grep_payload(checkout, revision, pattern)
    assert payload == {
        "rev": revision if worktree else shas["merge"],
        "hits": [{"path": name, "line": 2, "text": pattern}] if worktree else [],
        "truncated": False,
    }
    events = map(json.loads, trace.read_text().splitlines())
    (argv,) = [row["argv"] for row in events if row["event"] == "start" and "grep" in row["argv"]]
    flags = argv[: argv.index("grep")]
    assert "core.fsmonitor=false" in flags and "core.hooksPath=/dev/null" in flags
    assert argv[argv.index("grep") :] == [
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
        *(["--untracked", "--exclude-standard"] if worktree else []),
        "-e",
        pattern,
        "--end-of-options",
        *([] if worktree else [shas["merge"]]),
        "--",
    ]


def test_the_deadline_kills_and_reaps_a_silent_grep(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grep that never writes its first line cannot keep its worker alive after the deadline."""
    test_git_history.build_history(tmp_path)
    start = subprocess.Popen
    workers: list[subprocess.Popen[bytes] | subprocess.Popen[str]] = []

    def stall(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes] | subprocess.Popen[str]:
        if "grep" not in argv:
            return start(argv, **kwargs)
        proc = start([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        workers.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", stall)
    with pytest.raises(errors.GitError, match="git grep timed out after 3s"):
        git_model.grep_payload(tmp_path, "WORKTREE", "abc")
    assert [proc.returncode for proc in workers] == [-signal.SIGKILL]


def test_grep_marks_only_incomplete_results_and_skips_ignored_and_binary_files(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """The limit is global, and exactly a full page is not a cut search."""
    checkout, _ = repo
    route = "/api/git/grep?rev=WORKTREE&q=match"
    (checkout / ".git/info/exclude").write_text("ignored\n")
    (checkout / "ignored").write_text("match\n")
    (checkout / "image").write_bytes(b"\x00match")
    (checkout / "notes").write_text("match\n" * 100)
    (checkout / "link").symlink_to("notes")
    hits = [{"path": "notes", "line": i, "text": "match"} for i in range(1, 101)]
    assert test_web_git._ok(port, route) == {"rev": "WORKTREE", "hits": hits, "truncated": False}
    (checkout / "other").write_text("match\n")
    assert test_web_git._ok(port, route) == {"rev": "WORKTREE", "hits": hits, "truncated": True}
    (checkout / "notes").write_text("match" + "x" * (512 * 1024))
    large = test_web_git._ok(port, route)
    assert large["truncated"] and len(json.dumps(large)) < 300_000


@pytest.mark.parametrize(
    "query", ["q=", "q=a%00b", "q=a%0Ab", "q=" + "x" * 4097], ids=["empty", "nul", "lf", "long"]
)
def test_invalid_patterns_have_a_search_refusal(port: int, query: str) -> None:
    """An empty, multiline or oversized pattern cannot start a match-all search."""
    assert "search text" in test_web_git._refused(port, "/api/git/grep?rev=HEAD&" + query)


def test_grep_refuses_a_flag_shaped_revision(port: int) -> None:
    assert "revision cannot start" in test_web_git._refused(port, "/api/git/grep?rev=-n&q=word")
