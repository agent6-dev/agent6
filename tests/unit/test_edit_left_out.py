# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An edit the run's commits leave out says so: at the edit, and in the end summary."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any
from unittest import mock

import pytest

from agent6 import budget, git_ops, memory
from agent6.app import finalize, reporter
from agent6.harness import _chain, _conversation, _loop_state, _snapshot
from agent6.sessions import layout as sessions_layout
from agent6.tools import results
from tests.unit import test_harness
from tests.unit.turn_context import turn_context


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a repo ignoring `data/`, with `data/units.json` tracked all the same."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("data/\n", encoding="utf-8")
    (repo / "data" / "units.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "add", "-f", "data/units.json")
    _git(repo, "commit", "-qm", "seed")
    return repo


class _Events:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, **fields: Any) -> None:
        self.events.append({"type": event_type, **fields})


def test_run_chain_left_out_names_ignored_and_start_untracked_paths(
    tmp_path: pathlib.Path,
) -> None:
    """A new ignored file and a start-untracked file are named; tracked and new files are not."""
    repo = _repo(tmp_path)
    for rel in ("data/rates.json", "notes.txt", "new.py"):
        (repo / rel).write_text("x\n", encoding="utf-8")
    chain = _chain.RunChain(
        repo, ref="refs/agent6/t/head", untracked_at_start=frozenset({"notes.txt"})
    )
    paths = ("data/rates.json", "data/units.json", "notes.txt", "new.py")
    assert chain.left_out(paths) == (("data/rates.json",), ("notes.txt",))
    # A run that never commits leaves nothing out of its commits.
    assert _chain.RunChain(repo).left_out(paths) == ((), ())


def _dispatch_edit(
    repo: pathlib.Path,
    result: results.ToolResult,
    *,
    untracked: frozenset[str] = frozenset(),
    tool_path: str = "x",
    state_dir: pathlib.Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Dispatch one edit call whose result is `result`; return its wire content and the events.

    Args:
        repo: The workspace.
        result: What the dispatcher returns for the call.
        untracked: The run's untracked-at-start set.
        tool_path: The path the model gave the edit tool.
        state_dir: The run's state directory, whose memory store a write may address.
    """
    dispatcher = mock.MagicMock()
    dispatcher.dispatch.return_value = result
    events = _Events()
    wf = test_harness._wf(  # pyright: ignore[reportPrivateUsage]
        repo, mode="run", dispatcher=dispatcher, events=events
    )
    wf.chain = _chain.RunChain(
        repo, ref="refs/agent6/t/head", fallback_parent="HEAD", untracked_at_start=untracked
    )
    wf.state_dir = state_dir
    turn = _loop_state.TurnState(
        number=3,
        resp=test_harness._resp(""),  # pyright: ignore[reportPrivateUsage]
        assistant=_conversation.AssistantMessage(
            raw_content=(),
            tool_uses=(
                _conversation.ToolUse(id="tu1", name="apply_edit", input={"path": tool_path}),
            ),
        ),
    )
    wf._turn_dispatch_tools(  # pyright: ignore[reportPrivateUsage]
        test_harness._state(),  # pyright: ignore[reportPrivateUsage]
        turn,
        turn_context(),
    )
    item = turn.tool_results[0]
    assert isinstance(item, _conversation.ToolResultItem)
    return json.loads(item.content), events.events


def test_an_edit_to_a_gitignored_path_says_so_in_its_tool_result(tmp_path: pathlib.Path) -> None:
    """An edit to an ignored path was applied with no word that the commits leave it out."""
    repo = _repo(tmp_path)
    (repo / "data" / "rates.json").write_text("{}\n", encoding="utf-8")
    wire, events = _dispatch_edit(
        repo, results.EditResult(applied=("create",), path="data/rates.json", created=True)
    )
    assert wire["note"] == (
        "data/rates.json is ignored by git: the edit is on disk, and this run's commits"
        " leave it out."
    )
    left_out = [e for e in events if e["type"] == "loop.edit.left_out"]
    assert left_out == [
        {
            "type": "loop.edit.left_out",
            "turn": 3,
            "ignored": ["data/rates.json"],
            "untracked": [],
        }
    ]


def test_an_edit_to_a_start_untracked_file_says_so_in_its_tool_result(
    tmp_path: pathlib.Path,
) -> None:
    """An edit to a file untracked at run start was applied with no word either."""
    repo = _repo(tmp_path)
    (repo / "notes.txt").write_text("x\n", encoding="utf-8")
    (repo / "new.py").write_text("x\n", encoding="utf-8")
    patch = results.PatchResult(
        path="notes.txt", bytes_written=4, files=(("notes.txt", 2), ("new.py", 2))
    )
    wire, events = _dispatch_edit(repo, patch, untracked=frozenset({"notes.txt"}))
    assert wire["note"] == (
        "notes.txt was untracked when the run started: the edit is on disk, and this run's"
        " commits leave it out."
    )
    assert [e["untracked"] for e in events if e["type"] == "loop.edit.left_out"] == [["notes.txt"]]


def test_a_memory_store_write_carries_no_note(tmp_path: pathlib.Path) -> None:
    """A memory write's result names its file inside the store and carries no note.

    Taken as a repo path, the result's `notes.md` would match the operator's untracked
    `notes.md` and say the run's commits leave it out. The store is outside the repository.
    """
    repo = _repo(tmp_path)
    (repo / "notes.md").write_text("the operator's\n", encoding="utf-8")
    state = tmp_path / "state"
    store = memory.memory_dir(state)
    store.mkdir(parents=True)
    wire, events = _dispatch_edit(
        repo,
        results.EditResult(applied=("create",), path="notes.md", created=True),
        untracked=frozenset({"notes.md"}),
        tool_path=str(store / "notes.md"),
        state_dir=state,
    )
    assert "note" not in wire
    assert not [e for e in events if e["type"] == "loop.edit.left_out"]


def test_an_edit_the_commits_hold_carries_no_note(tmp_path: pathlib.Path) -> None:
    """A tracked path and a new unignored path get the plain result and no event."""
    repo = _repo(tmp_path)
    (repo / "new.py").write_text("x\n", encoding="utf-8")
    for rel in ("new.py", "data/units.json"):
        wire, events = _dispatch_edit(repo, results.EditResult(applied=("replace",), path=rel))
        assert wire == {"applied": ["replace"], "path": rel}
        assert not [e for e in events if e["type"] == "loop.edit.left_out"]


def _end_footer(
    tmp_path: pathlib.Path,
    repo: pathlib.Path,
    session_id: str,
    left_out: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Print the end block of a finished run whose journal holds one left-out event."""
    session_dir = tmp_path / "sessions" / "runs" / session_id
    session_dir.mkdir(parents=True)
    events: list[dict[str, object]] = [
        {"type": "session.start", "session_id": session_id, "user_task": "t"},
        {"type": "loop.edit.left_out", "turn": 1, **left_out},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (session_dir / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id=session_id)
    manifest = {
        "mode": "run",
        "session_id": session_id,
        "run_branch": f"agent6/{session_id}",
        "base_branch": "main",
    }
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    finalize.print_session_end(
        _snapshot.SessionResult(
            completed=True, reason="finish_session", summary="done", turns=1, tool_calls=1
        ),
        layout=layout,
        cwd=repo,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    return capsys.readouterr().out


def _commit_run_branch(repo: pathlib.Path, session_id: str, *force: str) -> None:
    """Commit `work.py`, and the `force` paths past the ignore rules, on the run's branch."""
    (repo / "work.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "switch", "-q", "-c", f"agent6/{session_id}")
    _git(repo, "add", "work.py")
    for rel in force:
        _git(repo, "add", "-f", rel)
    _git(repo, "commit", "-qm", "agent6 turn 1")
    _git(repo, "update-ref", git_ops.chain_ref_for(session_id), "HEAD")
    _git(repo, "switch", "-q", "main")
    # The run's work stays in the checkout, as a run leaves it.
    _git(repo, "checkout", f"agent6/{session_id}", "--", ".")


def test_end_summary_lists_the_edits_the_branch_leaves_out(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footer said the changes are on the branch and named no edit the branch lacks."""
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "data" / "rates.json").write_text("{}\n", encoding="utf-8")
    (repo / "notes.txt").write_text("x\n", encoding="utf-8")
    _commit_run_branch(repo, "s1")
    out = _end_footer(
        tmp_path,
        repo,
        "s1",
        {"ignored": ["data/rates.json", "data/gone.json"], "untracked": ["notes.txt"]},
        capsys,
    )
    assert "changes are on agent6/s1" in out
    # `data/gone.json` is no longer on disk, so it is not listed.
    assert (
        "  not in the run's commits: data/rates.json (ignored by git),"
        " notes.txt (untracked before the run)\n"
    ) in out


def test_end_summary_drops_a_left_out_path_that_reached_the_branch(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path the run's later commits hold is not reported as left out."""
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "data" / "rates.json").write_text("{}\n", encoding="utf-8")
    _commit_run_branch(repo, "s2", "data/rates.json")
    out = _end_footer(tmp_path, repo, "s2", {"ignored": ["data/rates.json"]}, capsys)
    assert "changes are on agent6/s2" in out
    assert "not in the run's commits" not in out


def test_end_summary_with_only_left_out_edits_lists_them(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose only edit is ignored read as recording nothing, with no word of the edit."""
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "data" / "rates.json").write_text("{}\n", encoding="utf-8")
    out = _end_footer(tmp_path, repo, "s3", {"ignored": ["data/rates.json"]}, capsys)
    assert "changes are on" not in out
    assert "  not in the run's commits: data/rates.json (ignored by git)\n" in out
