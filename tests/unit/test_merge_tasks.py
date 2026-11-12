# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`git.merge_method = "tasks"`: one commit per task range, named by the task."""

from __future__ import annotations

import json
import pathlib
import subprocess
from unittest import mock

import pytest

from agent6 import paths
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import run_chain, run_checkpoint
from agent6.sessions import journal
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import main
from agent6.vcs import git_ops, git_read


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: pathlib.Path) -> str:
    """Init a repository on main with one commit; return its sha."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return _git(tmp_path, "rev-parse", "HEAD")


def _checkpoint(repo: pathlib.Path, ref: str, base: str, name: str, task: str | None) -> str:
    """Write one file and commit the worktree on the chain, naming the task."""
    (repo / name).write_text(f"{name}\n", encoding="utf-8")
    sha = git_ops.chain_commit(
        repo,
        f"agent6 turn: {name}",
        ref=ref,
        fallback_parent=base,
        trailers={git_read.TASK_TRAILER: task} if task else None,
    )
    assert sha
    return sha


def _session(tmp_path: pathlib.Path, session_id: str, base: str) -> sessions_layout.SessionLayout:
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id=session_id
    )
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": session_id,
                "base_sha": base,
                "base_branch": "main",
                "run_branch": git_read.chain_ref_for(session_id),
                "user_task": "two tasks",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return layout


def _subtasks(layout: sessions_layout.SessionLayout, *titles: str) -> list[models.TaskNode]:
    curator = graph_curator.GraphCurator(layout)
    root = curator.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None,
            draft=models.TaskNodeDraft(title="two tasks", depends_on=(), created_by="planner"),
        )
    )
    return [
        curator.add_subtask(
            models.AddSubtaskIntent(
                parent_id=root.id,
                draft=models.TaskNodeDraft(
                    title=title,
                    acceptance=f"{title} is in place",
                    depends_on=(),
                    created_by="planner",
                ),
            )
        )
        for title in titles
    ]


def test_chain_task_ranges_group_consecutive_checkpoints_of_one_task(
    tmp_path: pathlib.Path,
) -> None:
    """A, A, none, B reads as three ranges, each ending at its last commit."""
    base = _repo(tmp_path)
    ref = "refs/agent6/x/head"
    _checkpoint(tmp_path, ref, base, "a1.txt", "A")
    a2 = _checkpoint(tmp_path, ref, base, "a2.txt", "A")
    none = _checkpoint(tmp_path, ref, base, "n.txt", None)
    b = _checkpoint(tmp_path, ref, base, "b.txt", "B")
    assert git_read.chain_task_ranges(tmp_path, base, b) == [("A", a2), ("", none), ("B", b)]
    assert git_read.chain_task_ranges(tmp_path, b, b) == []


def test_a_checkpoint_names_its_task_in_a_trailer_and_its_event(tmp_path: pathlib.Path) -> None:
    base = _repo(tmp_path)
    chain = run_chain.RunChain(tmp_path, ref="refs/agent6/x/head", fallback_parent=base)
    events: list[dict[str, object]] = []

    def emit(event: journal.Event, /) -> None:
        events.append({"type": event.TYPE, **event.model_dump(exclude_unset=True)})

    points = run_checkpoint.Checkpoints(
        chain=chain,
        style="agent6",
        enabled=True,
        provider=mock.MagicMock(),
        log=lambda _m: None,
        emit=emit,
    )
    (tmp_path / "e.txt").write_text("e\n", encoding="utf-8")
    sha = points.commit("agent6 turn 1: edit", turn=1, task_id="task-7")
    body = _git(tmp_path, "log", "-1", "--format=%B", sha)
    assert f"{git_read.TASK_TRAILER}: task-7" in body
    assert next(e for e in events if e["type"] == "checkpoint.committed")["task_id"] == "task-7"


def test_tasks_merge_lands_one_commit_per_task_range_under_its_title(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two subtasks worked in turn land as two commits named by them, in chain order."""
    monkeypatch.chdir(tmp_path)
    base = _repo(tmp_path)
    layout = _session(tmp_path, "run-TASK01", base)
    first, second = _subtasks(layout, "Add the parser", "Wire the parser into the CLI")
    ref = git_read.chain_ref_for("run-TASK01")
    _checkpoint(tmp_path, ref, base, "parser.py", first.id)
    _checkpoint(tmp_path, ref, base, "parser_test.py", first.id)
    tip = _checkpoint(tmp_path, ref, base, "cli.py", second.id)

    assert main(["sessions", "merge", "run-TASK01"]) == 0

    subjects = _git(tmp_path, "log", "--format=%s", "main").splitlines()
    assert subjects == ["Wire the parser into the CLI", "Add the parser", "init"]
    assert (
        _git(tmp_path, "log", "-1", "--format=%b", "main")
        .strip()
        .startswith("Wire the parser into the CLI is in place")
    )
    assert _git(tmp_path, "rev-parse", "main^{tree}") == _git(
        tmp_path, "rev-parse", f"{tip}^{{tree}}"
    )
    # The parser commit holds both parser files and not the CLI file.
    assert "cli.py" not in _git(tmp_path, "ls-tree", "--name-only", "main~1")
    assert "parser_test.py" in _git(tmp_path, "ls-tree", "--name-only", "main~1")
    merged = sessions_manifest.read_manifest(layout.session_dir).merged
    assert merged is not None and merged.method == "tasks"


def test_tasks_merge_under_the_model_style_takes_the_models_message_with_the_task_as_body(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured squash style names each range; the task's title and acceptance follow."""
    from agent6.app import merge_message, role_providers

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "model"\n', encoding="utf-8"
    )
    base = _repo(tmp_path)
    layout = _session(tmp_path, "run-TASK05", base)
    first, second = _subtasks(layout, "Add the parser", "Wire the parser into the CLI")
    ref = git_read.chain_ref_for("run-TASK05")
    _checkpoint(tmp_path, ref, base, "parser.py", first.id)
    _checkpoint(tmp_path, ref, base, "cli.py", second.id)
    asked: list[str] = []

    def draft(_provider: object, *, system: str, user: str, max_tokens: int) -> str:
        asked.append(user)
        if "Task: Add the parser" in user:
            return "feat(parser): add the parser\n\nOne pass over the grammar."
        return "feat(cli): wire the parser in\n\nThe CLI calls it."

    def provider(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(role_providers, "build_role_provider", provider)
    monkeypatch.setattr(merge_message, "call_for_text", draft)

    assert main(["sessions", "merge", "run-TASK05"]) == 0

    subjects = _git(tmp_path, "log", "--format=%s", "main").splitlines()
    assert subjects == ["feat(cli): wire the parser in", "feat(parser): add the parser", "init"]
    body = _git(tmp_path, "log", "-1", "--format=%b", "main")
    assert "The CLI calls it." in body
    assert "Wire the parser into the CLI\n\nWire the parser into the CLI is in place" in body
    # Each range's request carries its own diff, not the run's.
    assert len(asked) == 2 and "+parser.py" in asked[0] and "+parser.py" not in asked[1]


def test_tasks_merge_of_an_undecomposed_run_lands_one_commit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoints outside any subtask land as one squash with the squash style's message."""
    monkeypatch.chdir(tmp_path)
    base = _repo(tmp_path)
    _session(tmp_path, "run-TASK02", base)
    ref = git_read.chain_ref_for("run-TASK02")
    _checkpoint(tmp_path, ref, base, "a.txt", None)
    _checkpoint(tmp_path, ref, base, "b.txt", None)

    assert main(["sessions", "merge", "run-TASK02"]) == 0

    assert len(_git(tmp_path, "log", "--format=%s", "main").splitlines()) == 2
    assert (tmp_path / "b.txt").exists()


def test_tasks_merge_with_an_operator_message_lands_one_commit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _repo(tmp_path)
    layout = _session(tmp_path, "run-TASK03", base)
    first, second = _subtasks(layout, "One", "Two")
    ref = git_read.chain_ref_for("run-TASK03")
    _checkpoint(tmp_path, ref, base, "one.txt", first.id)
    _checkpoint(tmp_path, ref, base, "two.txt", second.id)

    assert main(["sessions", "merge", "run-TASK03", "-m", "both at once"]) == 0

    assert _git(tmp_path, "log", "--format=%s", "main").splitlines() == ["both at once", "init"]


def test_a_conflict_in_a_later_task_range_lands_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The squashes chain off the target; a conflict in range two leaves range one unlanded too."""
    monkeypatch.chdir(tmp_path)
    base = _repo(tmp_path)
    layout = _session(tmp_path, "run-TASK04", base)
    first, second = _subtasks(layout, "Add a file", "Rewrite the README")
    ref = git_read.chain_ref_for("run-TASK04")
    _checkpoint(tmp_path, ref, base, "added.txt", first.id)
    (tmp_path / "README.md").write_text("the run's README\n", encoding="utf-8")
    sha = git_ops.chain_commit(
        tmp_path, "agent6 turn: readme", ref=ref, fallback_parent=base,
        trailers={git_read.TASK_TRAILER: second.id},
    )  # fmt: skip
    assert sha
    # main moves under the run, on the file the second range rewrites.
    (tmp_path / "README.md").write_text("main's README\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "main edits the README")
    moved = _git(tmp_path, "rev-parse", "main")

    assert main(["sessions", "merge", "run-TASK04"]) == 1

    assert _git(tmp_path, "rev-parse", "main") == moved
    assert "added.txt" not in _git(tmp_path, "ls-tree", "--name-only", "main")
    assert sessions_manifest.read_manifest(layout.session_dir).merged is None
