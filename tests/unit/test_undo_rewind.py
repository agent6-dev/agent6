# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/undo` puts back to the checkpoint's tree every path the run changed.

A path changed after the run ended stays the operator's, and the later chain commits stay on
the undone session's ref.
"""

from __future__ import annotations

import json
import pathlib
import subprocess as sp

import pytest

from agent6 import errors, git_ops, paths
from agent6.app import reporter, undo
from agent6.harness import _snapshot
from agent6.sessions import layout as sessions_layout
from agent6.sessions import lock, process
from agent6.viewmodel import _session_log, _session_status


def _git(repo: pathlib.Path, *args: str) -> str:
    return sp.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _checkpoint(
    layout: sessions_layout.SessionLayout,
    turn: int,
    *,
    head_sha: str,
    ops: int,
    at: int | None = None,
) -> None:
    """A checkpoint at *turn* holding *ops* operator messages and *head_sha* as the workspace head.

    *at* is the file it sits in when that is not *turn* (a fork's seed in file 0).
    """
    messages: list[dict[str, object]] = [{"role": "user", "content": "do the thing"}]
    for i in range(1, ops):
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f"OPERATOR STEERING (x):\nsteer {i}"}],
            }
        )
    snap = _snapshot.SessionSnapshot(
        system="s",
        messages=messages,
        tool_calls=0,
        next_iteration=turn,
        root_task_id=None,
        original_task="do the thing",
        verify_command=(),
        head_sha=head_sha,
    )
    layout.checkpoint_path(turn if at is None else at).write_text(
        snap.model_dump_json(), encoding="utf-8"
    )


def _run_that_moved_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, str, str]:
    """A repo and a run whose turn-1 checkpoint sits at c1, with edits committed and in flight.

    The chain committed c2 (a.txt edited, gen.txt created), then a.txt and more.txt changed
    uncommitted; the operator's untracked notes.md was there from the start.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    (repo / "same.txt").write_text("same\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    (repo / "notes.md").write_text("mine\n", encoding="utf-8")
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-AAAA11")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "run-AAAA11",
                "mode": "run",
                "user_task": "do the thing",
                "base_sha": c1,
                "base_branch": "main",
                "run_branch": "agent6/run-AAAA11",
            }
        ),
        encoding="utf-8",
    )
    sessions_layout.write_untracked_at_start(layout.session_dir, {"notes.md"})
    _checkpoint(layout, 1, head_sha=c1, ops=1)
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    (repo / "gen.txt").write_text("generated\n", encoding="utf-8")
    c2 = git_ops.chain_commit(
        repo,
        "step",
        ref=git_ops.chain_ref_for("run-AAAA11"),
        fallback_parent=c1,
        also_branch="agent6/run-AAAA11",
        exclude={"notes.md"},
    )
    assert c2 is not None
    _checkpoint(layout, 2, head_sha=c2, ops=2)
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    (repo / "more.txt").write_text("in flight\n", encoding="utf-8")
    (layout.session_dir / "loop_state.json").write_text(
        layout.checkpoint_path(2).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return repo, c1, c2


def test_undo_of_a_model_controlled_run_refuses_before_committing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `git.control = "model"` run has no agent6 chain.

    The refusal sat in the fork step, after the undo commit, so a chain ref holding the tree was
    left behind: `auto_merge` landed it on the base, and the next run's dirty-tree question named a
    merge that refuses.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="model-AAAA11"
    )
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "model-AAAA11",
                "mode": "run",
                "user_task": "do the thing",
                "base_sha": c1,
                "base_branch": "main",
                "run_branch": None,
                "git_control": "model",
            }
        ),
        encoding="utf-8",
    )
    _checkpoint(layout, 1, head_sha=c1, ops=1)
    _checkpoint(layout, 2, head_sha=c1, ops=2)
    (repo / "a.txt").write_text("two\n", encoding="utf-8")  # the model's own uncommitted edit
    said: list[str] = []
    got = undo.undo_fork(
        (), "model-AAAA11", cwd=repo, reporter=reporter.Reporter(out=said.append, err=said.append)
    )
    assert got is None
    assert any("managed git itself" in s for s in said), said
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("model-AAAA11")) is None


def test_undo_puts_the_checkout_back_and_keeps_the_tree_as_it_stood(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tree as it stands goes onto the run's ref first, then every differing path is put back.

    The operator's untracked file, HEAD, the index and the later commit stay; rewinding straight
    from the worktree deleted the in-flight work with no commit holding it.
    """
    repo, c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    said: list[str] = []

    result = undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )

    assert result is not None
    child, undone_text = result
    assert undone_text == "steer 1"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert not (repo / "gen.txt").exists() and not (repo / "more.txt").exists()
    assert (repo / "same.txt").read_text(encoding="utf-8") == "same\n"
    assert (repo / "notes.md").read_text(encoding="utf-8") == "mine\n"
    assert _git(repo, "rev-parse", "HEAD") == c1  # HEAD and the index untouched
    assert _git(repo, "diff", "--cached", "--name-only") == ""
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("run-AAAA11"))
    assert tip is not None and tip != c2
    assert _git(repo, "rev-parse", f"{tip}^") == c2  # the pre-undo commit sits on c2
    assert _git(repo, "show", f"{tip}:a.txt") == "three"  # the in-flight edit, kept
    assert _git(repo, "show", f"{tip}:more.txt") == "in flight"  # the mid-run file, kept
    assert _git(repo, "rev-parse", "agent6/run-AAAA11") == tip
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for(child)) == c1
    text = "\n".join(said)
    assert "a.txt" in text and "gen.txt" in text and "more.txt" in text
    assert f"commit {tip[:12]} on agent6/run-AAAA11" in text
    assert "HEAD and the index are untouched" in text


def test_undo_of_a_finished_run_journals_its_undone_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source reads `undone` on every surface, as it does after a live run's own `/undo`.

    A finished run undone from the TUI or the web kept reading `passed · unmerged` with a
    merge line, and `sessions merge` landed the work the operator took back.
    """
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    events = [
        {"type": "session.start", "session_id": "run-AAAA11", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 2, "all_passed": True},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    result = undo.undo_fork((), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(print, print))

    assert result is not None
    child, _text = result
    tail = [json.loads(line) for line in layout.logs_path.read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in tail[-2:]] == ["session.undone", "session.end"]
    assert tail[-2]["new_session_id"] == child and tail[-2]["undone_text"] == "steer 1"
    assert tail[-1]["reason"] == "undone" and tail[-1]["iterations"] == 2
    scan = _session_log.scan_session_log(layout.logs_path)
    assert scan.undone_to == child
    word, _detail = _session_status.status_word(
        finished=scan.finished, all_passed=scan.all_passed, end_reason=scan.end_reason
    )
    assert word == "undone"


def test_undo_puts_back_only_the_paths_the_run_changed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished run's end records its tree; what differs from it since is the operator's.

    The rewind moved every path that differed from the checkpoint, so an operator's note left
    the checkout for the undo commit, in a file the run never touched or in one it changed.
    """
    repo, c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    # The run ended on the tree as it stands; the operator then edited two files.
    tree = git_ops.worktree_tree(repo, c2, {"notes.md"})
    events = [
        {"type": "session.start", "session_id": "run-AAAA11", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "tree": tree},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    (repo / "same.txt").write_text("same\noperator note\n", encoding="utf-8")
    (repo / "a.txt").write_text("three\noperator note\n", encoding="utf-8")
    said: list[str] = []

    assert undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )
    # The run's paths go back; an operator's edit since the end stays, on the run's paths too.
    assert (repo / "a.txt").read_text(encoding="utf-8") == "three\noperator note\n"
    assert not (repo / "gen.txt").exists() and not (repo / "more.txt").exists()
    assert (repo / "same.txt").read_text(encoding="utf-8") == "same\noperator note\n"
    assert _git(repo, "rev-parse", "HEAD") == c1
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("run-AAAA11"))
    assert tip is not None and _git(repo, "show", f"{tip}:a.txt") == "three\noperator note"
    text = "\n".join(said)
    assert "put 2 path(s) back" in text and "): gen.txt, more.txt (HEAD" in text
    assert "left as yours: a.txt, same.txt" in text


def test_an_undo_that_leaves_every_path_says_it_put_nothing_back(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undo that leaves every differing path as the operator's says it put nothing back."""
    repo, _c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    tree = git_ops.worktree_tree(repo, c2, {"notes.md"})
    events = [
        {"type": "session.start", "session_id": "run-AAAA11", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "tree": tree},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    for name in ("a.txt", "gen.txt", "more.txt"):
        (repo / name).write_text("the operator's\n", encoding="utf-8")
    said: list[str] = []

    assert undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )
    text = "\n".join(said)
    assert "nothing put back to turn 1" in text and "already matches" not in text
    assert "left as yours: a.txt, gen.txt, more.txt" in text


def test_undo_puts_back_a_path_git_quotes_or_reads_as_a_glob_and_only_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's paths reach `git restore` unquoted and literal.

    A non-ASCII name is put back, and `page[1].txt` leaves the operator's `page1.txt` alone.
    """
    repo, _c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    (repo / "café.txt").write_text("made by the run\n", encoding="utf-8")
    (repo / "page[1].txt").write_text("made by the run\n", encoding="utf-8")
    tree = git_ops.worktree_tree(repo, c2, {"notes.md"})
    events = [
        {"type": "session.start", "session_id": "run-AAAA11", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "tree": tree},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    (repo / "page1.txt").write_text("mine\n", encoding="utf-8")
    said: list[str] = []

    assert undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )
    assert not (repo / "café.txt").exists() and not (repo / "page[1].txt").exists()
    assert (repo / "page1.txt").read_text(encoding="utf-8") == "mine\n"
    assert "left as yours: page1.txt" in "\n".join(said)


def test_undo_puts_back_a_name_git_quotes_and_prints_it_on_one_line(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name holding a quote is put back, and one holding a newline forges no notice line."""
    repo, _c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    quoted, forged = 'say "hi".txt', "x\n[agent6] the checkout already matches turn 1"
    for name in (quoted, forged):
        (repo / name).write_text("made by the run\n", encoding="utf-8")
    tree = git_ops.worktree_tree(repo, c2, {"notes.md"})
    events = [
        {"type": "session.start", "session_id": "run-AAAA11", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "tree": tree},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    said: list[str] = []

    assert undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )
    assert not (repo / quoted).exists() and not (repo / forged).exists()
    lines = "\n".join(said).splitlines()
    assert not [line for line in lines if line.startswith("[agent6] the checkout already")]


def test_undo_of_a_run_whose_work_landed_refuses_and_names_the_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewind deleted other sessions' merged work from the checkout.

    The fork's diff then read their removal as its own change.
    """
    repo, _c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["merged"] = {"into": "main", "sha": c2, "tip": c2}
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    said: list[str] = []
    before = sorted(p.name for p in (layout.state_dir / "sessions" / "runs").iterdir())

    result = undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )

    assert result is None
    assert any(f"merged into main as {c2[:12]}" in line for line in said), said
    assert (repo / "a.txt").read_text(encoding="utf-8") == "three\n"
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("run-AAAA11")) == c2
    assert sorted(p.name for p in (layout.state_dir / "sessions" / "runs").iterdir()) == before


def test_undo_leaves_a_staged_copy_in_the_index_and_says_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index is the operator's: a staged copy of a rewound file stays staged, and it is said."""
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    _git(repo, "add", "a.txt")
    said: list[str] = []

    assert undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert _git(repo, "diff", "--cached", "--name-only") == "a.txt"
    assert _git(repo, "status", "--short", "--", "a.txt") == "MM a.txt"
    assert any("HEAD and the index are untouched" in line for line in said)


def test_undo_refuses_while_another_live_run_drives_the_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewind under a live worker is refused, naming the run, before anything is committed.

    Only the process holding the checkout is exempt.
    """
    repo, _c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    state = paths.state_dir(repo)
    for holder in ("other-LIVE11", "run-AAAA11"):
        said: list[str] = []
        fd = lock.acquire_repo_writer(state, repo, holder)
        try:
            result = undo.undo_fork(
                (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
            )
        finally:
            lock.release_single_writer(fd)
        assert result is None
        assert any(holder in line for line in said)
        assert (repo / "a.txt").read_text(encoding="utf-8") == "three\n"
        assert git_ops.chain_tip(repo, git_ops.chain_ref_for("run-AAAA11")) == c2
        assert not (state / "lineage.jsonl").exists()


def test_undo_of_a_live_plan_session_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live plan or ask session refuses /undo too, though it takes no checkout writer lock."""
    import subprocess as sp2

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(
        state_dir=state, session_id="plan-AAAA11", subdir="plans"
    )
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "plan-AAAA11",
                "mode": "plan",
                "user_task": "do the thing",
                "base_sha": c1,
                "base_branch": "main",
                "run_branch": None,
            }
        ),
        encoding="utf-8",
    )
    _checkpoint(layout, 1, head_sha=c1, ops=1)
    _checkpoint(layout, 2, head_sha=c1, ops=2)
    proc = sp2.Popen(["sleep", "30"])
    try:
        process.write_worker_pid(layout.session_dir, proc.pid)
        said: list[str] = []

        result = undo.undo_fork(
            (), "plan-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
        )

        assert result is None
        assert any("plan-AAAA11" in line for line in said)
        assert git_ops.chain_tip(repo, git_ops.chain_ref_for("plan-AAAA11")) is None
    finally:
        proc.terminate()
        proc.wait()


def test_undo_from_the_live_session_itself_is_allowed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop's own /undo runs inside the worker holding the checkout; its lock is no obstacle."""
    import os

    repo, c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    state = paths.state_dir(repo)
    process.write_worker_pid(
        sessions_layout.SessionLayout(state_dir=state, session_id="run-AAAA11").session_dir,
        os.getpid(),
    )
    fd = lock.acquire_repo_writer(state, repo, "run-AAAA11")
    try:
        result = undo.undo_fork(
            (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(print, print)
        )
    finally:
        lock.release_single_writer(fd)
    assert result is not None
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert _git(repo, "rev-parse", "HEAD") == c1


def test_undo_of_a_fork_whose_worktree_is_gone_refuses_before_creating_anything(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded worktree that is gone is named, with the recovery, before any child."""
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-AAAA11")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["worktree"] = str(tmp_path / "gone-worktree")
    manifest["worktree_git_dir"] = str((repo / ".git").resolve())
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    said: list[str] = []
    before = sorted(p.name for p in (state / "sessions" / "runs").iterdir())

    result = undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )

    assert result is None
    assert any("gone-worktree" in line and "agent6 fork run-AAAA11" in line for line in said)
    assert any("its commits are on agent6/run-AAAA11" in line for line in said)
    assert sorted(p.name for p in (state / "sessions" / "runs").iterdir()) == before
    assert not (state / "lineage.jsonl").exists()

    # Pruned after a merge (branch gone): the refusal points at the merge that
    # landed the commits, as `sessions commits` does, never at the gone branch.
    manifest["merged"] = {"into": "main", "sha": _c2, "tip": _c2}
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _git(repo, "branch", "-D", "agent6/run-AAAA11")
    said.clear()
    assert (
        undo.undo_fork(
            (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
        )
        is None
    )
    assert any(f"merged into main as {_c2[:12]}" in line for line in said)
    assert not any("agent6/run-AAAA11" in line or "refs/agent6" in line for line in said)


def test_undo_names_the_turn_every_other_surface_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An /undo resolved at a fork's seed checkpoint names the source's turn, not "turn 0"."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    state = paths.state_dir(repo)
    fork = sessions_layout.SessionLayout(state_dir=state, session_id="fork-BBBB22")
    fork.ensure()
    fork.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "fork-BBBB22",
                "mode": "run",
                "user_task": "do the thing",
                "base_sha": c1,
                "base_branch": "main",
                "run_branch": "agent6/fork-BBBB22",
                "parent_session_id": "src-AAAA11",
                "forked_from_turn": 3,
                "forked_from_sha": c1,
            }
        ),
        encoding="utf-8",
    )
    _checkpoint(fork, 3, head_sha=c1, ops=2, at=0)  # the seed: the source's turn 3
    _checkpoint(fork, 4, head_sha=c1, ops=3)  # the fork's own turn after a steer
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    said: list[str] = []

    result = undo.undo_fork(
        (), "fork-BBBB22", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )

    assert result is not None
    child, _text = result
    text = "\n".join(said)
    assert f"back to turn 3 ({c1[:12]})" in text and "turn 0" not in text
    child_manifest = json.loads(
        sessions_layout.SessionLayout(state_dir=state, session_id=child).manifest_path.read_text(
            encoding="utf-8"
        )
    )
    assert child_manifest["forked_from_turn"] == 3
    subject = _git(repo, "log", "-1", "--format=%s", git_ops.chain_ref_for("fork-BBBB22"))
    assert subject == "agent6 undo: the tree before turn 3 was taken back"


def test_an_undo_resolved_in_an_ancestor_keeps_the_undone_sessions_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork's /undo resolves in its parent, and the child keeps the undone fork's checkout.

    Recording the parent's checkout handed the child's model a writable checkout it was never given.
    """
    from agent6.app import stamps
    from agent6.config import Config

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "c2")
    c2 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    state = paths.state_dir(repo)
    parent = sessions_layout.SessionLayout(state_dir=state, session_id="parent-AAAA11")
    parent.ensure()
    stamps.write_session_manifest(
        parent,
        session_id="parent-AAAA11",
        user_task="do the thing",
        base_sha=c1,
        base_branch="main",
        run_branch="agent6/parent-AAAA11",
        cfg=Config(),
        mode="run",
    )
    _checkpoint(parent, 1, head_sha=c1, ops=1)
    _checkpoint(parent, 2, head_sha=c2, ops=2)
    worktree = tmp_path / "fork-wt"
    git_ops.add_worktree(repo, worktree, c2)
    fork = sessions_layout.SessionLayout(state_dir=state, session_id="fork-BBBB22")
    fork.ensure()
    stamps.write_session_manifest(
        fork,
        session_id="fork-BBBB22",
        user_task="do the thing",
        base_sha=c1,
        base_branch="main",
        run_branch="agent6/fork-BBBB22",
        cfg=Config(),
        mode="run",
        parent_session_id="parent-AAAA11",
        forked_from_turn=2,
        forked_from_sha=c2,
        worktree=worktree,
        worktree_git_dir=(repo / ".git").resolve(),
        workspace_set_up=True,
    )
    _checkpoint(fork, 0, head_sha=c2, ops=2)  # the seed: the parent's turn-2 conversation

    result = undo.undo_fork((), "fork-BBBB22", cwd=repo, reporter=reporter.Reporter(print, print))

    assert result is not None
    child, undone_text = result
    assert undone_text == "steer 1"
    child_manifest = json.loads(
        sessions_layout.SessionLayout(state_dir=state, session_id=child).manifest_path.read_text(
            encoding="utf-8"
        )
    )
    assert child_manifest["parent_session_id"] == "parent-AAAA11"
    assert child_manifest["worktree"] == str(worktree)
    # The child stays in the fork's worktree, so a resume must not run the setup command again.
    assert child_manifest["workspace_set_up"] is True
    assert (worktree / "a.txt").read_text(encoding="utf-8") == "one\n"  # the fork's, rewound
    assert (repo / "a.txt").read_text(encoding="utf-8") == "two\n"  # the operator's, untouched


def test_an_undo_fork_keeps_the_source_run_untracked_set(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An /undo fork continuing in the source's checkout keeps the run's untracked files."""
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    said: list[str] = []

    result = undo.undo_fork(
        (), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(said.append, said.append)
    )

    assert result is not None
    child, _text = result
    child_dir = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id=child
    ).session_dir
    assert sessions_layout.read_untracked_at_start(child_dir) == frozenset({"notes.md"}), (
        "the child inherits the operator's set, not the run's own output"
    )


def test_a_failed_rewind_still_journals_the_undone_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fork exists once the rewind fails, so the source must read undone all the same."""
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-AAAA11")
    events = [
        {"type": "session.start", "session_id": "run-AAAA11", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 2, "all_passed": True},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    def _fails(*_a: object, **_k: object) -> undo._Rewound:  # pyright: ignore[reportPrivateUsage]
        raise errors.GitError("restore refused")

    monkeypatch.setattr(undo, "_rewind_checkout", _fails)

    result = undo.undo_fork((), "run-AAAA11", cwd=repo, reporter=reporter.Reporter(print, print))

    assert result is not None
    child, _text = result
    assert "was not put back" in capsys.readouterr().out
    tail = [json.loads(line) for line in layout.logs_path.read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in tail[-2:]] == ["session.undone", "session.end"]
    assert tail[-2]["new_session_id"] == child and tail[-1]["reason"] == "undone"
