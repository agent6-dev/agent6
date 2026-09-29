# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One live run-mode worker per checkout: the checkout lock + park-and-resume flow.

Two concurrent run-mode workers share one working tree; each auto-commit is a
`git add -A` on whatever HEAD points at, so whichever run's branch was checked
out last received BOTH runs' commits. The repo-scoped flock refuses the second
worker up front, and the refused submission is PARKED (the manifest saves the
verbatim task) so the typed prompt is never dropped.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import Any
from unittest import mock

import pytest

from agent6 import git_ops, paths
from agent6.app import _execution as app__execution
from agent6.app import _session, _setup, stamps
from agent6.app import run as app_run
from agent6.config import Config
from agent6.models import validate
from agent6.sessions import layout as sessions_layout
from agent6.sessions import lock, manifest


def test_repo_writer_second_acquire_refused_and_holder_named(tmp_path: pathlib.Path) -> None:
    fd = lock.acquire_repo_writer(tmp_path, tmp_path, "run-A")
    assert fd is not None
    try:
        assert lock.acquire_repo_writer(tmp_path, tmp_path, "run-B") is None
        assert lock.repo_writer_holder(tmp_path, tmp_path) == "run-A"
        assert lock.repo_writer_held(tmp_path, tmp_path) is True
    finally:
        lock.release_single_writer(fd)
    # Released: the checkout is free again and the probe agrees.
    assert lock.repo_writer_held(tmp_path, tmp_path) is False
    fd2 = lock.acquire_repo_writer(tmp_path, tmp_path, "run-B")
    assert fd2 is not None
    lock.release_single_writer(fd2)


def test_one_probe_does_not_read_another_probe_as_a_live_run(tmp_path: pathlib.Path) -> None:
    """One probe does not read another probe as a live run.

    A probe that takes the exclusive lock to ask whether anyone holds it makes two at once (a web
    hub and a TUI, or two hub tabs) each report the other as a run driving the checkout, refusing a
    submission nothing was blocking.
    """
    import os
    import threading

    from agent6 import portable

    lock_path = lock.checkout_lock_path(tmp_path, tmp_path)
    created = lock.acquire_repo_writer(tmp_path, tmp_path, "run-A")  # creates the file
    assert created is not None
    lock.release_single_writer(created)  # the checkout is free: only a probe holds anything
    probing = threading.Event()
    done = threading.Event()

    def _hold_probe() -> None:
        fd = os.open(lock_path, os.O_RDWR)
        portable.lock_shared_nonblocking(fd)
        probing.set()
        done.wait(5.0)
        lock.release_single_writer(fd)

    thread = threading.Thread(target=_hold_probe)
    thread.start()
    try:
        assert probing.wait(5.0)
        assert lock.repo_writer_held(tmp_path, tmp_path) is False, "a probe read as a live writer"
    finally:
        done.set()
        thread.join(5.0)


def test_repo_writer_probe_does_not_hold(tmp_path: pathlib.Path) -> None:
    # The advisory probe must not itself keep the lock (it acquires + releases).
    assert lock.repo_writer_held(tmp_path, tmp_path) is False  # no lock file yet
    fd = lock.acquire_repo_writer(tmp_path, tmp_path, "run-A")
    assert fd is not None
    lock.release_single_writer(fd)
    assert lock.repo_writer_held(tmp_path, tmp_path) is False
    fd2 = lock.acquire_repo_writer(tmp_path, tmp_path, "run-C")
    assert fd2 is not None
    lock.release_single_writer(fd2)


def _init_repo(path: pathlib.Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A tmp git repo with isolated state + a minimal runnable global config."""
    gdir = tmp_path / "cfg"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # No provider key here: the lifecycle's route preflight passes.

    monkeypatch.setattr(_setup, "check_provider_keys", _no_missing_keys)
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    return repo


def _no_missing_keys(*_a: object, **_k: object) -> None:
    """check_provider_keys stand-in for a unit test with no real provider key."""
    return None


def _load_cfg() -> Config:
    from agent6.config import layer

    return layer.load_effective(pathlib.Path.cwd(), None).config


def test_second_run_parks_with_the_verbatim_task(repo: pathlib.Path) -> None:
    """A second run parks with the verbatim task.

    While a live worker holds the checkout, a second `run` submission is refused, but the exact
    typed prompt is saved as a parked, resumable run with no tree mutation (no stash, no branch
    cut).
    """
    from agent6.app import run

    state = paths.state_dir(repo)
    long_task = "fix the thing " + "x" * 5000  # > the 4000-char display cap
    holder_fd = lock.acquire_repo_writer(state, repo, "run-LIVE")
    try:
        rc = run.run_task(
            _load_cfg(),
            long_task,
            started_at=time.time(),
            frontend=mock.MagicMock(),
            session_id="run-PARKED",
            mode="run",
        )
    finally:
        lock.release_single_writer(holder_fd)
    assert rc == 2
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-PARKED")
    m = manifest.read_manifest(layout.session_dir)
    assert m.parked_task == long_task  # verbatim, not the truncated display twin
    assert m.run_branch is None
    # No branch was cut and the tree is untouched.
    branches = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "agent6/run-PARKED"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""
    # The parked dir survives (it is a saved run, not a discardable husk) and
    # the listing tells the truth about it.
    from agent6.viewmodel import listing

    row = listing.summarize_session_dir(layout.session_dir)
    assert (row.status, row.reason) == ("parked", "checkout busy")


def test_resume_starts_a_parked_run_with_the_saved_task(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 resume <parked-id>` starts a fresh run_task with the saved task under the same id.

    It releases its own locks first, so the fresh start can take them.
    """
    from agent6.app import resume as resume_mod
    from agent6.app import stamps

    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-PARKED2")
    layout.ensure()
    stamps.write_session_manifest(
        layout,
        session_id="run-PARKED2",
        user_task="do the saved thing",
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=_load_cfg(),
        mode="run",
    )
    stamps.stamp_parked(layout.session_dir, task="do the saved thing", reason="checkout busy")
    called: dict[str, Any] = {}

    def fake_run_task(cfg: Config, task: str, **kw: Any) -> int:
        called["task"] = task
        called["session_id"] = kw.get("session_id")
        called["mode"] = kw.get("mode")
        return 0

    monkeypatch.setattr(app_run, "run_task", fake_run_task)
    monkeypatch.setattr(_setup, "check_provider_keys", _no_missing_keys)
    rc = resume_mod.resume_task(
        None, "run-PARKED2", started_at=time.time(), frontend=mock.MagicMock(), force=False
    )
    assert rc == 0
    assert called == {"task": "do the saved thing", "session_id": "run-PARKED2", "mode": "run"}
    # The delegation released the run-dir lock before handing off, so a real
    # run_task can re-acquire it: prove the lock is free.

    fd = lock.acquire_single_writer(layout.session_dir)
    assert fd is not None
    lock.release_single_writer(fd)


def test_resume_refuses_while_another_run_drives_the_checkout(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resuming run B while run A's worker is live in the same checkout refuses.

    A resumed worker drives the tree exactly like a fresh one.
    """
    from agent6.app import resume as resume_mod

    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-B")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "run-B",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-B",
                "user_task": "t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    holder_fd = lock.acquire_repo_writer(state, repo, "run-A")
    try:
        rc = resume_mod.resume_task(
            None, "run-B", started_at=time.time(), frontend=mock.MagicMock(), force=False
        )
    finally:
        lock.release_single_writer(holder_fd)
    assert rc == 2
    err = capsys.readouterr().err
    assert "run-A" in err and "checkout" in err


def test_hub_new_work_preflight_refuses_while_checkout_busy(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hub refuses a New Work `run` submission up front while the checkout is busy.

    The refusal names the live run instead of spawning a detached run that parks and times out the
    locate; plan submissions are read-only and spawn freely.
    """
    from agent6.ui import spawn

    def must_not_spawn(*a: object, **k: object) -> tuple[pathlib.Path | None, str]:
        raise AssertionError("must not spawn")

    monkeypatch.setattr(spawn, "spawn_and_locate", must_not_spawn)
    state = paths.state_dir(repo)
    holder_fd = lock.acquire_repo_writer(state, repo, "run-LIVE")
    try:
        session_dir, err = spawn.spawn_new_work(repo, "run", "another task")
    finally:
        lock.release_single_writer(holder_fd)
    assert session_dir is None
    assert "run-LIVE" in err and "checkout" in err


def test_hub_new_work_fans_out_while_checkout_busy(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hub's New Work fans out while the checkout is busy.

    A fan-out takes no writer lock (its lanes clone the checkout), and the plain-run refusal tells
    the operator to `/parallel` instead, so the same check must not refuse the `/parallel` message
    before parsing it.
    """
    from agent6.ui import spawn

    spawned: list[str] = []

    def fake_spawn(
        cwd: pathlib.Path,
        mode: str,
        task: str,
        *,
        preset: str,
        model: str,
        spec: str,
        config_path: object = None,
    ) -> tuple[pathlib.Path | None, str]:
        spawned.append(f"{spec}:{task}")
        return repo / "run-P", ""

    monkeypatch.setattr(spawn, "_spawn_run", fake_spawn)

    def no_refusal(
        cwd: pathlib.Path,
        segments: object,
        config_path: object = None,
        *,
        preset: str = "",
        model: str = "",
    ) -> None:
        return None

    monkeypatch.setattr(validate, "directive_model_refusal", no_refusal)
    state = paths.state_dir(repo)
    holder_fd = lock.acquire_repo_writer(state, repo, "run-LIVE")
    try:
        session_dir, err = spawn.spawn_new_work(repo, "run", "/parallel 2 another task")
    finally:
        lock.release_single_writer(holder_fd)
    assert (session_dir, err) == (repo / "run-P", "")
    assert spawned == ["2:another task"]


def test_runs_show_reports_a_parked_run_as_parked(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` reports a parked run as parked.

    The refusal hands the operator a run id to resume, so the show page leads with the listing's
    word; "unknown (no events yet)" reads as a broken husk and hides the one action that starts it.
    """
    from agent6.app import run
    from agent6.ui.cli import sessions_show  # pyright: ignore[reportPrivateUsage]

    state = paths.state_dir(repo)
    holder_fd = lock.acquire_repo_writer(state, repo, "run-LIVE")
    try:
        rc = run.run_task(
            _load_cfg(),
            "add a retry to the fetch helper",
            started_at=time.time(),
            frontend=mock.MagicMock(),
            session_id="run-PARKED",
            mode="run",
        )
    finally:
        lock.release_single_writer(holder_fd)
    assert rc == 2
    capsys.readouterr()  # drop the refusal message

    assert sessions_show._cmd_status("run-PARKED") == 0
    out = capsys.readouterr().out
    assert "parked" in out
    assert "unknown" not in out


def test_parked_manifest_records_the_config_profile_not_the_sandbox_one(repo: pathlib.Path) -> None:
    """The parked manifest records the config preset, not the sandbox one.

    `harness.preset` is what resume feeds back to load_effective; the sandbox preset stamped there
    ('strict', 'hardened', 'none') makes `agent6 resume <parked-id>` die with "CONFIG ERROR: unknown
    preset 'strict'" on every sandboxed host.
    """
    from agent6.app import run
    from agent6.config import layer

    state = paths.state_dir(repo)
    holder_fd = lock.acquire_repo_writer(state, repo, "run-LIVE")
    try:
        rc = run.run_task(
            _load_cfg(),
            "do the thing",
            started_at=time.time(),
            frontend=mock.MagicMock(),
            session_id="run-PROF",
            mode="run",
        )
    finally:
        lock.release_single_writer(holder_fd)
    assert rc == 2
    m = manifest.read_manifest(
        sessions_layout.SessionLayout(state_dir=state, session_id="run-PROF").session_dir
    )
    assert m.harness.preset == _load_cfg().preset  # the CONFIG preset ("")
    # The exact call resume makes with it must not blow up on a sandbox word.
    layer.load_effective(repo, None, preset=m.harness.preset)


def test_parked_resume_passes_the_steer_through_to_run_task(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume --steer` on a PARKED run.

    The bridge files resume seeds are wiped by run_task's own stale-state clear, so the follow-up
    must ride the delegation (initial_steer) instead of dying on the floor.
    """
    from agent6.app import resume as resume_mod
    from agent6.app import stamps

    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-PSTEER")
    layout.ensure()
    stamps.write_session_manifest(
        layout,
        session_id="run-PSTEER",
        user_task="do the saved thing",
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=_load_cfg(),
        mode="run",
    )
    stamps.stamp_parked(layout.session_dir, task="do the saved thing", reason="checkout busy")
    called: dict[str, Any] = {}

    def fake_run_task(cfg: Config, task: str, **kw: Any) -> int:
        called["initial_steer"] = kw.get("initial_steer")
        return 0

    monkeypatch.setattr(app_run, "run_task", fake_run_task)
    monkeypatch.setattr(_setup, "check_provider_keys", _no_missing_keys)
    rc = resume_mod.resume_task(
        None,
        "run-PSTEER",
        started_at=time.time(),
        frontend=mock.MagicMock(),
        force=False,
        steer="also update the docs",
    )
    assert rc == 0
    assert called["initial_steer"] == "also update the docs"


def test_run_task_seeds_initial_steer_on_the_bridge(repo: pathlib.Path) -> None:
    """run_task's initial_steer lands on the bridge before the loop starts.

    Its first boundary poll finds it.
    """
    from agent6.app import run
    from agent6.sessions import ipc

    state = paths.state_dir(repo)
    holder_fd = lock.acquire_repo_writer(state, repo, "run-LIVE")
    try:
        rc = run.run_task(
            _load_cfg(),
            "do the thing",
            started_at=time.time(),
            frontend=mock.MagicMock(),
            session_id="run-STEERSEED",
            mode="run",
            initial_steer="focus on tests",
        )
    finally:
        lock.release_single_writer(holder_fd)
    assert rc == 2  # parked (checkout busy), but the steer already landed
    d = sessions_layout.SessionLayout(state_dir=state, session_id="run-STEERSEED").session_dir
    assert ipc.steer_request_pending(d)
    assert ipc.read_steer_answer(d) == "focus on tests"


def test_teardown_raise_still_releases_both_writer_locks(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raise inside run_task's teardown must still release both writer flocks.

    A CLI exit drops them with the process anyway, but the ACP front-end calls run_task IN-PROCESS
    and outlives the run, where a leaked flock refused every later run on the session/checkout until
    the server restarted.
    """
    from agent6.app import run as run_mod

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("fail with both writer locks held")

    # The first call past BOTH lock acquisitions on the clean-tree path.
    monkeypatch.setattr(stamps, "write_session_manifest", boom)
    frontend = mock.MagicMock()
    frontend.close_console_view.side_effect = OSError("teardown raise")
    with pytest.raises(OSError, match="teardown raise"):
        run_mod.run_task(
            _load_cfg(),
            "do a thing",
            started_at=time.time(),
            frontend=frontend,
            session_id="run-TD",
            mode="run",
        )
    # Both flocks are free for the next in-process run: the checkout's...
    state = paths.state_dir(repo)
    fd = lock.acquire_repo_writer(state, repo, "run-NEXT")
    assert fd is not None
    lock.release_single_writer(fd)
    # ...and the run dir's.
    fd2 = lock.acquire_single_writer(
        sessions_layout.SessionLayout(state_dir=state, session_id="run-TD").session_dir
    )
    assert fd2 is not None
    lock.release_single_writer(fd2)


def test_resume_teardown_raise_still_releases_both_writer_locks(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume teardown raise still releases both writer locks.

    A front-end teardown failure must not strand either resume flock in an in-process editor server;
    later runs must not wait for a process restart.
    """
    from agent6.app import _execution as app__execution
    from agent6.app import resume as resume_mod
    from agent6.harness import _snapshot

    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-RTD")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "run-RTD",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-RTD",
                "user_task": "t",
            }
        ),
        encoding="utf-8",
    )
    (layout.session_dir / "loop_state.json").write_text(
        _snapshot.SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=1,
            root_task_id=None,
            original_task="t",
            verify_command=(),
        ).model_dump_json(),
        encoding="utf-8",
    )

    def _none(*_a: object, **_k: object) -> None:
        return None

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _execution(*_a: object, **_k: object) -> app__execution.ExecutionEnd:
        return app__execution.ExecutionEnd(0)

    monkeypatch.setattr(_setup, "check_provider_keys", _none)
    monkeypatch.setattr(_session, "select_isolation", _strict)
    monkeypatch.setattr(app__execution, "run_execution", _execution)
    frontend = mock.MagicMock()
    frontend.close_console_view.side_effect = OSError("resume teardown raise")
    with pytest.raises(OSError, match="resume teardown raise"):
        resume_mod.resume_task(
            None, "run-RTD", started_at=time.time(), frontend=frontend, force=False
        )

    repo_fd = lock.acquire_repo_writer(state, repo, "run-NEXT")
    assert repo_fd is not None
    lock.release_single_writer(repo_fd)
    session_fd = lock.acquire_single_writer(layout.session_dir)
    assert session_fd is not None
    lock.release_single_writer(session_fd)


def test_resume_drops_a_stop_written_between_executions(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resume drops a stop written between executions.

    This execution's start is the threshold: a marker older than it is stale, whatever the journal
    says. Keeping every marker younger than the journal's last line lets a stop that landed after
    the previous execution ended (nobody honoured it) stop the next execution at its first step.
    """
    import os

    from agent6.app import resume as resume_mod
    from agent6.sessions import ipc

    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-C")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "run-C",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-C",
                "user_task": "t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        '{"type": "session.start", "mode": "run", "user_task": "t"}\n'
        '{"type": "session.end", "reason": "budget_exhausted", "all_passed": false}\n',
        encoding="utf-8",
    )
    execution_end = time.time() - 20
    os.utime(layout.logs_path, (execution_end, execution_end))
    (layout.session_dir / "steer.request").write_text("", encoding="utf-8")
    os.utime(layout.session_dir / "steer.request", (execution_end - 10, execution_end - 10))
    ipc.request_stop(layout.session_dir)
    os.utime(layout.session_dir / "stop.request", (execution_end + 10, execution_end + 10))
    holder_fd = lock.acquire_repo_writer(state, repo, "run-A")
    try:
        rc = resume_mod.resume_task(
            None, "run-C", started_at=time.time(), frontend=mock.MagicMock(), force=False
        )
    finally:
        lock.release_single_writer(holder_fd)
    assert rc == 2  # the checkout is busy: refused after the sweep
    assert "run-A" in capsys.readouterr().err
    assert not ipc.stop_request_pending(layout.session_dir)
    assert not (layout.session_dir / "steer.request").exists()


def test_a_reused_ask_dir_drops_the_previous_executions_markers_and_keeps_this_executions(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused ask dir drops the previous execution's markers and keeps this one's.

    An ask session reuses its dir under the same id (transient Q&A), so the second execution must
    not start on the first execution's leftover markers. A marker older than this execution's start
    is stale, even one younger than the journal (a steer typed after the execution ended); one
    written since the execution began (an editor's cancel while it came up) is this execution's.
    """
    import os

    from agent6.app import _execution as app__execution
    from agent6.app import run as run_mod
    from agent6.sessions import ipc

    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="chat", subdir="asks")
    layout.ensure()
    layout.logs_path.write_text(
        '{"type": "session.start", "mode": "ask", "user_task": "q"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )
    execution_end = time.time() - 20
    os.utime(layout.logs_path, (execution_end, execution_end))
    (layout.session_dir / "steer.request").write_text("", encoding="utf-8")
    os.utime(layout.session_dir / "steer.request", (execution_end + 10, execution_end + 10))
    started_at = time.time() - 5
    ipc.request_stop(layout.session_dir)  # the cancel, after the execution began
    seen: list[tuple[bool, bool]] = []

    def _execution(*_a: object, **_k: object) -> app__execution.ExecutionEnd:
        d = layout.session_dir
        seen.append((ipc.steer_request_pending(d), ipc.stop_request_pending(d)))
        return app__execution.ExecutionEnd(rc=0)

    monkeypatch.setattr(app__execution, "run_execution", _execution)
    rc = run_mod.run_task(
        _load_cfg(),
        "again?",
        started_at=started_at,
        frontend=mock.MagicMock(),
        session_id="chat",
        mode="ask",
    )
    assert rc == 0
    assert seen == [(False, True)]


def test_resume_treats_a_file_that_arrived_between_executions_as_the_operators(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume treats a file that arrived between executions as the operator's.

    The untracked set is recorded at the run's first execution, so a log or note the operator wrote
    between executions is untracked at the resume's start yet absent from the set, and the resumed
    execution's first checkpoint would commit it. At resume, every currently untracked file that no
    tool call of the run wrote joins the set; the run's own uncommitted file stays its own.
    """
    from agent6 import secrets
    from agent6.app import _execution as app__execution
    from agent6.app import _execution as execution_mod
    from agent6.app import resume as resume_mod
    from agent6.harness import _snapshot

    secrets.save_secret("anthropic", "x")  # the provider preflight runs before the execution
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="run-U")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "run-U",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-U",
                "user_task": "t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        '{"type": "session.start", "mode": "run", "user_task": "t"}\n'
        '{"type": "tool.result", "name": "apply_edit", "ok": true, "paths": ["mine.txt"]}\n'
        '{"type": "session.end", "reason": "budget_exhausted", "all_passed": false}\n',
        encoding="utf-8",
    )
    snapshot = _snapshot.SessionSnapshot(
        system="s",
        messages=[],
        tool_calls=0,
        next_iteration=1,
        root_task_id=None,
        original_task="t",
        verify_command=(),
    )
    (layout.session_dir / "loop_state.json").write_text(
        snapshot.model_dump_json(), encoding="utf-8"
    )
    (repo / "mine.txt").write_text("the run's, uncommitted\n", encoding="utf-8")
    # A file a command of the run created and a checkpoint recorded: chain
    # commits never touch the index, so it reads untracked for the whole run.
    (repo / "built.txt").write_text("a build output\n", encoding="utf-8")
    (repo / "réponse.txt").write_text("git quotes this name in ls-tree\n", encoding="utf-8")
    assert git_ops.chain_commit(
        repo, "iter 1", ref=git_ops.chain_ref_for("run-U"), fallback_parent=None
    )
    (repo / "note.md").write_text("the operator's\n", encoding="utf-8")
    seen: list[frozenset[str]] = []

    def _execution(
        cfg: object, layout: object, inputs: app__execution.ExecutionInputs, **_k: object
    ) -> app__execution.ExecutionEnd:
        seen.append(inputs.untracked_at_start)
        return app__execution.ExecutionEnd(rc=0)

    monkeypatch.setattr(app__execution, "run_execution", _execution)
    monkeypatch.setattr(execution_mod, "run_execution", _execution)
    rc = resume_mod.resume_task(
        None, "run-U", started_at=time.time(), frontend=mock.MagicMock(), force=False
    )
    assert rc == 0
    assert seen == [frozenset({"note.md"})]
    assert sessions_layout.read_untracked_at_start(layout.session_dir) == frozenset({"note.md"})
    # The check decides what the run may commit: a git failure refuses the
    # resume instead of quietly committing the operator's file as run work.
    layout.logs_path.write_text(
        layout.logs_path.read_text(encoding="utf-8")
        + '{"type": "loop.resume.start"}\n'
        + '{"type": "session.end", "reason": "budget_exhausted", "all_passed": false}\n',
        encoding="utf-8",
    )

    def _broken(_cwd: pathlib.Path) -> frozenset[str]:
        raise git_ops.GitError("git status failed: index.lock exists")

    monkeypatch.setattr(git_ops, "untracked_paths", _broken)
    seen.clear()
    rc = resume_mod.resume_task(
        None, "run-U", started_at=time.time(), frontend=mock.MagicMock(), force=False
    )
    assert rc == 2 and seen == []


def test_a_run_started_in_a_subdirectory_shares_the_checkouts_lock(tmp_path: pathlib.Path) -> None:
    """The lock is one per CHECKOUT, wherever the operator stood when they started the run.

    Keyed on the cwd, a run from `src/` took a lock of its own and drove the working tree beside the
    run holding the root's.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    state = tmp_path / "state"
    fd = lock.acquire_repo_writer(state, repo, "run-A")
    assert fd is not None
    try:
        assert lock.repo_writer_held(state, repo / "src") is True
        assert lock.repo_writer_holder(state, repo / "src") == "run-A"
        assert lock.acquire_repo_writer(state, repo / "src", "run-B") is None
    finally:
        lock.release_single_writer(fd)
