# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run writes worker.pid before it can ask the operator anything.

The pid is what every surface gates on: `sessions list` reads a run without
one as "created", and `agent6 answer` refuses it as "not running". A run
parked on its own dirty-tree start question is neither, it is a live worker
waiting for exactly that answer. Refusals BEFORE the first prompt still write
none, and the teardown clears it on every exit path.
"""

from __future__ import annotations

import pathlib
import subprocess as sp
import time
from unittest import mock

import pytest

from agent6 import paths
from agent6.app import _execution as app__execution
from agent6.app import _session
from agent6.app import run as run_mod
from agent6.config import Config
from agent6.sessions import ipc
from agent6.tools import operator_prompts


def _repo(root: pathlib.Path) -> None:
    root.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=root, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)


def test_run_writes_its_worker_pid_before_it_asks_the_operator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _repo(repo)
    monkeypatch.chdir(repo)
    order: list[str] = []

    def _pid(_session_dir: pathlib.Path, _pid: int) -> None:
        order.append("pid")

    def _isolation(*_a: object, **_k: object) -> str:
        return "none"

    monkeypatch.setattr(ipc, "write_worker_pid", _pid)
    monkeypatch.setattr(_session, "select_isolation", _isolation)

    def _execution(*_a: object, **_k: object) -> object:
        order.append("execution")
        raise RuntimeError("stop here")

    monkeypatch.setattr(app__execution, "run_execution", _execution)
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})

    def _cancel(_request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        order.append("ask")
        return operator_prompts.QuestionAnswer(("cancel",), "stdin")

    # A front-end whose operator cancels the dirty-tree start question.
    frontend = mock.MagicMock()
    frontend.build_questioner.return_value = _cancel
    # The pid lands BEFORE the question: while a run waits on it, `agent6
    # answer` and the listings must read it as the live worker it is.
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert run_mod.run_task(cfg, "t", started_at=time.time(), frontend=frontend, mode="run") == 2
    assert order == ["pid", "ask"]
    # A passing preflight writes the pid, then runs the execution -- in every mode:
    # an ask blocks on questions too, and `agent6 ps` and `steer` gate on the
    # same file. Only run mode took the checkout lock the write sat behind.
    sp.run(["git", "checkout", "-q", "--", "a.py"], cwd=repo, check=True)
    for mode in ("run", "plan", "ask"):
        order.clear()
        with pytest.raises(RuntimeError, match="stop here"):
            run_mod.run_task(cfg, "t", started_at=time.time(), frontend=frontend, mode=mode)
        assert order == ["pid", "execution"], mode


def test_a_cancelled_start_question_leaves_no_pid_behind(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled start question leaves no pid behind.

    The teardown clears it on every exit path, so a run that asked and was then cancelled does not
    go on reading as live.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _repo(repo)
    monkeypatch.chdir(repo)

    def _isolation(*_a: object, **_k: object) -> str:
        return "none"

    monkeypatch.setattr(_session, "select_isolation", _isolation)

    def _cancel(_request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        return operator_prompts.QuestionAnswer(("cancel",), "stdin")

    frontend = mock.MagicMock()
    frontend.build_questioner.return_value = _cancel
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")

    assert (
        run_mod.run_task(
            Config.model_validate({"sandbox": {"run_commands": "yes"}}),
            "t",
            started_at=time.time(),
            frontend=frontend,
            mode="run",
        )
        == 2
    )

    from agent6.sessions import layout

    runs = layout.bucket_dir(paths.state_dir(repo), "runs")
    pids = list(runs.glob("*/worker.pid"))
    assert [p for p in pids if p.read_text(encoding="utf-8").strip()] == []


def test_a_frontend_teardown_failure_still_clears_the_worker_pid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A front-end teardown failure still clears the worker pid.

    An in-process front-end outlives the run, so its pid must not remain the session's worker
    identity when closing its console view fails.
    """
    from agent6.app import _execution as app__execution

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _repo(repo)
    monkeypatch.chdir(repo)

    def _strict(*_args: object, **_kwargs: object) -> str:
        return "strict"

    def _finished(*_args: object, **_kwargs: object) -> app__execution.ExecutionEnd:
        return app__execution.ExecutionEnd(0)

    monkeypatch.setattr(_session, "select_isolation", _strict)
    monkeypatch.setattr(app__execution, "run_execution", _finished)
    frontend = mock.MagicMock()
    frontend.close_console_view.side_effect = OSError("console teardown failed")

    with pytest.raises(OSError, match="console teardown failed"):
        run_mod.run_task(
            Config.model_validate({"sandbox": {"run_commands": "yes"}}),
            "t",
            started_at=time.time(),
            frontend=frontend,
            session_id="pid-teardown",
            mode="run",
        )

    session = paths.state_dir(repo) / "sessions" / "runs" / "pid-teardown"
    assert not (session / "worker.pid").exists()


def test_a_frontend_teardown_failure_still_pops_the_auto_stash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stash pop shares the teardown with the pid clear.

    A console teardown that raises must not leave the operator's pre-run changes stashed with
    nothing said.
    """
    from agent6.app import _execution as app__execution

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _repo(repo)
    monkeypatch.chdir(repo)
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")

    def _strict(*_args: object, **_kwargs: object) -> str:
        return "strict"

    def _finished(*_args: object, **_kwargs: object) -> app__execution.ExecutionEnd:
        return app__execution.ExecutionEnd(0)

    monkeypatch.setattr(_session, "select_isolation", _strict)
    monkeypatch.setattr(app__execution, "run_execution", _finished)
    frontend = mock.MagicMock()
    frontend.close_console_view.side_effect = OSError("console teardown failed")
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "git": {"dirty_tree": "stash", "auto_stash_pop": True},
        }
    )

    with pytest.raises(OSError, match="console teardown failed"):
        run_mod.run_task(
            cfg,
            "t",
            started_at=time.time(),
            frontend=frontend,
            session_id="stash-teardown",
            mode="run",
        )

    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2\n"
    stashes = sp.run(
        ["git", "stash", "list"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert stashes == ""
