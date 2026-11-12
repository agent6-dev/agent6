# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child's life under its parent: spawn, wait, import, landing and cleanup (app/children.py).

Driven with a fake spawner that, for each ChildSpec, really clones the origin,
commits on the child's `agent6/<id>` branch, and fabricates a finished session dir
(manifest.json + logs.jsonl), so the clone-independent behavior (symlink live view,
import, lineage stamp, resilience to a failed child) is exercised on real tmp git repos
without spawning real runs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import io
import json
import os
import pathlib
import threading
import time
import types
from collections.abc import Callable, Generator
from typing import Any

import pytest

from agent6 import child_records, directive, kinds, paths, portable
from agent6.app import (
    child_launch,
    child_wait,
    compare,
    parallel_results,
    reporter,
    role_providers,
    stamps,
    worktrees,
)
from agent6.app import children as app_children
from agent6.config import Config
from agent6.context import memory
from agent6.harness import child_repo, judge
from agent6.providers import ProviderResponse
from agent6.sessions import ipc, process
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import child_agent, cli_main, interact, main
from agent6.ui.cli import parallel as parallel_cmd
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing


def _launch_of(env: dict[str, str]) -> child_launch.ChildLaunch:
    """Read the envelope a spawned child's environment names."""
    return child_launch.ChildLaunch.model_validate_json(
        pathlib.Path(env[child_launch.ENV]).read_text(encoding="utf-8")
    )


def _git(repo: pathlib.Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def _write_fake_run(
    session_dir: pathlib.Path,
    task: str,
    *,
    status: str,
    cost: float,
    parallel_id: str | None = None,
    number: int | None = None,
    parent: str | None = None,
    points: float | None = None,
    read_at: float = 0.0,
    base_sha: str = "",
) -> None:
    session_dir.mkdir(parents=True)
    # A real child's manifest records its lineage from birth (the spawn env);
    # the fakes mirror that for `<group>-l<N>` ids so post-import assertions
    # hold for the same reason they do live.
    manifest: dict[str, object] = {
        "version": 2,
        "session_id": session_dir.name,
        "mode": "run",
        "user_task": task,
    }
    if base_sha:
        manifest["base_sha"] = base_sha
    stem, sep, child_s = session_dir.name.rpartition("-l")
    if parallel_id is not None:
        group = parallel_id
    elif sep and stem and child_s.isdigit():
        group, number = stem, int(child_s)
    else:
        group, number = None, None
    if group is not None:
        manifest["parallel"] = {"group": group, "number": number, "parent": parent or group}
    (session_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": task},
        {"type": "budget.update", "usd_total": cost},
    ]
    if points is not None:
        # A child on a subscription plan: no dollars, the account's reading and its points.
        used = 40.0 + points
        events[1] = {
            "ts": datetime.datetime.fromtimestamp(read_at, tz=datetime.UTC).isoformat(),
            "type": "budget.update",
            "plan_used_percent": used,
            "plan_observed_points": points,
            "plan_windows": [
                {
                    "name": "primary",
                    "used_percent": used,
                    "window_minutes": 10080,
                    "resets_at": time.time() + 100 * 3600,
                }
            ],
        }
        manifest["models"] = {"driver": {"provider": "chatgpt", "model": "m"}}
        (session_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    if status == "passed":
        events.append(
            {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
        )
    elif status in ("failed", "setup_command_failed", "budget_exhausted"):
        reason = "provider_error" if status == "failed" else status
        events.append({"type": "session.stop", "reason": reason, "verification": "unverified"})
    elif status == "stale":
        # Died without a session.stop (OOM/SIGKILL): a recorded pid that is gone.
        (session_dir / "worker.pid").write_text("999999999", encoding="utf-8")
    elif status == "gated_fail":
        # A deliberate finish whose OWN verify ran this execution and failed (unlike
        # "finished", which never ran one): the compare must never crown it.
        events.append({"type": "verify.end", "exit_code": 1})
        events.append(
            {"type": "session.stop", "reason": "finish_session", "verification": "failed"}
        )
    else:  # "finished": a deliberate finish without a gate
        events.append(
            {"type": "session.stop", "reason": "finish_session", "verification": "not_applicable"}
        )
    (session_dir / "logs.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


_READ_AT = 1_800_000_000.0


class _FakeSpawner:
    """A synchronous stand-in for the bridge spawner.

    Clone, commit on the child branch, and fabricate a finished session dir in the origin's state.
    """

    def __init__(
        self,
        origin: pathlib.Path,
        origin_state: pathlib.Path,
        state_root: pathlib.Path,
        *,
        fail: set[int] | None = None,
        status_by_child: dict[int, str] | None = None,
        cost_by_child: dict[int, float] | None = None,
        pid_children: set[int] | None = None,
        group_id: str | None = None,
        points_by_child: dict[int, float] | None = None,
        read_at_by_child: dict[int, float] | None = None,
        no_commit: set[int] | None = None,
        branch_at_base: set[int] | None = None,
    ) -> None:
        # Children that create their branch and commit nothing: the branch's tip is the base.
        self.branch_at_base = branch_at_base or set()
        self.points_by_child = points_by_child or {}
        # A child's plan reading is stamped later for each later child unless a test says otherwise.
        self.read_at_by_child = read_at_by_child or {}
        self.no_commit = no_commit or set()
        self.origin = origin
        self.origin_state = origin_state
        self.state_root = state_root
        self.fail = fail or set()
        self.status_by_child = status_by_child or {}
        self.cost_by_child = cost_by_child or {}
        # Children whose fabricated session dir carries a LIVE worker.pid (this test
        # process), simulating the teardown window where session.stop is already in
        # logs.jsonl but the child process has not yet cleared its pid.
        self.pid_children = pid_children or set()
        # The group id the real bridge spawner stamps into the child env;
        # the fabricated manifest mirrors it (None = derive from the id).
        self.group_id = group_id
        self.prior_link_was_symlink: dict[int, bool] = {}
        self.tasks: list[str] = []

    def __call__(self, spec: child_records.ChildSpec, task: str) -> child_records.ChildResult:
        self.tasks.append(task)
        branch = f"agent6/{spec.session_id}"
        if spec.number > 1:  # observe the previous child's live symlink
            prefix = spec.session_id.rsplit("-l", 1)[0]
            prior = self.origin_state / "sessions" / "runs" / f"{prefix}-l{spec.number - 1}"
            self.prior_link_was_symlink[spec.number] = prior.is_symlink()
        if spec.number in self.fail:
            return child_records.ChildResult(
                spec=spec, session_dir=spec.worktree, branch=branch, ok=False, error="boom"
            )
        child_repo.clone_worktree(self.origin, spec.worktree)
        if spec.number in self.branch_at_base:
            git_ops.create_branch(spec.worktree, branch)
        elif spec.number not in self.no_commit:
            git_ops.create_branch(spec.worktree, branch)
            (spec.worktree / f"child{spec.number}.txt").write_text(
                f"child {spec.number}\n", encoding="utf-8"
            )
            _git(spec.worktree, "add", "-A")
            _git(spec.worktree, "commit", "-q", "-m", f"child {spec.number} work")
        # A child's record is in the origin's state from its first event.
        session_dir = self.origin_state / "sessions" / "runs" / spec.session_id
        _write_fake_run(
            session_dir,
            task,
            parallel_id=self.group_id,
            number=spec.number if self.group_id is not None else None,
            status=self.status_by_child.get(spec.number, "passed"),
            cost=self.cost_by_child.get(spec.number, 0.05),
            points=self.points_by_child.get(spec.number),
            read_at=self.read_at_by_child.get(spec.number, _READ_AT + spec.number),
            base_sha=git_read.status(self.origin).head_sha
            if spec.number in self.branch_at_base
            else "",
        )
        if spec.number in self.pid_children:
            (session_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
        return child_records.ChildResult(
            spec=spec, session_dir=session_dir, branch=branch, ok=True, error=""
        )


@pytest.fixture
def origin(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    o = tmp_path / "origin"
    _init_repo(o)
    return o


@pytest.fixture
def runtime() -> app_children.ChildRuntime:
    """The real front-end ChildRuntime the pipeline drives: detached spawn, reviewer and judging.

    A test faking one primitive uses `dataclasses.replace` on it (a fake `spawn`), or monkeypatches
    the module-level `worker_is_alive` (the session-dir bridge, imported directly).
    """
    return parallel_cmd.child_runtime()


def _specs(
    tmp_path: pathlib.Path, cfg: Config, group: str, spec: str
) -> list[child_records.ChildSpec]:
    """Plan a group's children as the dispatcher does: one per selector, numbered from 1."""
    settings = app_children.child_settings(
        cfg, directive.parse_spec(spec, limit=cfg.harness.parallel.max_children)
    )
    return [
        child_records.ChildSpec(
            number=i,
            session_id=f"{group}-l{i}",
            worktree=tmp_path / "work" / group / f"child-{i}",
            settings=chosen,
        )
        for i, chosen in enumerate(settings, start=1)
    ]


def _run_children(
    task: str,
    children: list[child_records.ChildSpec],
    *,
    cfg: Config,
    origin: pathlib.Path,
    runtime: app_children.ChildRuntime,
    group: str,
    spawner: child_records.ChildSpawner | None = None,
) -> list[child_records.ChildResult]:
    """Run each child to its end under one group, in child order."""
    return [
        app_children.run_child_to_completion(
            spec,
            task,
            cfg=cfg,
            explicit_leaves=frozenset(),
            origin=origin,
            record_state=paths.state_dir(origin),
            group=group,
            parent=group,
            runtime=runtime,
            spawner=spawner,
            poll_interval_s=0.01,
        )
        for spec in children
    ]


# ---------------------------------------------------------------------------
# Pre-spawn model validation (B3): a bogus model refuses before any clone.
# ---------------------------------------------------------------------------


def _write_models_cache(cache_home: pathlib.Path, provider: str, models: list[str]) -> None:
    p = cache_home / "models" / f"{provider}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"models": models}), encoding="utf-8")


def _provider_cfg(model: str = "moonshotai/kimi-k2.6", run_commands: str = "allow") -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"backend": "openai", "base_url": "https://x/v1"}},
            "models": {"default": {"model": "o/" + model}},
            "sandbox": {"run_commands": run_commands},
        }
    )


def test_a_parallel_spec_naming_an_unknown_model_is_refused_before_any_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in `--parallel` refuses with the closest id when a fresh listing confirms it."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_models_cache(
        tmp_path / "cache" / "agent6", "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"]
    )
    # A miss re-checks the live listing before refusing; stub it with the same ids.
    from agent6.models import validate as models_validate

    def _listing(*_a: object) -> list[str] | None:
        return ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"]

    monkeypatch.setattr(models_validate, "_fresh_listing", _listing)

    refusal = parallel_cmd.spec_refusal(_provider_cfg(), "moonshotai/kimi-k2.7")

    assert refusal is not None
    assert "unknown model 'o/moonshotai/kimi-k2.7'" in refusal
    assert "closest: o/moonshotai/kimi-k2.6" in refusal


def test_a_parallel_spec_no_cache_can_check_warns_and_proceeds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An offline machine is never blocked on a model cache it can regenerate."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # empty: no snapshot

    assert parallel_cmd.spec_refusal(_provider_cfg(), "made-up/model") is None
    assert "WARNING: unvalidated model" in capsys.readouterr().err


def test_a_parallel_spec_over_the_child_cap_is_refused() -> None:
    """A count over `[harness.parallel].max_children` refuses before any session exists."""
    cfg = Config.model_validate({"harness": {"parallel": {"max_children": 2}}})

    refusal = parallel_cmd.spec_refusal(cfg, "3")

    assert refusal is not None and "max_children = 2" in refusal


def test_children_take_the_parents_away_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """One owner decides a child's away-mode.

    A hub's marker on the parent, else `wait` with a terminal to attach from, else `deny`.
    """
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    monkeypatch.setattr(portable, "has_controlling_tty", lambda: True)
    assert interact.child_away_mode() == "wait"
    monkeypatch.setattr(portable, "has_controlling_tty", lambda: False)
    assert interact.child_away_mode() == "deny"
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    assert interact.child_away_mode() == "wait"


def test_parent_dispatch_refuses_unknown_model(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """The ui-built group dispatcher validates before cloning.

    An unknown model raises, and the loop's group-failure feedback (its `except Exception`) carries
    the message to the parent, so harness needs no models dependency.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_models_cache(tmp_path / "cache" / "agent6", "o", ["moonshotai/kimi-k2.6"])

    # The miss now re-checks the live listing before refusing; stub it with the
    # same ids so the refusal rests on "fresh" evidence (no real network).
    from agent6.models import validate as models_validate

    def _listing(*_a: object) -> list[str] | None:
        return ["moonshotai/kimi-k2.6"]

    monkeypatch.setattr(models_validate, "_fresh_listing", _listing)
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("clone must not happen before validation")

    monkeypatch.setattr(child_repo, "clone_worktree", _boom)
    dispatch = app_children.build_child_spawner(
        _provider_cfg(),
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        parent_session_id="parent",
        runtime=runtime,
    )
    with pytest.raises(
        app_children.ParallelError, match=r"unknown model 'o/moonshotai/kimi-k2\.7'"
    ):
        dispatch([child_records.ChildTask(task="do it", model="moonshotai/kimi-k2.7")], "p1")


def test_parent_dispatch_wait_honors_a_stop_request(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """The in-run parent's child wait honours a stop request.

    `stop` writes stop.request; a wait that polls only the separate immediate-abort answer blocks
    until every child ends on its own.
    """
    origin_state = tmp_path / "ostate"
    parent_dir = origin_state / "sessions" / "runs" / "parent"
    parent_dir.mkdir(parents=True)
    observed: list[bool] = []

    def fake_child(
        spec: child_records.ChildSpec, task: str, **kwargs: object
    ) -> child_records.ChildResult:
        del task

        assert ipc.request_stop(parent_dir)
        should_stop = kwargs["should_stop"]
        assert callable(should_stop)
        stopped = should_stop()
        assert isinstance(stopped, bool)
        observed.append(stopped)
        return child_records.ChildResult(
            spec, spec.worktree, f"agent6/{spec.session_id}", False, "stopped"
        )

    monkeypatch.setattr(app_children, "run_child_to_completion", fake_child)
    dispatch = app_children.build_child_spawner(
        _provider_cfg(),
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        parent_session_id="parent",
        runtime=runtime,
    )
    dispatch([child_records.ChildTask(task="a", model=None)], "p1")

    assert observed == [True]


def test_parent_dispatch_aborts_promptly_on_a_child_thread_raise(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """A child thread that raises aborts the group at once.

    A raise is a bug, not a child failure (those return ok=False); awaiting futures in submission
    order leaves the raise unobserved, and hard_stop unset, until each earlier child finishes on its
    own.
    """
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()
    seen: dict[str, bool] = {}

    def fake_child(
        spec: child_records.ChildSpec, task: str, **kwargs: object
    ) -> child_records.ChildResult:
        import threading

        hard_stop = kwargs["hard_stop"]
        assert isinstance(hard_stop, threading.Event)
        if spec.number == 1:
            # Stands in for a long-running healthy child: it unblocks only when
            # the dispatcher propagates the other child's failure.
            seen["child1_released_by_stop"] = hard_stop.wait(timeout=8.0)
            return child_records.ChildResult(
                spec=spec, session_dir=tmp_path / "l1", branch="b1", ok=True, error=""
            )
        raise RuntimeError("child thread blew up")

    monkeypatch.setattr(app_children, "run_child_to_completion", fake_child)
    dispatch = app_children.build_child_spawner(
        _provider_cfg(),
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        parent_session_id="parent",
        runtime=runtime,
    )
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="child thread blew up"):
        dispatch(
            [
                child_records.ChildTask(task="a", model=None),
                child_records.ChildTask(task="b", model=None),
            ],
            "p1",
        )
    assert time.monotonic() - start < 4.0  # not after child 1's own sweet time
    # The dispatcher re-raises without joining still-running child threads
    # (deliberate: teardown must not block on children), so give child 1's thread
    # a bounded moment to record that hard_stop released it.
    deadline = time.monotonic() + 2.0
    while "child1_released_by_stop" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen.get("child1_released_by_stop") is True  # hard_stop actually fired


def test_bridge_spawner_keeps_the_task_out_of_the_argv(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """The child's argv is the fixed bootstrap; the task and the cap go in the launch file.

    A task that looks like a flag is never parsed as one, since nothing parses it: the
    bootstrap reads the file `AGENT6_LAUNCH` names.
    """
    captured: list[list[str]] = []
    words = "Keep <file>literal</file>. " + "detail " * 800

    def fake_spawn(
        argv: list[str], workdir: pathlib.Path, **_k: object
    ) -> tuple[pathlib.Path, int, str]:
        captured.append(list(argv))
        env = _k["env"]
        assert isinstance(env, dict)
        launch = _launch_of(env)
        assert (launch.operator_task, launch.task, launch.max_usd) == (
            words,
            "--allow-root pwn",
            2.0,
        )
        assert launch.worktree == workdir and launch.session_id == "fan-l1"
        return workdir, 1, ""

    cfg = Config()
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=tmp_path / "work" / "child-1",
        settings=None,
    )
    app_children.bridge_spawner(
        spec,
        "--allow-root pwn",
        operator_task=words,
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=paths.state_dir(origin),
        max_usd=2.0,
        group_id="fan",
        runtime=dataclasses.replace(runtime, spawn=fake_spawn),
        parent="co",
    )

    argv = captured[-1]
    assert argv == child_launch.bootstrap_argv()
    assert argv[argv.index("-m") - 1 :: -1][-1] == argv[0]  # the interpreter leads
    assert not any("pwn" in word or "detail" in word for word in argv)


def test_a_child_is_seeded_with_the_repos_memory(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """A child clones the repo, so its state dir is new and its memory empty.

    The children ran blind to the facts and rulings every other run on that repo gets.
    """
    cfg = Config()
    origin_state = paths.state_dir(origin)
    memory.add(origin_state, "house-style", "Docstrings end in a period.")
    memory.record_decision(
        origin_state, question="Ship the rename?", answer="yes", session="s", when=0.0
    )

    def fake_spawn(
        _argv: list[str], workdir: pathlib.Path, **_k: object
    ) -> tuple[pathlib.Path, int, str]:
        return workdir, 1, ""

    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=tmp_path / "work" / "child-1",
        settings=None,
    )
    app_children.bridge_spawner(
        spec, "task", explicit_leaves=frozenset(), cfg=cfg, origin=origin,
        record_state=paths.state_dir(origin),
        max_usd=None,
        group_id="fan", runtime=dataclasses.replace(runtime, spawn=fake_spawn),
        parent="co",
    )  # fmt: skip

    child_state = paths.state_dir(spec.worktree)
    assert "house-style" in memory.index_text(child_state)
    assert "Docstrings end in a period." in memory.show(child_state, "house-style")
    assert "Ship the rename?" in memory.decisions_text(child_state)


def test_bridge_spawner_launch_carries_auto_approve_when_set(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """A parent or group started with `--auto-approve` forwards it to the child.

    Otherwise the child sits on `run_commands=ask` with nothing to answer it.
    """
    captured: list[child_launch.ChildLaunch] = []

    def fake_spawn(
        argv: list[str], workdir: pathlib.Path, **_k: object
    ) -> tuple[pathlib.Path, int, str]:
        env = _k["env"]
        assert isinstance(env, dict)
        captured.append(_launch_of(env))
        return workdir, 1, ""

    cfg = Config()
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=tmp_path / "work" / "child-1",
        settings=None,
    )
    app_children.bridge_spawner(
        spec, "do it", cfg=cfg, origin=origin,
        record_state=paths.state_dir(origin),
        max_usd=None, auto_approve=True,
        explicit_leaves=frozenset(),
        group_id="fan", runtime=dataclasses.replace(runtime, spawn=fake_spawn),
        parent="co",
    )  # fmt: skip

    assert captured[-1].auto_approve is True


def test_bridge_spawner_launch_omits_auto_approve_by_default(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    captured: list[child_launch.ChildLaunch] = []

    def fake_spawn(
        argv: list[str], workdir: pathlib.Path, **_k: object
    ) -> tuple[pathlib.Path, int, str]:
        env = _k["env"]
        assert isinstance(env, dict)
        captured.append(_launch_of(env))
        return workdir, 1, ""

    cfg = Config()
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=tmp_path / "work" / "child-1",
        settings=None,
    )
    app_children.bridge_spawner(
        spec, "do it", explicit_leaves=frozenset(), cfg=cfg, origin=origin,
        record_state=paths.state_dir(origin),
        max_usd=None,
        group_id="fan", runtime=dataclasses.replace(runtime, spawn=fake_spawn),
        parent="co",
    )  # fmt: skip

    assert captured[-1].auto_approve is False and captured[-1].max_usd is None


def test_run_child_to_completion_forwards_auto_approve_and_pins_to_the_default_spawner(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """`run_child_to_completion` forwards auto_approve and the pins to the spawner it builds.

    With no spawner injected (the parent's path), both reach the real bridge spawner
    exactly like max_usd does.
    """
    captured: list[dict[str, object]] = []

    def fake_bridge(
        spec: child_records.ChildSpec, task: str, **kw: object
    ) -> child_records.ChildResult:
        captured.append(kw)
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch="agent6/x", ok=False, error="stub"
        )

    monkeypatch.setattr(app_children, "bridge_spawner", fake_bridge)
    cfg = Config()
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "work" / "co-p1-l1",
        settings=None,
    )

    app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=tmp_path / "ostate",
        group="p1",
        parent="co",
        runtime=runtime,
        auto_approve=True,
        pins=("keep it",),
    )

    assert captured[-1]["auto_approve"] is True
    assert captured[-1]["pins"] == ("keep it",)


# ---------------------------------------------------------------------------
# Pending-ask probe: a "running" child blocked on an approval/question
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_code", [0, 3])
def test_child_setup_output_survives_import_and_a_failed_one_reaches_the_operator(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exit_code: int,
) -> None:
    """A child's setup output outlives its clone; a failed setup's is printed where it landed."""
    command = f"echo ran >> .setup-count; echo setup-out; echo setup-err >&2; exit {exit_code}"
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://127.0.0.1:1"}},
            "models": {"default": {"model": "p/m"}},
            "sandbox": {
                "isolation": "none",
                "home": "cache",
                "network": "host",
                "run_commands": "allow",
            },
            "harness": {
                "verify": {"infer": False},
            },
            "review": {"when": "never"},
            "git": {"worktree": {"setup_command": ["sh", "-c", command]}},
        }
    )
    calls: list[str] = []

    class FinishedProvider:
        def call(self, **_kw: Any) -> ProviderResponse:
            calls.append("called")
            return ProviderResponse(
                text="",
                tool_uses=(
                    {
                        "type": "tool_use",
                        "id": "end",
                        "name": "finish_session",
                        "input": {"summary": "done"},
                    },
                ),
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
                raw={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "end",
                            "name": "finish_session",
                            "input": {"summary": "done"},
                        }
                    ]
                },
            )

    def provider(*_a: Any, **_kw: Any) -> FinishedProvider:
        return FinishedProvider()

    monkeypatch.setattr(role_providers, "build_role_provider", provider)

    def spawn(
        argv: list[str],
        cwd: pathlib.Path,
        *,
        before: set[pathlib.Path],
        list_dirs: Any,
        env: dict[str, str],
    ) -> tuple[pathlib.Path, int, str]:
        del before, list_dirs
        # A detached child's stdout and stderr are not the group's own streams.
        child_output, child_errors = io.StringIO(), io.StringIO()
        with (
            monkeypatch.context() as patch,
            contextlib.redirect_stdout(child_output),
            contextlib.redirect_stderr(child_errors),
        ):
            patch.chdir(cwd)
            for key, value in env.items():
                if key.startswith("AGENT6_"):
                    patch.setenv(key, value)
            # The bootstrap exports the away mode; set through the patch so it is undone.
            patch.setenv("AGENT6_DETACHED_AWAY", "wait")
            assert argv == child_launch.bootstrap_argv()
            assert child_agent.main() == (0 if exit_code == 0 else 1), (
                child_errors.getvalue() + child_output.getvalue()
            )
            assert (cwd / ".setup-count").read_text(encoding="utf-8").splitlines() == ["ran"]
            launch = _launch_of(env)
            session_dir = launch.record_state / "sessions" / "runs" / launch.session_id
            assert session_dir.is_dir()
        return session_dir, 1, ""

    children = _specs(tmp_path, cfg, "setup-fan", "2")
    state = paths.state_dir(origin)
    results = _run_children(
        "task",
        children,
        cfg=cfg,
        origin=origin,
        runtime=dataclasses.replace(runtime, spawn=spawn),
        group="setup-fan",
    )
    assert [res.ok for res in results] == [exit_code == 0] * 2
    assert bool(calls) is (exit_code == 0)
    for child in children:
        imported = state / "sessions" / "runs" / child.session_id
        events = [json.loads(line) for line in (imported / "logs.jsonl").read_text().splitlines()]
        (setup,) = [event for event in events if event["type"] == "worktree.setup"]
        assert setup["stdout_tail"] == "setup-out\n" and setup["stderr_tail"] == "setup-err\n"
        if exit_code:
            (end,) = [event for event in events if event["type"] == "session.stop"]
            assert "setup-out" in end["summary"] and command in end["summary"]
    shown = capsys.readouterr().err
    # A setup that worked says nothing here; a failed one prints its command and output per child.
    assert (command in shown) is (exit_code != 0)
    assert shown.count("setup-out\n") == shown.count("setup-err\n") == (2 if exit_code else 0)
    for child in children:
        assert not child.worktree.exists()
        imported = state / "sessions" / "runs" / child.session_id
        events = [json.loads(line) for line in (imported / "logs.jsonl").read_text().splitlines()]
        assert sum(event["type"] == "worktree.setup" for event in events) == 1
        with monkeypatch.context() as showing:
            showing.chdir(origin)
            assert cli_main(["sessions", "show", child.session_id]) == 0
        shown = capsys.readouterr().out
        assert command in shown
        # A setup that worked shows as its command line; a failed one keeps its output.
        assert ("setup-out\n" in shown and "setup-err\n" in shown) is (exit_code != 0)


def test_a_children_memory_files_are_carried_into_the_origin_at_import(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A child's memory files are carried into the origin at import.

    The harness nudges a child to write memory like any run; an import that carries only its rulings
    before tearing the child's state dir down loses every memory file the child wrote.
    A name the origin already holds with other content (written before the child was seeded, not a
    seeded copy) is held back in the child's imported session dir, and the note says where.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    memory.add(origin_state, "repo-fact", "The build needs BUILD_ID set.")
    children = _specs(tmp_path, cfg, "fan", "1")
    child_state = paths.state_dir(children[0].worktree)
    memory.add(child_state, "child-fact", "The flaky test is test_clock.")
    memory.add(child_state, "repo-fact", "The build needs BUILD_ID set, and BUILD_NO.")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state")

    (result,) = _run_children(
        "do the task",
        children,
        cfg=cfg,
        origin=origin,
        runtime=runtime,
        group="fan",
        spawner=spawner,
    )
    assert result.ok
    assert "The flaky test is test_clock." in memory.show(origin_state, "child-fact")
    assert "child-fact" in memory.index_text(origin_state)
    assert (
        memory.index_text(origin_state).count("repo-fact") == 1
    )  # the seeded copy is not a second line
    assert "BUILD_NO" not in memory.show(origin_state, "repo-fact")
    held = (
        origin_state / "sessions" / "runs" / children[0].session_id / "memory-held" / "repo-fact.md"
    )
    assert "BUILD_NO" in held.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert (
        "child 1: memory 1 carried, held back (changed here too, or the name is taken):"
        f" repo-fact, kept at {held.parent}"
    ) in err


def test_compare_stamp_records_judge_rationale_truncated(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """The parent holds the ranking and judge cost once; children hold only placement."""
    origin_state = paths.state_dir(origin)
    cfg = Config()
    children = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state")
    long_rationale = "x" * 3000
    results = _run_children(
        "t", children, cfg=cfg, origin=origin, runtime=runtime, group="fan", spawner=spawner
    )
    parent_dir = origin_state / "sessions" / "runs" / "fan"
    _write_fake_run(parent_dir, "t", status="passed", cost=0.0)

    parallel_results.stamp_compare_outcomes(
        [
            judge.CandidateBrief(
                session_id=res.spec.session_id,
                task="t",
                diff="",
                verification="passed",
                cost_usd=0.05,
            )
            for res in results
        ],
        compare.RankOutcome(
            ("fan-l1", "fan-l2"),
            long_rationale,
            "judge",
            judge_cost_usd=0.0123,
            judge_cost_partial=True,
        ),
        winner="fan-l1",
        record_state=origin_state,
        parent_dir=parent_dir,
        group="fan",
    )

    m1 = json.loads(
        (origin_state / "sessions" / "runs" / "fan-l1" / "manifest.json").read_text("utf-8")
    )
    m2 = json.loads(
        (origin_state / "sessions" / "runs" / "fan-l2" / "manifest.json").read_text("utf-8")
    )
    assert m1["placement"] == {"group": "fan", "rank": 1, "of": 2, "winner": True}
    assert m2["placement"] == {"group": "fan", "rank": 2, "of": 2, "winner": False}
    assert m1["comparison"] is None and m2["comparison"] is None
    parent = json.loads(
        (origin_state / "sessions" / "runs" / "fan" / "manifest.json").read_text("utf-8")
    )
    assert parent["placement"] is None
    assert parent["comparison"] == {
        "group": "fan",
        "ranking": ["fan-l1", "fan-l2"],
        "ranked_by": "judge",
        "rationale": long_rationale[:2000],
        "judge_cost_usd": 0.0123,
        "judge_cost_partial": True,
    }


def test_landed_children_remove_their_emptied_worktree_levels(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """A group's clones sit at `<base>/<repo>/<group>/child-N`.

    The group dir and the per-repo dir go once empty, the base stays.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    per_repo = worktrees.worktree_base(cfg, origin)
    children = [
        dataclasses.replace(spec, worktree=per_repo / "fan" / f"child-{spec.number}")
        for spec in _specs(tmp_path, cfg, "fan", "2")
    ]
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state")

    results = _run_children(
        "t", children, cfg=cfg, origin=origin, runtime=runtime, group="fan", spawner=spawner
    )

    assert all(res.ok for res in results)
    assert not (per_repo / "fan").exists()
    assert not per_repo.exists()
    assert per_repo.parent.is_dir()


def test_the_ranked_table_spells_a_children_cost_and_outcome_as_the_listing_does(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ranked table prints each child's cost cell and status label, the listing's own.

    A plan-metered child read `$0.0000` in the table and `1pt` in `sessions`, and a child that
    finished over a red gate read `failed` in the table and `finished · gate red` there.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    children = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(
        origin,
        origin_state,
        tmp_path / "child-state",
        status_by_child={1: "gated_fail", 2: "passed"},
        points_by_child={1: 1.0, 2: 2.5},
    )

    results = _run_children(
        "t", children, cfg=cfg, origin=origin, runtime=runtime, group="fan", spawner=spawner
    )
    summaries = {
        res.spec.session_id: app_children.summarize_session_dir(res.session_dir) for res in results
    }
    capsys.readouterr()

    compare.print_ranked_candidates(
        [
            judge.CandidateBrief(
                session_id=sid,
                task="t",
                diff="",
                verification=row.verification,
                cost_usd=row.cost_usd,
            )
            for sid, row in summaries.items()
        ],
        compare.RankOutcome(("fan-l2", "fan-l1"), "", "mechanical"),
        rows=summaries,
        committed=dict.fromkeys(summaries, True),
    )

    ranked = [ln for ln in capsys.readouterr().out.splitlines() if "merge with:" in ln]
    rows = {ln.split()[1]: ln for ln in ranked}
    assert "passed" in rows["fan-l2"] and "2.5pt" in rows["fan-l2"]
    assert "finished · gate red" in rows["fan-l1"] and "1pt" in rows["fan-l1"]
    assert all("$0.0000" not in ln and "failed" not in ln for ln in ranked)


def test_a_child_with_no_commit_does_not_claim_a_branch_was_imported(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """A setup-failed child imports its record but claims no branch that git lacks."""
    state = paths.state_dir(origin)
    cfg = Config()
    (spec,) = _specs(tmp_path, cfg, "no-commit", "1")
    spawner = _FakeSpawner(
        origin,
        state,
        tmp_path / "child-state",
        status_by_child={1: "setup_command_failed"},
        no_commit={1},
    )
    (result,) = _run_children(
        "task", [spec], cfg=cfg, origin=origin, runtime=runtime, group="no-commit", spawner=spawner
    )
    assert not result.ok
    shown = result.error
    assert not git_read.branch_exists(origin, f"agent6/{spec.session_id}")
    assert (state / "sessions" / "runs" / spec.session_id / "logs.jsonl").exists()
    assert "no result (failed · setup command failed)" in shown
    assert "branch imported as" not in shown
    assert "no branch (the run made no commit)" in shown


def test_a_stamp_that_cannot_be_written_is_noted_and_the_others_still_land(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: app_children.ChildRuntime,
) -> None:
    """A disk-full or read-only mount at a ranking's stamps is noted per manifest, never raised."""
    origin_state = paths.state_dir(origin)
    cfg = Config()
    children = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state")
    results = _run_children(
        "t", children, cfg=cfg, origin=origin, runtime=runtime, group="fan", spawner=spawner
    )
    parent_dir = origin_state / "sessions" / "runs" / "fan"
    _write_fake_run(parent_dir, "t", status="passed", cost=0.0)

    def boom(_path: pathlib.Path, _m: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(parallel_results, "stamps", types.SimpleNamespace(write_manifest=boom))
    capsys.readouterr()

    parallel_results.stamp_compare_outcomes(
        [
            judge.CandidateBrief(
                session_id=res.spec.session_id,
                task="t",
                diff="",
                verification="passed",
                cost_usd=0.05,
            )
            for res in results
        ],
        compare.RankOutcome(("fan-l1", "fan-l2"), "", "mechanical"),
        winner="fan-l1",
        record_state=origin_state,
        parent_dir=parent_dir,
        group="fan",
    )

    err = capsys.readouterr().err
    assert "group [fan]: the comparison stamp failed" in err
    assert err.count("imported, but the placement stamp failed") == 2
    assert err.count("disk full") == 3


# ---------------------------------------------------------------------------
# Teardown race + cleanup safety
# ---------------------------------------------------------------------------


def test_await_waits_for_worker_pid_to_clear(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """session.stop lands in logs.jsonl BEFORE the child's teardown clears worker.pid.

    The await gate must keep waiting through that window (terminal = non-running status AND pid
    cleared/dead); importing inside it would misread the child as still running and cleanup would
    destroy its only copy.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    children = _specs(tmp_path, cfg, "fan", "1")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state", pid_children={1})

    # Clear the live pid only on the SECOND status poll of the child's session dir,
    # so poll 1 exercises the race window deterministically.
    real_summarize = child_wait.summarize_session_dir
    polls = {"n": 0}

    def summarize_then_clear_pid(session_dir: pathlib.Path) -> object:
        summary = real_summarize(session_dir)
        if session_dir.name == "fan-l1":
            polls["n"] += 1
            if polls["n"] >= 2:
                (session_dir / "worker.pid").unlink(missing_ok=True)
        return summary

    monkeypatch.setattr(child_wait, "summarize_session_dir", summarize_then_clear_pid)

    (result,) = _run_children(
        "t", children, cfg=cfg, origin=origin, runtime=runtime, group="fan", spawner=spawner
    )

    assert result.ok
    assert polls["n"] >= 2  # the gate really held through the live-pid poll
    assert git_read.branch_exists(origin, "agent6/fan-l1")
    imported = origin_state / "sessions" / "runs" / "fan-l1"
    assert imported.is_dir() and not imported.is_symlink()


def test_cleanup_preserves_unimported_child(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """A child whose import is refused keeps its clone; its record stays where it was.

    The clone holds the only copy of its branch; the refusal names it. Imported children are still
    cleaned up.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    children = _specs(tmp_path, cfg, "fan", "2")
    # Pre-existing branch in the origin makes child 1's import refuse.
    git_ops.create_branch(origin, "agent6/fan-l1")
    _git(origin, "checkout", "main")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state")
    # Each clone's persistent jail home, as the first command of a child under `partial` leaves it.
    homes = [paths.jail_cache_home(child.worktree) for child in children]
    for home in homes:
        home.mkdir(parents=True)

    results = _run_children(
        "t", children, cfg=cfg, origin=origin, runtime=runtime, group="fan", spawner=spawner
    )

    assert [res.ok for res in results] == [False, True]  # child 2 still imported
    # Child 1 kept: its clone with its branch, and its record.
    assert children[0].worktree.is_dir()
    assert git_read.branch_exists(children[0].worktree, "agent6/fan-l1")
    assert (origin_state / "sessions" / "runs" / "fan-l1").is_dir()
    # Child 2 imported and cleaned: its record stays, its clone is gone.
    assert (origin_state / "sessions" / "runs" / "fan-l2").is_dir()
    assert not children[1].worktree.exists()
    # A clone's jail home goes with the clone and stays with a kept one.
    assert homes[0].is_dir() and not homes[1].exists()
    # The refusal names the kept clone so the operator can act on it.
    assert str(children[0].worktree) in results[0].error


# ---------------------------------------------------------------------------
# Parent plumbing: run_child_to_completion + the group spawner
# ---------------------------------------------------------------------------


def test_run_child_to_completion_imports_and_stamps(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One child runs end to end: spawn, await, import and the lineage stamp.

    The child's record is a real dir in the origin's runs/ while it runs, so a hub can see and
    answer it, and nothing moves at the import.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state", group_id="p1")
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "work" / "co-p1-l1",
        settings=None,
    )
    memory.add(paths.state_dir(spec.worktree), "child-fact", "The flaky test is test_clock.")

    # The await polls summarize_session_dir; the child's record is in the origin's state then.
    record = origin_state / "sessions" / "runs" / "co-p1-l1"
    real_summarize = app_children.summarize_session_dir
    seen: dict[str, bool] = {}

    def observe(session_dir: pathlib.Path) -> object:
        seen.setdefault("record_during_life", record.is_dir() and not record.is_symlink())
        return real_summarize(session_dir)

    monkeypatch.setattr(app_children, "summarize_session_dir", observe)
    monkeypatch.setattr(child_wait, "summarize_session_dir", observe)

    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=origin_state,
        group="p1",
        parent="co",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok
    assert seen["record_during_life"] is True  # a hub could see + answer the child
    assert git_read.branch_exists(origin, "agent6/co-p1-l1")
    # The child's memory came back before its state dir went (the parent
    # path cleaned up with no carry at all).
    assert "The flaky test is test_clock." in memory.show(origin_state, "child-fact")
    assert "child 1: memory 1 carried" in capsys.readouterr().err
    imported = origin_state / "sessions" / "runs" / "co-p1-l1"
    assert imported.is_dir() and res.session_dir == imported
    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel"] == {"group": "p1", "number": 1, "parent": "p1"}


@pytest.mark.parametrize("shape", ["committed", "record_only", "branch_at_base"])
def test_run_child_to_completion_records_what_the_import_landed(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime, shape: str
) -> None:
    """The result says what landed: the ref, its tip and whether it holds commits past the base.

    A child that committed nothing (no branch, or a branch still at the base) is a success with
    `has_commits` False, so the parent joins it as `no_changes` and merges nothing.
    """
    origin_state = paths.state_dir(origin)
    base = git_read.status(origin).head_sha
    spawner = _FakeSpawner(
        origin,
        origin_state,
        tmp_path / "child-state",
        no_commit={1} if shape == "record_only" else set(),
        branch_at_base={1} if shape == "branch_at_base" else set(),
    )
    spec = child_records.ChildSpec(
        number=1, session_id="co-p1-l1", worktree=tmp_path / "work" / "co-p1-l1", settings=None
    )
    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=origin,
        record_state=origin_state,
        group="p1",
        parent="co",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )
    assert res.ok
    branch = "agent6/co-p1-l1"
    if shape == "committed":
        tip = git_read.branch_tip_sha(origin, branch)
        assert tip and tip != base
        assert (res.imported_ref, res.imported_tip, res.has_commits) == (branch, tip, True)
    elif shape == "record_only":
        assert not git_read.branch_exists(origin, branch)
        assert (res.imported_ref, res.imported_tip, res.has_commits) == ("", "", False)
    else:
        assert git_read.branch_tip_sha(origin, branch) == base
        assert (res.imported_ref, res.imported_tip, res.has_commits) == (branch, base, False)


def test_run_child_to_completion_failed_spawn_imports_nothing(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    origin_state = paths.state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state", fail={1})
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "work" / "co-p1-l1",
        settings=None,
    )

    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=origin_state,
        group="p1",
        parent="co",
        runtime=runtime,
        spawner=spawner,
    )

    assert not res.ok and res.error == "boom"
    assert not git_read.branch_exists(origin, "agent6/co-p1-l1")


def test_a_failed_child_never_joins_the_parent(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """A failed child never joins the parent.

    Candidacy is one question, `produced_result`, the same one the group asks; gating only on
    died-without-end lets a child that ended `provider_error` come back ok=True, merge its half-done
    branch into the parent's checkout and tell the model "joined at <sha>".
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(
        origin, origin_state, tmp_path / "child-state", status_by_child={1: "failed"}
    )
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-pf-l1",
        worktree=tmp_path / "work" / "co-pf-l1",
        settings=None,
    )

    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=origin_state,
        group="pf",
        parent="co",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok is False
    assert "failed" in res.error and "provider error" in res.error
    # The work is not lost: the branch was imported before the verdict.
    assert "agent6/co-pf-l1" in res.error
    assert git_read.branch_exists(origin, "agent6/co-pf-l1")


def test_a_crashed_child_never_joins_the_parent(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """The died-without-end half of the same gate (a child with no session.stop folds "stale").

    Imported, named in the error, never joined.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(
        origin, origin_state, tmp_path / "child-state", status_by_child={1: "stale"}
    )
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-ps-l1",
        worktree=tmp_path / "work" / "co-ps-l1",
        settings=None,
    )

    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=origin_state,
        group="ps",
        parent="co",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok is False
    assert "stale" in res.error and "agent6/co-ps-l1" in res.error
    assert git_read.branch_exists(origin, "agent6/co-ps-l1")


def test_build_child_spawner_over_cap_refused(
    origin: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runtime: app_children.ChildRuntime
) -> None:
    """A `/parallel` steer over `[harness.parallel].max_children` is refused before any spawn.

    The cap is a hard cap per group, as `run --parallel` enforces it before any session exists;
    the loop turns the raise into 'group dispatch failed' parent feedback.
    """
    cfg = Config.model_validate({"harness": {"parallel": {"max_children": 2}}})
    called: list[str] = []

    def fake_rltc(
        spec: child_records.ChildSpec, task: str, **kw: object
    ) -> child_records.ChildResult:
        called.append(spec.session_id)
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch="b", ok=True, error=""
        )

    monkeypatch.setattr(app_children, "run_child_to_completion", fake_rltc)
    dispatch = app_children.build_child_spawner(
        cfg,
        origin,
        paths.state_dir(origin),
        explicit_leaves=frozenset(),
        parent_session_id="co",
        runtime=runtime,
    )
    with pytest.raises(app_children.ParallelError, match="max_children"):
        dispatch([child_records.ChildTask(task="t", model=None)] * 3, "p1")
    assert called == []  # nothing cloned or spawned


def test_run_child_to_completion_imports_under_the_group_lock(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The group's *import_lock* is held ACROSS import_run.

    Two child threads fetching into one origin race on refs/objects.
    """
    origin_state = paths.state_dir(origin)
    lock = threading.Lock()
    held: list[bool] = []
    real_import_run = child_repo.import_run

    def observe(
        origin: pathlib.Path,
        child_worktree: pathlib.Path,
        branch: str,
        child_session_dir: pathlib.Path,
        *,
        model_control: bool = False,
    ) -> child_records.Imported:
        held.append(lock.locked())
        return real_import_run(
            origin, child_worktree, branch, child_session_dir, model_control=model_control
        )

    monkeypatch.setattr(child_repo, "import_run", observe)
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "work" / "co-p1-l1",
        settings=None,
    )
    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=origin,
        record_state=origin_state,
        group="p1",
        parent="co",
        runtime=runtime,
        spawner=_FakeSpawner(origin, origin_state, tmp_path / "child-state", group_id="p1"),
        import_lock=lock,
        poll_interval_s=0.01,
    )
    assert res.ok, res.error
    assert held == [True]


def test_run_child_to_completion_cleans_up_imported_clone(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """The parent path tears down an imported child's clone and state.

    Otherwise every `/parallel` group leaks a full repo clone, a state dir and a child config per
    child. Clones sit at `<base>/<repo>/<parent>/<group>/child-N`: every emptied level up to and
    including the per-repo dir goes, the base above it stays. Only the imported (ok=True) child is
    cleaned; a failed import keeps its clone.
    """
    origin_state = paths.state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "child-state")
    per_repo = worktrees.worktree_base(cfg, origin)
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p9-l1",
        worktree=per_repo / "co" / "grp" / "child-1",
        settings=None,
    )
    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=origin_state,
        group="p9",
        parent="co",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )
    assert res.ok
    assert not spec.worktree.exists()  # the clone is gone
    assert not (per_repo / "co" / "grp").exists()  # the emptied group dir is rmdir'd
    assert not (per_repo / "co").exists()  # the parent's dir above it
    assert not per_repo.exists()  # and the per-repo dir
    assert per_repo.parent.is_dir()  # never the base

    # Import refused (branch already exists): the child keeps its clone.
    git_ops.create_branch(origin, "agent6/co-p8-l1")
    git_ops.create_branch(origin, "main")
    spawner2 = _FakeSpawner(origin, origin_state, tmp_path / "child-state-2")
    spec2 = child_records.ChildSpec(
        number=1,
        session_id="co-p8-l1",
        worktree=tmp_path / "work" / "grp8" / "child-1",
        settings=None,
    )
    res2 = app_children.run_child_to_completion(
        spec2,
        "do it",
        explicit_leaves=frozenset(),
        cfg=cfg,
        origin=origin,
        record_state=origin_state,
        group="p8",
        parent="co",
        runtime=runtime,
        spawner=spawner2,
        poll_interval_s=0.01,
    )
    assert not res2.ok
    assert spec2.worktree.exists()  # unimported child retains its workspace
    assert (origin_state / "sessions" / "runs" / "co-p8-l1").is_dir()  # and its record


def test_build_child_spawner_builds_specs_and_preserves_order(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """The group dispatcher names children `<parent>-<group>-l<i>` and returns results in order.

    Children go under a per-group workdir; the order holds despite the pool.
    """
    origin_state = paths.state_dir(origin)
    cfg = _provider_cfg("kimi")
    seen: list[tuple[int, str, str, str, str]] = []

    def fake_rltc(
        spec: child_records.ChildSpec, task: str, **kw: object
    ) -> child_records.ChildResult:
        seen.append((spec.number, spec.session_id, task, str(kw["group"]), str(spec.worktree)))
        return child_records.ChildResult(
            spec=spec,
            session_dir=spec.worktree,
            branch=f"agent6/{spec.session_id}",
            ok=True,
            error="",
        )

    monkeypatch.setattr(app_children, "run_child_to_completion", fake_rltc)
    dispatch = app_children.build_child_spawner(
        cfg,
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        parent_session_id="co",
        runtime=runtime,
    )
    children = [
        child_records.ChildTask(task="task a", model="kimi"),
        child_records.ChildTask(task="task b", model=None),
    ]
    results = dispatch(children, "p2")

    assert [r.spec.session_id for r in results] == ["co-p2-l1", "co-p2-l2"]
    assert [r.spec.settings.route if r.spec.settings else None for r in results] == [
        kinds.ModelRoute("o", "kimi"),
        None,
    ]
    assert sorted(s[0] for s in seen) == [1, 2]  # every child ran once
    # One name for the group: `adopt_orphan_child` and the clone sweep both
    # derive a child's clone from the lineage its manifest recorded, so the dir
    # has to be exactly that. A bare "p2" put the clones a level deeper, and an
    # orphaned child was never adopted.
    assert all(group == "co-p2" for (_l, _r, _t, group, _w) in seen)
    assert all(
        str(worktrees.child_worktree_root(cfg, origin, group) / f"child-{child}") == workdir
        for (child, _r, _t, group, workdir) in seen
    )


def test_build_child_spawner_keeps_child_order_when_children_finish_out_of_order(
    origin: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runtime: app_children.ChildRuntime
) -> None:
    """Child 2 finishes before child 1; the results still come back in child order.

    The parent merges the list in order, so the join order is the dispatch order.
    """
    child_two_done = threading.Event()
    finished: list[int] = []

    def fake_rltc(
        spec: child_records.ChildSpec, task: str, **kw: object
    ) -> child_records.ChildResult:
        if spec.number == 1:
            assert child_two_done.wait(5.0)
        finished.append(spec.number)
        if spec.number == 2:
            child_two_done.set()
        return child_records.ChildResult(
            spec=spec,
            session_dir=spec.worktree,
            branch=f"agent6/{spec.session_id}",
            ok=True,
            error="",
        )

    monkeypatch.setattr(app_children, "run_child_to_completion", fake_rltc)
    dispatch = app_children.build_child_spawner(
        Config(),
        origin,
        paths.state_dir(origin),
        explicit_leaves=frozenset(),
        parent_session_id="co",
        runtime=runtime,
    )
    results = dispatch(
        [child_records.ChildTask("a", None), child_records.ChildTask("b", None)], "p3"
    )
    assert finished == [2, 1]
    assert [r.spec.session_id for r in results] == ["co-p3-l1", "co-p3-l2"]


def test_build_child_spawner_forwards_auto_approve(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    origin_state = paths.state_dir(origin)
    cfg = Config()
    seen: list[object] = []

    def fake_rltc(
        spec: child_records.ChildSpec, task: str, **kw: object
    ) -> child_records.ChildResult:
        seen.append(kw["auto_approve"])
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch="agent6/x", ok=True, error=""
        )

    monkeypatch.setattr(app_children, "run_child_to_completion", fake_rltc)
    dispatch = app_children.build_child_spawner(
        cfg,
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        parent_session_id="co",
        runtime=runtime,
        auto_approve=True,
    )
    dispatch([child_records.ChildTask(task="task a", model=None)], "p3")

    assert seen == [True]


def test_build_parent_spawner_forwards_auto_approve(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
) -> None:
    """A parent started with `--auto-approve` dispatches children that inherit it.

    One started without does not; `build_parent_spawner` passes it to `build_child_spawner` as
    it passes max_usd.
    """
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()
    cfg = Config()
    captured: list[object] = []

    def fake_build_child_spawner(*_a: object, **kw: object) -> object:
        captured.append(kw.get("auto_approve"))
        return "dispatcher"

    monkeypatch.setattr(app_children, "build_child_spawner", fake_build_child_spawner)

    app_children.build_parent_spawner(
        cfg,
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        mode="run",
        session_id="co",
        runtime=runtime,
        auto_approve=True,
    )
    app_children.build_parent_spawner(
        cfg,
        origin,
        origin_state,
        explicit_leaves=frozenset(),
        mode="run",
        session_id="co",
        runtime=runtime,
    )

    assert captured == [True, False]


# ---------------------------------------------------------------------------
# Interrupting a child await (parent stop channel / group teardown)
# ---------------------------------------------------------------------------


def test_await_child_returns_when_should_stop_fires(
    tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """The single-child await honors should_stop.

    Without it, the poll loop blocked until the child ended on its own, so a parent stop (or
    Ctrl-C teardown) sat on a child that might run for hours.
    """
    import os

    child_dir = tmp_path / "child-run"
    child_dir.mkdir()
    _write_fake_run(child_dir / "sub", "t", status="running", cost=0.0)

    session_dir = child_dir / "sub"
    # a live child: never terminal
    process.write_worker_pid(session_dir, os.getpid())
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "w",
        settings=None,
    )
    res = child_records.ChildResult(
        spec=spec, session_dir=session_dir, branch="agent6/x", ok=True, error=""
    )
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    assert child_wait.await_child(res, poll_interval_s=0.01, should_stop=stop_after_two) is False


def test_run_child_to_completion_interrupted_stops_child_and_skips_import(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """An interrupted await asks the child to stop and returns ok=False without importing.

    The child keeps running detached; with hard_stop set the bounded grace is skipped: teardown is
    prompt.
    """
    import os
    import threading

    child_dir = tmp_path / "child-run" / "sub"
    _write_fake_run(child_dir, "t", status="running", cost=0.0)
    # live forever from the test's view
    process.write_worker_pid(child_dir, os.getpid())
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "w",
        settings=None,
    )

    def fake_spawner(spec: child_records.ChildSpec, task: str) -> child_records.ChildResult:
        return child_records.ChildResult(
            spec=spec, session_dir=child_dir, branch="agent6/x", ok=True, error=""
        )

    hard_stop = threading.Event()
    hard_stop.set()
    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=origin,
        record_state=tmp_path / "ostate",
        group="p1",
        parent="co",
        runtime=runtime,
        spawner=fake_spawner,
        poll_interval_s=0.01,
        should_stop=hard_stop.is_set,
        hard_stop=hard_stop,
    )
    assert res.ok is False
    assert "interrupted" in res.error
    assert ipc.stop_request_is_now(child_dir) is True  # the child was asked to stop at once
    assert child_dir.is_dir()  # nothing was moved/imported


def test_child_config_forces_a_run_branch(tmp_path: pathlib.Path) -> None:
    """A child's config forces a run branch.

    The import fetches `agent6/<session_id>`, so a child config that materialises the origin's
    `[git].branch_per_run=false` completes and bills, then fails at import on a missing ref.
    """
    import tomllib

    base = Config()
    cfg = base.model_copy(update={"git": base.git.model_copy(update={"branch_per_run": False})})
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=tmp_path / "clone",
        settings=None,
    )
    path = worktrees.write_child_config(
        cfg,
        spec,
        explicit_leaves=frozenset({"git.branch_per_run"}),
        into=spec.worktree.parent / "launch",
    ).path
    written = tomllib.loads(path.read_text(encoding="utf-8"))
    assert written["git"]["branch_per_run"] is True


def test_child_config_carries_the_child_route(tmp_path: pathlib.Path) -> None:
    """A child's route names its provider, so a child on another provider loads it."""
    import tomllib

    cfg = Config.model_validate(
        {
            "providers": {
                "o": {"backend": "openai", "base_url": "https://x/v1"},
                "p": {"backend": "openai", "base_url": "https://p/v1"},
            },
            "models": {"default": {"model": "o/base"}},
        }
    )
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=tmp_path / "clone",
        settings=kinds.ModelSettings(kinds.ModelRoute("p", "other")),
    )
    written = tomllib.loads(
        worktrees.write_child_config(
            cfg, spec, explicit_leaves=frozenset(), into=spec.worktree.parent / "launch"
        ).path.read_text(encoding="utf-8")
    )
    assert "provider" not in written["models"]["worker"]
    assert written["models"]["worker"]["model"] == "p/other"


def test_the_judge_is_capped_like_a_child(tmp_path: pathlib.Path) -> None:
    """The judge is capped like a child.

    The group advertises "$X/child x N + judge = $Y total"; a judge tracker on the full config
    budget quietly exceeds the printed ceiling. `rank` caps the judge at the child cap when one is
    given, with the config budget as the fallback.
    """
    import contextlib

    from agent6 import budget as agent6_budget
    from agent6.harness import judge
    from agent6.providers import ProviderError

    seen: list[agent6_budget.BudgetTracker] = []

    def build(
        cfg: Config, sink: object, budget: agent6_budget.BudgetTracker, events: object
    ) -> object:
        seen.append(budget)
        raise ProviderError("captured; fall back to mechanical")

    @contextlib.contextmanager
    def status() -> Generator[None]:
        yield

    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "api_key_env": "K"}},
            "models": {
                "default": {"model": "p/m"},
                "reviewer": {"model": "p/judge"},
            },
            "budget": {"max_usd": 10.0},
        }
    )
    briefs = [
        judge.CandidateBrief(
            session_id=f"s{i}", task="t", diff="", verification="passed", cost_usd=0.0
        )
        for i in range(2)
    ]
    outcome = compare.rank(
        cfg,
        briefs,
        transcript_dir=tmp_path,
        build_provider=build,  # pyright: ignore[reportArgumentType]
        judging_status=status,
        max_usd=0.25,
    )
    assert outcome.ranked_by == "mechanical"
    assert [b.max_usd for b in seen] == [0.25]  # the child cap, not the $10 config budget


def test_an_unattended_group_does_not_park_its_children_on_a_question(
    origin: pathlib.Path, tmp_path: pathlib.Path, runtime: app_children.ChildRuntime
) -> None:
    """An unattended group does not park its children on a question.

    `bridge_spawner` stamps each child with the away-mode it is given; a child run with `wait` in a
    pipe or a cron parks on a model's question until someone attaches, which nobody can.
    """
    captured: list[dict[str, str]] = []

    def fake_spawn(
        argv: list[str], workdir: pathlib.Path, **kw: object
    ) -> tuple[pathlib.Path, int, str]:
        env = kw.get("env")
        assert isinstance(env, dict)
        captured.append(env)
        return workdir, 1, ""

    for child, away in enumerate(("wait", "deny"), start=1):
        spec = child_records.ChildSpec(
            number=child,
            session_id=f"fan-l{child}",
            worktree=tmp_path / "work" / f"child-{child}",
            settings=None,
        )
        result = app_children.bridge_spawner(
            spec, "do it", cfg=Config(), origin=origin,
            record_state=paths.state_dir(origin),
            max_usd=None, child_away=away,
            explicit_leaves=frozenset(),
            group_id="fan", parent="co",
            runtime=dataclasses.replace(runtime, spawn=fake_spawn),
        )  # fmt: skip
        assert result.ok, result.error
        assert _launch_of(captured[-1]).away == away


def test_child_is_self_describing_from_birth(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child is self-describing from birth.

    The spawn env carries the group lineage and the manifest's one writer records it, so the
    grouping survives a parent death; a stamp written only after import leaves orphaned children
    listed as unrelated runs.
    """
    from agent6.sessions import layout as sessions_layout

    captured: list[dict[str, str]] = []

    def fake_spawn(
        argv: list[str], workdir: pathlib.Path, **kw: object
    ) -> tuple[pathlib.Path, int, str]:
        env = kw.get("env")
        assert isinstance(env, dict)
        captured.append(env)
        return workdir, 1, ""

    spec = child_records.ChildSpec(
        number=2,
        session_id="fan-l2",
        worktree=tmp_path / "work" / "child-2",
        settings=None,
    )
    app_children.bridge_spawner(
        spec, "do it", explicit_leaves=frozenset(), cfg=Config(), origin=origin,
        record_state=paths.state_dir(origin),
        max_usd=None,
        group_id="fan", parent="co", runtime=dataclasses.replace(runtime, spawn=fake_spawn),
        gate_source="inferred",
    )  # fmt: skip
    envelope = _launch_of(captured[-1])
    assert (envelope.parent, envelope.group, envelope.number) == ("co", "fan", 2)
    assert envelope.gate_source == "inferred"

    # The writer records it from the envelope, exactly as the spawned child would.
    monkeypatch.setenv(child_launch.ENV, captured[-1][child_launch.ENV])
    layout = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="fan-l2")
    layout.ensure()
    stamps.write_session_manifest(
        layout,
        session_id="fan-l2",
        user_task="do it",
        base_sha="s",
        base_branch="main",
        run_branch="agent6/fan-l2",
        cfg=Config(),
    )
    m = sessions_manifest.read_manifest(layout.session_dir)
    assert m.parallel == sessions_manifest.ParallelLineage(group="fan", number=2, parent="co")

    # An ordinary run records no lineage.
    monkeypatch.delenv(child_launch.ENV)
    layout2 = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="plain-run")
    layout2.ensure()
    stamps.write_session_manifest(
        layout2,
        session_id="plain-run",
        user_task="t",
        base_sha="s",
        base_branch="main",
        run_branch=None,
        cfg=Config(),
    )
    assert sessions_manifest.read_manifest(layout2.session_dir).parallel is None


def test_child_sampling_notes_survive_import_and_reach_both_reports(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The production spawn passes a models-only snapshot and durable loss notes to its child."""
    from agent6.app import finalize
    from agent6.app.session import setup as session_setup
    from agent6.config import RoleModel
    from agent6.harness import loop_snapshot
    from agent6.sessions import layout as sessions_layout

    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {"default": {"model": "p/base", "effort": "high", "temperature": 0.7}},
        }
    )
    cfg = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"worker": RoleModel(effort=None, temperature=None)}
            )
        }
    )
    note = "models.worker.temperature: selected None; TOML key omitted; reload uses 0.0."
    spec = child_records.ChildSpec(
        number=1,
        session_id="notes-l1",
        worktree=tmp_path / "clone",
        settings=None,
    )

    def child(argv: list[str], cwd: pathlib.Path, **kwargs: Any) -> tuple[pathlib.Path, int, str]:
        assert argv == child_launch.bootstrap_argv()
        env = kwargs["env"]
        config_path = _launch_of(env).config
        with monkeypatch.context() as patch:
            for name, value in env.items():
                patch.setenv(name, value)
            effective = session_setup.load_session_config(cwd, (config_path,), mode="run")
            worker = effective.config.models.resolve("worker")
            assert worker is not None and worker.effort is None and worker.temperature == 0.0
            layout = sessions_layout.SessionLayout(
                state_dir=_launch_of(env).record_state, session_id=spec.session_id
            )
            layout.ensure()
            stamps.write_session_manifest(
                layout,
                session_id=spec.session_id,
                user_task="t",
                base_sha=git_read.status(cwd).head_sha,
                base_branch="main",
                run_branch=f"agent6/{spec.session_id}",
                cfg=effective.config,
            )
            assert sessions_manifest.read_manifest(layout.session_dir).model_dump()["parallel"][
                "config_notes"
            ] == (note,)
            stamps.stamp_execution(layout.session_dir, effective.config, "run", "none")
        git_ops.create_branch(cwd, f"agent6/{spec.session_id}")
        layout.logs_path.write_text(
            json.dumps({"type": "session.stop", "reason": "finish_session"}) + "\n"
        )
        finalize.print_session_end(
            loop_snapshot.SessionResult(
                completed=True, reason="finish_session", summary="done", turns=1, tool_calls=0
            ),
            layout=layout,
            cwd=cwd,
            budget=session_setup.budget_tracker(cfg),
            console_stream=False,
            reporter=reporter.STDIO_REPORTER,
        )
        assert note in capsys.readouterr().out
        return layout.session_dir, 1, ""

    result = app_children.bridge_spawner(
        spec,
        "t",
        cfg=cfg,
        explicit_leaves=frozenset(),
        origin=origin,
        record_state=paths.state_dir(origin),
        max_usd=None,
        group_id="notes",
        parent="notes",
        runtime=dataclasses.replace(runtime, spawn=child),
    )
    assert result.ok, result.error
    landed = app_children.land_child(
        result,
        parallel_results.import_child(result, origin, None),
        cfg=cfg,
        origin=origin,
        group="notes",
        reporter=reporter.STDIO_REPORTER,
    )
    assert landed.ok, landed.error
    assert note in "".join(capsys.readouterr())
    imported = paths.state_dir(origin) / "sessions" / "runs" / spec.session_id
    record = sessions_manifest.read_manifest(imported)
    assert record.model_dump()["parallel"]["config_notes"] == (note,)
    assert not (spec.worktree.parent / "child-1-config.toml").exists()


def test_sweep_keeps_a_clone_holding_unmerged_commits(
    origin: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep is content-safe by commit proof.

    A group dir whose child clone holds any commit the origin lacks is kept whole (the clone may be
    the only copy); one whose every child tip the origin holds is deleted. The scan covers only this
    repo's `<repo-id>` subdir: another repo's clones can neither be swept by a commit proof made
    against the wrong origin nor kept forever with advice that cannot apply here.
    """
    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(workdir)}}})
    scoped = workdir / paths.repo_id(origin)

    unmerged = scoped / "fan-a" / "child-1"
    unmerged.parent.mkdir(parents=True)
    child_repo.clone_worktree(origin, unmerged)
    git_ops.create_branch(unmerged, "agent6/fan-a-l1")
    (unmerged / "x.txt").write_text("x\n", encoding="utf-8")
    _git(unmerged, "add", "-A")
    _git(unmerged, "commit", "-q", "-m", "child work")

    merged = scoped / "fan-b" / "child-1"
    merged.parent.mkdir(parents=True)
    child_repo.clone_worktree(origin, merged)
    git_ops.create_branch(merged, "agent6/fan-b-l1")  # tip == origin HEAD: nothing unique

    foreign = workdir / "other-repo-0" / "fan-c" / "child-1"
    foreign.mkdir(parents=True)
    child_repo.clone_worktree(origin, foreign)
    git_ops.create_branch(foreign, "agent6/fan-c-l1")

    for clone in (unmerged, merged):
        paths.jail_cache_home(clone).mkdir(parents=True)

    swept, kept = worktrees.sweep_group_clones(origin, cfg)
    assert (swept, kept) == (1, 1)  # the foreign repo's dir is not counted
    assert (scoped / "fan-a").is_dir()  # unique commits: kept whole
    assert not (scoped / "fan-b").exists()
    # A swept clone's persistent jail home goes with it; a kept clone's stays.
    assert paths.jail_cache_home(unmerged).is_dir()
    assert not paths.jail_cache_home(merged).exists()
    assert (foreign / ".git").exists()  # out of scope: untouched


def test_prune_names_a_failed_start_clones_missing_record_and_removal(
    origin: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: app_children.ChildRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Prune names a failed spawn's missing child record and manual cleanup, not a live child."""
    monkeypatch.chdir(origin)
    cfg = Config()
    children = [
        child_records.ChildSpec(
            number=1,
            session_id="failed11-l1",
            worktree=worktrees.place(cfg, origin, "failed11") / "child-1",
            settings=None,
        )
    ]

    def fail_spawn(
        _argv: list[str],
        cwd: pathlib.Path,
        *,
        before: set[pathlib.Path],
        list_dirs: Callable[[], list[pathlib.Path]],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        assert cwd == children[0].worktree
        assert not before and not list_dirs()
        envelope = _launch_of(env)
        assert (envelope.parent, envelope.group, envelope.number) == ("failed11", "failed11", 1)
        assert worktrees.sweep_group_clones(origin, cfg) == (0, 1)
        return None, 1, "child refused before creating a session"

    failed_runtime = dataclasses.replace(runtime, spawn=fail_spawn)

    def spawn_child(spec: child_records.ChildSpec, task: str) -> child_records.ChildResult:
        return app_children.bridge_spawner(
            spec,
            task,
            cfg=cfg,
            explicit_leaves=frozenset(),
            origin=origin,
            record_state=paths.state_dir(origin),
            max_usd=None,
            group_id="failed11",
            parent="failed11",
            runtime=failed_runtime,
        )

    (result,) = _run_children(
        "t",
        children,
        cfg=cfg,
        origin=origin,
        runtime=failed_runtime,
        group="failed11",
        spawner=spawn_child,
    )
    assert not result.ok and "child refused before creating a session" in result.error
    capsys.readouterr()
    child = children[0].worktree
    assert child.is_dir()
    assert not listing.session_dirs(paths.state_dir(child))
    for _ in range(2):
        assert main(["sessions", "prune"]) == 0
        out = capsys.readouterr().out
        assert child.is_dir()
        assert "no child session record" in out
        assert "a child may still be starting" in out
        assert f"rm -r -- {child}" in out and str(child.parent) + '"' not in out
        assert "clone groups: swept 0, kept 1" in out
        assert "holding a live child" not in out
        assert "merge or archive their children" not in out


def test_sweep_keeps_a_commitless_clone_only_while_its_child_lives(
    origin: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The sweep keeps a commitless clone only while its child lives.

    A child checkpoints nothing before its first run ref; kept unconditionally, each child that died
    before its first checkpoint (and each plan or ask child) holds a full repo clone for good, under
    a prune message about commits it never held.
    """
    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(workdir)}}})
    child = workdir / paths.repo_id(origin) / "fan-starting" / "child-1"
    child.parent.mkdir(parents=True)
    child_repo.clone_worktree(origin, child)
    # No session dir yet: a child between its clone and its first write looks
    # like one that never started, and a prune in that window must not win.
    assert worktrees.sweep_group_clones(origin, cfg) == (0, 1)
    # The child's record is in the origin's state, named for its group and number.
    session = paths.state_dir(origin) / "sessions" / "runs" / "fan-starting-l1"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text("", encoding="utf-8")  # a record, not a husk
    process.write_worker_pid(session, os.getpid())

    assert worktrees.sweep_group_clones(origin, cfg) == (0, 1)
    assert child.is_dir()

    (session / "worker.pid").unlink()
    assert worktrees.sweep_group_clones(origin, cfg) == (1, 0)
    assert not child.parent.exists()

    # A child that died in preflight leaves a husk (a session dir with no
    # record and no worker): ended, so its clone goes too.
    child_repo.clone_worktree(origin, child)
    (session / "logs.jsonl").unlink()
    assert worktrees.sweep_group_clones(origin, cfg) == (1, 0)


def test_sweep_keeps_a_clone_whose_tip_the_origin_cannot_reach(
    origin: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The sweep keeps a clone whose tip the origin cannot reach.

    `rev-parse <sha>^{commit}` succeeds for a loose object no ref reaches, exactly what the operator
    is left with after the `git branch -D` prune's own message tells them to run; deleting the clone
    on that proof sends the work with the next `git gc`.
    """
    import subprocess

    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(workdir)}}})
    scoped = workdir / paths.repo_id(origin)
    child = scoped / "fan-x" / "child-1"
    child.parent.mkdir(parents=True)
    child_repo.clone_worktree(origin, child)
    git_ops.create_branch(child, "agent6/fan-x-l1")
    (child / "x.txt").write_text("child work\n", encoding="utf-8")
    _git(child, "add", "-A")
    _git(child, "commit", "-q", "-m", "child work")
    tip = subprocess.run(
        ["git", "-C", str(child), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # The origin has the OBJECT (fetched) but no ref reaching it.
    subprocess.run(
        ["git", "-C", str(origin), "fetch", str(child), tip], check=True, capture_output=True
    )

    swept, kept = worktrees.sweep_group_clones(origin, cfg)

    assert (swept, kept) == (0, 1)
    assert child.is_dir(), "the last ref reaching the child's work was deleted"


def test_sweep_leaves_a_dir_that_is_not_a_group_alone(
    origin: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a group dir holding `child-*` clones is agent6's to judge by commit proof.

    A directory the operator put under the workdir scope (a clone named otherwise, a plain dir with
    files) holds no child clone, and the sweep read that as "nothing to prove" and deleted it.
    """
    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(workdir)}}})
    scoped = workdir / paths.repo_id(origin)
    theirs = scoped / "my-clone"
    theirs.parent.mkdir(parents=True)
    child_repo.clone_worktree(origin, theirs)
    (theirs / "draft.txt").write_text("uncommitted\n", encoding="utf-8")
    notes = scoped / "notes"
    notes.mkdir()
    (notes / "todo.md").write_text("mine\n", encoding="utf-8")

    swept, kept = worktrees.sweep_group_clones(origin, cfg)
    assert (swept, kept) == (0, 0)
    assert (theirs / "draft.txt").read_text(encoding="utf-8") == "uncommitted\n"
    assert (notes / "todo.md").read_text(encoding="utf-8") == "mine\n"


def test_carry_back_names_the_kept_dir_only_when_it_holds_something(
    origin: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The carry-back note names the kept dir only when it holds something.

    A held-back deletion keeps nothing, so a note pointing at memory-held/ would name a directory
    that does not exist.
    """
    origin_state = paths.state_dir(origin)
    memory.add(origin_state, "d-fact", "D first.")
    child_state = tmp_path / "child-state"
    memory.seed_store(origin_state, child_state)
    memory.remove(child_state, "d-fact")
    (memory.memory_dir(origin_state) / "d-fact.md").write_text("D per the origin.\n")
    dest = tmp_path / "dest"
    parallel_results.carry_back(
        child_state, origin_state, dest, number=2, reporter=reporter.STDIO_REPORTER
    )
    err = capsys.readouterr().err
    assert "child 2: memory held back (changed here too, or the name is taken): d-fact" in err
    assert "kept at" not in err and not (dest / "memory-held").exists()


def test_a_plain_stop_escalates_to_now_when_the_operator_stops_again(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-run parent lets a child finish its turn; a second stop hands its command back."""
    import os

    child_dir = tmp_path / "child-run" / "sub"
    _write_fake_run(child_dir, "t", status="running", cost=0.0)
    process.write_worker_pid(child_dir, os.getpid())  # live forever from the test's view
    spec = child_records.ChildSpec(
        number=1,
        session_id="co-p1-l1",
        worktree=tmp_path / "w",
        settings=None,
    )
    asked: list[bool] = []
    real_request = ipc.request_stop

    def recording_request(session_dir: pathlib.Path, *, now: bool = False) -> bool:
        asked.append(now)
        return real_request(session_dir, now=now)

    monkeypatch.setattr(ipc, "request_stop", recording_request)
    monkeypatch.setattr(child_wait, "STOP_GRACE_S", 0.0)
    res = app_children.run_child_to_completion(
        spec,
        "do it",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=origin,
        record_state=tmp_path / "ostate",
        group="p1",
        parent="co",
        runtime=runtime,
        spawner=lambda spec, task: child_records.ChildResult(spec, child_dir, "agent6/x", True, ""),
        poll_interval_s=0.01,
        should_stop=lambda: True,
        stop_is_now=lambda: bool(asked),  # the second stop arrives after the first was written
    )
    assert asked == [False, True]
    assert res.ok is False and "interrupted" in res.error
    assert ipc.stop_request_is_now(child_dir)


def test_a_relative_worktree_base_resolves_to_an_absolute_path(
    origin: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative `git.worktree.base_dir` reached a fork's manifest as a relative worktree path."""
    monkeypatch.chdir(tmp_path)
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": "wt"}}})

    assert worktrees.worktree_base(cfg, origin) == tmp_path.resolve() / "wt" / paths.repo_id(origin)


@pytest.mark.parametrize("head_kind", ["branch", "taken", "detached"])
def test_model_parallel_import_keeps_child_commits(
    origin: pathlib.Path,
    tmp_path: pathlib.Path,
    runtime: app_children.ChildRuntime,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    head_kind: str,
) -> None:
    """Model-owned child commits survive clone cleanup under a durable origin ref."""
    from agent6.config import layer
    from agent6.viewmodel import wire

    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://127.0.0.1:1"}},
            "models": {"default": {"model": "p/m"}},
            "git": {
                "control": "model",
                "worktree": {"base_dir": str(tmp_path / "clones")},
            },
            "sandbox": {
                "isolation": "none",
                "home": "cache",
                "network": "host",
                "run_commands": "allow",
            },
            "harness": {
                "verify": {"infer": False},
            },
            "review": {"when": "never"},
        }
    )
    tips: dict[str, str] = {}
    clones: list[pathlib.Path] = []
    base = git_read.status(origin).head_sha

    class Provider:
        def call(self, **_kw: Any) -> ProviderResponse:
            cwd = pathlib.Path.cwd()
            name = _launch_of(dict(os.environ)).session_id
            if name not in tips:
                if head_kind == "detached":
                    _git(cwd, "checkout", "--detach")
                else:
                    _git(cwd, "checkout", "-b", name)
                (cwd / "work.txt").write_text(name)
                _git(cwd, "add", "work.txt")
                _git(cwd, "commit", "-qm", "child work")
                tips[name] = git_read.status(cwd).head_sha
                if head_kind == "taken":
                    _git(origin, "branch", name, base)
            tool = {
                "type": "tool_use",
                "id": "end",
                "name": "finish_session",
                "input": {"summary": "committed child work"},
            }
            return ProviderResponse(
                text="",
                tool_uses=(tool,),
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
                raw={"content": [tool]},
            )

    def provider(*_a: Any, **_kw: Any) -> Provider:
        return Provider()

    monkeypatch.setattr(role_providers, "build_role_provider", provider)

    one_at_a_time = threading.Lock()

    def spawn(argv: list[str], cwd: pathlib.Path, **kw: Any) -> tuple[pathlib.Path, int, str]:
        clones.append(cwd)
        # A child runs in this process, which has one cwd and one environment.
        with one_at_a_time, monkeypatch.context() as child:
            child.chdir(cwd)
            for key, value in kw["env"].items():
                if key.startswith("AGENT6_"):
                    child.setenv(key, value)
            child.setenv("AGENT6_DETACHED_AWAY", "wait")
            assert argv == child_launch.bootstrap_argv()
            assert child_agent.main() == 0
            launch = _launch_of(kw["env"])
            session = launch.record_state / "sessions" / "runs" / launch.session_id
        return session, 1, ""

    def no_judge(*_a: Any, **_kw: Any) -> Any:
        from agent6.providers import ProviderError

        raise ProviderError("mechanical ranking for this test")

    monkeypatch.setattr(
        parallel_cmd,
        "child_runtime",
        lambda: dataclasses.replace(runtime, spawn=spawn, build_provider=no_judge),
    )
    config = tmp_path / "model.toml"
    config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
    monkeypatch.chdir(origin)
    assert (
        cli_main(["--config", str(config), "run", "--auto-approve", "/parallel 2 -- do work"]) == 0
    )
    out = capsys.readouterr().out
    assert len(tips) == 2
    for name, tip in tips.items():
        chain = git_read.chain_ref_for(name)
        assert git_read.chain_tip(origin, chain) == tip
        m = sessions_manifest.read_manifest(paths.state_dir(origin) / "sessions" / "runs" / name)
        expected = name if head_kind == "branch" else chain
        assert wire.commits_ref(m, origin) == expected
        table = out.split("ranked candidates (best first):", 1)[1]
        assert f"work is on {expected}" in table
        assert cli_main(["sessions", "show", name]) == 0
        shown = capsys.readouterr().out
        assert f"work is on {expected}" in shown
        assert "merge with:" not in shown
        if head_kind == "taken":
            assert git_read.branch_tip_sha(origin, name) == base
    group = next(iter(tips)).rsplit("-l", 1)[0]
    assert cli_main(["--config", str(config), "sessions", "compare", group]) == 0
    compared = capsys.readouterr().out
    for name in tips:
        expected = name if head_kind == "branch" else git_read.chain_ref_for(name)
        assert f"work is on {expected}" in compared
    assert "merge with:" not in compared
    assert all(not clone.exists() for clone in clones)
    assert "merge with:" not in out
    assert "nothing to merge" not in out


@pytest.mark.parametrize("checkout_base", [False, True])
def test_cleanup_keeps_unreachable_model_head(
    origin: pathlib.Path, tmp_path: pathlib.Path, checkout_base: bool
) -> None:
    """An imported record alone is not proof that a child's commits are safe."""
    cfg = Config()
    (spec,) = _specs(tmp_path, cfg, "unsafe", "1")
    _git(origin, "clone", "--quiet", str(origin), str(spec.worktree))
    _git(spec.worktree, "config", "user.name", "test")
    _git(spec.worktree, "config", "user.email", "test@example.org")
    _git(spec.worktree, "checkout", "-b", "model-work")
    (spec.worktree / "work.txt").write_text("keep me")
    _git(spec.worktree, "add", "work.txt")
    _git(spec.worktree, "commit", "-qm", "work")
    _git(origin, "fetch", str(spec.worktree), "HEAD")
    if checkout_base:
        _git(spec.worktree, "checkout", "main")
    notes: list[str] = []
    worktrees.cleanup(
        [spec],
        origin=origin,
        record_state=paths.state_dir(origin),
        worktree_root=spec.worktree.parent,
        base=spec.worktree.parent,
        launch_dir=spec.worktree.parent / "launch",
        cfg=cfg,
        note_keep=lambda _p, why: notes.append(why),
    )
    assert spec.worktree.is_dir()
    assert notes and "cannot reach" in notes[0]
