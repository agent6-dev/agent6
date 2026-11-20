# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Real repositories and private launches for child tests."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from agent6 import budget, child_records, directive, kinds, paths
from agent6.app import (
    child_launch,
    children,
    worktrees,
)
from agent6.config import Config
from agent6.harness import (
    child_repo,
    loop_conversation,
    loop_state,
    parallel_dispatch,
    parallel_panel,
    run_chain,
)
from agent6.sessions import (
    child_dispatch,
    event_log,
    layout,
    process,
)
from agent6.ui.cli import parallel
from agent6.vcs.git import read as git_read


def ended(results: Sequence[child_records.ChildResult]) -> list[child_records.StartedChild]:
    """Return children that have already ended with these results, as a fake spawner starts them."""
    started: list[child_records.StartedChild] = []
    for result in results:
        future: concurrent.futures.Future[child_records.ChildResult] = concurrent.futures.Future()
        future.set_result(result)
        started.append(child_records.StartedChild(result.spec.session_id, future))
    return started


def blocking(
    run: Callable[..., Sequence[child_records.ChildResult]],
) -> child_records.GroupChildSpawner:
    """Adapt a fake that runs its group to the end into a spawner whose children have ended."""

    def start(
        children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        return ended(run(children, group, at=at))

    return start


@dataclasses.dataclass(frozen=True, slots=True)
class Completed:
    """A work dispatch run to completion.

    Attributes:
        started: What the dispatch started, or its refusal.
        children: Each child's recorded merge, in the order written.
        message: The completion notices, one per task.
    """

    started: child_records.Started
    children: list[child_records.ChildMerge]
    message: str


def completed(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    conversation: loop_conversation.Conversation,
    turn: int,
    state: loop_state.LoopState,
    segments: list[directive.Segment],
    **kwargs: Any,
) -> Completed:
    """Dispatch, wait for the children and complete the task without taking a turn."""
    started = dispatcher.dispatch(conversation, turn, state, segments, **kwargs)
    dispatcher.wait()
    dispatcher.settle(conversation, turn)
    assert dispatcher.session_dir is not None
    records = [
        child_dispatch.read_record(child_dispatch.record_path(dispatcher.session_dir, group))
        for group in started.groups
    ]
    return Completed(
        started,
        [merge for group in started.groups for merge in dispatcher.merges(group)],
        "\n".join(record.message for record in records),
    )


def results_of(started: Sequence[child_records.StartedChild]) -> list[child_records.ChildResult]:
    """Wait for started children and return their results, in child order."""
    return [child.result.result(timeout=120) for child in started]


def git(repo: pathlib.Path, *args: str) -> str:
    """Read git output from a temporary repository."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def init_repo(repo: pathlib.Path) -> None:
    """Initialize a committed temporary repository."""
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")


def create_record(
    repo: pathlib.Path, sid: str, *, parent: str = "", mode: str = "run"
) -> pathlib.Path:
    """Create a session record in the real repository state layout."""
    dest = layout.bucket_dir(paths.state_dir(repo), kinds.session_bucket(mode)) / sid
    dest.mkdir(parents=True)
    data: dict[str, Any] = {"version": 2, "session_id": sid, "mode": mode, "user_task": "t"}
    if parent:
        data["parallel"] = {"parent": parent, "group": parent + "-p1", "number": 1}
    (dest / "manifest.json").write_text(json.dumps(data))
    (dest / "logs.jsonl").write_text(json.dumps({"type": "session.start", "mode": mode}) + "\n")
    return dest


def config(url: str = "http://127.0.0.1:1/v1", *, tool: bool = False) -> Config:
    """Configure a local provider and a child's review authority."""
    return Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": url}},
            "models": {"default": {"model": "p/base"}},
            "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "allow"},
            "harness": {"verify": {"when": "never"}, "parallel": {"tool": tool}},
            "review": {"when": "verify_fail"},
        }
    )


def launch_worker(
    repo: pathlib.Path, clone: pathlib.Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, child_launch.ChildLaunch]:
    """Create a private worker launch and install its provenance in the environment."""
    child_repo.clone_worktree(repo, clone)
    sid = "parent-p1-l1"
    spec = child_records.ChildSpec(1, sid, clone, None)
    into = child_launch.launch_dir(paths.state_dir(repo), "parent-p1")
    config = worktrees.write_child_config(cfg, spec, explicit_leaves=frozenset(), into=into)
    launch = child_launch.ChildLaunch(
        worktree=clone,
        clone=child_launch.CloneReceipt.capture(clone),
        record_state=paths.state_dir(repo),
        session_id=sid,
        task="worker",
        config=config.path,
        parent="parent",
        group="parent-p1",
        number=1,
        limits=child_records.ChildLimits(max_turns=20, deadline_at=time.time() + 90),
    )
    path = child_launch.write(into, launch)
    monkeypatch.setenv(child_launch.ENV, str(path))
    monkeypatch.setenv("AGENT6_CHILD", "1")
    return path, launch


def dispatcher(
    repo: pathlib.Path, session_dir: pathlib.Path, cfg: Config, meter: budget.BudgetTracker
) -> parallel_dispatch.ParallelDispatcher:
    """Build the common dispatcher with the worker launch authority."""
    spawner = children.build_parent_spawner(
        cfg,
        repo,
        # A session's children keep their records where its own is.
        layout.layout_of(session_dir).state_dir,
        explicit_leaves=frozenset(),
        mode="run",
        session_id=session_dir.name,
        runtime=parallel.child_runtime(),
    )
    assert spawner is not None, "a trusted worker retains the common executor for its panel"
    return parallel_dispatch.ParallelDispatcher(
        chain=run_chain.RunChain(
            repo,
            ref=git_read.chain_ref_for(session_dir.name),
            fallback_parent=git(repo, "rev-parse", "HEAD"),
        ),
        curator=None,
        max_children=4,
        child_spawner=spawner,
        save_snapshot=lambda *_a, **_k: None,
        log=lambda _m: None,
        emit=event_log.EventSink(session_dir / "logs.jsonl").emit,
        emit_graph_snapshot=lambda: None,
        session_id=session_dir.name,
        execution_id="worker-execution",
        budget=meter,
        session_dir=session_dir,
        authority=child_launch.dispatch_authority(
            cfg, mode="run", session_id=session_dir.name, root=repo
        ),
    )


def run_panel_worker(url: str) -> None:
    """Run the internal panel seam in a worker process with real private launch provenance."""
    launch = child_launch.read()
    assert launch is not None
    directory = layout.bucket_dir(launch.record_state, "runs") / launch.session_id
    process.write_worker_pid(directory, os.getpid())
    meter = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
    executor = dispatcher(launch.worktree, directory, config(url), meter)
    parallel_panel.dispatch_panel(
        executor,
        [
            child_records.ChildTask(
                "review",
                "reviewer",
                contract=kinds.ConversationContract(
                    system="s",
                    opening="o",
                    max_turns=2,
                ),
            )
        ],
        panel_id="blocking",
        at=git(launch.worktree, "rev-parse", "HEAD"),
    )


def run_panel_parent(directory: str, url: str) -> None:
    """Supervise the worker while the test stops this ancestor through its visible record."""
    process.write_worker_pid(pathlib.Path(directory), os.getpid())
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests import child_env; child_env.run_panel_worker(sys.argv[1])",
            url,
        ],
        check=False,
        start_new_session=True,
    )
