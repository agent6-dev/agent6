# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Real repositories and private launches for subordinate lane tests."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import pytest

from agent6 import budget, kinds, lane_records, paths
from agent6.app import (
    lane_launch,
    lanes,
    workdirs,
)
from agent6.config import Config
from agent6.harness import parallel_dispatch, run_chain, subrun
from agent6.sessions import (
    event_log,
    layout,
    process,
)
from agent6.ui.cli import parallel
from agent6.vcs import git_read


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
        data["parallel"] = {"coordinator": parent, "group": parent + "-p1", "lane": 1}
    (dest / "manifest.json").write_text(json.dumps(data))
    (dest / "logs.jsonl").write_text(json.dumps({"type": "session.start", "mode": mode}) + "\n")
    return dest


def config(url: str = "http://127.0.0.1:1/v1", *, tool: bool = False) -> Config:
    """Configure a local provider and subordinate review authority."""
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
) -> tuple[pathlib.Path, lane_launch.LaneLaunch]:
    """Create a private worker launch and install its provenance in the environment."""
    subrun.clone_workspace(repo, clone)
    sid = "parent-p1-l1"
    spec = lane_records.LaneSpec(1, sid, clone, None)
    into = lane_launch.launch_dir(paths.state_dir(repo), "parent-p1")
    config = workdirs.write_lane_config(cfg, spec, explicit_leaves=frozenset(), into=into)
    launch = lane_launch.LaneLaunch(
        workdir=clone,
        session_id=sid,
        task="worker",
        config=config.path,
        coordinator="parent",
        group="parent-p1",
        lane=1,
        limits=lane_records.LaneLimits(max_turns=20, deadline_at=time.time() + 90),
    )
    path = lane_launch.write(into, launch)
    monkeypatch.setenv(lane_launch.ENV, str(path))
    monkeypatch.setenv("AGENT6_SUBRUN", "1")
    return path, launch


def dispatcher(
    repo: pathlib.Path, session_dir: pathlib.Path, cfg: Config, meter: budget.BudgetTracker
) -> parallel_dispatch.ParallelDispatcher:
    """Build the common dispatcher with the worker launch authority."""
    spawner = lanes.build_coordinator_spawner(
        cfg,
        repo,
        paths.state_dir(repo),
        explicit_leaves=frozenset(),
        mode="run",
        session_id=session_dir.name,
        runtime=parallel.lane_runtime(),
    )
    assert spawner is not None, "a trusted worker retains the common executor for its panel"
    return parallel_dispatch.ParallelDispatcher(
        chain=run_chain.RunChain(
            repo,
            ref=git_read.chain_ref_for(session_dir.name),
            fallback_parent=git(repo, "rev-parse", "HEAD"),
        ),
        curator=None,
        max_lanes=4,
        lane_spawner=spawner,
        save_snapshot=lambda *_a, **_k: None,
        log=lambda _m: None,
        emit=event_log.EventSink(session_dir / "logs.jsonl").emit,
        emit_graph_snapshot=lambda: None,
        session_id=session_dir.name,
        execution_id="worker-execution",
        budget=meter,
        session_dir=session_dir,
        authority=lane_launch.dispatch_authority(
            cfg, mode="run", session_id=session_dir.name, root=repo
        ),
    )


def run_panel_worker(url: str) -> None:
    """Run the internal panel seam in a worker process with real private launch provenance."""
    launch = lane_launch.read()
    assert launch is not None
    directory = layout.bucket_dir(paths.state_dir(launch.workdir), "runs") / launch.session_id
    process.write_worker_pid(directory, os.getpid())
    meter = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
    executor = dispatcher(launch.workdir, directory, config(url), meter)
    executor.dispatch_panel(
        [
            lane_records.LaneTask(
                "review",
                "reviewer",
                contract=kinds.ConversationContract(
                    kind="read_only",
                    system="s",
                    opening="o",
                    max_calls=2,
                    max_output_tokens=100,
                ),
            )
        ],
        panel_id="blocking",
        at=git(launch.workdir, "rev-parse", "HEAD"),
    )


def run_panel_parent(directory: str, url: str) -> None:
    """Supervise the worker while the test stops this ancestor through its visible record."""
    process.write_worker_pid(pathlib.Path(directory), os.getpid())
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests import subordinate_env; "
            "subordinate_env.run_panel_worker(sys.argv[1])",
            url,
        ],
        check=False,
        start_new_session=True,
    )
