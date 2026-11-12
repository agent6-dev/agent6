# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from agent6 import budget, child_records, paths
from agent6.app import children, parallel_results, stop
from agent6.sessions import child_dispatch, layout, process
from agent6.viewmodel import receipt
from tests import child_env, launch_env


def test_stop_ancestor_interrupts_a_panel_in_the_shared_executor(
    short_repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping the visible ancestor kills a blocked real panel and preserves its obligation."""
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    parent = child_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    entered, release = threading.Event(), threading.Event()

    def block_reply(_body: dict[str, Any]) -> None:
        entered.set()
        release.wait(60)

    with launch_env.scripted_provider(before_reply=block_reply) as (url, calls):
        cfg = child_env.config(url)
        _path, launch = child_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker = child_env.create_record(repo, launch.session_id, parent="parent")
        result = child_records.ChildResult(
            child_records.ChildSpec(1, worker.name, clone, None),
            worker,
            "",
            False,
            "stopped",
        )
        ancestor = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
        leaf: pathlib.Path | None = None
        log = short_repo / "worker.log"
        with (
            log.open("w") as output,
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; from tests import child_env; "
                    "child_env.run_panel_parent(*sys.argv[1:])",
                    str(parent),
                    url,
                ],
                cwd=pathlib.Path(__file__).resolve().parents[2],
                env={**os.environ, "AGENT6_ALLOW_ROOT": "1", "AGENT6_STREAM_TO_LOG": "1"},
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as supervisor,
        ):
            try:
                assert entered.wait(30), log.read_text()
                record = child_dispatch.read_records(worker)[0]
                assert record.state == "running" and len(calls) == 1
                leaf = (
                    layout.bucket_dir(paths.state_dir(repo), "reviewers")
                    / record.children[0].session_id
                )
                deadline = time.monotonic() + 5
                while not all(process.worker_is_alive(d) for d in (parent, worker, leaf)):
                    assert time.monotonic() < deadline, (parent, worker, leaf, log.read_text())
                    time.sleep(0.02)
                assert leaf.is_dir() and not leaf.is_symlink(), "the leaf's record is here"
                stopped = stop.stop_session(parent, wait_s=0, grace_s=0.05)
                assert stopped.ok, stopped.message
                supervisor.wait(timeout=10)
                assert all(not process.worker_is_alive(d) for d in (parent, worker, leaf))
                imported = parallel_results.import_child(result, repo, None)
                assert imported.session_dir == parent.parent / worker.name
                persisted = child_dispatch.read_records(imported.session_dir)[0]
                assert persisted.identity == record.identity
                usage = children.child_usage(imported.session_dir)
                ancestor.settle_child(worker.name, usage)
                if usage is None:
                    assert not ancestor.snapshot().settled_children
                    assert ancestor.available() == budget.BudgetAllowance(5, 1000, 20)
                else:
                    assert persisted.state == "complete" and all(
                        c.settled for c in persisted.children
                    )
                    assert ancestor.snapshot().unmetered_tokens == usage.tokens
                    assert ancestor.estimate_usd()[0] == usage.usd
                    assert ancestor.snapshot().settled_children == frozenset({worker.name})
                retained = layout.bucket_dir(paths.state_dir(repo), "reviewers") / leaf.name
                assert receipt.read_receipt(retained, repo).session_id == leaf.name
            finally:
                release.set()
                for directory in (leaf, worker, parent):
                    if directory is not None:
                        stop.stop_session(directory, wait_s=0, grace_s=0.05)
                if supervisor.poll() is None:
                    supervisor.kill()
                supervisor.wait(timeout=10)
