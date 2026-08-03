# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run branch's manifest is found wherever its session lives.

`agent6/<id>` is cut by any session that forks, and a forked PLAN lives in
plans/. Reading the owning manifest out of runs/ alone breaks the base-branch
chain walk, so a run cut from that branch records an `agent6/*` branch as its
base -- and `sessions merge` then defaults to merging INTO a run branch.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent6.app.preflight import resolve_base_branch
from agent6.sessions.layout import SessionLayout


def _session(state: Path, bucket: str, session_id: str, base_branch: str) -> None:
    layout = SessionLayout(state_dir=state, session_id=session_id, subdir=bucket)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": session_id,
                "mode": "plan" if bucket == "plans" else "run",
                "user_task": "t",
                "base_branch": base_branch,
            }
        ),
        encoding="utf-8",
    )


def test_the_chain_walks_through_a_forked_plan(tmp_path: Path) -> None:
    # A run cut from a plan's branch, which was itself cut from master.
    _session(tmp_path, "plans", "planny-fork-AAAAAA", "master")
    _session(tmp_path, "runs", "runny-child-BBBBBB", "agent6/planny-fork-AAAAAA")

    assert resolve_base_branch(tmp_path, "agent6/runny-child-BBBBBB") == "master"


def test_a_plain_run_chain_still_resolves(tmp_path: Path) -> None:
    _session(tmp_path, "runs", "first-run-AAAAAA", "master")
    _session(tmp_path, "runs", "second-run-BBBBB", "agent6/first-run-AAAAAA")

    assert resolve_base_branch(tmp_path, "agent6/second-run-BBBBB") == "master"


def test_a_base_branch_is_returned_unchanged(tmp_path: Path) -> None:
    assert resolve_base_branch(tmp_path, "master") == "master"
