# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The container a run holds reads from the journal: starting, then the lease, then its end."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from agent6.sessions import journal, manifest
from agent6.viewmodel import events, policy, state, state_types
from agent6.viewmodel.session import core as session_core
from agent6.viewmodel.session import log as session_log

_START: dict[str, Any] = {"type": "session.start", "execution_id": "e1", "user_task": "t"}
_STARTING: dict[str, Any] = {
    "type": "environment.starting",
    "execution_id": "e1",
    "command": ["devcontainer", "up"],
}
_ACQUIRED: dict[str, Any] = {
    "type": "environment.acquired",
    "execution_id": "e1",
    "lifecycle_ref": "k",
    "role": "owner",
    "reason": "started by this run; owned",
    "started": True,
    "start_s": 42.4,
}
_END: dict[str, Any] = {
    "type": "execution.environment",
    "execution_id": "e1",
    "target_id": "t",
    "outcome": "stopped",
    "reason": "owned; stopped",
}


def _journal(tmp_path: pathlib.Path, *rows: dict[str, Any]) -> pathlib.Path:
    logs = tmp_path / "logs.jsonl"
    with logs.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return logs


def test_the_fold_follows_the_container_through_start_lease_and_end() -> None:
    starting = state.fold_session([_START, _STARTING])
    assert starting.environment == state_types.EnvironmentView(
        execution_id="e1", starting=("devcontainer", "up")
    )
    held = state.fold_session([_START, _STARTING, _ACQUIRED])
    assert held.environment == state_types.EnvironmentView(
        execution_id="e1",
        role="owner",
        reason="started by this run; owned",
        started=True,
        start_s=42.4,
    )
    # The real order: the lease rows precede the start, which keeps its own execution's view.
    assert state.fold_session([_STARTING, _ACQUIRED, _START]).environment == held.environment
    other = {**_START, "execution_id": "e2"}
    assert state.fold_session([_STARTING, _ACQUIRED, other]).environment is None
    ended = state.fold_session([_START, _STARTING, _ACQUIRED, _END])
    assert ended.environment is not None and ended.environment.outcome == "stopped"
    resumed = state.fold_session(
        [_START, _ACQUIRED, _END, {"type": "session.resume", "execution_id": "e2"}]
    )
    assert resumed.environment is None


def test_the_tail_read_agrees_with_the_full_fold(tmp_path: pathlib.Path) -> None:
    """`sessions show` and the TUI header read the tail; the viewer folds everything."""
    logs = _journal(tmp_path, _START, _STARTING)
    assert (
        session_log.current_environment(logs) == state.fold_session([_START, _STARTING]).environment
    )
    _journal(tmp_path, _ACQUIRED, _END)
    full = session_log.scan_session_log(logs)
    assert session_log.current_environment(logs) == full.environment
    assert full.environment is not None and full.environment.outcome == "stopped"
    assert [a.lifecycle_ref for a in full.acquisitions] == ["k"]
    _journal(tmp_path, {"type": "session.resume", "execution_id": "e2"})
    assert session_log.current_environment(logs) is None
    assert session_log.scan_session_log(logs).environment is None


def test_a_lease_journaled_before_the_start_belongs_to_that_execution(
    tmp_path: pathlib.Path,
) -> None:
    """The console prints the policy row between the lease rows and `session.start`."""
    logs = _journal(tmp_path, _STARTING, _ACQUIRED)
    before = session_log.current_environment(logs)
    assert before is not None and before.role == "owner"
    _journal(tmp_path, _START)
    assert session_log.current_environment(logs) == before
    assert session_log.scan_session_log(logs).environment == before
    _journal(tmp_path, {"type": "session.stop", "execution_id": "e1", "reason": "finished"}, _END)
    assert session_log.current_environment(logs) == session_log.scan_session_log(logs).environment
    _journal(tmp_path, {**_ACQUIRED, "execution_id": "e2", "role": "borrower"})
    _journal(tmp_path, {"type": "session.resume", "execution_id": "e2"})
    now = session_log.current_environment(logs)
    assert now is not None and now.role == "borrower" and now.outcome == ""
    assert session_log.scan_session_log(logs).environment == now


def test_the_policy_row_names_the_container(tmp_path: pathlib.Path) -> None:
    assert policy.environment_lines(None) == ()
    assert policy.environment_lines(state_types.EnvironmentView()) == ()
    assert policy.environment_lines(state_types.EnvironmentView(starting=("up", "-d"))) == (
        "Container: starting (up -d)",
    )
    held = state_types.EnvironmentView(role="owner", reason="started by this run; owned")
    assert policy.environment_lines(held) == ("Container: owner (started by this run; owned)",)
    timed = state_types.EnvironmentView(
        role="owner", reason="started by this run; owned", started=True, start_s=42.4
    )
    assert policy.environment_lines(timed) == (
        "Container: owner (started by this run; owned); started by this run in 42s",
    )
    left = state_types.EnvironmentView(
        role="borrower", reason="already running; attached", outcome="left_running"
    )
    assert policy.environment_lines(left) == (
        "Container: borrower (already running; attached); left running",
    )
    _journal(tmp_path, _START, _ACQUIRED)
    (tmp_path / "manifest.json").write_text(
        manifest.SessionManifest(session_id="r").model_dump_json(), encoding="utf-8"
    )
    assert "Container: owner" in policy.session_policy(tmp_path).line()
    recorded = manifest.SessionManifest(session_id="r")
    assert policy.manifest_policy(recorded, environment=left).environment == (
        "Container: borrower (already running; attached); left running",
    )


def test_the_records_round_trip_through_the_reader() -> None:
    assert events.parse_event(_STARTING) == journal.EnvironmentStarting(
        execution_id="e1", command=("devcontainer", "up")
    )
    assert session_core.held_environment(
        journal.EnvironmentStarting(execution_id="e1", command=("up",)), "stopped"
    ) == state_types.EnvironmentView(execution_id="e1", starting=("up",), outcome="stopped")
    assert events.parse_event(_ACQUIRED) == journal.EnvironmentAcquired(
        execution_id="e1",
        lifecycle_ref="k",
        role="owner",
        reason="started by this run; owned",
        started=True,
        start_s=42.4,
    )
    assert events.parse_event(_END) == journal.ExecutionEnvironment(
        execution_id="e1", target_id="t", outcome="stopped", reason="owned; stopped"
    )
    assert events.parse_event({"type": "environment.acquired"}) == journal.EnvironmentAcquired(
        execution_id=None, lifecycle_ref="", role="", reason="", started=False, start_s=None
    )
