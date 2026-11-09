from __future__ import annotations

import json

import pytest

from agent6.sessions import layout
from agent6.viewmodel import policy, wire
from tests.unit import test_execution_evidence

saved = test_execution_evidence.saved
recorded = test_execution_evidence.recorded


@pytest.mark.parametrize("boundary", ["session.resume", "session.start", "session.stop"])
@pytest.mark.parametrize("execution_id", [None, "another-execution"])
def test_an_execution_without_a_target_cannot_borrow_one(
    recorded: layout.SessionLayout, boundary: str, execution_id: str | None
) -> None:
    """Missing target evidence at an execution boundary clears the displayed environment."""
    assert policy.session_policy(recorded.session_dir).environment
    with recorded.logs_path.open("a") as journal:
        journal.write(json.dumps({"type": boundary, "execution_id": execution_id}) + "\n")
    assert policy.session_policy(recorded.session_dir).environment == ()
    assert "Environment: local" not in wire.session_snapshot(recorded.session_dir)["policy"]


def test_a_manifest_cannot_supply_an_environment_without_its_journal(
    recorded: layout.SessionLayout,
) -> None:
    """An unreadable execution journal leaves the target unknown, even with mirrored history."""
    assert policy.session_policy(recorded.session_dir).environment
    recorded.logs_path.unlink()
    assert policy.session_policy(recorded.session_dir).environment == ()
    assert "Environment: local" not in wire.session_snapshot(recorded.session_dir)["policy"]


@pytest.mark.parametrize("damage", ["malformed", "conflicting", "unbound"])
def test_damaged_or_unbound_target_evidence_clears_the_displayed_environment(
    recorded: layout.SessionLayout, damage: str
) -> None:
    """A bad target record cannot leave an earlier known target displayed as current."""
    assert policy.session_policy(recorded.session_dir).environment
    event = next(
        row
        for line in recorded.logs_path.read_text().splitlines()
        if (row := json.loads(line))["type"] == "execution.target"
    )
    if damage == "malformed":
        event["target"] = None
    elif damage == "conflicting":
        event["target"]["home"] = "/different-home"
    else:
        event.pop("execution_id")
        event["target_id"] = "unbound"
    with recorded.logs_path.open("a") as journal:
        journal.write(json.dumps(event) + "\n")
    assert policy.session_policy(recorded.session_dir).environment == ()
    assert "Environment: local" not in wire.session_snapshot(recorded.session_dir)["policy"]
