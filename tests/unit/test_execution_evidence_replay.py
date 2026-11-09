from __future__ import annotations

import copy
import json
import pathlib
import uuid

import pytest

from agent6.sessions import layout
from agent6.viewmodel import receipt, session_log
from tests.unit import test_execution_evidence

saved = test_execution_evidence.saved
resumed = test_execution_evidence.resumed


@pytest.mark.parametrize(
    "event_type,field,value",
    [
        ("execution.target", "target", None),
        ("workspace.setup", "duration_s", "not-a-number"),
        ("execution.jobs", "outcome", "not-a-state"),
        ("execution.environment", "outcome", "not-a-state"),
        ("execution.target", "home", "/conflicting-home"),
    ],
    ids=[
        "malformed-target",
        "malformed-setup",
        "malformed-jobs",
        "malformed-environment",
        "conflict",
    ],
)
def test_damaged_environment_evidence_keeps_history_but_cannot_certify(
    resumed: layout.SessionLayout, event_type: str, field: str, value: object
) -> None:
    """Bad observations retain diagnostics and original facts while later valid records replay."""
    before = receipt.read_receipt(resumed.session_dir, pathlib.Path.cwd())
    assert not before.evidence_notes
    assert before.end_tree and before.final_verify_id
    final = next(v for v in before.verifications if v.verify_id == before.final_verify_id)
    assert final.outcome == "passed" and final.exit_code == 0
    assert final.tree_before == final.tree_after == before.end_tree
    assert "Final tree passed" in receipt.render(before)
    assert before.targets and before.setups and before.jobs and before.environments

    rows = [json.loads(line) for line in resumed.logs_path.read_text().splitlines()]
    bad = copy.deepcopy(next(row for row in reversed(rows) if row["type"] == event_type))
    bad["event_id"] = uuid.uuid4().hex
    if field == "home":
        assert bad["target"][field] != value
        bad["target"][field] = value
        diagnostic = f"Conflicting target stamp: {bad['target_id']}"
    else:
        bad[field] = value
        diagnostic = field
    index = next(
        i
        for i, row in enumerate(rows)
        if row["type"] == "verify.start" and row["verify_id"] == before.final_verify_id
    )
    rows.insert(index, bad)
    resumed.logs_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    scan = session_log.scan_session_log(resumed.logs_path)
    after = receipt.read_receipt(resumed.session_dir, pathlib.Path.cwd())
    assert scan.finished and scan.last_type == rows[-1]["type"]
    assert after.execution_id == before.execution_id
    assert after.end_tree == before.end_tree
    assert after.verifications == before.verifications
    assert after.targets == before.targets
    assert after.setups == before.setups
    assert after.jobs == before.jobs
    assert after.environments == before.environments
    assert len(scan.evidence_notes) == 1
    assert scan.evidence_notes[0].startswith(f"journal line {index + 1}:")
    assert diagnostic in scan.evidence_notes[0]
    assert after.evidence_notes == scan.evidence_notes
    assert scan.final_verify_id is None and after.final_verify_id is None
    text = receipt.render(after)
    assert f"Final tree not certified: {before.end_tree}" in text
    assert "Final tree passed" not in text
    assert scan.evidence_notes[0] in text
