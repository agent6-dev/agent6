from __future__ import annotations

from typing import Any

import pytest

from agent6.viewmodel import transcript, transcript_style


def attempt(*, model: bool = False, verify_id: str = "v1") -> list[dict[str, Any]]:
    """Return the journal order of an approved verification attempt."""
    call = {"call_id": 7} if model else {}
    rows: list[dict[str, Any]] = []
    if model:
        rows.append({"type": "tool.call", "name": "run_verify_command", "args_preview": {}, **call})
    rows.extend(
        [
            {
                "type": "verify.start",
                "verify_id": verify_id,
                "cmd": ["pytest", "-q"],
                "approval_required": True,
                **call,
            },
            {
                "type": "approval.prompt",
                "id": "approval-1",
                "prompt": "Allow run_verify_command: pytest -q",
                **call,
            },
            {"type": "approval.answer", "id": "approval-1", "approved": True},
            {
                "type": "verify.end",
                "verify_id": verify_id,
                "cmd": ["pytest", "-q"],
                "exit_code": 0,
                "outcome": "passed",
                "duration_s": 0.2,
            },
        ]
    )
    if model:
        rows.append({"type": "tool.result", "name": "run_verify_command", "ok": True, **call})
    return rows


@pytest.mark.parametrize("model", [False, True])
def test_one_verification_item_moves_from_approval_to_running_to_result(model: bool) -> None:
    """The harness and model gate each occupy one item, including before the approval prompt."""
    rows = attempt(model=model)
    offset = int(model)
    identities: set[str] = set()
    for length, text in [
        (1 + offset, "verify: waiting for approval"),
        (2 + offset, "verify: waiting for approval"),
        (3 + offset, "verify: running `pytest -q`"),
        (4 + offset, "verify: ✓ pass · 0.2s"),
        (len(rows), "verify: ✓ pass · 0.2s"),
    ]:
        (item,) = transcript.fold_transcript(rows[:length])
        assert item.kind == "verify"
        assert item.body == text
        assert text in transcript.restate(
            [{"type": "session.start", "user_task": "Verify"}, *rows[:length]]
        )
        identities.add(item.call_id)
        for detail in ("hidden", "collapsed", "expanded"):
            rendered = "".join(
                span
                for line in transcript_style.item_lines(item, detail=detail)
                for span, _ in line
            )
            assert text in rendered
    assert len(identities) == 1


def test_failed_verification_keeps_its_output_tail() -> None:
    """Replacing the model tool result with a verification item keeps the failure's output."""
    rows = attempt()
    rows[-1].update(exit_code=1, outcome="failed", stderr_tail="AssertionError: expected 2")
    (item,) = transcript.fold_transcript(rows)
    assert item.tail == "AssertionError: expected 2"
    expanded = transcript_style.item_lines(item, detail="expanded")
    assert "AssertionError: expected 2" in "".join(t for line in expanded for t, _ in line)


def test_web_payload_uses_the_shared_verification_lines() -> None:
    """The web payload has one verification entry at every phase, including when tools hide."""
    from agent6.ui.web import model

    rows = attempt()
    for length in range(1, len(rows) + 1):
        (entry,) = model.conversation_items(rows[:length])
        (item,) = transcript.fold_transcript(rows[:length])
        assert entry["kind"] == "verify"
        assert entry["lines"] == transcript_style.item_lines(item, detail="collapsed")


def test_repeated_and_idless_verification_attempts_stay_distinct() -> None:
    """Old journals without attempt ids and repeated commands still yield one item per attempt."""
    rows = attempt() + attempt(verify_id="v2")
    assert len(transcript.fold_transcript(rows)) == 2
    for row in rows:
        row.pop("verify_id", None)
    assert len(transcript.fold_transcript(rows)) == 2


def test_verification_without_approval_shows_the_command_at_start() -> None:
    """An allowed command never claims to be waiting for an operator."""
    (item,) = transcript.fold_transcript(
        [{"type": "verify.start", "cmd": ["echo", "two words"], "approval_required": False}]
    )
    assert item.body == "verify: running `echo 'two words'`"


def test_denial_and_worker_death_do_not_leave_a_running_verification() -> None:
    """A denial or dead worker replaces the pending verification without adding another item."""
    rows = attempt()[:2]
    denied = [
        *rows,
        {"type": "approval.answer", "id": "approval-1", "approved": False},
        {
            "type": "verify.end",
            "verify_id": "v1",
            "outcome": "denied",
            "reason": "not approved",
            "exit_code": None,
        },
    ]
    (item,) = transcript.fold_transcript(denied)
    assert "denied" in item.body and "not approved" in item.body
    assert item.ok is False
    (item,) = transcript.fold_transcript(rows, worker_dead=True)
    assert item.ok is False and "no result (the run died)" in item.body
