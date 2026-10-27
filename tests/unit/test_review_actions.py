from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib

import pytest

from agent6 import event_log, review_records
from agent6.app import review_actions
from agent6.sessions import ipc, lock, process
from agent6.ui.cli import main
from agent6.viewmodel import _session_log
from tests import review_helpers


def _resolve(request_id: str = "resolve", version: int = 1) -> review_records.FindingAction:
    return review_records.FindingAction(
        request_id=request_id,
        expected_version=version,
        action="resolve",
        resolution=review_records.FindingResolution(
            disposition="fixed", reason="Reproduced and fixed"
        ),
    )


def test_idempotency_precedes_version_check_and_fixed_invents_no_verification(
    tmp_path: pathlib.Path,
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    first = review_actions.act(saved.session_dir, "f1", _resolve(), source="cli")
    before = saved.logs_path.read_bytes()
    assert review_actions.act(saved.session_dir, "f1", _resolve(), source="cli") == first
    assert saved.logs_path.read_bytes() == before
    view = review_actions.read_review(saved.session_dir)
    assert first.version == view.version == 2
    finding = view.findings[0]
    assert finding.status == "resolved"
    assert finding.resolution is not None and finding.resolution.verify_id is None
    assert finding.actions[-1].source == "cli"
    assert _session_log.scan_session_log(saved.logs_path).last_verify_rc is None


def test_stale_version_and_cross_finding_request_reuse_are_conflicts(
    tmp_path: pathlib.Path,
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    review_actions.act(saved.session_dir, "f1", _resolve(), source="cli")
    for action in (_resolve("another"), _resolve()):
        with pytest.raises(review_actions.ReviewActionError) as caught:
            review_actions.act(saved.session_dir, "f2", action, source="web")
        assert caught.value.status == 409
    assert review_actions.read_review(saved.session_dir).version == 2


def test_reopen_records_its_reason_and_requires_current_version(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    review_actions.act(saved.session_dir, "f1", _resolve(), source="cli")
    request = review_records.FindingAction(
        request_id="reopen", expected_version=2, action="reopen", reason="Still reproducible"
    )
    review_actions.act(saved.session_dir, "f1", request, source="web")
    finding = review_actions.read_review(saved.session_dir).findings[0]
    assert finding.status == "open" and finding.resolution is None
    assert finding.actions[-1].reason == "Still reproducible"


@pytest.mark.parametrize("duplicate_of", ["f1", "elsewhere"])
def test_duplicate_must_name_another_finding_in_this_session(
    tmp_path: pathlib.Path, duplicate_of: str
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    action = _resolve().model_copy(
        update={
            "resolution": review_records.FindingResolution(
                disposition="duplicate", reason="Same problem", duplicate_of=duplicate_of
            )
        }
    )
    with pytest.raises(review_actions.ReviewActionError) as caught:
        review_actions.act(saved.session_dir, "f1", action, source="cli")
    assert caught.value.status == 422
    assert review_actions.read_review(saved.session_dir).version == 1


def test_concurrent_dispositions_use_the_existing_worker_lock(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)

    def resolve(name: str) -> bool:
        try:
            review_actions.act(saved.session_dir, name, _resolve(name), source="cli")
        except review_actions.ReviewActionError as exc:
            assert exc.status == 409
            return False
        return True

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(resolve, ("f1", "f2"))) == [False, True]
    view = review_actions.read_review(saved.session_dir)
    assert view.version == 2 and view.read_failure is None


def test_live_actions_are_applied_only_by_the_worker(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    fd = lock.acquire_single_writer(saved.session_dir)
    process.write_worker_pid(saved.session_dir, os.getpid())
    before = saved.logs_path.read_bytes()
    try:
        queued = review_actions.act(saved.session_dir, "f1", _resolve(), source="web", wait_s=0)
        assert queued.status == "queued"
        assert saved.logs_path.read_bytes() == before
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
        result = review_actions.act(saved.session_dir, "f1", _resolve(), source="web", wait_s=0)
        assert result.status == "resolved"
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)


def test_fix_uses_stored_evidence_and_refuses_a_second_pending_steer(
    tmp_path: pathlib.Path,
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())
    try:
        request = review_records.FindingAction(request_id="fix", expected_version=1, action="fix")
        review_actions.act(saved.session_dir, "f1", request, source="cli", wait_s=0)
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
        text = (saved.session_dir / ipc.STEER_ANSWER_FILE).read_text()
        for evidence in (
            "f1",
            "bt",
            "ht",
            "head",
            "parser.py:3-4",
            "block",
            "Defect f1",
            "Reproduce",
        ):
            assert evidence in text
        view = review_actions.read_review(saved.session_dir)
        second = request.model_copy(update={"request_id": "fix2", "expected_version": view.version})
        with pytest.raises(review_actions.ReviewActionError, match="pending"):
            review_actions.act(saved.session_dir, "f2", second, source="web", wait_s=0)
        assert (saved.session_dir / ipc.STEER_ANSWER_FILE).read_text() == text
        assert view.findings[0].fix_state == "submitted"
        assert view.findings[0].status == "open"
        rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
        assert [row["version"] for row in rows[-2:]] == [2, 3]
        assert rows[-1]["execution_id"] == "live"
    finally:
        process.clear_worker_pid(saved.session_dir)


def test_crash_after_request_never_retries_delivery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())

    def crash(*args: object, **kwargs: object) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(ipc, "submit_steer", crash)
    request = review_records.FindingAction(request_id="fix", expected_version=1, action="fix")
    try:
        review_actions.act(saved.session_dir, "f1", request, source="cli", wait_s=0)
        with pytest.raises(KeyboardInterrupt):
            review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
    finally:
        process.clear_worker_pid(saved.session_dir)
    result = review_actions.act(saved.session_dir, "f1", request, source="cli")
    assert result.status == "delivery_unknown"
    assert "delivery unknown" in result.message
    assert review_actions.read_review(saved.session_dir).findings[0].status == "open"


def test_finished_fix_requires_explicit_resume_and_a_destination(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    request = review_records.FindingAction(request_id="fix", expected_version=1, action="fix")
    with pytest.raises(review_actions.ReviewActionError, match="Resume to fix"):
        review_actions.act(saved.session_dir, "f1", request, source="cli")
    with pytest.raises(review_actions.ReviewActionError, match="resume"):
        review_actions.act(
            saved.session_dir, "f1", request.model_copy(update={"resume": True}), source="cli"
        )
    assert review_actions.read_review(saved.session_dir).version == 1


def test_cli_resolve_reopen_and_findings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "sessions",
                "resolve",
                saved.session_id,
                "f1",
                "--as",
                "fixed",
                "--reason",
                "Corrected",
                "--request-id",
                "cli-1",
                "--expected-version",
                "1",
            ]
        )
        == 0
    )
    assert main(["sessions", "findings", saved.session_id, "--json"]) == 0
    output = capsys.readouterr().out
    assert '"finding_id": "f2"' in output
    assert '"finding_id": "f1"' not in output
    assert main(["sessions", "reopen", saved.session_id, "f1", "--reason", "Still occurs"]) == 0
    assert review_actions.read_review(saved.session_dir).findings[0].status == "open"


def test_cli_finding_fix_refuses_a_finished_run_without_starting_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["steer", saved.session_id, "--finding", "f1"]) == 2
    assert "Resume to fix" in capsys.readouterr().err
    assert review_actions.read_review(saved.session_dir).version == 1


def test_worker_rechecks_versions_for_two_queued_dispositions(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())
    try:
        for name in ("f1", "f2"):
            result = review_actions.act(
                saved.session_dir, name, _resolve(name), source="web", wait_s=0
            )
            assert result.status == "queued"
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="worker")
        view = review_actions.read_review(saved.session_dir)
        assert view.version == 2 and view.read_failure is None
        assert [f.status for f in view.findings] == ["resolved", "open"]
        reply = ipc.read_review_reply(saved.session_dir, "f2")
        assert reply is not None
        assert json.loads(reply)["status"] == 409
    finally:
        process.clear_worker_pid(saved.session_dir)


def test_failed_fix_records_delivery_failure_without_resolving(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())

    def refuse(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(ipc, "submit_steer", refuse)
    request = review_records.FindingAction(request_id="failed", expected_version=1, action="fix")
    try:
        review_actions.act(saved.session_dir, "f1", request, source="cli", wait_s=0)
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="worker")
    finally:
        process.clear_worker_pid(saved.session_dir)
    result = review_actions.act(saved.session_dir, "f1", request, source="cli")
    assert result.status == "failed" and result.version == 3
    assert review_actions.read_review(saved.session_dir).findings[0].status == "open"


def test_resume_failure_is_recorded_and_not_retried(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    request = review_records.FindingAction(
        request_id="resume", expected_version=1, action="fix", resume=True
    )
    attempts: list[str] = []

    def resume(text: str) -> str:
        attempts.append(text)
        return "no resumable snapshot"

    first = review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume)
    assert first.status == "failed" and first.message == "no resumable snapshot"
    assert (
        review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume) == first
    )
    assert len(attempts) == 1
