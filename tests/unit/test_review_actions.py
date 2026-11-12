from __future__ import annotations

import dataclasses
import json
import os
import pathlib

import pytest

from agent6 import review_records
from agent6.app.review import actions as review_actions
from agent6.sessions import event_log, ipc, journal, lock, process
from agent6.ui.cli import main
from agent6.viewmodel.session import log as session_log
from tests import review_helpers


def test_cli_lists_line_evidence_without_lowering_severity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = review_helpers.reviewed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["sessions", "findings", saved.session_id]) == 0
    text = capsys.readouterr().out
    assert text.count("Pre-existing line") == 1
    assert "old.txt:2" in text and text.count("  block  ") == 4
    assert "Blame:" in text
    assert main(["sessions", "findings", saved.session_id, "--json"]) == 0
    wire = json.loads(capsys.readouterr().out)
    assert wire["findings"][0]["line_origin"]["kind"] == "pre_existing"
    assert wire["findings"][2]["line_origin"]["kind"] == "changed"


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
    assert session_log.scan_session_log(saved.logs_path).last_verify_rc is None


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


def test_offline_action_refuses_a_held_writer_lock_without_changing_the_journal(
    tmp_path: pathlib.Path,
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    fd = lock.acquire_single_writer(saved.session_dir)
    assert fd is not None
    before = saved.logs_path.read_bytes()
    try:
        with pytest.raises(review_actions.ReviewActionError, match="writer is busy") as caught:
            review_actions.act(saved.session_dir, "f1", _resolve(), source="cli")
        assert caught.value.status == 409
        assert saved.logs_path.read_bytes() == before
    finally:
        lock.release_single_writer(fd)
    result = review_actions.act(saved.session_dir, "f1", _resolve(), source="cli")
    assert result.status == "resolved" and result.version == 2


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


@pytest.mark.parametrize("resume_confirmed", [False, True])
def test_fix_uses_stored_evidence_and_queues_a_second_fix(
    tmp_path: pathlib.Path, resume_confirmed: bool
) -> None:
    """A Fix names its finding; a Fix of another finding queues behind it, not against it."""
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())
    try:
        request = review_records.FindingAction(
            request_id="fix", expected_version=1, action="fix", resume=resume_confirmed
        )

        def resume(text: str) -> str:
            pytest.fail("a live Fix must not launch another worker")

        review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume, wait_s=0)
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
        result = review_actions.act(
            saved.session_dir, "f1", request, source="cli", resume=resume, wait_s=0
        )
        assert result.status == "submitted" and result.message == "steer submitted"
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
        assert view.findings[0].fix_state == "submitted"
        assert view.findings[0].status == "open"
        rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
        assert [row["version"] for row in rows[-2:]] == [2, 3]
        assert rows[-1]["execution_id"] == "live"
        # The version the caller read does not bind a Fix: the finding does.
        second = request.model_copy(update={"request_id": "fix2", "expected_version": 1})
        review_actions.act(saved.session_dir, "f2", second, source="web", wait_s=0)
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
        queued = review_actions.act(saved.session_dir, "f2", second, source="web", wait_s=0)
        assert queued.status == "submitted"
        assert (saved.session_dir / ipc.STEER_ANSWER_FILE).read_text() == text
        (entry,) = ipc.queued_steers(saved.session_dir)
        assert entry.findings == ("f2",) and "Review finding f2" in entry.text
        # A second Fix of a finding whose Fix still waits is refused.
        third = request.model_copy(update={"request_id": "fix3"})
        with pytest.raises(review_actions.ReviewActionError, match="already queued"):
            review_actions.act(saved.session_dir, "f1", third, source="web", wait_s=0)
    finally:
        process.clear_worker_pid(saved.session_dir)


def test_an_ask_session_reads_and_resolves_its_findings_but_cannot_fix(
    tmp_path: pathlib.Path,
) -> None:
    """A `/review` ask has no edit tools: Fix is not offered and a request names the run."""
    from agent6.app import stamps
    from agent6.sessions import manifest

    saved = review_helpers.reviewed_session(tmp_path)
    stamps.write_manifest(
        saved.manifest_path, manifest.SessionManifest(session_id=saved.session_id, mode="ask")
    )
    view = review_actions.read_review(saved.session_dir)
    assert view.findings and review_actions.fixable_findings(saved.session_dir, view) == ()
    request = review_records.FindingAction(
        request_id="fix", expected_version=1, action="fix", resume=True
    )
    with pytest.raises(review_actions.ReviewActionError, match="ask session has no edit tools"):
        review_actions.act(
            saved.session_dir, "f1", request, source="cli", resume=lambda _t: "", wait_s=0
        )
    resolve = review_records.FindingAction(
        request_id="resolve",
        expected_version=1,
        action="resolve",
        resolution=review_records.FindingResolution(disposition="fixed", reason="by hand"),
    )
    review_actions.act(saved.session_dir, "f1", resolve, source="cli", wait_s=0)
    assert review_actions.read_review(saved.session_dir).findings[0].status == "resolved"


def test_one_steer_carries_several_findings_with_a_result_each(tmp_path: pathlib.Path) -> None:
    """A batch Fix writes one instruction naming every finding and records each one's request."""
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())
    try:
        request = review_records.FindingAction(request_id="both", expected_version=1, action="fix")
        view = review_actions.read_review(saved.session_dir)
        assert review_actions.fixable_findings(saved.session_dir, view) == ("f1", "f2")
        review_actions.act_many(saved.session_dir, ("f1", "f2"), request, source="cli", wait_s=0)
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
        results = review_actions.act_many(
            saved.session_dir, ("f1", "f2"), request, source="cli", wait_s=0
        )
        assert [(r.finding_id, r.status) for r in results] == [
            ("f1", "submitted"),
            ("f2", "submitted"),
        ]
        text = (saved.session_dir / ipc.STEER_ANSWER_FILE).read_text()
        assert text.startswith("Review findings f1, f2: each report below is one finding")
        assert text.count("Reproduce this report before changing code.") == 2
        assert "Review finding f1\n" in text and "Review finding f2\n" in text
        view = review_actions.read_review(saved.session_dir)
        assert [f.fix_state for f in view.findings] == ["submitted", "submitted"]
        assert review_actions.fixable_findings(saved.session_dir, view) == ()
        with pytest.raises(review_actions.ReviewActionError, match="already queued \\(f2\\)"):
            review_actions.act_many(
                saved.session_dir,
                ("f2",),
                request.model_copy(update={"request_id": "again"}),
                source="cli",
                wait_s=0,
            )
        with pytest.raises(review_actions.ReviewActionError, match="named twice"):
            review_actions.act_many(
                saved.session_dir,
                ("f1", "f1"),
                request.model_copy(update={"request_id": "twice"}),
                source="cli",
                wait_s=0,
            )
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


def _alive(_session_dir: pathlib.Path) -> bool:
    return True


def _dead(_session_dir: pathlib.Path) -> bool:
    return False


def test_a_live_fix_in_delivery_reads_queued_not_unknown(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Fix seen between the worker's "requested" and "submitted" records is not unknown."""
    saved = review_helpers.reviewed_session(tmp_path)
    view = review_actions.read_review(saved.session_dir)
    finding = view.findings[0]
    request = review_actions.ActionRequest(
        finding_ids=(finding.finding_id,),
        source="cli",
        action=review_records.FindingAction(
            request_id="live", expected_version=view.version, action="fix"
        ),
    )
    events = event_log.EventSink(saved.logs_path)
    review_actions._emit(  # pyright: ignore[reportPrivateUsage]
        events,
        request,
        "fix.requested",
        finding_id=finding.finding_id,
        target=dataclasses.asdict(
            review_actions._target(view, finding)  # pyright: ignore[reportPrivateUsage]
        ),
        action="live-steer",
    )
    monkeypatch.setattr(process, "worker_is_alive", _alive)
    (result,) = review_actions._wait(  # pyright: ignore[reportPrivateUsage]
        saved.session_dir, request, 0.2
    )
    assert result.status == "queued"
    assert "queued" in result.message and "unknown" not in result.message
    monkeypatch.setattr(process, "worker_is_alive", _dead)
    (dead,) = review_actions._wait(  # pyright: ignore[reportPrivateUsage]
        saved.session_dir, request, 0.2
    )
    assert dead.status == "delivery_unknown"


def test_finished_fix_requires_explicit_resume_and_a_destination(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    request = review_records.FindingAction(request_id="fix", expected_version=1, action="fix")
    for source in ("cli", "web"):
        with pytest.raises(review_actions.ReviewActionError, match="Resume to fix") as caught:
            review_actions.act(saved.session_dir, "f1", request, source=source)
        assert '`--resume` on the CLI, `"resume": true` over HTTP' in str(caught.value)
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


def test_rejected_resolve_is_not_requeued_at_a_later_version(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    fd = lock.acquire_single_writer(saved.session_dir)
    assert fd is not None
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
        before = saved.logs_path.read_bytes()
        with pytest.raises(
            review_actions.ReviewActionError, match="stale review version"
        ) as caught:
            review_actions.act(
                saved.session_dir, "f2", _resolve("f2", version=2), source="web", wait_s=0
            )
        assert caught.value.status == 409
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="worker")
        assert saved.logs_path.read_bytes() == before
        assert ipc.review_requests(saved.session_dir) == []
        assert ipc.read_review_reply(saved.session_dir, "f2") == reply
        result = review_actions.act(
            saved.session_dir, "f2", _resolve("fresh", version=2), source="web", wait_s=0
        )
        assert result.status == "queued"
        review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="worker")
        assert review_actions.read_review(saved.session_dir).findings[1].status == "resolved"
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)


def test_a_fix_queues_behind_an_ordinary_steer_and_follows_it_into_the_slot(
    tmp_path: pathlib.Path,
) -> None:
    """A Fix sent while a steer waits is submitted, not refused; it reaches the model next."""
    saved = review_helpers.reviewed_session(tmp_path)
    fd = lock.acquire_single_writer(saved.session_dir)
    assert fd is not None
    process.write_worker_pid(saved.session_dir, os.getpid())
    request = review_records.FindingAction(request_id="fix", expected_version=1, action="fix")
    try:
        queued = review_actions.act(saved.session_dir, "f1", request, source="cli", wait_s=0)
        assert queued.status == "queued"
        assert ipc.submit_steer(saved.session_dir, "ordinary steer")
        events = event_log.EventSink(saved.logs_path)
        review_actions.drain(events, execution_id="worker")
        reply = ipc.read_review_reply(saved.session_dir, "fix")
        assert reply is not None and json.loads(reply)["results"][0]["status"] == "submitted"
        assert ipc.take_steer_answer(saved.session_dir) == "ordinary steer"
        ipc.clear_steer_request(saved.session_dir)
        assert ipc.promote_queued_steer(saved.session_dir)
        assert "Review finding f1" in (saved.session_dir / ipc.STEER_ANSWER_FILE).read_text()
        before = saved.logs_path.read_bytes()
        retry = review_actions.act(saved.session_dir, "f1", request, source="cli", wait_s=0)
        assert retry.status == "submitted"
        assert saved.logs_path.read_bytes() == before
        assert ipc.queued_steers(saved.session_dir) == ()
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)


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


@pytest.mark.parametrize("acknowledge_before_return", [False, True])
def test_resume_receipt_distinguishes_launch_from_worker_acknowledgement(
    tmp_path: pathlib.Path, acknowledge_before_return: bool
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    request = review_records.FindingAction(
        request_id="resume", expected_version=1, action="fix", resume=True
    )
    steers: list[str] = []
    fd: int | None = None
    events = event_log.EventSink(saved.logs_path)

    def resume(text: str) -> str:
        nonlocal fd
        fd = lock.acquire_single_writer(saved.session_dir)
        assert fd is not None
        process.write_worker_pid(saved.session_dir, os.getpid())
        steers.append(text)
        if acknowledge_before_return:
            review_actions.drain(events, execution_id="resumed", resume_steer=text)
        return ""

    try:
        result = review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume)
        view = review_actions.read_review(saved.session_dir)
        finding = view.findings[0]
        expected_types = ["review.fix.requested"]
        if acknowledge_before_return:
            expected_types.append("review.fix.submitted")
        assert [action.type for action in finding.actions] == expected_types
        assert finding.actions[0].action == "resume"
        assert finding.status == "open"
        assert result.request_id == "resume" and result.finding_id == "f1"
        assert result.version == view.version == (3 if acknowledge_before_return else 2)
        assert result.message == "run resumed to fix finding f1" + (
            "" if acknowledge_before_return else "; submission acknowledgement pending"
        )
        assert result.status == ("submitted" if acknowledge_before_return else "queued")
        before = saved.logs_path.read_bytes()
        retry = review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume)
        assert retry.status == ("submitted" if acknowledge_before_return else "delivery_unknown")
        assert saved.logs_path.read_bytes() == before
        review_actions.drain(events, execution_id="resumed", resume_steer=steers[0])
        acknowledged = review_actions.act(
            saved.session_dir, "f1", request, source="cli", resume=resume
        )
        assert acknowledged.status == "submitted" and acknowledged.version == 3
        assert acknowledged.message == "run resumed to fix finding f1"
        finding = review_actions.read_review(saved.session_dir).findings[0]
        assert finding.status == "open"
        assert len(finding.actions) == 2 and finding.actions[-1].execution_id == "resumed"
        assert len(steers) == 1
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)


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
    assert first.version == review_actions.read_review(saved.session_dir).version == 3
    assert review_actions.read_review(saved.session_dir).findings[0].status == "open"
    assert (
        review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume) == first
    )
    assert len(attempts) == 1


def test_successful_launcher_preserves_recorded_failure(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    request = review_records.FindingAction(
        request_id="resume", expected_version=1, action="fix", resume=True
    )

    def resume(text: str) -> str:
        fd = lock.acquire_single_writer(saved.session_dir)
        assert fd is not None
        try:
            event_log.EventSink(saved.logs_path).emit(
                journal.ReviewFixFailed(
                    finding_id="f1",
                    request_id="resume",
                    source="cli",
                    version=3,
                    error="worker refused instruction",
                )
            )
        finally:
            lock.release_single_writer(fd)
        return ""

    result = review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume)
    assert result.status == "failed" and result.version == 3
    assert result.message == "worker refused instruction"
    assert review_actions.read_review(saved.session_dir).findings[0].status == "open"


def test_resume_fix_is_acknowledged_after_an_intervening_resolution(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    request = review_records.FindingAction(
        request_id="resume", expected_version=1, action="fix", resume=True
    )
    steers: list[str] = []

    def resume(text: str) -> str:
        steers.append(text)
        review_actions.act(saved.session_dir, "f1", _resolve(version=2), source="web")
        return ""

    pending = review_actions.act(saved.session_dir, "f1", request, source="cli", resume=resume)
    assert pending.status == "queued" and pending.version == 2
    assert pending.message == "run resumed to fix finding f1; submission acknowledgement pending"
    fd = lock.acquire_single_writer(saved.session_dir)
    assert fd is not None
    process.write_worker_pid(saved.session_dir, os.getpid())
    try:
        events = event_log.EventSink(saved.logs_path)
        review_actions.drain(events, execution_id="resumed", resume_steer=steers[0])
        result = review_actions.act(saved.session_dir, "f1", request, source="cli", wait_s=0)
        assert result.status == "submitted"
        assert result.request_id == "resume" and result.version == 4
        finding = review_actions.read_review(saved.session_dir).findings[0]
        assert finding.status == "resolved" and finding.fix_state == "submitted"
        assert [action.type for action in finding.actions] == [
            "review.fix.requested",
            "review.finding.resolved",
            "review.fix.submitted",
        ]
        assert finding.actions[-1].execution_id == "resumed"
        assert finding.actions[-1].source == "cli"
        before = saved.logs_path.read_bytes()
        review_actions.drain(events, execution_id="resumed", resume_steer=steers[0])
        assert saved.logs_path.read_bytes() == before
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)
