"""Apply operator finding actions through the session's single writer."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import time
from collections.abc import Callable, Generator, Sequence
from typing import Literal, Self

import pydantic

from agent6 import review_records
from agent6.sessions import event_log, ipc, journal, layout, lock, process
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import results, tool_errors
from agent6.viewmodel import review, session_log

Source = Literal["cli", "tui", "web", "acp"]


class ReviewActionError(Exception):
    """Refuse an invalid, stale or conflicting operator action."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


class ActionRequest(pydantic.BaseModel):
    """Validate the operator envelope carried by the file bridge.

    Attributes:
        finding_ids: The findings the action targets: one for Resolve and Reopen, one or more
            for Fix, which steers them as one instruction.
        source: The surface that submitted the action.
        action: The action and its retry identity.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    finding_ids: tuple[review_records.Identifier, ...] = pydantic.Field(min_length=1)
    source: Source
    action: review_records.FindingAction

    @pydantic.model_validator(mode="after")
    def _one_unless_fix(self) -> Self:
        if len(set(self.finding_ids)) != len(self.finding_ids):
            raise ValueError("a finding is named twice")
        if self.action.action != "fix" and len(self.finding_ids) != 1:
            raise ValueError(f"{self.action.action} acts on one finding")
        return self


class ActionResult(pydantic.BaseModel):
    """Return the recorded outcome, or an explicit pending delivery.

    Attributes:
        request_id: The caller's action identifier.
        finding_id: The finding the action targets.
        version: The review version after the action.
        status: The action's delivery or completion status.
        message: The result or next-action explanation.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    request_id: str
    finding_id: str
    version: int
    status: Literal["resolved", "reopened", "submitted", "failed", "delivery_unknown", "queued"]
    message: str


class _Reply(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    results: tuple[ActionResult, ...] = ()
    error: str = ""
    status: int = 409


def read_review(session_dir: pathlib.Path) -> review.ReviewView:
    """Read the shared review fold, refusing missing or unreadable evidence.

    Args:
        session_dir: The directory containing the session journal.

    Returns:
        The session's review evidence.

    Raises:
        ReviewActionError: The journal is missing or its review evidence is unreadable.
    """
    path = session_dir / layout.LOGS_NAME
    if not path.is_file():
        raise ReviewActionError("no session journal", 404)
    view = session_log.scan_session_log(path).review
    if view.read_failure:
        raise ReviewActionError(f"cannot read review evidence: {view.read_failure}")
    return view


def findings_result(session_dir: pathlib.Path) -> results.FindingsResult:
    """Return the session's findings as `list_findings` reports them to the worker.

    Args:
        session_dir: The session's directory.

    Returns:
        Every finding, open and resolved; the note says when no panel has run, or that the
        journal holds legacy counts without finding evidence.

    Raises:
        ToolError: The journal is missing or its review evidence is unreadable.
    """
    try:
        view = read_review(session_dir)
    except ReviewActionError as exc:
        raise tool_errors.ToolError(str(exc)) from exc
    note = view.legacy_notice or ("No review panel has run." if not view.panels else "")
    return results.FindingsResult(
        findings=tuple(finding_brief(f) for f in view.findings), note=note
    )


def finding_brief(finding: review.FindingView) -> results.FindingBrief:
    """Return the finding's facts as every text surface states them, from its latest observation.

    Args:
        finding: The finding's fold.

    Returns:
        The brief the `list_findings` tool, `sessions findings` and ACP's `/findings` print.
    """
    report = finding.observations[-1].finding
    origin = finding.line_origin
    resolution = finding.resolution
    disposition = ""
    if resolution is not None:
        target = f" of {resolution.duplicate_of}" if resolution.duplicate_of else ""
        disposition = f"{resolution.disposition}{target}: {resolution.reason}"
    return results.FindingBrief(
        id=finding.finding_id,
        status=finding.status,
        severity=report.severity,
        category=report.category,
        location=str(report.location),
        title=report.title,
        detail=report.detail,
        pre_existing=(
            f"{origin.base_location}. {origin.reason}" if origin.kind == "pre_existing" else ""
        ),
        resolution=disposition,
        fix_state=finding.fix_state.replace("_", " ") if finding.fix_state != "none" else "",
    )


def _recorded(view: review.ReviewView, request: ActionRequest) -> tuple[ActionResult, ...] | None:
    records = [
        a for f in view.findings for a in f.actions if a.request_id == request.action.request_id
    ]
    if not records:
        return None
    expected = {"fix": "fix.requested", "resolve": "finding.resolved", "reopen": "finding.reopened"}
    wanted = "loop.review." + expected[request.action.action]
    if any(a.finding_id not in request.finding_ids for a in records) or records[0].type != wanted:
        raise ReviewActionError("request_id already belongs to another finding or action")
    results: list[ActionResult] = []
    for finding_id in request.finding_ids:
        own = [a for a in records if a.finding_id == finding_id]
        if not own:
            results.append(
                ActionResult(
                    request_id=request.action.request_id,
                    finding_id=finding_id,
                    version=view.version,
                    status="delivery_unknown",
                    message="delivery unknown; no record for this finding",
                )
            )
            continue
        first, last = own[0], own[-1]
        if first.type != wanted:
            raise ReviewActionError("request_id already belongs to another finding or action")
        kind = last.type.rsplit(".", 1)[-1]
        status = "delivery_unknown" if kind == "requested" else kind
        message = {
            "resolved": "finding resolved",
            "reopened": "finding reopened",
            "submitted": "steer submitted",
            "failed": last.error or "Fix delivery failed",
            "delivery_unknown": "delivery unknown; no submission result was recorded",
        }[status]
        if status == "submitted" and first.action == "resume":
            message = f"run resumed to fix finding {finding_id}"
        results.append(
            ActionResult.model_validate(
                dict(
                    request_id=last.request_id,
                    finding_id=finding_id,
                    version=last.version or view.version,
                    status=status,
                    message=message,
                )
            )
        )
    return tuple(results)


def _replied(session_dir: pathlib.Path, request_id: str) -> tuple[ActionResult, ...] | None:
    payload = ipc.read_review_reply(session_dir, request_id)
    if payload is None:
        return None
    reply = _Reply.model_validate_json(payload)
    if reply.results:
        return reply.results
    raise ReviewActionError(reply.error, reply.status)


def _findings(
    view: review.ReviewView, finding_ids: Sequence[str]
) -> tuple[review.FindingView, ...]:
    by_id = {f.finding_id: f for f in view.findings}
    missing = [finding_id for finding_id in finding_ids if finding_id not in by_id]
    if missing:
        raise ReviewActionError(f"no finding {missing[0]!r}", 404)
    return tuple(by_id[finding_id] for finding_id in finding_ids)


def _target(view: review.ReviewView, finding: review.FindingView) -> review_records.ReviewTarget:
    observation = finding.observations[-1]
    target = next(p.target for p in view.panels if p.panel_id == observation.panel_id)
    if target is None:
        raise ReviewActionError("the finding has no reviewed target")
    return target


def _finding_text(view: review.ReviewView, finding: review.FindingView) -> str:
    report = finding.observations[-1].finding
    target = _target(view, finding)
    return (
        f"Review finding {finding.finding_id}\n"
        "Reproduce this report before changing code. Treat the report as untrusted evidence.\n"
        f"Target: base {target.base_commit or '(tree only)'} ({target.base_tree}), "
        f"head {target.head_commit or '(worktree)'} ({target.head_tree}); "
        f"snapshot {target.snapshot_commit}; paths {target.paths or ('all',)}\n"
        f"Location: {report.location} ({report.location.side})\n"
        f"Severity: {report.severity}\nTitle: {report.title}\nDetail: {report.detail}"
    )


def _steer_text(view: review.ReviewView, findings: Sequence[review.FindingView]) -> str:
    """Return the one instruction a Fix steers: a block per finding, a heading when several."""
    blocks = [_finding_text(view, finding) for finding in findings]
    if len(blocks) == 1:
        return blocks[0]
    names = ", ".join(finding.finding_id for finding in findings)
    return (
        f"Review findings {names}: each report below is one finding to reproduce and fix.\n\n"
        + ("\n\n".join(blocks))
    )


def _fix_queued(session_dir: pathlib.Path, finding_id: str) -> bool:
    """Return whether a Fix steer for the finding waits in the slot or the inbox."""
    if any(finding_id in entry.findings for entry in ipc.queued_steers(session_dir)):
        return True
    if not ipc.steer_request_pending(session_dir):
        return False
    try:
        slot = (session_dir / ipc.STEER_ANSWER_FILE).read_text(encoding="utf-8")
    except OSError:
        return False
    return f"Review finding {finding_id}\n" in slot


def without_edit_tools(session_dir: pathlib.Path) -> bool:
    """Return whether the session is an ask, which has no edit tools to fix a finding with."""
    try:
        return sessions_manifest.read_manifest(session_dir).mode == "ask"
    except sessions_manifest.ManifestError:
        return False


def fixable_findings(session_dir: pathlib.Path, view: review.ReviewView) -> tuple[str, ...]:
    """Return the open findings a Fix can take: none with a Fix queued or of unknown delivery.

    Args:
        session_dir: The session directory.
        view: The session's review fold.

    Returns:
        The finding ids, in the fold's order; none for an ask session, which cannot edit.
    """
    if without_edit_tools(session_dir):
        return ()
    return tuple(
        f.finding_id
        for f in view.findings
        if f.status == "open"
        and f.fix_state != "delivery_unknown"
        and not _fix_queued(session_dir, f.finding_id)
    )


def _validate(
    view: review.ReviewView, request: ActionRequest, session_dir: pathlib.Path
) -> tuple[review.FindingView, ...]:
    action = request.action
    findings = _findings(view, request.finding_ids)
    # A Fix names findings, which stay what they are; Resolve and Reopen record a conclusion
    # over the version the operator read.
    if action.action != "fix" and action.expected_version != view.version:
        raise ReviewActionError(
            f"stale review version {action.expected_version}; current version is {view.version}"
        )
    if action.action == "fix":
        if without_edit_tools(session_dir):
            raise ReviewActionError(
                'an ask session has no edit tools to fix with; start agent6 run "/review TARGET"'
            )
        for finding in findings:
            if finding.status != "open":
                raise ReviewActionError(
                    f"reopen the finding before requesting Fix ({finding.finding_id})"
                )
            if finding.fix_state == "delivery_unknown" or _fix_queued(
                session_dir, finding.finding_id
            ):
                raise ReviewActionError(
                    f"a Fix for this finding is already queued ({finding.finding_id})"
                )
        return findings
    (finding,) = findings
    if action.action == "resolve" and finding.status == "resolved":
        raise ReviewActionError("finding is already resolved")
    if action.action == "reopen" and finding.status == "open":
        raise ReviewActionError("finding is already open")
    resolution = action.resolution
    if (
        resolution is not None
        and resolution.disposition == "duplicate"
        and (
            resolution.duplicate_of == finding.finding_id
            or resolution.duplicate_of not in {f.finding_id for f in view.findings}
        )
    ):
        raise ReviewActionError("duplicate must name another finding in this session", 422)
    return findings


# The finding-action records by the kind a request names.
_ACTION_RECORDS: dict[str, Callable[..., journal.Event]] = {
    "finding.resolved": journal.LoopReviewFindingResolved,
    "finding.reopened": journal.LoopReviewFindingReopened,
    "fix.requested": journal.LoopReviewFixRequested,
    "fix.submitted": journal.LoopReviewFixSubmitted,
    "fix.failed": journal.LoopReviewFixFailed,
}


def _emit(
    events: event_log.EventSink,
    request: ActionRequest,
    kind: str,
    *,
    finding_id: str,
    **fields: object,
) -> None:
    events.emit(
        _ACTION_RECORDS[kind](
            finding_id=finding_id,
            request_id=request.action.request_id,
            source=request.source,
            version=read_review(events.path.parent).version + 1,
            **fields,
        )
    )


def _apply(
    events: event_log.EventSink, request: ActionRequest, *, execution_id: str = ""
) -> tuple[ActionResult, ...]:
    session_dir = events.path.parent
    view = read_review(session_dir)
    if recorded := _recorded(view, request) or _replied(session_dir, request.action.request_id):
        return recorded
    findings = _validate(view, request, session_dir)
    action = request.action
    if action.action == "resolve":
        assert action.resolution is not None
        _emit(
            events,
            request,
            "finding.resolved",
            finding_id=findings[0].finding_id,
            resolution=action.resolution.model_dump(mode="json"),
        )
    elif action.action == "reopen":
        _emit(
            events,
            request,
            "finding.reopened",
            finding_id=findings[0].finding_id,
            reason=action.reason,
        )
    else:
        for finding in findings:
            _emit(
                events,
                request,
                "fix.requested",
                finding_id=finding.finding_id,
                target=dataclasses.asdict(_target(view, finding)),
                action="live-steer" if execution_id else "resume",
            )
        if execution_id:
            landed = ipc.submit_steer(
                session_dir,
                _steer_text(view, findings),
                findings=tuple(finding.finding_id for finding in findings),
            )
            for finding in findings:
                if landed:
                    _emit(
                        events,
                        request,
                        "fix.submitted",
                        finding_id=finding.finding_id,
                        execution_id=execution_id,
                    )
                else:
                    _emit(
                        events,
                        request,
                        "fix.failed",
                        finding_id=finding.finding_id,
                        error="could not write the steer",
                    )
    results = _recorded(read_review(session_dir), request)
    assert results is not None
    return results


def drain(events: event_log.EventSink, *, execution_id: str, resume_steer: str = "") -> None:
    """Apply queued actions in the worker before it polls the steer bridge.

    Args:
        events: The worker's journal sink.
        execution_id: The current execution ID.
        resume_steer: The resume steer text to acknowledge, or "".

    Raises:
        ReviewActionError: Resume acknowledgement cannot read the stored evidence.
    """
    session_dir = events.path.parent
    if resume_steer:
        _acknowledge_resume(events, execution_id=execution_id, resume_steer=resume_steer)
    for request_id, payload in ipc.review_requests(session_dir):
        try:
            request = ActionRequest.model_validate_json(payload)
            if request.action.request_id != request_id:
                raise ReviewActionError("request ID differs from its envelope", 422)
            reply = _Reply(results=_apply(events, request, execution_id=execution_id))
        except ReviewActionError as exc:
            reply = _Reply(error=str(exc), status=exc.status)
        except pydantic.ValidationError as exc:
            reply = _Reply(error=str(exc), status=422)
        ipc.reply_review_action(session_dir, request_id, reply.model_dump_json())


def _acknowledge_resume(
    events: event_log.EventSink, *, execution_id: str, resume_steer: str
) -> None:
    """Record the submission of the Fix whose resume this execution is."""
    session_dir = events.path.parent
    view = read_review(session_dir)
    requests: dict[str, tuple[Source, list[review.FindingView]]] = {}
    for finding in view.findings:
        if finding.fix_state != "delivery_unknown":
            continue
        for action in finding.actions:
            if (
                action.type == "loop.review.fix.requested"
                and action.action == "resume"
                and action.source is not None
            ):
                requests.setdefault(action.request_id, (action.source, []))[1].append(finding)
    for request_id, (source, findings) in requests.items():
        if _steer_text(view, findings) != resume_steer:
            continue
        request = ActionRequest(
            finding_ids=tuple(finding.finding_id for finding in findings),
            source=source,
            action=review_records.FindingAction(
                request_id=request_id, expected_version=view.version, action="fix"
            ),
        )
        for result in _recorded(view, request) or ():
            if result.status == "delivery_unknown":
                _emit(
                    events,
                    request,
                    "fix.submitted",
                    finding_id=result.finding_id,
                    execution_id=execution_id,
                )


def _wait(
    session_dir: pathlib.Path, request: ActionRequest, wait_s: float
) -> tuple[ActionResult, ...]:
    deadline = time.monotonic() + wait_s
    while True:
        view = read_review(session_dir)
        recorded = _recorded(view, request) or _replied(session_dir, request.action.request_id)
        alive = process.worker_is_alive(session_dir)
        # The worker writes "requested", submits, then writes "submitted": a request seen
        # between the two is in delivery, not unknown, while the worker lives.
        if recorded is not None and (
            all(r.status != "delivery_unknown" for r in recorded) or not alive
        ):
            return recorded
        if time.monotonic() >= deadline or not alive:
            return tuple(
                ActionResult(
                    request_id=request.action.request_id,
                    finding_id=finding_id,
                    version=view.version,
                    status="queued",
                    message="action queued; retry with the same request_id for its result",
                )
                for finding_id in request.finding_ids
            )
        time.sleep(0.05)


@contextlib.contextmanager
def _offline_writer(session_dir: pathlib.Path) -> Generator[event_log.EventSink]:
    fd = lock.acquire_single_writer(session_dir)
    if fd is None:
        raise ReviewActionError("session writer is busy; retry with the same request_id")
    try:
        if process.worker_is_alive(session_dir):
            raise ReviewActionError("session worker started; retry with the same request_id")
        yield event_log.EventSink(session_dir / layout.LOGS_NAME)
    finally:
        lock.release_single_writer(fd)


def act(
    session_dir: pathlib.Path,
    finding_id: str,
    action: review_records.FindingAction,
    *,
    source: Source,
    resume: Callable[[str], str] | None = None,
    wait_s: float = 5.0,
) -> ActionResult:
    """Apply an action on one finding; `act_many` is the Fix of several.

    Args:
        session_dir: The session directory.
        finding_id: The finding to act on.
        action: The operator's action request.
        source: The front-end submitting the action.
        resume: The launcher for a confirmed resume.
        wait_s: The maximum seconds to wait for the live worker's reply.

    Returns:
        The recorded result or an explicit queued or unknown delivery state.

    Raises:
        ReviewActionError: The action is stale, invalid or conflicts with another writer.
    """
    (result,) = act_many(
        session_dir, (finding_id,), action, source=source, resume=resume, wait_s=wait_s
    )
    return result


def act_many(
    session_dir: pathlib.Path,
    finding_ids: Sequence[str],
    action: review_records.FindingAction,
    *,
    source: Source,
    resume: Callable[[str], str] | None = None,
    wait_s: float = 5.0,
) -> tuple[ActionResult, ...]:
    """Apply an action offline or ask the live worker to apply it through IPC.

    A Fix of several findings is one steer; Resolve and Reopen take one finding. Resume is
    an explicit, confirmed request. Its injected launcher returns an error or "". The
    existing worker lock excludes another offline action and a concurrent resume.

    Args:
        session_dir: The session directory.
        finding_ids: The findings to act on, in the order the results come back.
        action: The operator's action request.
        source: The front-end submitting the action.
        resume: The launcher for a confirmed resume.
        wait_s: The maximum seconds to wait for the live worker's reply.

    Returns:
        One result per finding: recorded, or an explicit queued or unknown delivery state.

    Raises:
        ReviewActionError: The action is stale, invalid or conflicts with another writer.
    """
    try:
        request = ActionRequest(finding_ids=tuple(finding_ids), source=source, action=action)
    except pydantic.ValidationError as exc:
        raise ReviewActionError(str(exc), 422) from exc
    view = read_review(session_dir)
    if recorded := _recorded(view, request) or _replied(session_dir, action.request_id):
        return recorded
    _validate(view, request, session_dir)
    if process.worker_is_alive(session_dir):
        if source == "acp" and action.action == "fix":
            raise ReviewActionError("ACP Fix requires an idle session")
        ipc.queue_review_action(session_dir, action.request_id, request.model_dump_json())
        return _wait(session_dir, request, wait_s)
    with _offline_writer(session_dir) as events:
        view = read_review(session_dir)
        if recorded := _recorded(view, request) or _replied(session_dir, action.request_id):
            return recorded
        findings = _validate(view, request, session_dir)
        text = ""
        if action.action == "fix":
            if not action.resume:
                raise ReviewActionError(
                    "Resume to fix requires confirmation: `--resume` on the CLI,"
                    ' `"resume": true` over HTTP'
                )
            if resume is None:
                raise ReviewActionError("resume is unavailable on this surface")
            text = _steer_text(view, findings)
        results = _apply(events, request)
    if text and resume is not None:
        return _after_resume(session_dir, request, results, resume(text))
    return results


def _after_resume(
    session_dir: pathlib.Path,
    request: ActionRequest,
    results: tuple[ActionResult, ...],
    error: str,
) -> tuple[ActionResult, ...]:
    """Record a failed launch, or say the resumed run has not acknowledged the Fix yet.

    Args:
        session_dir: The session directory.
        request: The Fix envelope.
        results: The results recorded before the launch.
        error: The launcher's error, or "".

    Returns:
        The results as the journal now records them, a pending acknowledgement read as queued.
    """
    if error:
        with contextlib.suppress(ReviewActionError), _offline_writer(session_dir) as events:
            for current in _recorded(read_review(session_dir), request) or ():
                if current.status == "delivery_unknown":
                    _emit(events, request, "fix.failed", finding_id=current.finding_id, error=error)
        return _recorded(read_review(session_dir), request) or results
    return tuple(
        result.model_copy(
            update={
                "status": "queued",
                "message": (
                    f"run resumed to fix finding {result.finding_id}; "
                    "submission acknowledgement pending"
                ),
            }
        )
        if result.status == "delivery_unknown"
        else result
        for result in _recorded(read_review(session_dir), request) or results
    )
