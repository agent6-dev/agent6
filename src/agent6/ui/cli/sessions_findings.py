"""Inspect findings and submit operator actions through the shared application path."""

from __future__ import annotations

import argparse
import json
import pathlib
import uuid
from typing import Literal

import pydantic

from agent6 import review_records
from agent6.app import review_actions
from agent6.sessions import id
from agent6.ui import spawn
from agent6.ui.cli import _common, completers
from agent6.viewmodel import review


def add_action_options(parser: argparse.ArgumentParser) -> None:
    """Add retry identity and optimistic concurrency options to an action parser."""
    parser.add_argument("--request-id", default="", help="Retry this action ID. Default: a new ID.")
    parser.add_argument(
        "--expected-version",
        type=int,
        default=None,
        help="Refuse a different review version. Default: the current version.",
    )


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add findings, resolve and reopen under sessions."""
    findings = _common._sub(
        sub, "findings", help="List open review findings and the review version."
    )
    _common._add_session_id(findings, completers._complete_session_ids)
    findings.add_argument("--all", action="store_true", help="Include resolved findings.")
    findings.add_argument("--json", action="store_true", help="Print structured review evidence.")
    for verb in ("resolve", "reopen"):
        parser = _common._sub(
            sub, verb, help=f"{verb.capitalize()} a finding with an operator reason."
        )
        session = parser.add_argument("session_id", help=_common.SESSION_ID)
        session.completer = completers._complete_session_ids  # type: ignore[attr-defined]
        finding = parser.add_argument("finding_id", help="The finding ID from sessions findings.")
        finding.completer = completers.complete_finding_ids  # type: ignore[attr-defined]
        parser.add_argument(
            "--reason", required=True, help="A nonblank reason, at most 2,000 characters."
        )
        if verb == "resolve":
            parser.add_argument(
                "--as",
                dest="disposition",
                required=True,
                choices=(
                    "duplicate",
                    "false_positive",
                    "fixed",
                    "intentional",
                    "pre_existing",
                    "wont_fix",
                ),
            )
            duplicate = parser.add_argument(
                "--duplicate-of", default=None, help="Another finding in this session."
            )
            duplicate.completer = completers.complete_finding_ids  # type: ignore[attr-defined]
        add_action_options(parser)


def cmd_findings(session_id: str, *, all_findings: bool, as_json: bool) -> int:
    """Print the shared review evidence, with resolved findings only when requested.

    Args:
        session_id: The session selector.
        all_findings: Whether to include resolved findings.
        as_json: Whether to print JSON instead of plain text.

    Returns:
        Zero on success, 2 when the session or review evidence is unreadable.
    """
    saved = _common.resolve_target(session_id)
    if saved is None:
        return 2
    try:
        view = review_actions.read_review(saved.session_dir)
    except review_actions.ReviewActionError as exc:
        _common.error(str(exc))
        return 2
    payload = review.as_dict(view)
    payload["findings"] = [f for f in payload["findings"] if all_findings or f["status"] == "open"]
    if as_json:
        # Panels retain observations of resolved findings; the listing includes only its rows.
        payload.pop("panels")
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    print(f"Review version {view.version}")
    for finding in view.findings:
        if all_findings or finding.status == "open":
            brief = review_actions.finding_brief(finding)
            print(f"{brief.id}  {brief.status}  {brief.severity}  {brief.location}: {brief.title}")
            print(f"  {brief.detail}")
            if brief.pre_existing:
                print(f"  Pre-existing line: {brief.pre_existing}")
                print(f"  Blame: {', '.join(finding.line_origin.commits)}")
            if brief.resolution:
                print(f"  {brief.resolution}")
            if brief.fix_state:
                print(f"  Fix: {brief.fix_state}")
    if view.legacy_notice:
        print(view.legacy_notice)
    return 0


def cmd_action(
    session_id: str,
    finding_id: str,
    action: Literal["fix", "resolve", "reopen"],
    *,
    reason: str | None = None,
    disposition: str = "",
    duplicate_of: str | None = None,
    request_id: str = "",
    expected_version: int | None = None,
    resume: bool = False,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> int:
    """Submit an action and print its delivery result, never an inferred fix.

    Args:
        session_id: The session selector.
        finding_id: The finding to act on.
        action: The action to submit.
        reason: The reason for resolving or reopening.
        disposition: The resolution category.
        duplicate_of: The original finding ID for a duplicate resolution.
        request_id: The retry identity, or "" to generate one.
        expected_version: The required review version, or None to use the current version.
        resume: Whether the operator confirmed resuming to fix.
        config_paths: The config overlays for a detached resume, in order.

    Returns:
        Zero for a recorded or queued action, 2 for a refusal or failed delivery.
    """
    try:
        saved = _common.resolve_session_layout(pathlib.Path.cwd(), session_id)
        view = review_actions.read_review(saved.session_dir)
        resolution = (
            review_records.FindingResolution.model_validate(
                dict(disposition=disposition, reason=reason, duplicate_of=duplicate_of)
            )
            if action == "resolve"
            else None
        )
        request = review_records.FindingAction(
            request_id=request_id or uuid.uuid4().hex,
            expected_version=view.version if expected_version is None else expected_version,
            action=action,
            resolution=resolution,
            reason=reason if action == "reopen" else None,
            resume=resume,
        )
        result = review_actions.act(
            saved.session_dir,
            finding_id,
            request,
            source="cli",
            resume=lambda text: spawn.spawn_detached_resume(
                pathlib.Path.cwd(), saved.session_id, steer=text, config_paths=config_paths
            ),
        )
    except (
        review_actions.ReviewActionError,
        id.SessionIdError,
        pydantic.ValidationError,
        OSError,
        ValueError,
    ) as exc:
        _common.error(str(exc))
        return 2
    print(
        f"{saved.session_id}: {result.message} "
        f"(request {result.request_id}, version {result.version})"
    )
    return 2 if result.status == "failed" else 0
