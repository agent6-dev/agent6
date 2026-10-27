"""Serve session findings and operator actions through the shared application path."""

from __future__ import annotations

import pathlib
from typing import Any

import pydantic

from agent6 import review_records
from agent6.app import review_actions
from agent6.ui import spawn
from agent6.viewmodel import review


def findings(session_dir: pathlib.Path, status: str) -> tuple[int, dict[str, Any]]:
    """Return open findings, or all findings when status is all."""
    if status not in {"open", "all"}:
        return 422, {"error": "status must be open or all"}
    try:
        payload = review.as_dict(review_actions.read_review(session_dir))
    except review_actions.ReviewActionError as exc:
        return exc.status, {"error": str(exc)}
    payload["findings"] = [
        f for f in payload["findings"] if status == "all" or f["status"] == "open"
    ]
    return 200, payload


def act(
    cwd: pathlib.Path,
    session_dir: pathlib.Path,
    finding_id: str,
    verb: str,
    body: dict[str, Any],
    config_paths: tuple[pathlib.Path, ...],
) -> tuple[int, dict[str, Any]]:
    """Validate an HTTP action and return its result with the application status code.

    Returns:
        An HTTP status and the result or refusal payload.
    """
    if verb not in {"fix", "resolve", "reopen"}:
        return 404, {"error": "unknown finding action"}
    try:
        if "action" in body:
            return 422, {"error": "the route supplies the action"}
        request = review_records.FindingAction.model_validate({**body, "action": verb})
        result = review_actions.act(
            session_dir,
            finding_id,
            request,
            source="web",
            resume=lambda text: spawn.spawn_detached_resume(
                cwd, session_dir.name, steer=text, config_paths=config_paths
            ),
        )
    except review_actions.ReviewActionError as exc:
        return exc.status, {"error": str(exc)}
    except (pydantic.ValidationError, ValueError) as exc:
        return 422, {"error": str(exc)}
    return 202 if result.status == "queued" else 200, result.model_dump(mode="json")
