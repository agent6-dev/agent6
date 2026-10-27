"""Serve session findings and operator actions through the shared application path."""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import re
from collections.abc import Iterator
from typing import Any

import pydantic

from agent6 import errors, paths, review_records
from agent6.app import review_actions
from agent6.sessions import layout
from agent6.ui import spawn
from agent6.ui.web import git_model, highlight
from agent6.vcs import git_files, git_history, git_read
from agent6.viewmodel import listing, review, scan_session_log


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


def _records(cwd: pathlib.Path) -> Iterator[tuple[str, review.ReviewView]]:
    state = paths.state_dir(cwd)
    for directory in listing.session_dirs(state):
        yield directory.name, scan_session_log(directory / layout.LOGS_NAME).review
    for journal in sorted((state / "reviews").glob("*.jsonl")):
        yield "", scan_session_log(journal).review


def _tree(cwd: pathlib.Path, revision: str) -> str:
    commit = git_history.resolve_commit(cwd, revision)
    if not commit and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise git_model.GitRequestError(f"unknown revision {revision!r}")
    return git_read.tree_sha(cwd, commit or revision)


def comparison(cwd: pathlib.Path, base: str, head: str) -> dict[str, Any]:
    """Return findings with exact comparison matches and links to their recorded targets."""
    base_tree = _tree(cwd, base)
    head_tree = "" if head == git_history.WORKTREE else _tree(cwd, head)
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for session_id, view in _records(cwd):
        if view.read_failure:
            failures.append(f"{session_id or 'saved review'}: {view.read_failure}")
        payload = review.as_dict(view)
        panels = {p["panel_id"]: p for p in payload["panels"]}
        for finding in payload["findings"]:
            observations = finding["observations"]
            candidates: list[dict[str, Any]] = []
            for observation in observations:
                panel = panels[observation["panel_id"]]
                target = panel["target"]
                matches = bool(
                    target and target["base_tree"] == base_tree and target["head_tree"] == head_tree
                )
                candidates.append(
                    dict(
                        session_id=session_id,
                        version=view.version,
                        panel=panel,
                        finding=finding,
                        observation=observation,
                        matches=matches,
                    )
                )
            entries.append(next((e for e in reversed(candidates) if e["matches"]), candidates[-1]))
    return {"findings": entries, "read_failures": failures}


def _file(
    cwd: pathlib.Path, target: review_records.ReviewTarget, side: str, path: str, whole: bool
) -> tuple[int, dict[str, Any]]:
    location = review_records.ReviewLocation.model_validate({"path": path, "side": side})
    tree = target.base_tree if location.side == "base" else target.head_tree
    blob = git_files.blob_at(
        cwd, _tree(cwd, tree), location.path, max_bytes=git_model.BLOB_MAX_BYTES
    )
    if blob is None:
        return 404, {"error": f"no regular file {path!r} on the {side} side"}
    if blob.data is None:
        return 413, {"error": f"file exceeds {git_model.BLOB_MAX_BYTES} bytes"}
    data = blob.data
    cap = None if whole else git_model.TEXT_MAX_BYTES
    binary = git_history.looks_binary(data)
    truncated = not binary and cap is not None and len(data) > cap
    shown = data[: data.rfind(b"\n", 0, cap) + 1] if truncated else data
    text = "" if binary else shown.decode("utf-8", errors="replace")
    language = highlight.language_of(path)
    return 200, dict(
        rev=tree,
        path=path,
        text=text,
        binary=binary,
        truncated=truncated,
        max_bytes=cap,
        digest=hashlib.sha256(data).hexdigest(),
        lines=len(text.splitlines()),
        language=language,
        spans=highlight.spans(text, language) if text else [],
    )


def _diff(
    cwd: pathlib.Path,
    base: str,
    head: str,
    files: tuple[git_history.ChangedFile, ...],
    path: str,
    whole: bool,
) -> tuple[int, dict[str, Any]]:
    location = review_records.ReviewLocation(path=path, side="head")
    changed = next((f for f in files if path in (f.path, f.old_path)), None)
    new_path, old_path = (changed.path, changed.old_path) if changed else (location.path, "")
    if not any(git_history.tree_has_entry(cwd, tree, path) for tree in (base, head)):
        return 404, {"error": f"no file {path!r} in the reviewed trees"}
    cap = None if whole else git_model.DIFF_MAX_BYTES
    patch, truncated = git_history.file_patch(cwd, base, head, new_path, old_path, max_bytes=cap)
    return 200, dict(
        base=base,
        head=head,
        path=new_path,
        old_path=old_path,
        patch=patch,
        binary=git_history.is_binary_patch(patch),
        truncated=truncated,
        max_bytes=cap,
    )


def panel(
    cwd: pathlib.Path,
    panel_id: str,
    sub: str = "",
    *,
    path: str = "",
    side: str = "",
    whole: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Read a recorded comparison or file without consulting a mutable checkout.

    Args:
        cwd: The repository root.
        panel_id: The recorded panel ID.
        sub: "file" or "diff" for file evidence, otherwise the panel overview.
        path: The repository-relative file path.
        side: The side for file evidence: base or head.
        whole: Whether to omit the display limit for file text or diffs.

    Returns:
        An HTTP status and the evidence or refusal payload.
    """
    for session_id, view in _records(cwd):
        found = next((p for p in view.panels if p.panel_id == panel_id), None)
        if found is None:
            continue
        if view.read_failure or found.target is None:
            return 409, {"error": view.read_failure or "panel has no reviewed target"}
        target = found.target
        try:
            if sub == "file":
                return _file(cwd, target, side, path, whole)
            base, head = _tree(cwd, target.base_tree), _tree(cwd, target.head_tree)
            files = git_history.changed_files(cwd, base, head)
            if sub == "diff":
                return _diff(cwd, base, head, files, path, whole)
            payload = review.as_dict(view)
            return 200, dict(
                session_id=session_id,
                review=payload,
                panel=next(p for p in payload["panels"] if p["panel_id"] == panel_id),
                files=[dataclasses.asdict(f) for f in files[: git_model.FILES_MAX]],
                truncated=len(files) > git_model.FILES_MAX,
            )
        except (pydantic.ValidationError, git_model.GitRequestError, errors.GitError) as exc:
            return 422, {"error": str(exc)}
    return 404, {"error": f"no review panel {panel_id!r}"}


def act(
    cwd: pathlib.Path,
    session_dir: pathlib.Path,
    finding_id: str,
    verb: str,
    body: dict[str, Any],
    config_paths: tuple[pathlib.Path, ...],
) -> tuple[int, dict[str, Any]]:
    """Validate an HTTP action and return its result with the application status code.

    Args:
        cwd: The repository root.
        session_dir: The session directory.
        finding_id: The finding to act on.
        verb: The action named by the route.
        body: The action fields from the JSON request.
        config_paths: The config overlays for a detached resume, in order.

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
