"""Derive testing, review and reversal evidence without running commands from a receipt."""

from __future__ import annotations

import dataclasses
import pathlib
import shlex
from typing import Any, Literal

import pydantic

from agent6 import errors, evidence
from agent6.sessions import manifest
from agent6.vcs import commit_graph, git_read
from agent6.viewmodel import _session_log
from agent6.viewmodel import review as review_view


@dataclasses.dataclass(frozen=True, slots=True)
class RevertPlan:
    """Explain reversal from recorded git facts without executing it."""

    kind: Literal["not_recorded", "squash", "merge", "ff_only", "manual"]
    commits: tuple[str, ...] = ()
    mainline: int | None = None
    explanation: str = "cannot determine a safe revert"


def revert_plan(root: pathlib.Path, stamp: manifest.MergeStamp | None) -> RevertPlan:
    """Derive reversal guidance from a merge stamp and existing commit objects.

    Args:
        root: The repository containing the landed objects.
        stamp: The recorded merge, if any.

    Returns:
        A verified plan or an explanation of the missing evidence.
    """
    if stamp is None:
        return RevertPlan(
            "not_recorded", explanation="no merge recorded; cannot determine a safe revert"
        )
    if stamp.sha == manifest.NO_MERGE_COMMIT:
        return RevertPlan(
            "manual",
            explanation=(
                "The recorded landing added no commit; "
                "cannot determine a safe revert of this run alone."
            ),
        )
    if stamp.via:
        return RevertPlan(
            "manual",
            explanation=(
                "This run landed through another run; "
                "cannot determine a safe revert of this run alone."
            ),
        )
    if not stamp.method or not stamp.target_before:
        return RevertPlan("manual")
    plan = RevertPlan(
        "manual",
        explanation=(
            "The recorded target-parent relationship could not be verified; "
            "cannot determine a safe revert."
        ),
    )
    try:
        parents = commit_graph.parents(root, stamp.sha)
        commit_graph.parents(root, stamp.target_before)
        if stamp.method == "squash" and parents == (stamp.target_before,):
            plan = RevertPlan(
                "squash", (stamp.sha,), explanation="Revert the recorded squash commit."
            )
        elif (
            stamp.method == "merge"
            and len(parents) == 2
            and parents.count(stamp.target_before) == 1
        ):
            plan = RevertPlan(
                "merge",
                (stamp.sha,),
                parents.index(stamp.target_before) + 1,
                "Keep the verified target parent when reverting the merge commit.",
            )
        elif stamp.method == "ff_only" and git_read.is_ancestor(
            root, stamp.target_before, stamp.sha
        ):
            commits = commit_graph.landed_commits(root, stamp.target_before, stamp.sha)
            if commits and all(len(commit_graph.parents(root, c)) == 1 for c in commits):
                plan = RevertPlan(
                    "ff_only",
                    commits,
                    explanation="Revert landed commits in reverse topological order.",
                )
            else:
                plan = RevertPlan(
                    "manual",
                    explanation=(
                        "The fast-forward contains merges or no new commits; "
                        "cannot determine a safe revert without an explicit plan."
                    ),
                )
    except (errors.GitError, OSError):
        plan = RevertPlan(
            "manual",
            explanation=(
                "Recorded git objects are missing or unreadable; cannot determine a safe revert."
            ),
        )
    return plan


@dataclasses.dataclass(frozen=True, slots=True)
class ReceiptView:
    """Keep a session's evidence derived from its journal and manifest."""

    session_id: str
    execution_id: str | None
    source_tip: str | None
    end_tree: str | None
    verifications: tuple[evidence.VerificationEvidence, ...]
    final_verify_id: str | None
    review: review_view.ReviewView
    revert: RevertPlan
    evidence_notes: tuple[str, ...]
    merged: manifest.MergeStamp | None = None


def read_receipt(session_dir: pathlib.Path, root: pathlib.Path) -> ReceiptView:
    """Read evidence without calling a provider or running verification.

    Args:
        session_dir: The saved session directory.
        root: The repository containing the recorded git objects.

    Returns:
        A receipt with explicit notes for unreadable or absent evidence.
    """
    notes: list[str] = []
    try:
        saved = manifest.read_manifest(session_dir)
    except manifest.ManifestError as exc:
        saved = manifest.SessionManifest()
        notes.append(f"Manifest unreadable: {exc}")
    fold = _session_log.scan_session_log(session_dir / "logs.jsonl")
    notes.extend(fold.evidence_notes)
    if not fold.verifications:
        notes.append("No verification recorded.")
    review = fold.review
    if fold.end_tree and fold.review.panels and fold.review.panels[-1].target:
        target = fold.review.panels[-1].target
        review = fold.review.for_comparison(
            dataclasses.replace(target, head_tree=fold.end_tree, paths=())
        )
    source_tip = fold.source_tip or (saved.merged.tip if saved.merged else None)
    return ReceiptView(
        session_dir.name,
        fold.execution_id,
        source_tip,
        fold.end_tree or None,
        tuple(fold.verifications),
        fold.final_verify_id,
        review,
        revert_plan(root, saved.merged),
        tuple(notes),
        saved.merged,
    )


def as_dict(view: ReceiptView) -> dict[str, Any]:
    """Return the JSON-safe receipt fields.

    Args:
        view: The derived receipt.

    Returns:
        Its structured wire form.
    """
    payload = pydantic.TypeAdapter(ReceiptView).dump_python(view, mode="json")
    payload["revert_request"] = revert_request(view)
    return payload


def verification_summary(record: evidence.VerificationEvidence) -> str:
    """Describe an attempt's command, tree, scope and capture limits.

    Args:
        record: One observed verification attempt.

    Returns:
        A plain-text summary without output bodies.
    """
    scope = "scoped" if record.scoped else "full gate"
    baseline = "baseline; " if record.baseline else ""
    code = f"exit {record.exit_code}, " if record.exit_code is not None else ""
    reason = f"Reason: {record.reason}\n" if record.reason else ""
    return (
        f"{record.verify_id or 'unrecorded id'}: {baseline}{record.outcome}, {code}{scope}\n"
        f"{reason}"
        f"Command: {shlex.join(record.cmd)}\n"
        f"Tested tree: {record.tree_before or 'unknown'}; after: {record.tree_after or 'unknown'}\n"
        f"Selected paths: {', '.join(record.selected_paths) or 'none'}\n"
        f"Duration: {record.duration_s:.2f}s; timeout: {record.timeout_s:g}s\n"
        "Output cap: 2000 characters per stream; "
        f"stdout {evidence.capture_label(record.stdout_truncated)}; "
        f"stderr {evidence.capture_label(record.stderr_truncated)}"
    )


def revert_request(view: ReceiptView) -> str:
    """Return a task for existing steering or new-work controls, not a command to execute.

    Args:
        view: The inspected receipt.

    Returns:
        The proposed task, which asks the worker to inspect and verify a reversal.
    """
    lines = [f"Prepare a reversal of the landed change from session {view.session_id}."]
    if stamp := view.merged:
        lines.extend(
            (
                f"Recorded landing method: {stamp.method or 'not recorded'}",
                f"Target: {stamp.into or 'not recorded'}",
                f"Target before landing: {stamp.target_before or 'not recorded'}",
                f"Landed commit: {stamp.sha or 'not recorded'}",
                f"Source tip: {stamp.tip or 'not recorded'}",
            )
        )
        if stamp.via:
            lines.append(f"Included through run: {stamp.via}")
    if view.revert.kind in ("manual", "not_recorded"):
        lines.append(f"No safe plan determined: {view.revert.explanation}")
    else:
        lines.append("Commits to revert in order: " + ", ".join(view.revert.commits))
        if view.revert.mainline:
            lines.append(f"Mainline: {view.revert.mainline}")
        lines.append(view.revert.explanation)
    lines.extend(
        (
            "Check the current target and these recorded git facts before editing.",
            "Preserve unrelated changes and run the configured verification.",
            "Do not assume the original checks certify the reversal.",
        )
    )
    return "\n".join(lines)


def render(view: ReceiptView, *, include_output: bool = True) -> str:
    """Render the shared receipt without interpreting model prose as test evidence.

    Args:
        view: The derived receipt.
        include_output: Whether to include the retained stdout and stderr bodies.

    Returns:
        Testing, review, risk and reversal sections as plain text.
    """
    final = next(
        (
            v
            for v in reversed(view.verifications)
            if v.verify_id and v.verify_id == view.final_verify_id
        ),
        None,
    )
    certified = final is not None and final.certifies(view.end_tree)
    word = "Final tree passed" if certified else "Final tree not certified"
    if certified and final is not None and final.scoped:
        word += " (scoped only)"
    lines = [
        f"Receipt {view.session_id}",
        f"Execution: {view.execution_id or 'unknown'}",
        f"Source tip: {view.source_tip or 'unknown'}",
        "",
        "Testing",
        f"{word}: {view.end_tree or 'unknown'}",
    ]
    for record in view.verifications:
        lines.append(verification_summary(record))
        if include_output:
            lines.extend((f"stdout:\n{record.stdout_tail}", f"stderr:\n{record.stderr_tail}"))
    lines.extend(view.evidence_notes)
    lines.extend(
        (
            "",
            "Review",
            f"{view.review.completion}; freshness: {view.review.freshness}",
            f"Open findings: {view.review.open_counts.block} block, "
            f"{view.review.open_counts.warn} warn, {view.review.open_counts.nit} nit",
        )
    )
    for panel in view.review.panels:
        target = panel.target
        lines.append(
            f"Panel {panel.panel_id}: {panel.completion}; "
            f"reviewed head {target.head_tree if target else 'unknown'}"
        )
    if view.review.read_failure:
        lines.append(view.review.read_failure)
    lines.extend(("", "Risk"))
    seats = [(p.panel_id, seat) for p in view.review.panels for seat in p.seats]
    for panel_id, seat in seats:
        lines.append(f"{panel_id} / {seat.model_route}:\n{seat.risk or 'not recorded'}")
        if seat.risk_truncated:
            lines.append("Risk truncated at 1000 characters.")
    if not seats:
        lines.append("not recorded")
    lines.extend(
        (
            "",
            "How to revert",
            view.revert.explanation,
            "Git commands below are an explanation only, not an action.",
        )
    )
    for commit in view.revert.commits:
        mainline = f" -m {view.revert.mainline}" if view.revert.mainline else ""
        lines.append(f"git revert{mainline} {commit}")
    lines.append("Ask a run to revert through new-work or steering controls.")
    return "\n".join(lines)
