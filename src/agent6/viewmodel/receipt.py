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
from agent6.viewmodel import review as review_view
from agent6.viewmodel import session_log


@dataclasses.dataclass(frozen=True, slots=True)
class RevertPlan:
    """Explain reversal from recorded git facts without executing it.

    Attributes:
        kind: The reversal method supported by the recorded evidence.
        commits: The commits to reverse in order.
        mainline: The merge parent to keep, or null when not needed.
        explanation: The evidence supporting the plan or requiring manual inspection.
    """

    kind: Literal["not_recorded", "squash", "tasks", "merge", "ff_only", "manual"]
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
        elif stamp.method in ("tasks", "ff_only") and git_read.is_ancestor(
            root, stamp.target_before, stamp.sha
        ):
            # Both land a straight line of single-parent commits on the target.
            commits = commit_graph.landed_commits(root, stamp.target_before, stamp.sha)
            if commits and all(len(commit_graph.parents(root, c)) == 1 for c in commits):
                plan = RevertPlan(
                    stamp.method,
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
    """Keep a session's evidence derived from its journal and manifest.

    Attributes:
        session_id: The session identifier.
        execution_id: The last execution identifier, or null when unrecorded.
        source_tip: The source checkpoint commit, or null when unrecorded.
        end_tree: The final observed tree, or null when unrecorded.
        verifications: The recorded VerificationEvidence attempts.
        final_verify_id: The attempt covering the final tree, or null when absent.
        review: The review evidence compared against the final tree.
        revert: The reversal plan derived from recorded merge facts.
        evidence_notes: Diagnostics for absent or unreadable evidence.
        merged: The MergeStamp, or null without a recorded landing.
        targets: The recorded execution targets, without runtime queries.
        setups: Every recorded setup result.
        jobs: Job-settlement observations.
        environments: Environment lifecycle observations.
    """

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
    targets: tuple[evidence.ExecutionTargetEvidence, ...] = ()
    setups: tuple[evidence.SetupEvidence, ...] = ()
    jobs: tuple[evidence.JobsObservation, ...] = ()
    environments: tuple[evidence.EnvironmentObservation, ...] = ()


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
    fold = session_log.scan_session_log(session_dir / "logs.jsonl")
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
        fold.targets,
        fold.setups,
        fold.jobs,
        fold.environments,
    )


def as_dict(view: ReceiptView) -> dict[str, Any]:
    """Return the JSON-safe receipt fields.

    Attributes:
        revert_request: A proposed task for preparing a reversal, not a command to execute.
        text: The shared receipt rendering without retained output bodies.

    Args:
        view: The derived receipt.

    Returns:
        Its structured wire form, with the shared rendering (output bodies left out) as `text`.
    """
    payload = pydantic.TypeAdapter(ReceiptView).dump_python(view, mode="json")
    payload["revert_request"] = revert_request(view)
    payload["text"] = render(view, include_output=False)
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
    capture = [
        f"{name} truncated"
        for name, cut in (("stdout", record.stdout_truncated), ("stderr", record.stderr_truncated))
        if cut
    ]
    if record.stdout_truncated is None or record.stderr_truncated is None:
        capture.append("capture completeness unknown")
    output = "\nOutput: " + "; ".join(capture) if capture else ""
    return (
        f"{record.verify_id or 'unrecorded id'}: {baseline}{record.outcome}, {code}{scope}\n"
        f"{reason}"
        f"Command: {shlex.join(record.cmd)}\n"
        f"Environment: {record.target_id or 'unknown'}; "
        f"execution: {record.execution_id or 'unknown'}\n"
        f"Tested tree: {record.tree_before or 'unknown'}; after: {record.tree_after or 'unknown'}\n"
        f"Selected paths: {', '.join(record.selected_paths) or 'none'}\n"
        f"Duration: {record.duration_s:.2f}s; timeout: {record.timeout_s:g}s{output}"
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


def environment_dict(view: ReceiptView | session_log.LogScan) -> dict[str, Any]:
    """Return the shared environment evidence for session inspection surfaces."""
    return {
        "targets": [evidence.TARGET.dump_python(v, mode="json") for v in view.targets],
        "setups": [evidence.SETUP.dump_python(v, mode="json") for v in view.setups],
        "jobs": [evidence.JOBS.dump_python(v, mode="json") for v in view.jobs],
        "environments": [
            evidence.ENVIRONMENT.dump_python(v, mode="json") for v in view.environments
        ],
        "text": environment_summary(view),
    }


def environment_summary(view: ReceiptView | session_log.LogScan) -> str:
    """Render target, setup and lifecycle facts from the shared journal fold.

    Args:
        view: The shared session or receipt fold.

    Returns:
        Stored observations as plain text, with unknown facts named.
    """
    lines: list[str] = []
    for stamp in view.targets:
        lines.extend(evidence.target_lines(stamp.target_id, stamp.target))
        lines.append(f"Execution: {stamp.execution_id or 'unknown'}")
    if not view.targets:
        lines.append("Environment: unknown")
    for setup in view.setups:
        lines.append(
            f"Setup: {shlex.join(setup.cmd)}; "
            f"{evidence.result_label(setup.exit_code, 'unavailable')}; "
            f"Environment: {setup.target_id or 'unknown'}; "
            f"execution: {setup.execution_id or 'unknown'}"
        )
    for jobs in view.jobs:
        label = "Run jobs settled" if jobs.outcome == "settled" else "Run job settlement unknown"
        lines.append(
            f"{label} (target {jobs.target_id or 'unknown'})"
            + (f": {jobs.reason}" if jobs.reason else "")
        )
    if not view.jobs:
        lines.append("Run job settlement unknown")
    for env in view.environments:
        label = {
            "left_running": "Environment left running",
            "stopped": "Environment stopped",
            "unknown": "Environment lifecycle unknown",
            "not_applicable": "Environment lifecycle not managed",
        }[env.outcome]
        lines.append(
            f"{label} (target {env.target_id or 'unknown'})"
            + (f": {env.reason}" if env.reason else "")
        )
    if not view.environments:
        lines.append("Environment lifecycle unknown")
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
        environment_summary(view),
        "",
        "Testing (output cap: 2000 characters per stream)",
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
            f"Review: {view.review.summary}",
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
