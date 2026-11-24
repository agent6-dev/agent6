# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Merge imported children and project their dispatch outcomes into the task graph."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable, Sequence

import pydantic

from agent6 import child_records, errors, terminal
from agent6.graph import curator as graph_curator
from agent6.graph import models as graph_models
from agent6.harness import dag_focus
from agent6.sessions import child_dispatch
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import format


def merge_child_result(
    root: pathlib.Path,
    res: child_records.ChildResult,
    *,
    ref: str,
    fallback_parent: str | None,
    identity: git_read.CommitIdentity | None,
    also_branch: str | None,
) -> child_records.ChildMerge:
    """Merge one returned child's branch onto the parent's chain.

    `chain_merge` leaves HEAD and the operator's checkout alone; the worktree gains the child's
    files. Never raises; the DAG stamp is the segment's (`segment_stamp`). Stopped work
    reaches this helper only through an explicit `merge_child` request.

    Args:
        root: The repository root.
        res: The child's result.
        ref: The chain ref that receives the merge.
        fallback_parent: The parent an unborn ref merges onto, or None.
        identity: The merge commit's identity, or None for the repo's.
        also_branch: A checked-out branch to move with the ref, or None.

    Returns:
        `merged` with the sha, `no_changes` for a child whose imported tip holds no commit
        beyond its base (nothing merged), `conflict` naming the files, or `failed` for a
        child that imported nothing.
    """
    rid = res.spec.session_id
    if not res.ok:
        return child_records.ChildMerge(
            rid, res.imported_ref, "failed", "", res.error, commits=res.commits
        )
    if not res.has_commits:
        return child_records.ChildMerge(rid, res.branch, "no_changes", "", "")
    rev = res.imported_ref or res.branch
    try:
        sha = git_ops.chain_merge(
            root,
            rev,
            f"merge {res.branch}",
            ref=ref,
            fallback_parent=fallback_parent,
            identity=identity,
            also_branch=also_branch,
        )
    except (errors.GitError, OSError) as exc:
        return child_records.ChildMerge(rid, res.branch, "failed", "", str(exc))
    if sha is None:
        try:
            tip = git_read.chain_tip(root, ref) or fallback_parent
            paths = git_ops.merge_conflicts(root, tip, rev) if tip else ()
        except (errors.GitError, OSError):
            paths = ()  # the conflict stands, its files unnamed
        names = ", ".join(terminal.printable_name(path) for path in paths)
        detail = f"merge conflict in {names}" if names else "merge conflict"
        return child_records.ChildMerge(rid, res.branch, "conflict", "", detail)
    return child_records.ChildMerge(rid, res.branch, "merged", sha, "")


def stopped_merge(res: child_records.ChildResult) -> child_records.ChildMerge:
    """Return the merge of a stopped child: nothing merges, and its branch holds its commits."""
    return child_records.ChildMerge(
        res.spec.session_id,
        res.imported_ref or res.branch,
        "stopped",
        "",
        res.stopped_by,
        commits=res.commits,
        context=res.context,
    )


def stopped_words(
    stopped_by: str,
    holder: str,
    commits: int,
    context: child_records.ChildContext,
) -> str:
    """Return how a stopped child stands: what stopped it, and what holds its work.

    Args:
        stopped_by: What stopped it (`stopped by its parent`).
        holder: The name of what holds its work: its branch, or `the branch`.
        commits: The commits it made beyond its base.
        context: Its last measured conversation size and activity.

    Returns:
        The words every surface of the parent uses for a stopped child.
    """
    held = f"{holder} holds {commits} commit(s)" if holder else f"{commits} commit(s); ref unknown"
    size = f"{context.tokens} tokens" if context.tokens is not None else "unknown"
    age = (
        f"last ran {format.format_age(time.time() - context.last_run_at)} ago"
        if context.last_run_at is not None
        else "last ran: unknown"
    )
    return f"{stopped_by}, not merged; {held}; conversation at last call: {size}; {age}"


def segment_stamp(
    children: Sequence[child_records.ChildMerge],
) -> tuple[graph_models.NodeStatus, str, str]:
    """Reduce one segment's child merges to its DAG stamp.

    A finished result can merge, have no changes, or retain its ref under model-owned git.
    Only a merge records a commit on the parent task.

    Args:
        children: The segment's merges.

    Returns:
        The status, the note and the sha ("" when failed).
    """
    merged = [j for j in children if j.status == "merged"]
    note = "; ".join(child_note(j) for j in children)
    if merged or any(j.status in ("no_changes", "retained") for j in children):
        return "passed", note, merged[-1].sha if merged else ""
    return "failed", note, ""


def merge_words(j: child_records.ChildMerge) -> str:
    """Return how a child's work merged, as the background roster states it."""
    if j.status == "merged":
        return f"merged into parent chain at {j.sha[:12]}"
    if j.status == "ranked":
        return f"ranked {j.detail}; not merged; {j.branch} holds its work"
    if j.status == "retained":
        rank = f"ranked {j.detail}; " if j.detail else ""
        noun = "commit" if j.commits == 1 else "commits"
        status = "; merge unknown" if j.branch == git_read.chain_ref_for(j.session_id) else ""
        return f"{rank}retained on {j.branch} ({j.commits} {noun}){status}"
    if j.status == "stopped":
        return stopped_words(j.detail, j.branch, j.commits, j.context)
    return "no changes" if j.status == "no_changes" else j.status


def child_note(j: child_records.ChildMerge) -> str:
    """Return one child's outcome as a DAG note line."""
    if j.status in ("merged", "retained"):
        return f"{j.session_id} {merge_words(j)}"
    if j.status == "no_changes":
        return f"{j.session_id} made no changes"
    if j.status == "conflict":
        return f"{j.session_id} conflicted; merge manually"
    if j.status == "ranked":
        return f"{j.session_id} ranked {j.detail}; not merged"
    if j.status == "stopped":
        return f"{j.session_id} {j.detail}; not merged"
    return f"{j.session_id} failed: {j.detail}"


def started_text(children: Sequence[str]) -> str:
    """Return the line that says what a task started and how its parent follows it.

    Args:
        children: The started children's session ids, in child order.

    Returns:
        The line: the tool's answer, or a steer's notice.
    """
    if len(children) == 1:
        return (
            f"[parallel] started child {children[0]}. A notice arrives when it ends and when"
            " its task completes; read_background reads it and stop_background stops it, by id."
        )
    return (
        f"[parallel] started {len(children)} attempts at one task: {', '.join(children)}."
        " A notice arrives as each ends and when the task completes after ranking in the"
        " background. read_background reads one and stop_background stops one, by id."
    )


def end_notice(res: child_records.ChildResult) -> str:
    """Return the notice that tells a parent's model one child has ended.

    Args:
        res: The child's result.

    Returns:
        How it ended, with its finish summary under it as its journal holds it (2,000
        characters at most). A stopped child's line says it did not merge and what holds
        its work.
    """
    sid = res.spec.session_id
    if not res.ok:
        return f"[parallel] child {sid} ended without a result: {res.error}"
    how = "finished"
    if res.stopped_by:
        how = stopped_words(
            res.stopped_by, res.imported_ref or res.branch, res.commits, res.context
        )
    head = f"[parallel] child {sid} {how}."
    if not res.stopped_by and res.imported_ref and res.commits:
        noun = "commit" if res.commits == 1 else "commits"
        head += f" {res.imported_ref} holds {res.commits} {noun}."
    report = res.report.strip()
    return f"{head}\nsummary: " + report.replace("\n", "\n  ") if report else head


def ranked_by(ranking: child_dispatch.RankRecord) -> str:
    """Return the line that says how a task's attempts were ranked.

    Args:
        ranking: The task's ranking.

    Returns:
        The judges' reason, or that the mechanical order ranked and why.
    """
    if ranking.ranked_by == "judge":
        return "Ranked by the judges: " + ranking.rationale.replace("\n", "\n  ")
    return f"Ranked by verify result, then cost: {ranking.fallback or 'no judge ranked'}."


def summary_text(
    group: str,
    children: list[child_records.ChildMerge],
    ranking: child_dispatch.RankRecord | None = None,
) -> str:
    """Return the one notice that says how each child of a completed dispatch merged.

    Args:
        group: The dispatch's group id.
        children: Each child's merge, in dispatch order.
        ranking: How its attempts were ranked; None for a task with one child and for a panel.

    Returns:
        One line per child, how the attempts were ranked, then what to do; a dispatch
        whose every child stopped says nothing merged.
    """
    lines = [f"[parallel] group {group} complete ({len(children)} child(ren)):"]
    for j in children:
        if j.status == "merged":
            lines.append(
                f"  - {j.session_id} ({j.branch}): merged into parent chain at {j.sha[:12]}"
            )
        elif j.status == "no_changes":
            # A panel's reviewer the deadline ended did not finish: the line says so.
            ended = j.detail or "finished"
            lines.append(f"  - {j.session_id} ({j.branch}): {ended} with no changes to merge.")
        elif j.status == "retained":
            lines.append(f"  - {j.session_id}: {merge_words(j)}.")
        elif j.status == "stopped":
            words = stopped_words(j.detail, "the branch", j.commits, j.context)
            lines.append(f"  - {j.session_id} ({j.branch}): {words}.")
        elif j.status == "conflict":
            # Git is agent6's in this run and `.git` is read-only in the jail: the operator merges.
            lines.append(
                f"  - {j.session_id} ({j.branch}): CONFLICT -- {j.detail or 'merge conflict'};"
                f" nothing merged. The branch holds its work (`git merge {j.branch}` for the"
                " operator); continue without it."
            )
        elif j.status == "ranked":
            lines.append(
                f"  - {j.session_id} ({j.branch}): ranked {j.detail}; not merged, the branch"
                " holds its work."
            )
        else:
            lines.append(f"  - {j.session_id} ({j.branch}): FAILED -- {j.detail}; nothing merged.")
    if ranking is not None and ranking.order:
        lines.append(ranked_by(ranking))
    if children and all(j.status == "stopped" for j in children):
        lines.append("Every child stopped: nothing merged.")
    elif any(j.status == "merged" for j in children):
        lines.append("Review what merged and continue.")
    else:
        lines.append(
            "Agent6 made no merge."
            if any(j.status == "retained" for j in children)
            else "Nothing merged."
        )
    return "\n".join(lines)


def rank_summary(children: Sequence[child_records.ChildMerge]) -> str:
    """Return what a rank merge did, for the operator: one line per child, in child order.

    Args:
        children: Each child's merge.

    Returns:
        The count line and each child's outcome: merged, changed nothing, its place with the
        branch that holds its work, stopped, or its failure.
    """
    ranked = sum(1 for j in children if j.status in ("merged", "no_changes", "ranked", "retained"))
    lines = [f"{ranked} of {len(children)} child(ren) ranked:"]
    for j in children:
        if j.status == "merged":
            what = f"first; merged into parent chain at {j.sha[:12]}"
        elif j.status == "no_changes":
            what = "first; it changed nothing"
        elif j.status == "ranked":
            what = f"{j.detail}; {j.branch} holds its work"
        elif j.status == "retained":
            what = merge_words(j)
        elif j.status == "conflict":
            what = f"first; its merge conflicted, {j.branch} holds its work"
        elif j.status == "stopped":
            what = stopped_words(j.detail, j.branch, j.commits, j.context)
        else:
            what = j.detail or "no result"
        lines.append(f"  {j.session_id}: {what}")
    return "\n".join(lines)


def reviewer_record(
    child: child_dispatch.ChildRecord, result: child_records.ChildResult | None, tip: str
) -> child_dispatch.ChildRecord:
    """Return the reviewer's recorded outcome without changing the parent's chain."""
    merge = child_dispatch.MergeRecord(
        status="no_changes" if result is not None and result.ok else "failed",
        detail=(result.error or result.stopped_by) if result is not None else "no imported record",
        target_before=tip,
        target_after=tip,
    )
    return child.model_copy(update={"merge": merge, "report": result.report if result else ""})


def merge_of(child: child_dispatch.ChildRecord) -> child_records.ChildMerge:
    """Return a recorded child's merge; a child never merged reads as failed."""
    merge = child.merge
    if merge is None:
        return child_records.ChildMerge(child.session_id, "", "failed", "", "not merged")
    return child_records.ChildMerge(
        child.session_id,
        merge.branch,
        merge.status,
        merge.sha,
        merge.detail,
        child.report,
        commits=merge.commits,
        context=merge.context,
    )


def parallel_parent_id(
    curator: graph_curator.GraphCurator | None, root_task_id: str | None
) -> str | None:
    """Return the parent for a dispatched subtask: the cursor's open node, else the run root."""
    if curator is None:
        return root_task_id
    return dag_focus.current_task_id(curator.nodes(), curator.cursor()) or root_task_id


def add_parallel_node(
    curator: graph_curator.GraphCurator | None,
    task: str,
    parent_id: str | None,
    *,
    source: child_records.DispatchSource = "operator",
    log: Callable[[str], None],
) -> str | None:
    """Add a steering-created DAG node for one dispatched task.

    Args:
        curator: The graph, or None.
        task: The task text; its first line is the title.
        parent_id: The parent node, or None.
        source: Who dispatched it, named in the node's rationale.
        log: The run's text logger.

    Returns:
        The node's id, or None when no curator is wired or the add failed; the dispatch
        proceeds either way.
    """
    if curator is None:
        return None
    title = next((ln.strip() for ln in task.splitlines() if ln.strip()), "")[:200]
    try:
        node = curator.add_subtask(
            graph_models.AddSubtaskIntent(
                parent_id=parent_id,
                draft=graph_models.TaskNodeDraft(
                    title=title or "(parallel task)",
                    rationale=f"dispatched through /parallel by the {source}",
                    created_by="steering",
                ),
            )
        )
        return node.id
    except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
        log(f"PARALLEL: DAG node add failed: {exc}")
        return None


def stamp_parallel_node(
    curator: graph_curator.GraphCurator | None,
    node_id: str | None,
    *,
    status: graph_models.NodeStatus,
    note: str,
    sha: str = "",
    log: Callable[[str], None],
) -> None:
    """Record a dispatched node's merge sha, when given, then its final status.

    A curator fault is logged, never raised.

    Args:
        curator: The graph, or None.
        node_id: The node, or None.
        status: The final status.
        note: The status note.
        sha: The merge sha, or "".
        log: The run's text logger.
    """
    if curator is None or node_id is None:
        return
    try:
        if sha:
            curator.record_commit(graph_models.RecordCommitIntent(id=node_id, sha=sha))
        if status == "passed" and curator.get(node_id).status == "passed":
            return
        curator.update_status(
            graph_models.UpdateStatusIntent(id=node_id, new_status=status, note=note)
        )
    except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
        log(f"PARALLEL: DAG node stamp failed for {node_id}: {exc}")


def stamp_segment_node(
    curator: graph_curator.GraphCurator | None,
    node_id: str | None,
    children: list[child_records.ChildMerge],
    *,
    log: Callable[[str], None],
) -> None:
    """Stamp one segment's DAG node from its children's merges (`segment_stamp`)."""
    status, note, sha = segment_stamp(children)
    stamp_parallel_node(curator, node_id, status=status, note=note, sha=sha, log=log)
