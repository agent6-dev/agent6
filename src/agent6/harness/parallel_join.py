# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Join imported children and project their dispatch outcomes into the task graph."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence

import pydantic

from agent6 import child_records, errors, terminal
from agent6.graph import curator as graph_curator
from agent6.graph import models as graph_models
from agent6.harness import dag_focus
from agent6.sessions import child_dispatch
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


def join_child_result(
    root: pathlib.Path,
    res: child_records.ChildResult,
    *,
    ref: str,
    fallback_parent: str | None,
    identity: git_read.CommitIdentity | None,
    also_branch: str | None,
) -> child_records.ChildJoin:
    """Join one returned child's branch onto the parent's chain.

    `chain_merge` leaves HEAD and the operator's checkout alone; the worktree gains the child's
    files. Never raises; the DAG stamp is the segment's (`segment_stamp`).

    Args:
        root: The repository root.
        res: The child's result.
        ref: The chain ref the merge lands on.
        fallback_parent: The parent an unborn ref merges onto, or None.
        identity: The merge commit's identity, or None for the repo's.
        also_branch: A checked-out branch to move with the ref, or None.

    Returns:
        `joined` with the sha, `no_changes` for a child whose imported tip holds no commit
        beyond its base (nothing merged), `conflict` naming the files, or `failed` for a
        child that imported nothing.
    """
    rid = res.spec.session_id
    if not res.ok:
        return child_records.ChildJoin(rid, res.branch, "failed", "", res.error)
    if not res.has_commits:
        return child_records.ChildJoin(rid, res.branch, "no_changes", "", res.stopped_by)
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
        return child_records.ChildJoin(rid, res.branch, "failed", "", str(exc))
    if sha is None:
        try:
            tip = git_read.chain_tip(root, ref) or fallback_parent
            paths = git_ops.merge_conflicts(root, tip, rev) if tip else ()
        except (errors.GitError, OSError):
            paths = ()  # the conflict stands, its files unnamed
        names = ", ".join(terminal.printable_name(path) for path in paths)
        detail = f"merge conflict in {names}" if names else "merge conflict"
        return child_records.ChildJoin(rid, res.branch, "conflict", "", detail)
    return child_records.ChildJoin(rid, res.branch, "joined", sha, res.stopped_by)


def segment_stamp(
    children: list[child_records.ChildJoin],
) -> tuple[graph_models.NodeStatus, str, str]:
    """Reduce one segment's child joins to its DAG stamp.

    A segment passes when any child joined or finished with no changes, recording the last
    joined sha (none for no changes); the note names every child. A conflict counts as not
    joined, and so does a child its parent stopped before it changed anything.

    Args:
        children: The segment's joins.

    Returns:
        The status, the note and the sha ("" when failed).
    """
    joined = [j for j in children if j.status == "joined"]
    note = "; ".join(child_note(j) for j in children)
    if joined or any(j.status == "no_changes" and not j.detail for j in children):
        return "passed", note, joined[-1].sha if joined else ""
    return "failed", note, ""


def join_words(j: child_records.ChildJoin) -> str:
    """Return how a child's work joined, as the background roster states it."""
    if j.status == "joined":
        return f"joined at {j.sha[:12]}"
    if j.status == "ranked":
        return f"ranked {j.detail}"
    return "no changes" if j.status == "no_changes" else j.status


def child_note(j: child_records.ChildJoin) -> str:
    """Return one child's outcome as a DAG note line."""
    stopped = f" ({j.detail})" if j.detail else ""
    if j.status == "joined":
        return f"{j.session_id} joined at {j.sha[:12]}{stopped}"
    if j.status == "no_changes":
        return f"{j.session_id} made no changes{stopped}"
    if j.status == "conflict":
        return f"{j.session_id} conflicted; merge manually"
    if j.status == "ranked":
        return f"{j.session_id} ranked {j.detail}; not joined"
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
            " its work lands; read_background reads it and stop_background stops it, by this id."
        )
    return (
        f"[parallel] started {len(children)} attempts at one task: {', '.join(children)}."
        " A notice arrives as each ends; once all have, they are ranked and the first lands."
        " read_background reads one and stop_background stops one, by id."
    )


def end_notice(res: child_records.ChildResult) -> str:
    """Return the notice that tells a parent's model one child has ended.

    Args:
        res: The child's result.

    Returns:
        How it ended, with its finish summary under it as its journal holds it (2,000
        characters at most).
    """
    sid = res.spec.session_id
    if not res.ok:
        return f"[parallel] child {sid} ended without a result: {res.error}"
    head = f"[parallel] child {sid} {res.stopped_by or 'finished'}."
    report = res.report.strip()
    return f"{head}\nsummary: " + report.replace("\n", "\n  ") if report else head


def summary_text(group: str, children: list[child_records.ChildJoin]) -> str:
    """Return the one notice that says how each child of a landed dispatch joined.

    Args:
        group: The dispatch's group id.
        children: Each child's join, in dispatch order.

    Returns:
        One line per child.
    """
    lines = [f"[parallel] group {group} complete ({len(children)} child(ren)):"]
    for j in children:
        if j.status == "joined":
            stopped = f" ({j.detail})" if j.detail else ""
            lines.append(f"  - {j.session_id} ({j.branch}): joined at {j.sha[:12]}{stopped}")
        elif j.status == "no_changes":
            # A child its parent stopped did not finish: the line says what stopped it.
            ended = j.detail or "finished"
            lines.append(f"  - {j.session_id} ({j.branch}): {ended} with no changes to join.")
        elif j.status == "conflict":
            # Git is agent6's in this run and `.git` is read-only in the jail: the operator merges.
            lines.append(
                f"  - {j.session_id} ({j.branch}): CONFLICT -- {j.detail or 'merge conflict'};"
                f" nothing landed. The branch holds its work (`git merge {j.branch}` for the"
                " operator); continue without it."
            )
        elif j.status == "ranked":
            lines.append(
                f"  - {j.session_id} ({j.branch}): ranked {j.detail}; not joined, the branch"
                " holds its work."
            )
        else:
            lines.append(f"  - {j.session_id} ({j.branch}): FAILED -- {j.detail}; nothing joined.")
    lines.append("Review what landed and continue.")
    return "\n".join(lines)


def rank_summary(children: Sequence[child_records.ChildJoin]) -> str:
    """Return what a rank join did, for the operator: one line per child, in child order.

    Args:
        children: Each child's join.

    Returns:
        The count line and each child's outcome: joined, changed nothing, its place with the
        branch that holds its work, or its failure.
    """
    ranked = sum(1 for j in children if j.status in ("joined", "no_changes", "ranked"))
    lines = [f"{ranked} of {len(children)} child(ren) ranked:"]
    for j in children:
        if j.status == "joined":
            what = f"first; joined at {j.sha[:12]}"
        elif j.status == "no_changes":
            what = "first; it changed nothing"
        elif j.status == "ranked":
            what = f"{j.detail}; {j.branch} holds its work"
        elif j.status == "conflict":
            what = f"first; its merge conflicted, {j.branch} holds its work"
        else:
            what = j.detail or "no result"
        lines.append(f"  {j.session_id}: {what}")
    return "\n".join(lines)


def join_of(child: child_dispatch.ChildRecord) -> child_records.ChildJoin:
    """Return a recorded child's join; a child never joined reads as failed."""
    join = child.join
    if join is None:
        return child_records.ChildJoin(child.session_id, "", "failed", "", "not joined")
    return child_records.ChildJoin(
        child.session_id, join.branch, join.status, join.sha, join.detail, child.report
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
    """Record a dispatched node's join sha, when given, then its final status.

    A curator fault is logged, never raised.

    Args:
        curator: The graph, or None.
        node_id: The node, or None.
        status: The final status.
        note: The status note.
        sha: The join sha, or "".
        log: The run's text logger.
    """
    if curator is None or node_id is None:
        return
    try:
        if sha:
            curator.record_commit(graph_models.RecordCommitIntent(id=node_id, sha=sha))
        curator.update_status(
            graph_models.UpdateStatusIntent(id=node_id, new_status=status, note=note)
        )
    except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
        log(f"PARALLEL: DAG node stamp failed for {node_id}: {exc}")


def stamp_segment_node(
    curator: graph_curator.GraphCurator | None,
    node_id: str | None,
    children: list[child_records.ChildJoin],
    *,
    log: Callable[[str], None],
) -> None:
    """Stamp one segment's DAG node from its children's joins (`segment_stamp`)."""
    status, note, sha = segment_stamp(children)
    stamp_parallel_node(curator, node_id, status=status, note=note, sha=sha, log=log)
