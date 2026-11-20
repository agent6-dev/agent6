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
    files. Never raises; the DAG stamp is the segment's (`segment_stamp`). A stopped child
    never comes here: the dispatcher records it `stopped` and merges nothing.

    Args:
        root: The repository root.
        res: The child's result.
        ref: The chain ref that receives the merge.
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
        return child_records.ChildJoin(rid, res.branch, "no_changes", "", "")
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
    return child_records.ChildJoin(rid, res.branch, "joined", sha, "")


def stopped_join(res: child_records.ChildResult) -> child_records.ChildJoin:
    """Return the join of a stopped child: nothing merges, and its branch holds its commits."""
    return child_records.ChildJoin(
        res.spec.session_id,
        res.imported_ref or res.branch,
        "stopped",
        "",
        res.stopped_by,
        commits=res.commits,
    )


def stopped_words(stopped_by: str, holder: str, commits: int) -> str:
    """Return how a stopped child stands: what stopped it, and what holds its work.

    Args:
        stopped_by: What stopped it (`stopped by its parent`).
        holder: The name of what holds its work: its branch, or `the branch`.
        commits: The commits it made beyond its base.

    Returns:
        The words every surface of the parent uses for a stopped child.
    """
    held = f"{holder} holds {commits} commit(s)" if commits else "it made no commit"
    return f"{stopped_by}, not merged; {held}"


def segment_stamp(
    children: list[child_records.ChildJoin],
) -> tuple[graph_models.NodeStatus, str, str]:
    """Reduce one segment's child joins to its DAG stamp.

    A segment passes when any child joined or finished with no changes, recording the last
    joined sha (none for no changes); the note names every child. A conflict counts as not
    joined, and so does a stopped child.

    Args:
        children: The segment's joins.

    Returns:
        The status, the note and the sha ("" when failed).
    """
    joined = [j for j in children if j.status == "joined"]
    note = "; ".join(child_note(j) for j in children)
    if joined or any(j.status == "no_changes" for j in children):
        return "passed", note, joined[-1].sha if joined else ""
    return "failed", note, ""


def join_words(j: child_records.ChildJoin) -> str:
    """Return how a child's work joined, as the background roster states it."""
    if j.status == "joined":
        return f"joined at {j.sha[:12]}"
    if j.status == "ranked":
        return f"ranked {j.detail}; not merged; {j.branch} holds its work"
    if j.status == "unmerged":
        return f"not merged; {j.branch} holds {j.commits} commit(s)"
    if j.status == "stopped":
        return stopped_words(j.detail, j.branch, j.commits)
    return "no changes" if j.status == "no_changes" else j.status


def child_note(j: child_records.ChildJoin) -> str:
    """Return one child's outcome as a DAG note line."""
    if j.status in ("joined", "unmerged"):
        return f"{j.session_id} {join_words(j)}"
    if j.status == "no_changes":
        return f"{j.session_id} made no changes"
    if j.status == "conflict":
        return f"{j.session_id} conflicted; merge manually"
    if j.status == "ranked":
        return f"{j.session_id} ranked {j.detail}; not joined"
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
            " its work merges; read_background reads it and stop_background stops it, by this id."
        )
    return (
        f"[parallel] started {len(children)} attempts at one task: {', '.join(children)}."
        " A notice arrives as each ends; once all have, they are ranked in the background and"
        " the first merges. read_background reads one and stop_background stops one, by id."
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
        how = stopped_words(res.stopped_by, res.imported_ref or res.branch, res.commits)
    head = f"[parallel] child {sid} {how}."
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
    children: list[child_records.ChildJoin],
    ranking: child_dispatch.RankRecord | None = None,
) -> str:
    """Return the one notice that says how each child of a completed dispatch joined.

    Args:
        group: The dispatch's group id.
        children: Each child's join, in dispatch order.
        ranking: How its attempts were ranked; None for a task with one child and for a panel.

    Returns:
        One line per child, how the attempts were ranked, then what to do; a dispatch
        whose every child stopped says nothing merged.
    """
    lines = [f"[parallel] group {group} complete ({len(children)} child(ren)):"]
    for j in children:
        if j.status == "joined":
            lines.append(f"  - {j.session_id} ({j.branch}): joined at {j.sha[:12]}")
        elif j.status == "no_changes":
            # A panel's reviewer the deadline ended did not finish: the line says so.
            ended = j.detail or "finished"
            lines.append(f"  - {j.session_id} ({j.branch}): {ended} with no changes to join.")
        elif j.status == "unmerged":
            lines.append(
                f"  - {j.session_id}: {join_words(j)}. git.control = model: merge with git."
            )
        elif j.status == "stopped":
            words = stopped_words(j.detail, "the branch", j.commits)
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
                f"  - {j.session_id} ({j.branch}): ranked {j.detail}; not joined, the branch"
                " holds its work."
            )
        else:
            lines.append(f"  - {j.session_id} ({j.branch}): FAILED -- {j.detail}; nothing joined.")
    if ranking is not None and ranking.order:
        lines.append(ranked_by(ranking))
    if children and all(j.status == "stopped" for j in children):
        lines.append("Every child stopped: nothing merged.")
    elif any(j.status == "joined" for j in children):
        lines.append("Review what merged and continue.")
    else:
        lines.append("Nothing merged.")
    return "\n".join(lines)


def rank_summary(children: Sequence[child_records.ChildJoin]) -> str:
    """Return what a rank join did, for the operator: one line per child, in child order.

    Args:
        children: Each child's join.

    Returns:
        The count line and each child's outcome: joined, changed nothing, its place with the
        branch that holds its work, stopped, or its failure.
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
        elif j.status == "stopped":
            what = stopped_words(j.detail, j.branch, j.commits)
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
        child.session_id,
        join.branch,
        join.status,
        join.sha,
        join.detail,
        child.report,
        commits=join.commits,
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
