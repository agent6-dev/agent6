# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Alignment-guard sub-agent.

A small, cheap call (usually Haiku) that checks whether a proposed action on
a `TaskNode` is still aligned with the original root task. Triggered:

  - Before a worker executes a node (catches "this subtask is premature").
  - When any sub-agent calls `add_subtask` (catches scope creep).
  - Periodically (every N nodes) against the original root task (drift check).
  - On `agent6 run --resume` with a `ResumeDiff` (catches "user changed code
    under our feet").

Returns an `AlignmentVerdict` discriminated by `.verdict`. See
`agent6.models.AlignmentVerdict` for the verdict semantics.
"""

from __future__ import annotations

from agent6.agents._common import call_for_model
from agent6.graph.models import ResumeDiff, TaskNode
from agent6.models import AlignmentAction, AlignmentVerdict
from agent6.providers import Provider

_SYSTEM = """You are the alignment guard for a coding agent.

Your job is to check whether a proposed action on a task node is still
aligned with the user's original root task. You see:
  - the original root task,
  - the current task node (title + rationale + acceptance),
  - the path of ancestor nodes from the root,
  - the proposed action ("expand", "execute", "add_subtask", "skip", "resume"),
  - optionally a structured ResumeDiff describing what the user changed
    under the agent's feet between runs.

Return one verdict:
  - "proceed"          : carry on with the proposed action.
  - "reorder"          : keep the node but execute siblings in a different
                          order; populate suggested_reorder with sibling ids.
  - "reject"           : this node is misaligned; mark it obsolete.
  - "re-plan-subtree"  : this subtree is invalid; planner should revise it.
  - "re-plan-root"     : the entire plan is invalid; back to plan mode.
  - "ask"              : diff is large or ambiguous; ask the user.

Be conservative. "proceed" is the default when the node is plainly on-task.
Use "reject" only for clearly off-topic or out-of-scope nodes. Always
populate `reasoning` (one short paragraph)."""


def _format_node(node: TaskNode) -> str:
    lines = [f"  id: {node.id}", f"  title: {node.title}"]
    if node.rationale:
        lines.append(f"  rationale: {node.rationale}")
    if node.acceptance:
        lines.append(f"  acceptance: {node.acceptance}")
    if node.relevant_paths:
        lines.append(f"  relevant_paths: {list(node.relevant_paths)}")
    lines.append(f"  status: {node.status}")
    lines.append(f"  created_by: {node.created_by}")
    return "\n".join(lines)


def _format_resume_diff(diff: ResumeDiff) -> str:
    parts = [
        f"  snapshot_head: {diff.snapshot_head}",
        f"  current_head:  {diff.current_head}",
        f"  snapshot_missing: {diff.snapshot_missing}",
        f"  guard_summary: {diff.guard_summary}",
    ]
    cd = diff.committed_delta
    if cd.files:
        parts.append(f"  committed_delta: {cd.from_sha[:8]}..{cd.to_sha[:8]} ({len(cd.files)})")
        for f in cd.files[:20]:
            parts.append(f"    - {f}")
    if diff.uncommitted_diff:
        parts.append(f"  uncommitted_diff: {len(diff.uncommitted_diff)} file(s)")
        for u in diff.uncommitted_diff[:20]:
            parts.append(f"    - {u.path} ({u.note or 'changed'})")
    return "\n".join(parts)


def alignment_check(
    provider: Provider,
    *,
    node: TaskNode,
    parent_path: tuple[TaskNode, ...],
    original_task: str,
    proposed_action: AlignmentAction,
    resume_diff: ResumeDiff | None = None,
) -> AlignmentVerdict:
    ancestors = (
        "\n".join(f"  {i + 1}. {p.title}" for i, p in enumerate(parent_path))
        if parent_path
        else "  (root)"
    )
    diff_block = (
        f"\n\nRESUME DIFF:\n{_format_resume_diff(resume_diff)}" if resume_diff is not None else ""
    )
    user = (
        f"ORIGINAL ROOT TASK:\n{original_task}\n\n"
        f"PARENT PATH (root -> node parent):\n{ancestors}\n\n"
        f"CURRENT NODE:\n{_format_node(node)}\n\n"
        f"PROPOSED ACTION: {proposed_action}{diff_block}\n\n"
        "Emit your AlignmentVerdict."
    )
    return call_for_model(
        provider,
        system=_SYSTEM,
        user=user,
        output_model=AlignmentVerdict,
        max_tokens=1024,
    )
