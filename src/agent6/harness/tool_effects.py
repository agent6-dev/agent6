# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Note what a dispatched tool did to the turn.

The verify result, the metric sample, the tree edit, the memory touch, the finish and the
error each feed the commit, the review panel and the finish gates through the turn's state.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent6 import paths, task_text
from agent6.harness import dag_focus, finish_gates, loop_state, memory_touch, nudges
from agent6.sessions import journal
from agent6.tools import mcp_client, results, tool_errors

if TYPE_CHECKING:
    from agent6.harness import loop


def left_out_note(
    wf: loop.Harness,
    turn: loop_state.TurnState,
    name: str,
    result: results.ToolResult,
    tool_input: Any,
) -> str:
    """Return the sentence for an edit the run's commits leave out, and log the paths.

    Args:
        wf: The harness driving the execution.
        turn: This turn's state.
        name: The tool that ran.
        result: The dispatched tool's result.
        tool_input: The model's input to the tool.

    Returns:
        One sentence per left-out path; "" when the commits hold every path written.
    """
    if wf.mode != "run":
        return ""
    # A memory write names its file inside the store, which is no path of the repository.
    touch = memory_touch.memory_store_touch(wf.state_dir, name, result, tool_input)
    if touch is not None and touch.whole_store:
        return ""
    store_paths = () if touch is None else touch.store_paths
    if isinstance(result, results.EditResult):
        paths: tuple[str, ...] = (result.path,)
    elif isinstance(result, results.PatchResult):
        written = tuple(p for p, _ in result.files) or (result.path,)
        paths = tuple(p for p in written if p not in result.deleted and p not in store_paths)
    else:
        return ""
    ignored, untracked = wf.chain.left_out(paths)
    if not (ignored or untracked):
        return ""
    wf.emit(
        journal.CheckpointLeftOut(
            turn=turn.number,
            ignored=list(ignored),
            untracked=list(untracked),
        )
    )
    tail = ": the edit is on disk, and this run's commits leave it out."
    return " ".join(
        [f"{p} is ignored by git{tail}" for p in ignored]
        + [f"{p} was untracked when the run started{tail}" for p in untracked]
    )


def tree_before_command(wf: loop.Harness, name: str) -> str:
    """Return the tree's content sha before a child-process tool's call.

    The operator's own gates are excluded: the caches they drop must not invalidate
    the pass they just produced.

    Args:
        wf: The harness driving the execution.
        name: The tool about to run.

    Returns:
        The sha, or "" for a tool that cannot touch the tree.
    """
    if name != "run_command" and not name.startswith(mcp_client.MCP_TOOL_PREFIX):
        return ""
    return wf.chain.tree_sha()


def _left_the_tree_dirty(wf: loop.Harness, tree_before: str) -> bool:
    """Report whether a child-process tool changed the tree.

    Git decides, so a read-only probe costs its pass nothing and gitignored build
    output never counts.

    Args:
        wf: The harness driving the execution.
        tree_before: The sha from `tree_before_command`; "" reads as unchanged.

    Returns:
        True when the tree's content sha differs from `tree_before`.
    """
    if not tree_before:
        return False
    after = wf.chain.tree_sha()
    return bool(after) and after != tree_before


def note(
    wf: loop.Harness,
    state: loop_state.LoopState,
    turn: loop_state.TurnState,
    name: str,
    result: results.ToolResult,
    tool_input: Any,
    tree_before: str = "",
) -> None:
    """Note a dispatched tool's effects on the turn.

    Verify results, metric samples, tree edits and graph mutations feed the commit,
    the review panel and the finish gates.

    Args:
        wf: The harness driving the execution.
        state: The execution's state.
        turn: This turn's state.
        name: The tool that ran.
        result: Its result.
        tool_input: Its input.
        tree_before: The sha from `tree_before_command` for a child-process tool.
    """
    if (
        name == "ask_user"
        and isinstance(result, results.AnswersResult)
        and result.answers is not None
    ):
        # The result carries the questions the dispatcher accepted; the input is not reparsed.
        for question, answer in zip(result.asked, result.answers, strict=False):
            wf.record_decision(state, question, answer)
    state.command_cut = state.command_cut or (
        isinstance(result, results.ExecResult) and result.stopped_by_operator
    )
    if name == "run_verify_command" and isinstance(result, results.ExecResult):
        # A timed-out model gate gets the scoped follow-up, whatever the gate policy.
        if not (
            wf.mode == "run"
            and result.timed_out
            and not state.verify.scoped
            and wf.gate.scoped_followup(state, turn) is not None
        ):
            if result.returncode == 0:
                # The model's call runs the full argv, so its green is a full pass.
                state.verify.scoped = False
            wf.gate.note_result(state, turn, result)
    elif name == "run_metric_command" and isinstance(result, results.MetricResult):
        turn.metric_sampled = True
        # The tree this reading covers, so the auto path does not sample it again.
        state.metric.tree = wf.chain.tree_sha()
        turn.metric_feedback = wf.metrics.record(
            state.metric.history,
            result,
            turn=turn.number,
            label=f"manual turn {turn.number}",
            sha="",
        )
        if turn.verify_just_passed:
            turn.metric_plateau_finish = wf.metrics.plateau_finish(state.metric.history)
    turn.waited_on_background |= (
        name == "run_command"
        and isinstance(result, results.ExecResult)
        and bool(result.background_id)
    ) or (
        name == "read_background"
        and isinstance(result, results.BackgroundResult)
        and result.running
    )
    if name in ("apply_edit", "apply_patch") and isinstance(result, results.PreviewResult):
        return  # a dry run writes nothing: no memory write, no tree edit
    if _note_memory_touch(wf, state, name, result, tool_input):
        # A memory write is not workspace work: the tree bookkeeping below does not apply.
        turn.wrote_memory = True
        return
    if name in ("apply_edit", "apply_patch"):
        turn.edited = True
        state.ever_edited = True
        # A same-turn verify pass no longer covers this tree.
        turn.edit_since_verify_pass = True
        state.verify.note_edit()
    elif _left_the_tree_dirty(wf, tree_before):
        # A command or an MCP tool changed the tree, so a green verify no longer covers it.
        turn.edit_since_verify_pass = True
        state.verify.note_edit()
    if name in dag_focus.DAG_MUTATING_TOOLS:
        turn.dag_mutated = True  # snapshot once after the turn


def _note_memory_touch(
    wf: loop.Harness,
    state: loop_state.LoopState,
    name: str,
    result: results.ToolResult,
    tool_input: Any,
) -> bool:
    """Count a read of a memory fact, or record an edit tool's write to the store.

    Args:
        wf: The harness driving the execution.
        state: The execution's state; holds the memory bookkeeping.
        name: The tool that ran.
        result: Its result.
        tool_input: Its input.

    Returns:
        True for a write to the store, which is not workspace work.
    """
    touch = memory_touch.memory_store_touch(wf.state_dir, name, result, tool_input)
    if touch is None:
        return False
    if name == "read_file":
        for fact in touch.facts:
            state.memory.read[fact] = state.memory.read.get(fact, 0) + 1
        return False
    if name in ("apply_edit", "apply_patch"):
        state.memory.written = True
        for fact, op in touch.facts.items():
            state.memory.note_write(fact, op)
        return touch.whole_store
    return False


def capture_finish(
    wf: loop.Harness,
    state: loop_state.LoopState,
    turn: loop_state.TurnState,
    name: str,
    tool_input: Any,
) -> None:
    """Record a dispatched finish on the turn; a finish_planning also writes its plan.

    The finish gates may still revoke it; a gate proposal it carried stays for the end.

    Args:
        wf: The harness driving the execution.
        state: The execution's state.
        turn: This turn's state.
        name: The tool that ran.
        tool_input: Its input.
    """
    finish = finish_gates.FinishCall.parse(name, tool_input)
    if finish is None:
        return
    turn.finish = finish
    if finish.stale_gate:
        state.gates.stale_gate = finish.stale_gate
    if finish.kind != "finish_planning":
        return
    if finish.plan_salvaged:
        wf.log("  plan salvaged: folded summary into a title-only plan_markdown")
    if wf.plan_output_path is None or not finish.plan_markdown:
        return
    plan_md = finish_gates.with_original_task(
        finish.plan_markdown,
        state.operator_task or task_text.operator_task_text(state.original_task),
    )
    try:
        paths.mkdir_for_real_user(wf.plan_output_path.parent)
        wf.plan_output_path.write_text(plan_md, encoding="utf-8")
        wf.log(f"  plan written: {wf.plan_output_path} ({len(plan_md)} chars)")
        wf.emit(
            journal.PlanWritten(
                path=str(wf.plan_output_path),
                bytes=len(plan_md.encode("utf-8")),
            )
        )
    except OSError as exc:
        wf.log(f"  plan write failed: {exc}")
        wf.emit(journal.PlanWriteFailed(path=str(wf.plan_output_path), error=str(exc)))


def note_tool_error(
    wf: loop.Harness,
    state: loop_state.LoopState,
    name: str,
    tool_input: dict[str, Any],
    exc: tool_errors.ToolError,
) -> str:
    """Note one failed dispatch for the spiral guard and the reachability note.

    Args:
        wf: The harness driving the execution.
        state: The execution's state.
        name: The tool that failed.
        tool_input: Its input.
        exc: The error.

    Returns:
        The error content served as the tool result.
    """
    content = json.dumps({"error": str(exc)})
    wf.log(f"  tool_error: {name}: {exc}")
    state.spiral.note_error(
        nudges.tool_error_signature(name, str(exc)),
        denial=isinstance(exc, tool_errors.ToolDeniedError),
        content=content,
    )
    return content
