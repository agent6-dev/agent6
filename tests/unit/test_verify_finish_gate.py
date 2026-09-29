# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The verify finish gate: no 'passed' over a red or stale verify; `verify_when` runs it.

A red finish returns `verify_retries` times; both ground on VerifyGate.tree_green.
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal
from unittest import mock

import pytest

from agent6.config import Config
from agent6.harness import _chain, _finish_gates, _loop_state, _snapshot, _verify_verdict, loop
from agent6.prompts import loop as prompts_loop
from agent6.viewmodel import listing


def _wf(
    *,
    verify: bool,
    mode: Literal["run", "plan", "ask", "agent"] = "run",
    root: pathlib.Path = pathlib.Path("/tmp"),
) -> loop.Harness:
    data: dict[str, Any] = {"harness": {"verify_command": ["true"]}} if verify else {}
    return loop.Harness(
        chain=_chain.RunChain(root),
        config=Config.model_validate(data),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        mode=mode,
    )


def _green(wf: loop.Harness, **verdict_kw: Any) -> bool | None:
    state = _loop_state.LoopState(
        original_task="t", tool_calls=0, verify=_verify_verdict.VerifyVerdict(**verdict_kw)
    )
    return wf.gate.tree_green(state.verify)


def test_no_verify_command_is_not_gated() -> None:
    # Nothing to gate on -> None -> finish is always an honest pass.
    assert _green(_wf(verify=False), last_ok=None) is None
    assert _green(_wf(verify=False), last_ok=False) is None


def test_green_only_when_last_verify_passed_and_tree_unedited() -> None:
    wf = _wf(verify=True)
    assert _green(wf, last_ok=True, edited_since=False) is True
    # Never verified, or last verify failed -> not green.
    assert _green(wf, last_ok=None) is False
    assert _green(wf, last_ok=False) is False
    # A green verify that has since been edited over is stale -> not green.
    assert _green(wf, last_ok=True, edited_since=True) is False


def test_the_harness_gate_defaults_to_finish_with_two_returns() -> None:
    wf = Config().harness
    assert (wf.verify_when, wf.verify_retries) == ("finish", 2)


def test_a_red_gate_at_the_untouched_base_is_not_returned_to_the_worker() -> None:
    wf = _wf(verify=True)
    state = _loop_state.LoopState(
        original_task="t",
        tool_calls=0,
        verify=_verify_verdict.VerifyVerdict(last_ok=False, baseline_ok=False),
    )
    assert not _finish_gates.red_gate_returns(
        wf.config.harness.verify_when,
        wf.config.harness.verify_retries,
        state.verify,
        state.gates,
        gate_present=wf.gate.present(state.verify),
    )
    assert "untouched base" in prompts_loop.V2_VERIFY_WHEN["finish"]


def _verified(wf: loop.Harness, **verdict_kw: Any) -> str:
    state = _loop_state.LoopState(
        original_task="t", tool_calls=0, verify=_verify_verdict.VerifyVerdict(**verdict_kw)
    )
    return wf.gate.verification(state.verify)


def test_verification_carries_the_same_verdict_the_event_does() -> None:
    """SessionResult.verified copies session.end.all_passed; "failed" means an observed red.

    Exit code, auto-merge and the notify hook read it instead of `completed`; folding "no
    verify ran" into failed sent the operator to bisect a failure that never happened.
    """
    assert _verified(_wf(verify=True), last_ok=True, edited_since=False) == "passed"
    assert _verified(_wf(verify=True), last_ok=False) == "failed"
    # Red, then edited without re-verifying: the red observation stands.
    assert _verified(_wf(verify=True), last_ok=False, edited_since=True) == "failed"
    # Green but edited since: no observation covers the final tree.
    assert _verified(_wf(verify=True), last_ok=True, edited_since=True) == "unverified"
    # Never observed this execution: not red, not green.
    assert _verified(_wf(verify=True), last_ok=None) == "unverified"
    # Gateless: nothing ever gated this run, so there is no verdict to claim.
    assert _verified(_wf(verify=False), last_ok=None) == "not_applicable"


def test_a_gateless_end_and_its_verdict_agree() -> None:
    """all_passed=True needs an observed green; the gateless end carries None on the wire.

    The grounded end turned the gateless None into True while `_verification` mapped it to
    not_applicable, so the run read "passed" on every surface though nothing gated it. None
    words as "finished", never "passed" or "failed".
    """
    emitted: list[dict[str, Any]] = []

    def _capture(_type: str, **fields: Any) -> None:
        emitted.append(fields)

    cases: tuple[tuple[bool, _verify_verdict.VerifyVerdict, bool | None, str], ...] = (
        (False, _verify_verdict.VerifyVerdict(last_ok=None), None, "not_applicable"),
        (True, _verify_verdict.VerifyVerdict(last_ok=True, edited_since=False), True, "passed"),
        (True, _verify_verdict.VerifyVerdict(last_ok=False), False, "failed"),
    )
    for verify, verify_verdict, all_passed, verdict in cases:
        wf = _wf(verify=verify)
        wf.events = mock.MagicMock(emit=_capture)
        wf.events.emit = _capture  # type: ignore[method-assign]
        state = _loop_state.LoopState(original_task="t", tool_calls=0, verify=verify_verdict)
        emitted.clear()
        wf._finish(  # pyright: ignore[reportPrivateUsage]
            state,
            _snapshot.End(
                "finish_session", "", completed=True, verdict="grounded", checkpoint=False
            ),
            iteration=1,
        )
        assert emitted and emitted[-1]["all_passed"] is all_passed
        assert wf.gate.verification(state.verify) == verdict
        # The invariant the docstring promises: the event and the result agree.
        assert (emitted[-1]["all_passed"] is True) == (
            wf.gate.verification(state.verify) == "passed"
        )


def test_the_end_event_carries_whether_the_certifying_gate_ran_scoped() -> None:
    """A scoped green is a pass with a qualifier, worded "passed · scoped gate" through status_word.

    A run that ended over a red or a stale scoped gate is "finished"; the qualifier belongs to a
    pass only.
    """
    emitted: list[dict[str, Any]] = []

    def _capture(_type: str, **fields: Any) -> None:
        emitted.append(fields)

    wf = _wf(verify=True)
    wf.events = mock.MagicMock(emit=_capture)
    wf.events.emit = _capture  # type: ignore[method-assign]
    for scoped in (True, False):
        state = _loop_state.LoopState(
            original_task="t",
            tool_calls=0,
            verify=_verify_verdict.VerifyVerdict(last_ok=True, scoped=scoped),
        )
        wf._finish(  # pyright: ignore[reportPrivateUsage]
            state,
            _snapshot.End(
                "finish_session", "", completed=True, verdict="grounded", checkpoint=False
            ),
            iteration=1,
        )
        assert (emitted[-1]["all_passed"], emitted[-1]["scoped"]) == (True, scoped)
    assert listing.status_word(
        finished=True, all_passed=True, end_reason="finish_session", scoped=True
    ) == ("passed", "scoped gate")
    assert listing.status_word(
        finished=True, all_passed=False, end_reason="finish_session", scoped=True, gate_red=True
    ) == ("finished", "gate red")
    assert listing.status_word(finished=True, all_passed=False, end_reason="finish_session") == (
        "finished",
        "unverified",
    )


def test_plan_and_ask_are_never_gated_on_verify() -> None:
    """Plan and ask end clean whatever the tree looks like and report no verify verdict.

    Reporting one made `agent6 plan` exit 4: preflight infers a verify command for plan, which
    never runs it, so the tree read as red while every listing said passed.
    """
    for mode in ("plan", "ask"):
        assert _verified(_wf(verify=True, mode=mode), last_ok=None) == "not_applicable"
        assert _verified(_wf(verify=True, mode=mode), last_ok=False) == "not_applicable"


def test_a_command_that_dirties_the_tree_invalidates_the_verify_pass(
    tmp_path: pathlib.Path,
) -> None:
    """A green verify does not survive a run_command that changed the tree.

    edited_since_verify was set only by the edit tools, so a model could verify green, mutate
    through run_command or an MCP tool, and finish "passed". Grounded on git, a read-only command
    keeps the pass.
    """
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)

    wf = _wf(verify=True, root=tmp_path)
    dirty = wf._left_the_tree_dirty  # pyright: ignore[reportPrivateUsage]
    before = wf._tree_before_command  # pyright: ignore[reportPrivateUsage]

    sha = before("run_command")
    assert sha and before("mcp__srv__write") == sha
    assert dirty(sha) is False  # a read-only probe costs nothing
    assert before("read_file") == ""  # never asked of in-process read tools
    (tmp_path / "a.txt").write_text("mutated\n", encoding="utf-8")
    assert dirty(sha) is True
    # verify/metric are the operator's own gates; their caches must not
    # invalidate the pass they just produced.
    assert before("run_verify_command") == ""
    assert before("run_metric_command") == ""
    assert dirty("") is False


def test_a_read_only_command_over_uncommitted_work_keeps_the_verify_pass(
    tmp_path: pathlib.Path,
) -> None:
    """The edited check asks whether the command changed the tree, not whether it is uncommitted.

    Over work the chain had not recorded, every `rg` through run_command re-marked the tree
    edited, so `step` mode re-ran the gate on identical bytes.
    """
    _git_seed(tmp_path)
    (tmp_path / "a.txt").write_text("uncommitted\n", encoding="utf-8")
    wf = _wf(verify=True, root=tmp_path)
    assert wf.chain.dirty() is True
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_edit()
    state.verify.note_pass()
    turn = _turn()
    before = wf._tree_before_command("run_command")  # pyright: ignore[reportPrivateUsage]
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_command", _exec(0), {"argv": ["git", "diff"]}, tree_before=before
    )
    assert state.verify.green_and_untouched is True
    assert turn.edit_since_verify_pass is False
    (tmp_path / "a.txt").write_text("the command wrote this\n", encoding="utf-8")
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_command", _exec(0), {"argv": ["sed", "-i"]}, tree_before=before
    )
    assert state.verify.green_and_untouched is False
    assert turn.edit_since_verify_pass is True


def _git_seed(tmp_path: pathlib.Path) -> str:
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    out = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _snap(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "system": "s",
        "messages": [],
        "tool_calls": 0,
        "next_iteration": 1,
        "root_task_id": None,
        "original_task": "t",
        "verify_command": ("true",),
    }
    return _snapshot.SessionSnapshot(**{**base, **kw})


def _resumed_state(wf: loop.Harness, snap: Any) -> _loop_state.LoopState:
    from agent6.harness import _conversation

    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    wf._seed_carryover(state, _conversation.Conversation.from_wire([]), snap)  # pyright: ignore[reportPrivateUsage]
    return state


def test_a_resumed_execution_carries_the_verify_verdict_over_an_unmoved_tree(
    tmp_path: pathlib.Path,
) -> None:
    """The verdict carries across executions when HEAD is the snapshot's and the worktree is clean.

    Execution-scoped, resuming a green-finished run and finishing without edits read "unverified";
    baseline_ok is about the base commit, which resume never moves, so it always carries.
    """
    head = _git_seed(tmp_path)
    wf = _wf(verify=True, root=tmp_path)
    snap = _snap(head_sha=head, last_verify_ok=True, edited_since_verify=False, baseline_ok=False)
    state = _resumed_state(wf, snap)
    assert state.verify.last_ok is True
    assert state.verify.edited_since is False
    assert state.verify.baseline_ok is False
    assert wf.gate.verification(state.verify) == "passed"
    # A red observation carries the same way: the resumed execution stays answerable.
    red = _resumed_state(wf, _snap(head_sha=head, last_verify_ok=False))
    assert red.verify.last_ok is False


def test_the_carried_verdict_is_dropped_when_the_tree_moved(tmp_path: pathlib.Path) -> None:
    """An operator commit or edit between executions starts the execution unobserved.

    No observation covers this tree, so it fails closed like the baseline probe.
    """
    import subprocess as sp

    head = _git_seed(tmp_path)
    wf = _wf(verify=True, root=tmp_path)
    green = {"last_verify_ok": True, "edited_since_verify": False, "baseline_ok": True}

    # Worktree dirtied between executions.
    (tmp_path / "a.txt").write_text("edited\n", encoding="utf-8")
    state = _resumed_state(wf, _snap(head_sha=head, **green))
    assert state.verify.last_ok is None
    assert state.verify.baseline_ok is True  # the base commit did not move

    # HEAD moved forward between executions.
    sp.run(["git", "commit", "-qam", "operator work"], cwd=tmp_path, check=True)
    assert _resumed_state(wf, _snap(head_sha=head, **green)).verify.last_ok is None

    # No head recorded at write time: nothing to compare against.
    assert _resumed_state(wf, _snap(head_sha="", **green)).verify.last_ok is None


def test_a_resumed_execution_carries_the_scoped_gate(tmp_path: pathlib.Path) -> None:
    """After the full gate overran once, a resumed execution goes straight to the scoped form.

    The fact is about the suite, so it carries whatever the tree did; the verdict still drops
    when the tree moved.
    """
    _git_seed(tmp_path)
    wf = _wf(verify=True, root=tmp_path)
    (tmp_path / "a.txt").write_text("edited\n", encoding="utf-8")
    state = _resumed_state(wf, _snap(verify_scoped=True, last_verify_ok=True))
    assert state.verify.scoped is True
    assert state.verify.last_ok is None


# ---- the harness-run gate (`[harness].verify_when`) -------------------------


def _exec(rc: int, out: str = "") -> Any:
    from agent6.tools import results

    return results.ExecResult(
        returncode=rc, stdout=out, stderr="", duration_s=1.0, exec_failed=False
    )


def _harness_wf(
    when: str, retries: int = 2, *, policy: str = "yes"
) -> tuple[loop.Harness, mock.MagicMock]:
    """A run-mode loop over a gate, and the mock dispatcher that owns `run_verify`."""
    data: dict[str, Any] = {
        "harness": {"verify_command": ["true"], "verify_when": when, "verify_retries": retries}
    }
    dispatcher = mock.MagicMock()
    dispatcher.command_policy.return_value = policy
    wf = loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate(data),
        provider=mock.MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    return wf, dispatcher


def _turn(*, finishing: bool = False, edited: bool = False) -> Any:
    turn = _loop_state.TurnState(iteration=3, resp=mock.MagicMock(), assistant=mock.MagicMock())
    if finishing:
        turn.finish = _finish_gates.FinishCall("finish_session", "done")
    if edited:
        turn.edited = True
        turn.edit_since_verify_pass = True
    return turn


def _verify_gate(wf: loop.Harness, state: _loop_state.LoopState, turn: Any) -> None:
    """The verify gate's answer over *turn*, applied through the loop."""
    ctx = wf._turn_context(state, iteration=turn.iteration, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    wf._refuse(state, turn, _finish_gates.verify_finish(turn, state, ctx))  # pyright: ignore[reportPrivateUsage]


def _notices(turn: Any) -> list[str]:
    from agent6.harness import _conversation

    return [r.text for r in turn.tool_results if isinstance(r, _conversation.Notice)]


def test_finish_mode_runs_the_gate_when_a_finish_arrives_over_an_unverified_tree() -> None:
    """`verify_when = "finish"` runs the gate on finish_session, and a green run certifies the tree.

    The verdict's own bookkeeping, so the finish gate sees green and auto-commit sees a pass.
    """
    wf, dispatcher = _harness_wf("finish")
    dispatcher.run_verify.return_value = _exec(0, "3 passed")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)

    assert wf.gate.harness_verify(state, turn) is None

    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.green_and_untouched and turn.verify_just_passed
    assert _notices(turn) == ["[harness verify] finish: verify_command passed (1s).\n3 passed"]
    _verify_gate(wf, state, turn)
    assert turn.finish == _finish_gates.FinishCall("finish_session", "done")


def test_a_red_finish_certification_returns_to_the_model_verify_retries_times() -> None:
    """A red gate at finish returns the finish `verify_retries` times; the next red stands."""
    wf, dispatcher = _harness_wf("finish", retries=2)
    dispatcher.run_verify.return_value = _exec(1, "1 failed")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    seen: list[str | None] = []
    notices: list[str] = []
    for _ in range(3):
        turn = _turn(finishing=True, edited=True)
        wf.gate.harness_verify(state, turn)
        _verify_gate(wf, state, turn)
        seen.append(turn.finish.summary if turn.finish is not None else None)
        notices.extend(_notices(turn))
    assert seen == [None, None, "done"]
    assert state.gates.verify_retries_used == 2
    assert wf.gate.tree_green(state.verify) is False
    assert any("(return 1 of 2); 1 more red finish returns" in n for n in notices)
    assert any("(return 2 of 2); the next red finish ends the run" in n for n in notices)


def test_zero_retries_lets_the_first_red_finish_stand() -> None:
    wf, dispatcher = _harness_wf("finish", retries=0)
    dispatcher.run_verify.return_value = _exec(1)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn)
    _verify_gate(wf, state, turn)
    assert turn.finish == _finish_gates.FinishCall("finish_session", "done")
    assert wf.gate.verification(state.verify) == "failed"


def test_a_tree_the_model_already_certified_is_not_judged_twice() -> None:
    """Green and untouched since: the finish needs no second run.

    And a turn whose own run_verify_command judged the tree is never judged on top.
    """
    wf, dispatcher = _harness_wf("finish")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_pass()
    turn = _turn(finishing=True)
    wf.gate.harness_verify(state, turn)
    dispatcher.run_verify.assert_not_called()

    # And a RED verdict the run already holds for this tree is not re-run
    # (the finish reports the red it knows), where a green-only skip re-judged
    # it and fed the no-progress streak the one red a second time.
    wf2, dispatcher2 = _harness_wf("finish")
    state2 = _loop_state.LoopState(original_task="t", tool_calls=0)
    state2.verify.note_edit()
    state2.verify.note_fail("sig")  # the model's own red verify, tree untouched since
    wf2.gate.harness_verify(state2, _turn(finishing=True))
    dispatcher2.run_verify.assert_not_called()


def test_step_mode_judges_every_editing_turn_and_finish_mode_does_not() -> None:
    for when, calls in (("step", 1), ("finish", 0), ("never", 0)):
        wf, dispatcher = _harness_wf(when)
        dispatcher.run_verify.return_value = _exec(0)
        state = _loop_state.LoopState(original_task="t", tool_calls=0)
        wf.gate.harness_verify(state, _turn(edited=True))
        assert dispatcher.run_verify.call_count == calls, when


def test_never_mode_leaves_a_finish_over_an_unverified_tree_alone() -> None:
    """`never`: the measured model-driven shape.

    The harness neither runs the gate nor returns the finish; the end is reported finished, not
    passed.
    """
    wf, dispatcher = _harness_wf("never")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn)
    _verify_gate(wf, state, turn)
    dispatcher.run_verify.assert_not_called()
    assert turn.finish == _finish_gates.FinishCall("finish_session", "done")
    assert wf.gate.verification(state.verify) == "unverified"


def test_run_commands_no_withholds_the_gate_from_the_harness_too() -> None:
    wf, dispatcher = _harness_wf("finish", policy="no")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn)
    _verify_gate(wf, state, turn)
    dispatcher.run_verify.assert_not_called()
    assert turn.finish == _finish_gates.FinishCall("finish_session", "done")


def test_a_denied_gate_is_withheld_for_the_run_and_the_finish_stands() -> None:
    """A gate not approved under `ask` is withheld for the rest of the run, and the finish stands.

    The model is told so; bouncing the finish against a denial burned every retry on a wall
    nobody could open.
    """
    from agent6.tools import errors

    wf, dispatcher = _harness_wf("finish", retries=2)
    dispatcher.run_verify.side_effect = errors.ToolDeniedError("run_verify_command not approved")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn)
    assert _notices(turn) == [
        "[harness verify] finish: not run: run_verify_command not approved."
        " The gate is withheld for the rest of the run; the run ends unverified."
    ]
    _verify_gate(wf, state, turn)
    assert turn.finish == _finish_gates.FinishCall("finish_session", "done")  # no bounce
    assert state.gates.verify_retries_used == 0
    assert wf.gate.verification(state.verify) == "unverified"

    # A later end never re-asks: the withheld gate stays withheld.
    turn2 = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn2)
    assert dispatcher.run_verify.call_count == 1
    assert _notices(turn2) == []


def test_the_prompt_states_when_the_harness_runs_the_gate() -> None:
    from agent6 import kinds
    from agent6.harness import _prompt_blocks

    repo = kinds.RepoSummary(
        root=pathlib.Path("/tmp"),
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )

    def block(when: str, mode: Literal["run", "plan"] = "run") -> str:
        cfg = Config.model_validate(
            {"harness": {"verify_command": ["true"], "verify_when": when, "verify_retries": 1}}
        )
        return _prompt_blocks.build_system_prompt(config=cfg, repo=repo, mode=mode, skills=None)

    assert "The harness runs it when finish_session is called" in block("finish")
    assert "returns to you 1 time(s)" in block("finish")
    assert "after every turn that edits the tree" in block("step")
    assert "The harness never runs it; only your run_verify_command calls do." in block("never")
    # plan and ask never run the gate, whatever the knob says
    assert "The harness never runs it" in block("finish", mode="plan")
    # the commit fact follows: a finish-certified run commits each editing turn
    assert "commits each editing turn automatically" in block("finish")
    assert "commits pending changes automatically after each passing" in block("never")
    assert "a passing run auto-commits the step" not in block("never")


def test_a_verify_followed_by_an_edit_in_one_turn_is_judged_again() -> None:
    """Under `step` an edit after a green gate in the same turn makes the harness run it again."""
    wf, dispatcher = _harness_wf("step")
    dispatcher.run_verify.return_value = _exec(0)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(edited=True)
    turn.verify_just_passed = True  # the model's own green, then the edit
    turn.edit_since_verify_pass = True
    wf.gate.harness_verify(state, turn)
    dispatcher.run_verify.assert_called_once_with(extra_argv=())


def _scoped_wf(
    root: pathlib.Path, command: list[str], *, when: str = "finish"
) -> tuple[loop.Harness, mock.MagicMock]:
    """A harness-gated loop whose root holds pkg/mod.py + tests/test_mod.py."""
    for rel in ("pkg/mod.py", "tests/test_mod.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")
    data: dict[str, Any] = {"harness": {"verify_command": command, "verify_when": when}}
    dispatcher = mock.MagicMock()
    dispatcher.command_policy.return_value = "yes"
    wf = loop.Harness(
        chain=_chain.RunChain(root),
        config=Config.model_validate(data),
        provider=mock.MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    return wf, dispatcher


def _fake_diff(_self: loop.Harness) -> str:
    return "diff --git a/pkg/mod.py b/pkg/mod.py\n"


def test_a_timed_out_gate_reruns_scoped_to_the_nearest_tests(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A full gate that overran verify_timeout_s is re-run scoped to the tests nearest the diff.

    The verdict comes from the scoped run and the notice names the scope; later gates go straight
    to the scoped form. Big-repo executions had finished over a gate that certified nothing.
    """
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    emitted: list[tuple[str, dict[str, Any]]] = []

    def _capture(event_type: str, **fields: Any) -> None:
        emitted.append((event_type, fields))

    wf.events = mock.MagicMock(emit=_capture)
    dispatcher.run_verify.side_effect = [_exec(124), _exec(0)]
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf.gate.harness_verify(state, turn)
    assert [c.kwargs["extra_argv"] for c in dispatcher.run_verify.call_args_list] == [
        (),
        ("tests/test_mod.py",),
    ]
    assert state.verify.scoped is True
    assert turn.verify_just_passed is True
    notice = turn.tool_results[-1].text
    assert "the gate ran scoped to the tests nearest the run's change (tests/test_mod.py)" in notice
    assert "not a full-suite pass" in notice
    assert ("loop.verify_scoped", {"paths": ["tests/test_mod.py"], "iteration": 3}) in emitted
    # The next gate skips the doomed full run.
    dispatcher.run_verify.reset_mock(side_effect=True)
    dispatcher.run_verify.return_value = _exec(0)
    state.verify.note_edit()
    turn2 = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn2)
    dispatcher.run_verify.assert_called_once_with(extra_argv=("tests/test_mod.py",))


def test_a_timed_out_non_pytest_gate_stays_a_plain_timeout(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """Only pytest takes file-path selection; another gate that times out is reported as is."""
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["make", "test"])
    dispatcher.run_verify.return_value = _exec(124)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf.gate.harness_verify(state, turn)
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.scoped is False
    assert "scoped" not in turn.tool_results[-1].text


@pytest.mark.parametrize(
    "command",
    [
        ["sh", "-c", "uv run ruff check && uv run pytest"],
        ["python", "-m", "pytest", "-q", "tests"],
    ],
    ids=["sh-c-pipeline", "pytest-naming-a-path"],
)
def test_a_gate_that_cannot_take_appended_paths_stays_a_plain_timeout(
    tmp_path: pathlib.Path, monkeypatch: Any, command: list[str]
) -> None:
    """Neither a `sh -c` script nor `pytest tests` scopes.

    The script binds appended paths as $0 and $1, and the dir unions with the files: an identical
    full command, a second timeout, and a false "ran scoped" notice.
    """
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, command)
    dispatcher.run_verify.return_value = _exec(124)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf.gate.harness_verify(state, turn)
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.scoped is False
    assert _notices(turn) == ["[harness verify] finish: verify_command exit 124 (1s)."]


def test_a_timeout_with_no_nearby_tests_stands(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """With nothing near the change to run, the timeout is the verdict and scoping never arms."""

    def no_tests_diff(_self: loop.Harness) -> str:
        return "diff --git a/docs/page.md b/docs/page.md\n"

    monkeypatch.setattr(_chain.RunChain, "diff_since_base", no_tests_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.return_value = _exec(124)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf.gate.harness_verify(state, turn)
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.scoped is False
    assert turn.verify_just_failed is True


def test_a_models_own_timed_out_gate_gets_the_scoped_followup(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """run_verify_command exit 124 from the model's OWN call gets the scoped follow-up too.

    The harness-gate fallback alone never reached this flow (a self-judged turn is not re-judged),
    so pilot executions timed out at the full budget with no scoped re-run ever firing.
    """
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.return_value = _exec(0)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn()
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(124), {}
    )
    dispatcher.run_verify.assert_called_once_with(extra_argv=("tests/test_mod.py",))
    assert state.verify.scoped is True
    assert turn.verify_just_passed is True  # the scoped green stands
    assert "not a full-suite pass" in turn.tool_results[-1].text
    # One verdict per turn, as on the harness path: the 124 is not noted as a
    # fail beside the scoped green (an on_verify_fail panel and the memory
    # flip nudge key on those flags).
    assert turn.verify_just_failed is False
    assert turn.verify_flipped_green is False
    assert state.verify.fail_streak == 0


def test_never_mode_leaves_the_models_timed_out_gate_alone(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """Under `never` a timeout in the model's own run_verify_command gets no harness re-run."""
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"], when="never")
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn()
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(124), {}
    )
    dispatcher.run_verify.assert_not_called()
    assert state.verify.scoped is False
    assert turn.verify_just_failed is True


def test_a_full_green_from_the_models_own_gate_unarms_scoping(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A green from the model's own full run_verify_command ends scoping and reads "passed"."""
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.return_value = _exec(0)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, _turn(), "run_verify_command", _exec(124), {}
    )
    assert state.verify.scoped is True
    turn = _turn()
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(0), {}
    )
    assert state.verify.scoped is False
    assert turn.verify_just_passed is True
    dispatcher.run_verify.assert_called_once_with(extra_argv=("tests/test_mod.py",))


def test_a_denied_scoped_rerun_withholds_the_gate_for_the_run(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """One denial means the same on both call sites, and a later finish never asks again."""
    from agent6.tools import errors

    monkeypatch.setattr(_chain.RunChain, "diff_since_base", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.side_effect = [
        _exec(124),
        errors.ToolDeniedError("run_verify_command not approved"),
    ]
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf.gate.harness_verify(state, turn)
    assert state.verify.denied is True
    assert (
        "[verify] scoped re-run: not run: run_verify_command not approved."
        " The gate is withheld for the rest of the run; the run ends unverified."
    ) in _notices(turn)
    turn2 = _turn(finishing=True, edited=True)
    wf.gate.harness_verify(state, turn2)
    assert dispatcher.run_verify.call_count == 2
    _verify_gate(wf, state, turn2)
    assert turn2.finish == _finish_gates.FinishCall("finish_session", "done")


def test_a_silent_finish_over_a_standing_red_is_handed_back() -> None:
    """The model ran the gate itself and it was red; its next turn is prose with no tool call.

    The harness gate is skipped (that red already covers the untouched tree), so no verify fails on
    THIS turn, and the end was accepted as if the gate had never been red. The standing verdict
    decides.
    """
    from agent6.tools import results

    dispatcher = mock.MagicMock()
    dispatcher.command_policy.return_value = "yes"
    dispatcher.run_verify.return_value = results.ExecResult(
        returncode=1, stdout="1 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    wf = loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate(
            {"harness": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 2}}
        ),
        provider=mock.MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_edit()
    state.verify.note_fail("sig")
    turn = _loop_state.TurnState(iteration=4, resp=mock.MagicMock(), assistant=mock.MagicMock())

    wf._end_gates(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        wf._turn_context(state, iteration=turn.iteration, execution_start=1),  # pyright: ignore[reportPrivateUsage]
        ending="silent_finish",
        gates=_finish_gates.SILENT_END_GATES,
    )

    assert dispatcher.run_verify.call_count == 0, "the standing red covers the tree"
    assert turn.end_returned is True
    assert state.gates.verify_retries_used == 1


@pytest.mark.parametrize(("policy", "denied"), [("no", False), ("ask", True)])
def test_a_silent_end_is_not_handed_back_over_a_gate_the_model_cannot_run(
    policy: str, denied: bool
) -> None:
    """The standing red decides the hand-back, with finish_session's own guards.

    A withheld or denied gate is not the model's to fix, and bouncing the end told it to.
    """
    dispatcher = mock.MagicMock()
    dispatcher.command_policy.return_value = policy
    wf = loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate(
            {"harness": {"verify_command": ["true"], "verify_when": "step", "verify_retries": 2}}
        ),
        provider=mock.MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_edit()
    state.verify.note_fail("sig")
    state.verify.denied = denied
    state.verify.note_edit()
    turn = _loop_state.TurnState(iteration=7, resp=mock.MagicMock(), assistant=mock.MagicMock())

    wf._end_gates(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        wf._turn_context(state, iteration=turn.iteration, execution_start=1),  # pyright: ignore[reportPrivateUsage]
        ending="silent_finish",
        gates=_finish_gates.SILENT_END_GATES,
    )

    assert turn.end_returned is False
    assert dispatcher.run_verify.call_count == 0
