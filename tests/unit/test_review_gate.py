# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-loop review gate: how the loop schedules and honours panel verdicts.

The panel itself (grounding, block_on modes, dedup) is pinned in
test_review_panel.py; here the LOOP's plumbing is the unit, a NEEDS-WORK
verdict revokes finish_session and injects the findings, the rejection cap
disarms the gate, trigger "off" never runs a panel, and the periodic /
`verify_fail` panels fire only on their schedule. The panel is stubbed at
the `Reviewer.critique` seam.
"""

from __future__ import annotations

import json
import pathlib
import types
from typing import Any
from unittest import mock

import pytest

from agent6 import review_records
from agent6.harness import (
    loop,
    loop_conversation,
    loop_state,
    provider_call,
    quiet_turns,
    review_gate,
    run_chain,
)
from agent6.providers import ProviderResponse
from agent6.sessions import event_log, journal
from agent6.tools import results
from tests import review_lane_env

# The `[git]` surface the loop reads: the checkpoint message and the commit
# identity (`commit_identity`), empty as a real Config carries it unset.
_GIT_STUB = types.SimpleNamespace(
    commit=types.SimpleNamespace(
        checkpoint=types.SimpleNamespace(message="agent6"), name="", email="", trailer=""
    )
)


def _silent(_msg: str) -> None:
    return None


def _wf(
    root: pathlib.Path | None = None,
    *,
    ref: str | None = None,
    fallback_parent: str | None = None,
    branch: str | None = None,
    per_turn: bool = True,
    base_sha: str = "",
    **kw: Any,
) -> loop.Harness:
    defaults: dict[str, Any] = {
        "chain": run_chain.RunChain(
            root or pathlib.Path("/tmp"),
            ref=ref,
            branch=branch,
            fallback_parent=fallback_parent,
            per_turn=per_turn,
            base_sha=base_sha,
        ),
        "config": mock.MagicMock(
            git=_GIT_STUB,
            prompt=mock.MagicMock(system_prompt_file=""),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                settled_stop_after=12,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_at_start=True,
                verify_retries=2,
            ),
        ),
        "provider": mock.MagicMock(),
        "dispatcher": mock.MagicMock(
            last_verification=None,
            execution_id="execution",
            verify_withheld=False,
        ),
        "logger": _silent,
        "call": provider_call.CallSettings(retry_delay_s=0.01),
        "review": review_gate.ReviewSettings(reviewers=[mock.MagicMock()]),
    }
    defaults.update(kw)
    return loop.Harness(**defaults)


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def _finish_tool_use(tool_id: str, summary: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": "finish_session",
        "input": {"summary": summary},
    }


def _resp_with_tool_use(text: str, tool_use: dict[str, Any]) -> ProviderResponse:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks.append({"type": "tool_use", **tool_use})
    return ProviderResponse(
        text=text,
        tool_uses=(tool_use,),
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": blocks},
    )


class _PanelScript:
    """A scripted `Reviewer.critique` stand-in.

    Pops verdicts in order and counts how often the loop consulted the panel.
    """

    def __init__(self, verdicts: list[review_gate.CritiqueResult | None]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    def __call__(self, _state: Any, *, trigger: str, turn: int) -> Any:
        self.calls += 1
        return self.verdicts.pop(0) if self.verdicts else None


_MSGS: list[dict[str, Any]] = [
    {"role": "user", "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}]}
]


def test_before_finish_panel_revokes_finish_and_injects_findings() -> None:
    """A NEEDS-WORK panel verdict on finish_session revokes the finish and injects the findings.

    The tool_result still returns so the call is not half-applied; the findings ride into the next
    user message under [review].
    """
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("attempting to finish", _finish_tool_use("tu1", "wrap up")),
        _resp("ok, looks good"),
        _resp("ok, looks good"),
        _resp("ok, looks good"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = results.RawResult({"ok": True})
    panel = _PanelScript(
        [
            review_gate.CritiqueResult(text="* still TODOs left", satisfied=False),
            review_gate.CritiqueResult(text="* now fine", satisfied=True),
        ]
    )
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        conversation = loop_conversation.Conversation.from_wire(_MSGS)
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.turns == 4
    assert result.reason == "silent_finish"
    assert panel.calls == 2
    iter1_user_msg = conversation.to_wire()[2]
    assert iter1_user_msg["role"] == "user"
    blocks = [b for b in iter1_user_msg["content"] if b.get("type") == "text"]
    assert any("[review]" in b["text"] for b in blocks)
    assert any("TODOs" in b["text"] for b in blocks)


def test_before_finish_panel_satisfied_accepts_finish() -> None:
    worker = mock.MagicMock()
    worker.call.return_value = _resp_with_tool_use(
        "wrapping up", _finish_tool_use("tu1", "all done")
    )
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = results.RawResult({"ok": True})
    panel = _PanelScript([review_gate.CritiqueResult(text="* clean", satisfied=True)])
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=loop_conversation.Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "finish_session"
    assert result.turns == 1
    assert panel.calls == 1


def test_before_finish_rejection_cap_lets_finish_through() -> None:
    """After `max_review_rejections` rejected finishes the next one goes through.

    The findings are still injected; the worker cannot bounce forever. The panel past the cap
    reports its block as `capped`, which the gate reads as the end standing.
    """
    worker = mock.MagicMock()
    worker.call.return_value = _resp_with_tool_use("finishing", _finish_tool_use("tu1", "done"))
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = results.RawResult({"ok": True})
    rejected = review_gate.CritiqueResult(text="* nope", satisfied=False)
    capped = review_gate.CritiqueResult(text="* nope", satisfied=True, capped=True)
    panel = _PanelScript([rejected, rejected, capped])
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(
            when="finish", max_review_rejections=2, reviewers=[mock.MagicMock()]
        ),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=loop_conversation.Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "finish_session"
    assert panel.calls == 3  # two rejections, then the cap admits the third


def test_trigger_off_never_runs_a_panel() -> None:
    worker = mock.MagicMock()
    worker.call.return_value = _resp_with_tool_use("done", _finish_tool_use("tu1", "d"))
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = results.RawResult({"ok": True})
    panel = _PanelScript([])
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="never", reviewers=[mock.MagicMock()]),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=loop_conversation.Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "finish_session"
    assert panel.calls == 0


def test_silent_finish_panel_revokes_and_continues() -> None:
    """A prose-only finish is also gated.

    Rejected once, the run continues with the findings visible; a later pass exits.
    """
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use(
            "edit",
            {
                "type": "tool_use",
                "id": "t1",
                "name": "apply_edit",
                "input": {"path": "a", "edits": []},
            },
        ),
        _resp("i think we are done"),
        _resp("still done"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = results.RawResult({"ok": True})
    panel = _PanelScript(
        [
            review_gate.CritiqueResult(text="* not yet", satisfied=False),
            review_gate.CritiqueResult(text="* fine now", satisfied=True),
        ]
    )
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        conversation = loop_conversation.Conversation.from_wire(_MSGS)
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "silent_finish"
    assert panel.calls == 2
    wire = conversation.to_wire()
    joined = " ".join(b.get("text", "") for m in wire for b in m["content"] if isinstance(b, dict))
    assert "[review]" in joined and "not yet" in joined


def _verify_pass_tool_use(tu_id: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tu_id,
        "name": "run_verify_command",
        "input": {},
    }


def _exec(returncode: int, stderr: str = "") -> results.ExecResult:
    return results.ExecResult(
        returncode=returncode, stdout="", stderr=stderr, duration_s=0.0, exec_failed=False
    )


def test_periodic_panel_fires_every_n_turns() -> None:
    """`review_when=periodic` with `review_period=2` runs the panel on turns 2 and 4 only.

    The turn-5 finish_session is not gated under periodic.
    """
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),
        _resp_with_tool_use("t3", _verify_pass_tool_use("v3")),
        _resp_with_tool_use("t4", _verify_pass_tool_use("v4")),
        _resp_with_tool_use("done", _finish_tool_use("f", "summary")),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = _exec(0)
    panel = _PanelScript([review_gate.CritiqueResult(text="* fine", satisfied=True)] * 2)
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(
            when="periodic", interval_turns=2, reviewers=[mock.MagicMock()]
        ),
    )
    # Each verify pass commits real progress (the normal success path), so the
    # verify-settled detector stays dormant and all 5 turns run.
    with (
        mock.patch.object(review_gate.Reviewer, "critique", panel),
        mock.patch("agent6.vcs.git_ops.chain_commit", return_value="sha"),
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=loop_conversation.Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.turns == 5
    assert result.reason == "finish_session"
    assert panel.calls == 2


def test_periodic_panel_injects_text_into_next_user_msg() -> None:
    """Periodic findings are advisory: injected under [review] even on a satisfied verdict."""
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = _exec(0)
    panel = _PanelScript([review_gate.CritiqueResult(text="* CONSIDER X", satisfied=True)])
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(
            when="periodic", interval_turns=1, reviewers=[mock.MagicMock()]
        ),
    )
    conversation = loop_conversation.Conversation.from_wire(_MSGS)
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    iter1_user_msg = conversation.to_wire()[2]
    text_blocks = [b for b in iter1_user_msg["content"] if b.get("type") == "text"]
    assert any("[review]" in b["text"] for b in text_blocks)
    assert any("CONSIDER X" in b["text"] for b in text_blocks)


def test_on_verify_fail_panel_fires_only_on_nonzero_exit() -> None:
    """`review.when = "verify_fail"` runs the panel only where run_verify_command exited non-zero.

    Passing verifies never fire it.
    """
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),  # passes
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),  # FAILS
        _resp_with_tool_use("t3", _verify_pass_tool_use("v3")),  # passes
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.side_effect = [
        _exec(0),
        _exec(1, stderr="test x failed"),
        _exec(0),
        results.RawResult({"ok": True}),
    ]
    panel = _PanelScript([review_gate.CritiqueResult(text="* hmm", satisfied=False)])
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="verify_fail", reviewers=[mock.MagicMock()]),
    )
    conversation = loop_conversation.Conversation.from_wire(_MSGS)
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.turns == 4
    assert result.reason == "finish_session"
    assert panel.calls == 1
    iter2_user_msg = conversation.to_wire()[4]
    text_blocks = [b for b in iter2_user_msg["content"] if b.get("type") == "text"]
    assert any("[review]" in b["text"] for b in text_blocks)


def test_on_verify_fail_panel_skipped_when_no_verify_call() -> None:
    """A turn with no run_verify_command call never consults the panel under `verify_fail`.

    There is no failure signal.
    """
    edit_tool = {"type": "tool_use", "id": "e1", "name": "list_dir", "input": {"path": "."}}
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("editing", edit_tool),
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.return_value = results.RawResult({"entries": []})
    panel = _PanelScript([])
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="verify_fail", reviewers=[mock.MagicMock()]),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=loop_conversation.Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert panel.calls == 0


# ---- the settled end passes the same gates a finish_session does ------------


def _settled_state() -> Any:
    from agent6.harness import loop_state

    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.settled.gateless_ever_edited = True
    state.settled.idle = 12
    return state


def _idle_turn() -> Any:
    from agent6.harness import loop_state

    return loop_state.TurnState(number=9, resp=mock.MagicMock(), assistant=mock.MagicMock())


def _settle(wf: loop.Harness, state: Any, turn: Any) -> Any:
    """The settled advisor's answer, applied through the loop (a stop runs the end gates)."""
    from agent6.harness import guards

    ctx = wf._turn_context(state, turn=turn.number, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    return wf._take(state, turn, ctx, guards.verify_settled(turn, state, ctx))  # pyright: ignore[reportPrivateUsage]


def test_a_settled_end_is_reviewed_like_a_finish() -> None:
    """A settled end is reviewed like a finish.

    A gateless run that commits and goes idle ends "settled" without calling finish_session; the
    before-finish panel judges that end too: a rejection hands the findings to the model and
    restarts the idle count, an approval lets the end stand.
    """
    wf = _wf(review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]))
    wf.mode = "run"
    wf.config.harness.metric = None
    panel = _PanelScript(
        [
            review_gate.CritiqueResult(text="* a test is missing", satisfied=False),
            review_gate.CritiqueResult(text="* fine", satisfied=True),
        ]
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        state = _settled_state()
        turn = _idle_turn()
        assert _settle(wf, state, turn) is None
        assert turn.stops == []
        assert state.settled.idle == 0
        # The turn's notices went out before the settled check, so the panel's
        # findings ride the settled path's own notice.
        assert turn.review_text is None
        assert any(
            isinstance(item, loop_conversation.Notice) and "a test is missing" in item.text
            for item in turn.tool_results
        )
        state = _settled_state()
        turn = _idle_turn()
        _settle(wf, state, turn)
        assert [stop.soft for stop in turn.stops] == ["verify_settled"]
    assert panel.calls == 2


def test_a_periodic_finding_on_a_settling_turn_is_delivered_once() -> None:
    """A periodic finding on a settling turn is delivered once.

    The turn's notices deliver a periodic panel's text, and the settled gate delivers whatever the
    turn still holds; the same finding must not go out twice.
    """
    wf = _wf(
        review=review_gate.ReviewSettings(
            when="periodic", interval_turns=1, reviewers=[mock.MagicMock()]
        )
    )
    wf.mode = "run"
    wf.config.harness.metric = None
    panel = _PanelScript(
        [review_gate.CritiqueResult(text="* one periodic finding", satisfied=True)]
    )
    state = _settled_state()
    turn = _idle_turn()
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        wf.reviewer.triggers(state, turn)
        wf._turn_notices(state, turn)  # pyright: ignore[reportPrivateUsage]
        _settle(wf, state, turn)
    delivered = [
        item
        for item in turn.tool_results
        if isinstance(item, loop_conversation.Notice) and "one periodic finding" in item.text
    ]
    assert len(delivered) == 1 and panel.calls == 1


def test_a_rejected_plateau_end_is_named_as_one() -> None:
    """The panel's rejection tells the worker which ending it rejected.

    The metric plateau's wording names the plateau, not a settled end.
    """
    wf = _wf(review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]))
    wf.mode = "run"
    panel = _PanelScript(
        [review_gate.CritiqueResult(text="* the gain is unmeasured", satisfied=False)]
    )
    turn = _idle_turn()
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        assert wf.reviewer.end_rejected(_settled_state(), turn, ending="metric_plateau")
    assert turn.review_text is not None
    assert turn.review_text.startswith("The review panel rejected the end at the metric plateau")
    assert "the gain is unmeasured" in turn.review_text


def test_a_settled_end_is_certified_by_the_harness_gate() -> None:
    """A settled end is certified by the harness gate.

    Under `verify_when = "finish"` a settled end over an unverified tree runs the gate like a
    finish: red returns to the model with the output (bounded by verify_retries), green lets the end
    stand.
    """
    from agent6.config import Config

    cfg = Config.model_validate(
        {"harness": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 1}}
    )
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.command_policy.return_value = "allow"
    dispatcher.run_verify.return_value = results.ExecResult(
        returncode=1, stdout="1 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    wf = _wf(config=cfg, dispatcher=dispatcher, review=review_gate.ReviewSettings(reviewers=[]))
    wf.mode = "run"
    state = _settled_state()
    turn = _idle_turn()
    assert _settle(wf, state, turn) is None
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert turn.stops == [] and state.settled.idle == 0
    assert state.gates.settled_returns_used == 1
    notices = [r.text for r in turn.tool_results if hasattr(r, "text")]
    assert any("[harness verify] finish" in n for n in notices)
    assert any("the next settled stop ends the run" in n for n in notices)
    # The return is spent: the next settled end stands, red and all.
    state.settled.idle = 12
    turn = _idle_turn()
    _settle(wf, state, turn)
    assert [stop.soft for stop in turn.stops] == ["verify_settled"]
    # A green gate certifies the end at once.
    dispatcher.run_verify.return_value = results.ExecResult(
        returncode=0, stdout="", stderr="", duration_s=1.0, exec_failed=False
    )
    state = _settled_state()
    turn = _idle_turn()
    _settle(wf, state, turn)
    assert turn.stops and state.verify.green_and_untouched


class _Events:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: journal.Event) -> None:
        self.events.append(
            {"type": event.TYPE, **event.model_dump(mode="json", exclude_unset=True)}
        )


def test_a_red_settled_stop_names_itself_and_spends_no_finish_return() -> None:
    """The settled stop over a red gate told the model "finish_session not honoured".

    The model had called no finish, and the return came out of its own `verify_retries`.
    """
    from agent6.config import Config
    from agent6.harness import finish_gates

    cfg = Config.model_validate(
        {"harness": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 2}}
    )
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.command_policy.return_value = "allow"
    dispatcher.run_verify.return_value = results.ExecResult(
        returncode=1, stdout="1 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    events = _Events()
    wf = _wf(
        config=cfg,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(reviewers=[]),
        events=events,
    )
    wf.mode = "run"
    state = _settled_state()
    turn = _idle_turn()
    assert _settle(wf, state, turn) is None
    assert turn.stops == [] and turn.end_returned
    assert state.gates.verify_retries_used == 0 and state.gates.settled_returns_used == 1
    notices = [r.text for r in turn.tool_results if hasattr(r, "text")]
    assert not any("finish_session" in n for n in notices)
    assert notices[-1] == (
        "[harness settled] The tree stopped changing, but the verify gate is red, so the run"
        " goes on (return 1 of 2); 1 more settled stop returns before the run ends red."
    )
    gated = [e for e in events.events if e["type"] == "loop.verify_finish.gated"]
    assert gated == [
        {"type": "loop.verify_finish.gated", "turn": 9, "nudges_used": 1, "trigger": "settled"}
    ]
    # The model's own red finish still has both of its returns.
    ctx = wf._turn_context(state, turn=10, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    finishing = _idle_turn()
    finishing.finish = finish_gates.FinishCall("finish_session", "done")
    finishing.ending = "finish_session"
    refusal = finish_gates.verify_finish(finishing, state, ctx)
    assert refusal is not None and "finish_session not honoured" in refusal.text
    assert "(return 1 of 2)" in refusal.text
    assert isinstance(refusal.event, journal.LoopVerifyFinishGated)
    assert refusal.event.trigger == "finish_session"


def test_a_silent_finish_is_certified_and_reviewed_like_a_finish() -> None:
    """A prose turn with no tool call is an end too.

    Under `verify_when = "finish"` the harness runs the gate first (red returns the worker to work
    with the output, in the conversation since there are no tool results), then the before-finish
    panel judges it.
    """
    from agent6.config import Config
    from agent6.harness import loop_state

    cfg = Config.model_validate(
        {"harness": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 1}}
    )
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.command_policy.return_value = "allow"
    dispatcher.run_verify.return_value = results.ExecResult(
        returncode=1, stdout="2 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    panel = _PanelScript([review_gate.CritiqueResult(text="* fine", satisfied=True)])
    wf = _wf(
        config=cfg,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]),
    )
    wf.mode = "run"
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.ever_edited = True
    state.verify.note_pass()
    state.verify.note_edit()  # green once, edited since: stale
    conv = loop_conversation.Conversation()
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        turn = loop_state.TurnState(number=5, resp=_resp("Done."), assistant=mock.MagicMock())
        ctx = wf._turn_context(state, turn=5, execution_start=1)  # pyright: ignore[reportPrivateUsage]
        assert quiet_turns.handle_silent_finish(wf, "Done.", conv, state, turn, ctx) is None
        texts = [b["text"] for m in conv.to_wire() for b in m["content"] if b.get("type") == "text"]
        assert any("[harness verify] finish: verify_command exit 1" in t for t in texts)
        assert any("the next red finish ends the run" in t for t in texts)
        assert panel.calls == 0  # a red gate returns the end before the panel sits
        # The return is spent: the next silent finish stands, the panel sits and approves.
        turn = loop_state.TurnState(number=6, resp=_resp("Done."), assistant=mock.MagicMock())
        ended = quiet_turns.handle_silent_finish(wf, "Done.", conv, state, turn, ctx)
    assert ended is not None and ended.reason == "silent_finish"
    assert ended.verification == "failed"
    assert panel.calls == 1


def test_before_finish_panel_reviews_a_model_controlled_run(tmp_path: pathlib.Path) -> None:
    """A model-controlled run gets its panel: the checkout, not a missing chain, is the target.

    The panel skipped with `no_target`, and the receipt read "reviewed head unknown".
    """
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@t"),
        ("config", "user.name", "t"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
    base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "seed.txt").write_text("the model's edit\n", encoding="utf-8")
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("done", _finish_tool_use("tu1", "wrap up")),
        _resp("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None, verify_withheld=False, execution_id="e" * 32
    )
    dispatcher.dispatch.return_value = results.RawResult({"ok": True})
    panel = _PanelScript([review_gate.CritiqueResult(text="* fine", satisfied=True)])
    log = tmp_path / "events.jsonl"
    wf = _wf(
        repo,
        ref=None,
        fallback_parent=base,
        base_sha=base,
        provider=worker,
        dispatcher=dispatcher,
        review=review_gate.ReviewSettings(when="finish", reviewers=[mock.MagicMock()]),
        events=event_log.EventSink(log),
    )
    with mock.patch.object(review_gate.Reviewer, "critique", panel):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=loop_conversation.Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_turn=1,
            root_task_id=None,
            original_task="t",
        )
    assert panel.calls == 1
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["reason"] for row in rows if row["type"] == "loop.review.skipped"] == []


_TARGET_DIFF = (
    "diff --git a/a.py b/a.py\n"
    "index 0000000..1111111 100644\n"
    "--- a/a.py\n"
    "+++ b/a.py\n"
    "@@ -1 +1 @@\n"
    "-x = 1\n"
    "+x = 2\n"
)
_SEAT_REPLY = json.dumps(
    {
        "verdict": "block",
        "summary": "one defect",
        "findings": [
            {
                "category": "correctness",
                "severity": "block",
                "title": "x = 2 breaks the invariant",
                "detail": "the caller expects 1",
                "location": {"path": "a.py", "side": "head", "start_line": 1, "end_line": 1},
            }
        ],
    }
)


def _target_reviewer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, events: list[Any]
) -> review_gate.Reviewer:
    """Return a reviewer with one scripted seat and no git behind its grounding."""
    provider = mock.MagicMock()
    provider.call.return_value = _resp(_SEAT_REPLY)
    seat = review_lane_env.seat(focus="correctness", model="p/m", provider=provider)

    def rules(*_args: Any, **_kwargs: Any) -> review_records.ReviewRules:
        return review_records.ReviewRules()

    def matching(*_args: Any) -> tuple[dict[str, str], tuple[review_records.Finding, ...]]:
        return {}, ()

    monkeypatch.setattr(review_gate.repo_context, "review_rules", rules)
    monkeypatch.setattr(review_gate.review_history, "matching_findings", matching)

    def observations(
        panel_id: str, verdicts: Any, known: Any, **_kw: Any
    ) -> tuple[review_records.FindingObservation, ...]:
        return tuple(
            review_records.FindingObservation("f-1", panel_id, f, (v.seat_index,))
            for v in verdicts
            for f in v.findings
        )

    monkeypatch.setattr(review_gate.review_history, "observations", observations)
    return review_gate.Reviewer(
        settings=review_gate.ReviewSettings(when="finish", reviewers=[seat]),
        chain=mock.MagicMock(root=tmp_path),
        hide=(),
        executor=mock.MagicMock,
        budget_remaining=lambda: None,
        log=_silent,
        emit=events.append,
    )


def _state() -> loop_state.LoopState:
    return loop_state.LoopState(
        original_task="/review feature-x",
        execution_id="e1",
        review_panels=(),
        operator_task="",
        tool_calls=0,
        root_task_id=None,
        system="",
    )


def test_a_review_session_panel_sits_on_its_target_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel records the reviewed range as its target; the same range is reused, not re-sat."""
    events: list[Any] = []
    reviewer = _target_reviewer(tmp_path, monkeypatch, events)
    target = review_records.ReviewTarget("b", "bt", "h", "ht", "h", (), "revisions")
    state = _state()
    critique = reviewer.review_target(state, target, _TARGET_DIFF)
    assert critique is not None and critique.satisfied
    assert "x = 2 breaks the invariant" in critique.text
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["LoopReviewStart", "LoopReviewSeat", "LoopReviewPanel"]
    start, _seat, panel = events
    assert start.trigger == "review_target" and start.turn == 0
    assert start.target["head_tree"] == "ht" and start.target["base_tree"] == "bt"
    assert panel.trigger == "review_target" and panel.block_on == "none" and not panel.blocked
    assert [o["finding"]["title"] for o in panel.findings] == ["x = 2 breaks the invariant"]
    assert len(state.review_panels) == 1
    # A resume of the session reads the recorded panel; the trees did not change.
    again = reviewer.review_target(state, target, _TARGET_DIFF)
    assert again is not None and "reused" in again.text
    assert type(events[-1]).__name__ == "LoopReviewReused"


def test_a_review_target_without_a_diff_is_skipped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []
    reviewer = _target_reviewer(tmp_path, monkeypatch, events)
    target = review_records.ReviewTarget("b", "bt", "h", "bt", "h", (), "revisions")
    assert reviewer.review_target(_state(), target, "  \n") is None
    (skipped,) = events
    assert type(skipped).__name__ == "LoopReviewSkipped" and skipped.reason == "no_diff"


@pytest.fixture(autouse=True)
def panel_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_gate, "run_panel", review_lane_env.run_panel)
