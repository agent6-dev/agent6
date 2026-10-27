# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one gate every operator prompt goes through (`tools.operator_prompts`).

It mints the ids, journals the prompt and answer pair, names the call a prompt gates, and no front-
end journals a copy.
"""

from __future__ import annotations

import json
import pathlib
import re
import threading
from concurrent import futures
from typing import Any

import pytest

from agent6 import event_log
from agent6.config import Config
from agent6.sessions import ipc
from agent6.tools import dispatch, errors, operator_prompts, schema

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "agent6"


def _journal(session_dir: pathlib.Path) -> list[dict[str, Any]]:
    lines = (session_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _of(session_dir: pathlib.Path, event_type: str) -> list[dict[str, Any]]:
    return [e for e in _journal(session_dir) if e["type"] == event_type]


def _deny(_request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
    return operator_prompts.ApprovalAnswer(False, "stdin")


def _sink(session_dir: pathlib.Path) -> event_log.EventSink:
    session_dir.mkdir(parents=True, exist_ok=True)
    return event_log.EventSink(session_dir / "logs.jsonl")


def _prompts(
    session_dir: pathlib.Path, events: event_log.EventSink, **kw: Any
) -> operator_prompts.OperatorPrompts:
    return operator_prompts.OperatorPrompts(journal=events.emit, session_dir=session_dir, **kw)


def _dispatcher(
    session_dir: pathlib.Path,
    events: event_log.EventSink,
    prompts: operator_prompts.OperatorPrompts,
) -> dispatch.ToolDispatcher:
    """A run's wiring: the gate and the dispatcher journal to the one sink."""
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "harness": {"verify_command": ["true"]}}
    )
    return dispatch.ToolDispatcher(
        root=session_dir.parent, config=cfg, prompts=prompts, events=events, session_dir=session_dir
    )


def test_an_approval_is_journaled_with_the_call_it_gates(tmp_path: pathlib.Path) -> None:
    """An approval is journaled with the call it gates.

    The dispatcher journals tool.call, then the gate journals the prompt stamped with that call's id
    and the answer with the approver's source.
    """
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    d = _dispatcher(session_dir, events, _prompts(session_dir, events, approver=_deny))
    with pytest.raises(errors.ToolDeniedError):
        d.dispatch("run_command", {"argv": ["ls"]})
    (call,) = _of(session_dir, "tool.call")
    (prompt,) = _of(session_dir, "approval.prompt")
    (answer,) = _of(session_dir, "approval.answer")
    assert prompt["call_id"] == call["call_id"]
    assert (prompt["id"], prompt["prompt"], prompt["standing"]) == (
        "approval-1",
        "Allow run_command: ls",
        True,
    )
    assert (answer["id"], answer["approved"], answer["source"]) == ("approval-1", False, "stdin")


def test_a_question_is_journaled_with_the_call_it_gates(tmp_path: pathlib.Path) -> None:
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    seen: list[operator_prompts.QuestionRequest] = []

    def _pick(request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        seen.append(request)
        return operator_prompts.QuestionAnswer(("b",), "stdin")

    d = _dispatcher(session_dir, events, _prompts(session_dir, events, questioner=_pick))
    out = d.dispatch("ask_user", {"questions": [{"question": "which?", "options": ["a", "b"]}]})
    assert out.to_wire() == {"id": "question-1", "status": "answered", "answers": ["b"]}
    (call,) = _of(session_dir, "tool.call")
    (prompt,) = _of(session_dir, "question.prompt")
    (answer,) = _of(session_dir, "question.answer")
    assert prompt["call_id"] == call["call_id"] == seen[0].call_id
    assert prompt["id"] == "question-1" == seen[0].id
    assert prompt["questions"] == [{"question": "which?", "options": ["a", "b"]}]
    assert (answer["answers"], answer["source"]) == (["b"], "stdin")


def test_concurrent_seats_each_name_their_own_call(tmp_path: pathlib.Path) -> None:
    """Concurrent seats each name their own call.

    Two seats dispatching at once on one dispatcher, both past their stamp before either reads it:
    each prompt names the call its own thread is dispatching (a stamp shared across threads would
    name the other seat's).
    """
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    named: dict[str, int | None] = {}

    def _record(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        named[request.prompt] = request.call_id
        return operator_prompts.ApprovalAnswer(False, "stdin")

    d = _dispatcher(session_dir, events, _prompts(session_dir, events, approver=_record))
    both_stamped = threading.Barrier(2)
    policy = d.command_policy

    def _rendezvous() -> str:
        both_stamped.wait(timeout=10)
        return policy()

    d.command_policy = _rendezvous  # read after the stamp, before the gate
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        seats = [pool.submit(d.dispatch, "run_command", {"argv": ["ls", arg]}) for arg in "ab"]
        for seat in seats:
            with pytest.raises(errors.ToolDeniedError):
                seat.result(timeout=30)
    stamped = {" ".join(e["args"]["argv"]): e["call_id"] for e in _of(session_dir, "tool.call")}
    assert named == {f"Allow run_command: {argv}": cid for argv, cid in stamped.items()}
    assert len(set(named.values())) == 2


def test_a_verify_the_harness_runs_gates_no_call(tmp_path: pathlib.Path) -> None:
    """A verify the harness runs gates no call.

    `run_verify` outside a dispatch (the harness's own certification) goes through the same gate and
    carries no call; so does a question asked before the loop.
    """
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    prompts = _prompts(session_dir, events, approver=_deny)
    d = _dispatcher(session_dir, events, prompts)
    with pytest.raises(errors.ToolDeniedError):
        d.run_verify()
    prompts.ask((schema.UserQuestion(question="stash?", options=("stash", "cancel")),))
    (approval,) = _of(session_dir, "approval.prompt")
    (question,) = _of(session_dir, "question.prompt")
    assert approval["prompt"] == "Allow run_verify_command: true"
    assert approval["call_id"] is None and question["call_id"] is None


def test_a_standing_grant_answers_without_a_prompt(tmp_path: pathlib.Path) -> None:
    """An "Allow all" for a scope answers that scope's later prompts itself.

    No front-end is asked and no prompt is journaled, only the answer. The id is still consumed, so
    the sequence stays in step on every surface.
    """
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    seen: list[str] = []

    def _record(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        seen.append(request.id)
        return operator_prompts.ApprovalAnswer(False, "stdin")

    prompts = _prompts(session_dir, events, approver=_record)
    ipc.set_session_allow(session_dir, ipc.COMMAND_SCOPE)
    assert prompts.approve("Allow run_command: ls", scope=ipc.COMMAND_SCOPE) is True
    assert seen == [] and _of(session_dir, "approval.prompt") == []
    (answer,) = _of(session_dir, "approval.answer")
    assert (answer["id"], answer["approved"], answer["source"]) == ("approval-1", True, "session")
    prompts.approve("Allow fetch: evil.example /x")  # no scope: the grant covers nothing
    assert seen == ["approval-2"]


def test_a_premature_answer_is_cleared_before_the_prompt_is_journaled(
    tmp_path: pathlib.Path,
) -> None:
    """A premature answer is cleared before the prompt is journaled.

    Ids are predictable counters, so an answer written ahead of its prompt must be gone before any
    front-end could read the slot: the gate clears it, then journals the prompt, then asks.
    """
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    ipc.write_answer(session_dir, "approval-1", "yes")  # the premature POST
    slot = session_dir / "approvals" / "approval-1.answer"
    assert slot.exists()
    observed: list[tuple[bool, list[str]]] = []

    def _observe(_request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        observed.append((slot.exists(), [e["type"] for e in _journal(session_dir)]))
        return operator_prompts.ApprovalAnswer(False, "stdin")

    prompts = _prompts(session_dir, events, approver=_observe)
    assert prompts.approve("Allow run_command: ls", scope=ipc.COMMAND_SCOPE) is False
    assert observed == [(False, ["approval.prompt"])]


def test_answers_align_to_the_questions(tmp_path: pathlib.Path) -> None:
    """Answers align to the questions.

    A front-end that answered fewer questions leaves the rest unanswered (the TUI writes no answers
    for a dismissed modal); more answers than questions is a front-end defect and fails loudly.
    """
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    questions = (schema.UserQuestion(question="a?"), schema.UserQuestion(question="b?"))

    def _short(_request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        return operator_prompts.QuestionAnswer((), "frontend")

    assert _prompts(session_dir, events, questioner=_short).ask(questions).answers == ("", "")
    (answer,) = _of(session_dir, "question.answer")
    assert answer["answers"] == ["", ""]

    def _long(_request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        return operator_prompts.QuestionAnswer(("x", "y", "z"), "frontend")

    with pytest.raises(ValueError, match="2 questions with 3 answers"):
        _prompts(session_dir, events, questioner=_long).ask(questions)


def test_every_prompt_event_has_exactly_one_emitter() -> None:
    """Every prompt event has exactly one emitter.

    The prompt and answer pairs are journaled in one place; a front-end that journals its own copy
    drifts (its own counter, its own idea of the gated call).
    """
    emits = re.compile(r'(?:emit|journal)\(\s*"(approval|question)\.(prompt|answer)"')
    sites: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        for match in emits.finditer(path.read_text(encoding="utf-8")):
            event = f"{match.group(1)}.{match.group(2)}"
            sites.setdefault(event, set()).add(str(path.relative_to(_SRC)))
    assert sites == {
        "approval.prompt": {"tools/operator_prompts.py"},
        "approval.answer": {"tools/operator_prompts.py"},
        "question.prompt": {"tools/operator_prompts.py"},
        "question.answer": {"tools/operator_prompts.py"},
    }


def test_a_queued_question_returns_pending_and_journals_no_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`wait=false` journals the prompt, calls no questioner and returns the pending status."""
    from agent6 import portable

    session_dir = tmp_path / "s"
    events = _sink(session_dir)

    def _never(_request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        pytest.fail("a queued question reaches no questioner")

    prompts = _prompts(session_dir, events, questioner=_never)
    questions = (schema.UserQuestion(question="which?", options=("a", "b")),)
    monkeypatch.setattr(portable, "has_controlling_tty", lambda: False)
    answer = prompts.ask(questions, call_id=3, wait=False)
    assert answer.status == "pending" and answer.answers == () and answer.id == "question-1"
    assert answer.note == (
        "no front-end or terminal attached; away mode unset; queued: the answers arrive as a"
        " message at a later turn, or on resume"
    )
    (prompt,) = _of(session_dir, "question.prompt")
    assert prompt["wait"] is False and prompt["call_id"] == 3
    assert _of(session_dir, "question.answer") == []
    assert [p.id for p in prompts.pending()] == ["question-1"]


def test_late_answers_are_consumed_once_and_journaled_as_late(tmp_path: pathlib.Path) -> None:
    """An answer file for a queued question is taken once, journaled as `late`, and gone."""
    session_dir = tmp_path / "s"
    events = _sink(session_dir)
    prompts = _prompts(session_dir, events)
    first = (schema.UserQuestion(question="first?", options=("a", "b")),)
    second = (schema.UserQuestion(question="second?"), schema.UserQuestion(question="third?"))
    prompts.ask(first, wait=False)
    prompts.ask(second, wait=False)
    assert prompts.collect_answers() == ()
    assert ipc.write_question_answers(session_dir, "question-2", ["x", "y"])
    (late,) = prompts.collect_answers()
    assert late.id == "question-2" and late.answers == ("x", "y") and late.questions == second
    (row,) = _of(session_dir, "question.answer")
    assert row["id"] == "question-2" and row["source"] == "late" and row["unseen"] is False
    assert row["answers"] == ["x", "y"]
    assert prompts.collect_answers() == ()
    assert [p.id for p in prompts.pending()] == ["question-1"]
    assert not ipc.question_answers_written(session_dir, "question-2")


def test_question_ids_continue_from_the_number_given(tmp_path: pathlib.Path) -> None:
    """A later execution numbers its questions after the journal's, so an id names one prompt."""
    session_dir = tmp_path / "s"
    events = _sink(session_dir)
    seeded = operator_prompts.QuestionRequest(
        id="question-2", questions=(schema.UserQuestion(question="old?"),), call_id=1, wait=False
    )
    prompts = _prompts(session_dir, events, first_question=4, pending=(seeded,))
    prompts.ask((schema.UserQuestion(question="new?"),), wait=False)
    assert [p.id for p in prompts.pending()] == ["question-2", "question-4"]
    assert ipc.write_question_answers(session_dir, "question-2", ["kept"])
    (late,) = prompts.collect_answers()
    assert late.id == "question-2" and late.questions == seeded.questions


def test_an_unanswered_question_carries_the_state_in_its_result(tmp_path: pathlib.Path) -> None:
    """A questioner that could reach nobody says so; a blank a person left is an answer."""
    session_dir = tmp_path / "s"
    events = _sink(session_dir)

    def _nobody(request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        return operator_prompts.QuestionAnswer(
            tuple("" for _ in request.questions),
            "away-deny",
            status="unanswered",
            note="no front-end attached; away mode deny: not queued, no answer",
        )

    def _blank(request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        return operator_prompts.QuestionAnswer(tuple("" for _ in request.questions), "stdin")

    questions = (schema.UserQuestion(question="which?", options=("a", "b")),)
    unanswered = _prompts(session_dir, events, questioner=_nobody).ask(questions)
    assert unanswered.status == "unanswered" and "away mode deny" in unanswered.note
    blank = _prompts(session_dir, events, questioner=_blank).ask(questions)
    assert blank.status == "answered" and blank.note == "" and blank.answers == ("",)
    rows = _of(session_dir, "question.answer")
    assert [r["unseen"] for r in rows] == [True, False]
    assert not hasattr(operator_prompts, "UNANSWERED_NOTE")


def test_undelivered_answers_come_first_and_are_not_journaled_again(
    tmp_path: pathlib.Path,
) -> None:
    """An answer journaled by an earlier execution and never read is delivered once, unjournaled."""
    session_dir = tmp_path / "s"
    events = _sink(session_dir)
    seeded = operator_prompts.LateAnswer(
        "question-2", (schema.UserQuestion(question="old?"),), ("kept",)
    )
    prompts = _prompts(session_dir, events, undelivered=(seeded,))
    assert prompts.collect_answers() == (seeded,)
    assert prompts.collect_answers() == ()
    assert not (session_dir / "logs.jsonl").exists()


def test_presence_names_the_claim_the_terminal_and_the_away_mode(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The presence the model reads: the front-end claim, else the terminal, else the away mode."""
    import os

    from agent6 import portable
    from agent6.sessions import process

    session_dir = tmp_path / "s"
    prompts = _prompts(session_dir, _sink(session_dir))
    monkeypatch.setattr(portable, "has_controlling_tty", lambda: False)
    assert prompts.presence() == "no front-end or terminal attached; away mode unset"
    ipc.set_away_mode(session_dir, "wait")
    assert prompts.presence() == "no front-end or terminal attached; away mode wait"
    monkeypatch.setattr(portable, "has_controlling_tty", lambda: True)
    assert prompts.presence() == "a terminal is attached"
    process.register_frontend(session_dir, os.getpid())
    assert prompts.presence() == "a front-end is attached"
    assert operator_prompts.OperatorPrompts().presence() == ""


def test_a_late_answer_is_journaled_before_its_file_goes(tmp_path: pathlib.Path) -> None:
    """A journal that fails leaves the answer file in place, so nothing is lost."""
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    calls: list[str] = []

    def failing(event_type: str, /, **fields: Any) -> None:
        calls.append(event_type)
        if event_type == "question.answer":
            raise OSError("disk full")

    prompts = operator_prompts.OperatorPrompts(journal=failing, session_dir=session_dir)
    prompts.ask((schema.UserQuestion(question="a?"),), wait=False)
    assert ipc.write_question_answers(session_dir, "question-1", ["x"])
    with pytest.raises(OSError):
        prompts.collect_answers()
    assert ipc.question_answers_written(session_dir, "question-1")
    assert [p.id for p in prompts.pending()] == ["question-1"]
