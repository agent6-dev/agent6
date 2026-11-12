# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A parent answers its child's prompts: the relay, and the wait loop that calls it."""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from agent6 import lane_records
from agent6.app import lane_watch, prompt_relay
from agent6.sessions import ipc, journal
from agent6.tools import operator_prompts, schema
from agent6.ui.cli import interact


def _child(tmp_path: pathlib.Path, *rows: dict[str, object]) -> pathlib.Path:
    """A lane's session dir whose journal holds the rows and whose worker is this process."""
    child = tmp_path / "child"
    child.mkdir()
    events = [{"type": "session.start", "mode": "run", "user_task": "t"}, *rows]
    (child / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (child / "manifest.json").write_text(json.dumps({"mode": "run", "session_id": "child"}))
    (child / "worker.pid").write_text(str(os.getpid()))
    return child


def _prompt(scope: str | None = "command") -> dict[str, object]:
    return {
        "type": "approval.prompt",
        "id": "approval-1",
        "prompt": "Allow run_command: pytest -q",
        "standing": bool(scope),
        "scope": scope,
    }


class _Parent:
    """A parent's prompts over a scripted operator, with the journal it wrote."""

    def __init__(
        self,
        tmp_path: pathlib.Path,
        answer: str = "yes",
        *,
        answers: operator_prompts.QuestionAnswer | None = None,
    ) -> None:
        self.dir = tmp_path / "parent"
        self.dir.mkdir()
        self.asked: list[operator_prompts.ApprovalRequest] = []
        self.questions: list[operator_prompts.QuestionRequest] = []
        self.rows: list[journal.Event] = []
        self._answer = answer
        self._answers = answers or operator_prompts.QuestionAnswer((), "frontend")
        self.think_s = 0.0
        self.prompts = operator_prompts.OperatorPrompts(
            approver=self._approve,
            questioner=self._ask,
            journal=self.rows.append,
            session_dir=self.dir,
        )

    def _ask(self, request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        self.questions.append(request)
        time.sleep(self.think_s)
        return self._answers

    def _approve(
        self, request: operator_prompts.ApprovalRequest, /
    ) -> operator_prompts.ApprovalAnswer:
        self.asked.append(request)
        time.sleep(self.think_s)
        # A front-end's literal answer sets the standing choice as it is recorded.
        approved = ipc.record_answer(self.dir, self._answer, request.scope)
        return operator_prompts.ApprovalAnswer(approved=approved, source="frontend")

    def relay(self) -> prompt_relay.PromptRelay:
        return prompt_relay.PromptRelay(self.prompts, self.dir)


def _answer(child: pathlib.Path) -> str | None:
    path = ipc.approvals_path(child) / "approval-1.answer"
    return path.read_text() if path.exists() else None


def test_a_parent_puts_its_childs_approval_to_its_own_operator(tmp_path: pathlib.Path) -> None:
    """The child's prompt is asked on the parent, named, and answered where a front-end would.

    A lane under `sandbox.run_commands = "ask"` parked until someone attached to that lane.
    """
    child = _child(tmp_path, _prompt())
    parent = _Parent(tmp_path, "yes")
    parent.relay()(child, "lane 2 child")
    (asked,) = parent.asked
    assert asked.prompt == "[lane 2 child] Allow run_command: pytest -q"
    assert asked.scope == "command"
    assert _answer(child) == "yes"
    kinds = [type(row).__name__ for row in parent.rows]
    assert kinds == ["ApprovalPrompt", "ApprovalAnswer"]


def test_a_denial_goes_down_as_a_denial(tmp_path: pathlib.Path) -> None:
    child = _child(tmp_path, _prompt())
    parent = _Parent(tmp_path, "no")
    parent.relay()(child, "lane 1 child")
    assert _answer(child) == "no"


def test_a_standing_answer_at_the_parent_is_handed_down(tmp_path: pathlib.Path) -> None:
    """An "allow all" given on the parent stops the child asking; a "deny all" asks nobody."""
    child = _child(tmp_path, _prompt())
    parent = _Parent(tmp_path, "session")
    parent.relay()(child, "lane 1 child")
    assert _answer(child) == "session"

    other = tmp_path / "second"
    other.mkdir()
    again = _child(other, _prompt())
    parent.relay()(again, "lane 2 child")
    assert len(parent.asked) == 1, "the parent's standing allow answered without asking"
    assert _answer(again) == "session"

    refusing = _Parent(other, "session-deny")
    third = tmp_path / "third"
    third.mkdir()
    denied = _child(third, _prompt())
    refusing.relay()(denied, "lane 1 child")
    assert _answer(denied) == "session-deny"
    fourth = tmp_path / "fourth"
    fourth.mkdir()
    refusing.relay()(_child(fourth, _prompt()), "lane 2 child")
    assert len(refusing.asked) == 1, "a standing deny answers without asking"


def test_a_prompt_with_no_scope_is_asked_every_time(tmp_path: pathlib.Path) -> None:
    """`fetch` offers no standing answer; an "allow all" on it grants that call only."""
    child = _child(tmp_path, _prompt(scope=None))
    parent = _Parent(tmp_path, "session")
    parent.relay()(child, "lane 1 child")
    assert parent.asked[0].scope is None
    assert _answer(child) == "yes"


def test_nothing_open_asks_nothing(tmp_path: pathlib.Path) -> None:
    """An answered prompt, or one whose answer is already on disk, is left alone."""
    answered = _child(
        tmp_path,
        _prompt(),
        {"type": "approval.answer", "id": "approval-1", "approved": True, "source": "frontend"},
    )
    parent = _Parent(tmp_path, "yes")
    parent.relay()(answered, "lane 1 child")
    other = tmp_path / "second"
    other.mkdir()
    taken = _child(other, _prompt())
    assert ipc.write_answer(taken, "approval-1", "no")
    parent.relay()(taken, "lane 2 child")
    assert parent.asked == [] and _answer(taken) == "no"


def test_the_wait_calls_the_relay_while_the_lane_waits(tmp_path: pathlib.Path) -> None:
    """The loop that waits on a lane is where its parent answers it."""
    child = _child(tmp_path, _prompt())
    res = lane_records.LaneResult(
        spec=lane_records.LaneSpec(lane=1, session_id="child", workdir=tmp_path, settings=None),
        session_dir=child,
        branch="",
        ok=True,
        error="",
    )
    calls: list[int] = []

    def on_waiting() -> None:
        calls.append(1)

    ended = lane_watch.await_lane(
        res, poll_interval_s=0.0, should_stop=lambda: len(calls) >= 2, on_waiting=on_waiting
    )
    assert not ended and len(calls) == 2


def _question(*, wait: bool = True) -> dict[str, object]:
    return {
        "type": "question.prompt",
        "id": "question-1",
        "questions": [
            {"question": "Which port?", "options": ["80", "8080"]},
            {"question": "Which host?", "options": []},
        ],
        "wait": wait,
    }


def _reply(child: pathlib.Path) -> ipc.QuestionReply | None:
    return ipc.read_question_reply(child, "question-1", timeout_s=0.0)


def test_a_parent_puts_its_childs_question_to_its_own_operator(tmp_path: pathlib.Path) -> None:
    """The questions a child is paused on are asked on the parent, named, and answered below.

    A lane's `ask_user` parked until someone attached to that lane.
    """
    child = _child(tmp_path, _question())
    parent = _Parent(
        tmp_path, answers=operator_prompts.QuestionAnswer(("8080", "localhost"), "frontend")
    )
    parent.relay()(child, "lane 2 child")
    (asked,) = parent.questions
    assert asked.questions == (
        schema.UserQuestion(question="[lane 2 child] Which port?", options=("80", "8080")),
        schema.UserQuestion(question="[lane 2 child] Which host?"),
    )
    assert _reply(child) == ipc.QuestionReply(("8080", "localhost"))
    assert [type(row).__name__ for row in parent.rows] == ["QuestionPrompt", "QuestionAnswer"]


def test_no_answer_at_the_parent_goes_down_as_no_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent whose operator gave no answer tells the child so, with the state it met.

    Blank answers would read as a person who answered blank; no answer would park the child
    for good.
    """
    note = "no front-end attached; away mode deny: not queued, no answer"
    child = _child(tmp_path, _question())
    parent = _Parent(
        tmp_path,
        answers=operator_prompts.QuestionAnswer(
            ("", ""), "away-deny", status="unanswered", note=note
        ),
    )
    parent.relay()(child, "lane 1 child")
    assert ipc.peek_question_answers(child, "question-1") == ()

    # The child's own questioner, parked with nobody attached, reads the outcome.
    monkeypatch.setattr("agent6.portable.has_controlling_tty", lambda: False)
    ipc.set_away_mode(child, "wait")
    request = operator_prompts.QuestionRequest(
        id="question-1",
        questions=(
            schema.UserQuestion(question="Which port?", options=("80", "8080")),
            schema.UserQuestion(question="Which host?"),
        ),
        call_id=None,
    )
    assert interact.build_questioner(child)(request) == operator_prompts.QuestionAnswer(
        ("", ""), "parent", status="unanswered", note=note
    )


def test_a_queued_question_is_not_relayed(tmp_path: pathlib.Path) -> None:
    """A question the child did not wait on pauses nothing and stays on the child's row."""
    child = _child(tmp_path, _question(wait=False))
    parent = _Parent(tmp_path)
    parent.relay()(child, "lane 1 child")
    assert parent.questions == [] and _reply(child) is None


def test_a_question_answered_on_the_child_is_left_alone(tmp_path: pathlib.Path) -> None:
    child = _child(tmp_path, _question())
    assert ipc.write_question_answers(child, "question-1", ("80", "here"))
    parent = _Parent(tmp_path)
    parent.relay()(child, "lane 1 child")
    assert parent.questions == [] and _reply(child) == ipc.QuestionReply(("80", "here"))


def test_a_relayed_question_keeps_within_the_length_a_question_may_have(
    tmp_path: pathlib.Path,
) -> None:
    long = "x" * schema.QUESTION_MAX_CHARS
    child = _child(
        tmp_path,
        {
            "type": "question.prompt",
            "id": "question-1",
            "questions": [{"question": long, "options": []}],
            "wait": True,
        },
    )
    parent = _Parent(tmp_path, answers=operator_prompts.QuestionAnswer(("ok",), "frontend"))
    parent.relay()(child, "lane 1 child")
    (asked,) = parent.questions
    text = asked.questions[0].question
    assert len(text) == schema.QUESTION_MAX_CHARS
    assert text.startswith("[lane 1 child] xxx") and text.endswith("x\u2026")


def test_a_relayed_prompts_wait_counts_as_the_parents_operator_wait(tmp_path: pathlib.Path) -> None:
    """The time the operator takes over a child's prompt is the parent's operator wait.

    Only the parent's own tool calls counted it, so a parent waiting on a lane's approval
    read eleven minutes of that as time with no edit and no verify.
    """
    parent = _Parent(tmp_path)
    parent.think_s = 0.05
    parent.relay()(_child(tmp_path, _prompt()), "lane 1 child")
    after_approval = parent.prompts.waited_s
    assert after_approval >= 0.05
    other = tmp_path / "second"
    other.mkdir()
    parent.relay()(_child(other, _question()), "lane 2 child")
    assert parent.prompts.waited_s >= after_approval + 0.05
