# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A parent watches a running child: it answers its prompts and counts its spend."""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

import pytest

from agent6 import budget as agent6_budget
from agent6 import child_records
from agent6.app import child_wait, child_watch
from agent6.sessions import ipc, journal
from agent6.tools import operator_prompts, schema
from agent6.ui.cli import interact
from agent6.viewmodel import listing
from agent6.viewmodel.session import log as session_log
from agent6.viewmodel.session import records as state_types
from agent6.viewmodel.session import state as session_fold
from agent6.viewmodel.session import view as state_view


def _child(tmp_path: pathlib.Path, *rows: dict[str, object]) -> pathlib.Path:
    """A child's session dir whose journal holds the rows and whose worker is this process."""
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

    def watch(self) -> child_watch.ChildWatch:
        return child_watch.ChildWatch(self.prompts, self.dir)


def _answer(child: pathlib.Path) -> str | None:
    path = ipc.approvals_path(child) / "approval-1.answer"
    return path.read_text() if path.exists() else None


def test_a_parent_puts_its_childs_approval_to_its_own_operator(tmp_path: pathlib.Path) -> None:
    """The child's prompt is asked on the parent, named, and answered where a front-end would.

    A child under `sandbox.run_commands = "ask"` parked until someone attached to that child.
    """
    child = _child(tmp_path, _prompt())
    parent = _Parent(tmp_path, "yes")
    parent.watch()(child, "child 2 child", "waiting")
    (asked,) = parent.asked
    assert asked.prompt == "[child 2 child] Allow run_command: pytest -q"
    assert asked.scope == "command"
    assert _answer(child) == "yes"
    kinds = [type(row).__name__ for row in parent.rows]
    assert kinds == ["ApprovalPrompt", "ApprovalAnswer"]


def test_a_denial_goes_down_as_a_denial(tmp_path: pathlib.Path) -> None:
    child = _child(tmp_path, _prompt())
    parent = _Parent(tmp_path, "no")
    parent.watch()(child, "child 1 child", "waiting")
    assert _answer(child) == "no"


def test_a_prompt_before_the_start_report_reaches_the_parent(tmp_path: pathlib.Path) -> None:
    """A child waiting for a preflight answer cannot report its start yet."""
    child = _child(tmp_path)
    (child / "logs.jsonl").write_text(json.dumps(_prompt()) + "\n")
    parent = _Parent(tmp_path, "no")
    parent.watch()(child, "attempt 1 child", child_watch.ChildStartup(os.getpid(), 0))
    assert _answer(child) == "no"


def test_preflight_does_not_rewind_into_a_stopped_executions_prompt(tmp_path: pathlib.Path) -> None:
    """A short journal cannot supply a prompt from before the launch offset."""
    child = _child(tmp_path, _prompt())
    log = child / "logs.jsonl"
    previous = log.read_bytes()
    log.write_bytes(previous + b'{"type":"session.stop","reason":"interrupted"}\n')
    startup = child_watch.ChildStartup(os.getpid(), log.stat().st_size)
    log.write_bytes(previous)
    parent = _Parent(tmp_path)
    parent.watch()(child, "attempt 1 child", startup)
    assert not parent.asked and _answer(child) is None


@pytest.mark.parametrize("startup", [False, True], ids=["waiting", "startup"])
def test_polling_an_open_approval_reads_each_row_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, startup: bool
) -> None:
    """The parent's answer wait reads appended rows, not the child's history on each tick."""
    child = _child(tmp_path)
    log = child / "logs.jsonl"
    status = child_watch.ChildStartup(os.getpid(), log.stat().st_size) if startup else "waiting"
    note = '{"type":"loop.note","text":"still working"}\n'
    with log.open("a") as fh:
        fh.write(note * 64 + json.dumps(_prompt()) + "\n")
    apply = session_fold.apply_event
    rows_folded = 0

    def counted(state: state_types.SessionState, row: dict[str, Any]) -> state_types.SessionState:
        nonlocal rows_folded
        rows_folded += 1
        return apply(state, row)

    def approve(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        for _ in range(8):
            with log.open("a") as fh:
                fh.write(note)
            assert not request.interrupted()
        return operator_prompts.ApprovalAnswer(True, "frontend")

    monkeypatch.setattr(session_fold, "apply_event", counted)
    watch = child_watch.ChildWatch(operator_prompts.OperatorPrompts(approver=approve), tmp_path)
    watch(child, "child", status)
    assert _answer(child) == "yes"
    watch(child, "child", status)
    assert rows_folded == 64 + 1 + 8 + (not startup)


@pytest.mark.parametrize(
    "change",
    [
        "answered",
        "consumed",
        "reasked",
        "rotated",
        "truncated",
        "missing",
        "worker",
        "stop",
        "ended",
    ],
)
def test_a_relay_writes_only_to_the_same_open_slot(tmp_path: pathlib.Path, change: str) -> None:
    """An answer in flight cannot authorize another slot with the same id and text."""
    child = _child(tmp_path, _prompt())
    log = child / "logs.jsonl"

    def approve(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        assert not request.interrupted()
        if change in {"answered", "consumed"}:
            assert ipc.write_answer(child, "approval-1", "no")
        elif change == "reasked":
            with log.open("a") as fh:
                fh.write('{"type":"approval.answer","id":"approval-1","approved":false}\n')
                fh.write(json.dumps(_prompt()) + "\n")
        elif change == "rotated":
            replacement = child / "replacement"
            replacement.write_bytes(log.read_bytes())
            replacement.replace(log)
        elif change == "truncated":
            log.write_text(json.dumps(_prompt()) + "\n")
        elif change == "missing":
            log.unlink()
        elif change == "worker":
            (child / "worker.pid").unlink()
        elif change == "ended":
            with log.open("a") as fh:
                fh.write('{"type":"session.stop","reason":"interrupted"}\n')
        else:
            assert ipc.request_stop(child, now=True)
        assert request.interrupted()
        if change == "consumed":
            assert ipc.read_answer(child, "approval-1", timeout_s=0) == "no"
        return operator_prompts.ApprovalAnswer(True, "frontend")

    watch = child_watch.ChildWatch(operator_prompts.OperatorPrompts(approver=approve), tmp_path)
    watch(child, "child", "waiting")
    assert _answer(child) == ("no" if change == "answered" else None)


@pytest.mark.parametrize("startup", [False, True], ids=["waiting", "startup"])
def test_regrowth_cannot_hide_a_consumed_answer(tmp_path: pathlib.Path, startup: bool) -> None:
    """The same inode can outgrow the cursor with an answer already behind it."""
    child = _child(tmp_path)
    log = child / "logs.jsonl"
    prefix = log.read_bytes()
    status = child_watch.ChildStartup(os.getpid(), len(prefix)) if startup else "waiting"
    prompt = json.dumps(_prompt()).encode() + b"\n"
    note = b'{"type":"loop.note","text":"unrelated"}\n'
    log.write_bytes(prefix + prompt + note * 12)
    offers = 0

    def approve(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        nonlocal offers
        offers += 1
        assert not request.interrupted()
        assert ipc.write_answer(child, "approval-1", "no")
        assert ipc.read_answer(child, "approval-1", timeout_s=0) == "no"
        answer = b'{"type":"approval.answer","id":"approval-1","approved":false}\n'
        # Even the original prompt row and the last note match at the retained offsets.
        log.write_bytes(prefix + prompt + answer.ljust(len(note) * 3 - 1) + b"\n" + note * 15)
        request.interrupted()  # poll the new suffix before the final check
        return operator_prompts.ApprovalAnswer(True, "frontend")

    watch = child_watch.ChildWatch(operator_prompts.OperatorPrompts(approver=approve), tmp_path)
    watch(child, "child", status)
    assert _answer(child) is None
    watch(child, "child", status)  # the next poll no longer offers the closed prompt
    assert offers == 1


def test_a_standing_answer_at_the_parent_is_handed_down(tmp_path: pathlib.Path) -> None:
    """An "allow all" given on the parent stops the child asking; a "deny all" asks nobody."""
    child = _child(tmp_path, _prompt())
    parent = _Parent(tmp_path, "session")
    parent.watch()(child, "child 1 child", "waiting")
    assert _answer(child) == "session"

    other = tmp_path / "second"
    other.mkdir()
    again = _child(other, _prompt())
    parent.watch()(again, "child 2 child", "waiting")
    assert len(parent.asked) == 1, "the parent's standing allow answered without asking"
    assert _answer(again) == "session"

    refusing = _Parent(other, "session-deny")
    third = tmp_path / "third"
    third.mkdir()
    denied = _child(third, _prompt())
    refusing.watch()(denied, "child 1 child", "waiting")
    assert _answer(denied) == "session-deny"
    fourth = tmp_path / "fourth"
    fourth.mkdir()
    refusing.watch()(_child(fourth, _prompt()), "child 2 child", "waiting")
    assert len(refusing.asked) == 1, "a standing deny answers without asking"


@pytest.mark.parametrize("startup", [False, True], ids=["waiting", "startup"])
def test_regrowth_cannot_hide_a_new_blocking_question(
    tmp_path: pathlib.Path, startup: bool
) -> None:
    child = _child(tmp_path)
    log = child / "logs.jsonl"
    prefix = log.read_bytes()
    status = child_watch.ChildStartup(os.getpid(), len(prefix)) if startup else "waiting"
    note = b'{"type":"loop.note","text":"padding"}\n'
    log.write_bytes(prefix + note * 20)
    parent = _Parent(
        tmp_path, answers=operator_prompts.QuestionAnswer(("8080", "here"), "frontend")
    )
    watch = parent.watch()
    watch(child, "child", status)
    assert not parent.questions
    question = json.dumps(_question()).encode()
    log.write_bytes(prefix + question.ljust(len(note) * 8 - 1) + b"\n" + note * 16)
    for _ in range(3):
        watch(child, "child", status)
    assert len(parent.questions) == 1 and _reply(child) == ipc.QuestionReply(("8080", "here"))


def test_a_prompt_with_no_scope_is_asked_every_time(tmp_path: pathlib.Path) -> None:
    """`fetch` offers no standing answer; an "allow all" on it grants that call only."""
    child = _child(tmp_path, _prompt(scope=None))
    parent = _Parent(tmp_path, "session")
    parent.watch()(child, "child 1 child", "waiting")
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
    parent.watch()(answered, "child 1 child", "waiting")
    other = tmp_path / "second"
    other.mkdir()
    taken = _child(other, _prompt())
    assert ipc.write_answer(taken, "approval-1", "no")
    parent.watch()(taken, "child 2 child", "waiting")
    assert parent.asked == [] and _answer(taken) == "no"


def test_the_wait_calls_the_watch_on_every_poll(tmp_path: pathlib.Path) -> None:
    """The loop that waits on a child is where its parent watches it; a True ends the wait."""
    child = _child(tmp_path, _prompt())
    res = child_records.ChildResult(
        spec=child_records.ChildSpec(
            number=1, session_id="child", worktree=tmp_path, settings=None
        ),
        session_dir=child,
        branch="",
        ok=True,
        error="",
    )
    seen: list[str] = []

    def on_poll(status: str) -> bool:
        seen.append(status)
        return len(seen) >= 2

    assert not child_wait.await_child(res, poll_interval_s=0.0, on_poll=on_poll)
    assert seen == ["waiting", "waiting"]


def _spend(usd: float, *, running: float = 0.0) -> dict[str, object]:
    return {"type": "budget.update", "usd_total": usd, "usd_running_children": running}


def test_a_parent_counts_a_running_childs_spend_and_ends_it_at_the_cap(
    tmp_path: pathlib.Path,
) -> None:
    """A running child's spend so far counts against its parent's cap on each poll.

    Each child held an even share of what its parent had left, so one child's unused share
    was closed to its siblings and to the parent until the child settled.
    """
    parent = _Parent(tmp_path)
    budget = agent6_budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=-1, max_plan_points=-1)
    watch = child_watch.ChildWatch(parent.prompts, parent.dir, budget)
    child = _child(tmp_path, _spend(0.25))
    assert watch(child, "child 1 child", "running") is False
    assert budget.available().usd == pytest.approx(0.75)
    assert budget.estimate_usd() == (0.0, False), "a running child's spend is not yet the run's"

    # The reading replaces the one before; the child's own running children count with it.
    (child / "logs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (_spend(0.25), _spend(0.5, running=0.25)))
    )
    assert watch(child, "child 1 child", "running") is False
    assert budget.available().usd == pytest.approx(0.25)

    second = tmp_path / "second"
    second.mkdir()
    sibling = _child(second, _spend(0.5)).rename(second / "sibling")  # a session id of its own
    assert watch(sibling, "child 2 child", "running") is True, "together they reached the cap"
    assert "0.75 of it in running children" not in budget.snapshot().exhausted_reason
    assert "running children" in budget.snapshot().exhausted_reason
    assert parent.asked == [], "a spent budget ends the child before its prompt is relayed"


def test_the_watch_journals_the_meter_as_running_children_move_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent's journal carries its running children's spend, so its own parent reads it.

    A child waiting on its reviewers journaled nothing until they settled.
    """
    parent = _Parent(tmp_path)
    budget = agent6_budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=-1, max_plan_points=-1)
    rows: list[journal.Event] = []
    watch = child_watch.ChildWatch(parent.prompts, parent.dir, budget, rows.append)
    child = _child(tmp_path, _spend(0.25))
    watch(child, "reviewer 1 child", "running")
    watch(child, "reviewer 1 child", "running")
    (row,) = rows
    assert isinstance(row, journal.BudgetUpdate)
    assert (row.usd_total, row.usd_running_children) == (0.0, 0.25)

    # A move inside the interval waits for it; the cap does not.
    (child / "logs.jsonl").write_text(json.dumps(_spend(0.5)) + "\n")
    watch(child, "reviewer 1 child", "running")
    assert len(rows) == 1
    monkeypatch.setattr(child_watch, "ROW_INTERVAL_S", 0.0)
    watch(child, "reviewer 1 child", "running")
    assert [r.usd_running_children for r in rows if isinstance(r, journal.BudgetUpdate)] == [
        0.25,
        0.5,
    ]
    monkeypatch.setattr(child_watch, "ROW_INTERVAL_S", 3600.0)
    (child / "logs.jsonl").write_text(json.dumps(_spend(1.5)) + "\n")
    assert watch(child, "reviewer 1 child", "running") is True
    last = rows[-1]
    assert isinstance(last, journal.BudgetUpdate) and last.usd_running_children == 1.5

    # The parent's own parent reads the subtree from that journal.
    logs = tmp_path / "grandparent-view"
    logs.mkdir()
    (logs / "logs.jsonl").write_text(json.dumps({"type": last.TYPE, **last.model_dump()}) + "\n")
    assert child_watch.running_usage(logs).usd == 1.5


def test_a_runs_usage_cell_holds_its_running_children(tmp_path: pathlib.Path) -> None:
    """Every hub's usage cell counts the children still running, so it does not step at settlement.

    The cell showed a run's own dollar while its children had spent five, then jumped.
    """
    run = _child(tmp_path, {**_spend(1.0, running=5.0), "usd_cap": 10.0})
    summary = listing.summarize_session_dir(run)
    assert summary.cost_usd == 6.0 and summary.usage_cell == "$6.00"
    view = state_view.budget_dict(session_log.scan_session_log(run / "logs.jsonl").budget)
    assert view["usd_text"] == "$6.00 / $10.00"
    assert view["execution"]["usd"] == 1.0 and view["execution"]["usd_running_children"] == 5.0

    settled = [{**_spend(1.0, running=5.0), "usd_cap": 10.0}, {**_spend(6.5), "usd_cap": 10.0}]
    (run / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in settled))
    assert listing.summarize_session_dir(run).usage_cell == "$6.50"


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

    A child's `ask_user` parked until someone attached to that child.
    """
    child = _child(tmp_path, _question())
    parent = _Parent(
        tmp_path, answers=operator_prompts.QuestionAnswer(("8080", "localhost"), "frontend")
    )
    parent.watch()(child, "child 2 child", "waiting")
    (asked,) = parent.questions
    assert asked.questions == (
        schema.UserQuestion(question="[child 2 child] Which port?", options=("80", "8080")),
        schema.UserQuestion(question="[child 2 child] Which host?"),
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
    parent.watch()(child, "child 1 child", "waiting")
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
    parent.watch()(child, "child 1 child", "waiting")
    assert parent.questions == [] and _reply(child) is None


def test_a_question_answered_on_the_child_is_left_alone(tmp_path: pathlib.Path) -> None:
    child = _child(tmp_path, _question())
    assert ipc.write_question_answers(child, "question-1", ("80", "here"))
    parent = _Parent(tmp_path)
    parent.watch()(child, "child 1 child", "waiting")
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
    parent.watch()(child, "child 1 child", "waiting")
    (asked,) = parent.questions
    text = asked.questions[0].question
    assert len(text) == schema.QUESTION_MAX_CHARS
    assert text.startswith("[child 1 child] xxx") and text.endswith("x\u2026")


def test_a_relayed_prompts_wait_counts_as_the_parents_operator_wait(tmp_path: pathlib.Path) -> None:
    """The time the operator takes over a child's prompt is the parent's operator wait.

    Only the parent's own tool calls counted it, so a parent waiting on a child's approval
    read eleven minutes of that as time with no edit and no verify.
    """
    parent = _Parent(tmp_path)
    parent.think_s = 0.05
    parent.watch()(_child(tmp_path, _prompt()), "child 1 child", "waiting")
    after_approval = parent.prompts.waited_s
    assert after_approval >= 0.05
    other = tmp_path / "second"
    other.mkdir()
    parent.watch()(_child(other, _question()), "child 2 child", "waiting")
    assert parent.prompts.waited_s >= after_approval + 0.05
