# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A parent answers its child's approval: the relay, and the wait loop that calls it."""

from __future__ import annotations

import json
import os
import pathlib

from agent6 import lane_records
from agent6.app import lane_watch, prompt_relay
from agent6.sessions import ipc, journal
from agent6.tools import operator_prompts


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

    def __init__(self, tmp_path: pathlib.Path, answer: str) -> None:
        self.dir = tmp_path / "parent"
        self.dir.mkdir()
        self.asked: list[operator_prompts.ApprovalRequest] = []
        self.rows: list[journal.Event] = []
        self._answer = answer
        self.prompts = operator_prompts.OperatorPrompts(
            approver=self._approve, journal=self.rows.append, session_dir=self.dir
        )

    def _approve(
        self, request: operator_prompts.ApprovalRequest, /
    ) -> operator_prompts.ApprovalAnswer:
        self.asked.append(request)
        # A front-end's literal answer sets the standing choice as it is recorded.
        approved = ipc.record_answer(self.dir, self._answer, request.scope)
        return operator_prompts.ApprovalAnswer(approved=approved, source="frontend")

    def relay(self) -> prompt_relay.ApprovalRelay:
        return prompt_relay.ApprovalRelay(self.prompts, self.dir)


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
