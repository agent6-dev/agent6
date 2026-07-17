# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Interactive `agent6 attach` attaches as a CLI front-end: an unanswered
run_command approval / ask_user question in the streamed log is prompted on the
terminal and the answer is written back over the file bridge. Historical and
already-answered prompts are not re-asked on the replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent6.runs.bridge import approvals_dir, questions_dir
from agent6.ui.cli import plan_watch


def _view() -> Any:
    class _V:
        def pause(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

    return _V()


def _write_log(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_open_prompt_at_attach_is_answered_and_written(tmp_path: Path, monkeypatch: Any) -> None:
    # A run already waiting at approval-1 when you attach: the front-end prompts
    # and writes the answer the worker is blocked reading.
    def _yes(_prompt: str) -> str:
        return "yes"

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _yes)
    log = tmp_path / "logs.jsonl"
    _write_log(
        log,
        [
            {"type": "run.start"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "run `ls`?"},
        ],
    )
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    opens = fe.open_prompts_at_attach(log)
    assert [(k, i) for k, i, _ in opens] == [("approval", "approval-1")]
    for kind, pid, content in opens:
        fe.handle(kind, pid, content)
    assert (approvals_dir(tmp_path) / "approval-1.answer").read_text() == "yes"


def test_already_answered_prompt_is_not_reasked(tmp_path: Path, monkeypatch: Any) -> None:
    # approval-1 was emitted AND answered in history: not open at attach, and the
    # replay must not re-prompt it (the approver would fail the test if called).
    def _forbidden(_p: object) -> str:
        raise AssertionError("must not prompt for an already-answered approval")

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _forbidden)
    log = tmp_path / "logs.jsonl"
    _write_log(
        log,
        [
            {"type": "approval.prompt", "id": "approval-1", "prompt": "x"},
            {"type": "approval.answer", "id": "approval-1", "approved": True},
        ],
    )
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    assert fe.open_prompts_at_attach(log) == []
    # replay through react(): the historical prompt is skipped (answered).
    replay: list[dict[str, Any]] = [
        {"type": "approval.prompt", "id": "approval-1", "prompt": "x"},
        {"type": "approval.answer", "id": "approval-1", "approved": True},
    ]
    for ev in replay:
        fe.react(ev)  # no exception == not re-prompted


def test_react_answers_a_new_live_question(tmp_path: Path, monkeypatch: Any) -> None:
    def _beta(_qs: object) -> tuple[str, ...]:
        return ("beta",)

    monkeypatch.setattr(plan_watch, "default_stdin_questioner", _beta)
    log = tmp_path / "logs.jsonl"
    _write_log(log, [{"type": "run.start"}])
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    fe.open_prompts_at_attach(log)
    event: dict[str, Any] = {
        "type": "question.prompt",
        "id": "question-1",
        "questions": [{"question": "which?", "options": ["alpha", "beta"]}],
    }
    fe.react(event)
    assert json.loads((questions_dir(tmp_path) / "question-1.answer").read_text()) == ["beta"]
