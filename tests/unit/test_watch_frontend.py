# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Interactive `agent6 attach` prompts unanswered approvals and questions on the terminal.

The answer goes back over the file bridge; historical and answered prompts are not re-asked.
"""

from __future__ import annotations

import json
import os
import pathlib
import pty
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import pytest

from agent6.sessions import ipc
from agent6.ui.cli import _interact, plan_watch
from agent6.viewmodel import tail

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)


def _view() -> Any:
    class _V:
        def __init__(self) -> None:
            self.notices: list[str] = []

        def pause(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

        def notice(self, msg: str) -> None:
            self.notices.append(msg)

    return _V()


def _write_log(path: pathlib.Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _append_log(path: pathlib.Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def _follow(fe: Any, log: pathlib.Path) -> None:
    """Feed the log to the front-end as the follow loop does: each event's offset, then it."""
    for event in tail.tail_events(log, follow=False, on_position=fe.heard):
        fe.react(event)


def _typist(answer: str, asked: list[str]) -> Callable[..., str | None]:
    """Return a terminal approver that types `answer` unless the prompt closed first."""

    def approve(
        prompt: str, *, standing: bool = True, until: Callable[[], bool] | None = None
    ) -> str | None:
        if until is not None and until():
            return None
        asked.append(prompt)
        return answer

    return approve


def _attach_under_pty(session_dir: pathlib.Path) -> tuple[int, int]:
    """Fork a child under a fresh pty that asks the session's open prompts on its terminal.

    The child prints its notices once every prompt is handled, then exits 0.
    """
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        code = 1
        try:
            view = _view()
            fe = plan_watch._CliFrontEnd(session_dir, view)  # pyright: ignore[reportPrivateUsage]
            for event in fe.open_prompts_at_attach(session_dir / "logs.jsonl"):
                fe.handle(event)
            os.write(1, "".join(f"{n}\n" for n in view.notices).encode() + b"DONE\n")
            code = 0
        finally:
            os._exit(code)
    return pid, master


def _read_pty(master: int, until: bytes, seconds: float = 15.0) -> bytes:
    """Return the pty's output up to `until`, or what arrived before the deadline or EOF."""
    buf = b""
    deadline = time.monotonic() + seconds
    while until not in buf and time.monotonic() < deadline:
        if select.select([master], [], [], 0.2)[0]:
            try:
                chunk = os.read(master, 4096)
            except OSError:  # the child exited and the pty closed
                break
            if not chunk:
                break
            buf += chunk
    return buf


def _reap(pid: int) -> int:
    """Return the child's exit code, killing it when it is still at its prompt after 2 s."""
    deadline = time.monotonic() + 2.0
    done, status = os.waitpid(pid, os.WNOHANG)
    while done == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        done, status = os.waitpid(pid, os.WNOHANG)
    if done == 0:
        os.kill(pid, signal.SIGKILL)
        _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def test_open_prompt_at_attach_is_answered_and_written(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    # A run already waiting at approval-1 when you attach: the front-end prompts
    # and writes the answer the worker is blocked reading.
    def _yes(_prompt: str, *, standing: bool = True, until: object = None) -> str:
        return "yes"

    monkeypatch.setattr(_interact, "default_stdin_approver", _yes)
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    _write_log(
        log,
        [
            {"type": "session.start"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "run `ls`?"},
        ],
    )
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    opens = fe.open_prompts_at_attach(log)
    assert [(e["type"], e["id"]) for e in opens] == [("approval.prompt", "approval-1")]
    for event in opens:
        fe.handle(event)
    assert (ipc.approvals_dir(tmp_path) / "approval-1.answer").read_text() == "yes"


def test_already_answered_prompt_is_not_reasked(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # approval-1 was emitted AND answered in history: not open at attach, and the
    # replay must not re-prompt it (the approver would fail the test if called).
    def _forbidden(_p: object, *, standing: bool = True) -> str:
        raise AssertionError("must not prompt for an already-answered approval")

    monkeypatch.setattr(_interact, "default_stdin_approver", _forbidden)
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


def test_react_answers_a_new_live_question(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    def _beta(_qs: object, until: object = None) -> tuple[str, ...]:
        return ("beta",)

    monkeypatch.setattr(_interact, "default_stdin_questioner", _beta)
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    history: list[dict[str, Any]] = [{"type": "session.start"}]
    _write_log(log, history)
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    fe.open_prompts_at_attach(log)
    for ev in history:  # the follow loop replays the scanned prefix first
        fe.react(ev)
    event: dict[str, Any] = {
        "type": "question.prompt",
        "id": "question-1",
        "questions": [{"question": "which?", "options": ["alpha", "beta"]}],
    }
    fe.react(event)
    assert json.loads((ipc.questions_dir(tmp_path) / "question-1.answer").read_text()) == ["beta"]


def test_attach_replay_does_not_reask_an_answered_prompt(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """The pre-scan's answered ids survive the follow loop's replay of session.start.

    Cleared there, attaching to any run that had answered a prompt re-asked it and blocked.
    """
    asked: list[str] = []

    def _yes(prompt: str, *, standing: bool = True) -> str:
        asked.append(prompt)
        return "yes"

    monkeypatch.setattr(_interact, "default_stdin_approver", _yes)
    log = tmp_path / "logs.jsonl"
    history: list[dict[str, Any]] = [
        {"type": "session.start", "user_task": "t"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "ANSWERED LONG AGO"},
        {"type": "approval.answer", "id": "approval-1", "approved": True},
        {"type": "role.call", "role": "worker", "model": "m"},
    ]
    _write_log(log, history)

    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    assert fe.open_prompts_at_attach(log) == []  # nothing open
    for ev in history:  # _watch_transcript replays from the start
        fe.react(ev)
    assert asked == []  # the answered prompt is history, not a question


def test_resumed_execution_reuses_prompt_ids_and_is_still_answered(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """The answered set clears at the session boundary, since prompt ids restart per execution.

    An attached CLI dropped the new execution's first prompt and the run hung forever.
    """
    asked: list[str] = []

    def _yes(prompt: str, *, standing: bool = True, until: object = None) -> str:
        asked.append(prompt)
        return "yes"

    def _answer(_qs: Any, until: object = None) -> tuple[str, ...]:
        asked.append("question")
        return ("a",)

    monkeypatch.setattr(_interact, "default_stdin_approver", _yes)
    monkeypatch.setattr(_interact, "default_stdin_questioner", _answer)
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    leg1: list[dict[str, Any]] = [
        {"type": "session.start"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "execution 1 ok?"},
        {"type": "approval.answer", "id": "approval-1", "approved": True},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "q?"}]},
        {"type": "question.answer", "id": "question-1", "answers": ["x"]},
        {"type": "session.end", "reason": "steer_abort"},
    ]
    _write_log(log, leg1)
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    assert fe.open_prompts_at_attach(log) == []  # execution 1 is fully answered
    for ev in leg1:  # the follow loop replays the whole log first
        fe.react(ev)
    assert asked == []  # nothing historical is re-asked

    # The resumed execution restarts the id counters and prompts again, live.
    leg2: tuple[dict[str, object], ...] = (
        {"type": "loop.resume.start", "iteration": 2},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "execution 2 ok?"},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "q2?"}]},
    )
    for ev in leg2:
        fe.react(ev)
    assert asked == ["execution 2 ok?", "question"]  # both prompted, neither swallowed
    assert (ipc.approvals_dir(tmp_path) / "approval-1.answer").read_text(encoding="utf-8") == "yes"
    assert (ipc.questions_dir(tmp_path) / "question-1.answer").exists()


def test_an_answer_that_lost_to_another_surface_is_reported(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A prompt answered elsewhere while this terminal asked keeps the first answer, and says so."""

    def _yes(_prompt: str, *, standing: bool = True, until: object = None) -> str:
        return "yes"

    monkeypatch.setattr(_interact, "default_stdin_approver", _yes)
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    event: dict[str, Any] = {"type": "approval.prompt", "id": "approval-1", "prompt": "run `ls`?"}
    _write_log(log, [{"type": "session.start"}, event])
    assert ipc.write_answer(tmp_path, "approval-1", "no")
    view = _view()
    fe = plan_watch._CliFrontEnd(tmp_path, view)  # pyright: ignore[reportPrivateUsage]
    for opened in fe.open_prompts_at_attach(log):
        fe.handle(opened)
    assert view.notices == [f"[agent6] {ipc.ANSWERED_ELSEWHERE}"]
    assert (ipc.approvals_dir(tmp_path) / "approval-1.answer").read_text() == "no"


def test_an_unscoped_approval_offers_no_session_choice(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A gate with no scope to grant offers no "allow all" answer on the foreground prompt.

    `fetch` journals `standing: false` and `record_answer` drops a grant it cannot scope.
    """
    seen: list[bool] = []

    def _approver(_prompt: str, *, standing: bool = True, until: object = None) -> str:
        seen.append(standing)
        return "no"

    monkeypatch.setattr(_interact, "default_stdin_approver", _approver)
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    history: list[dict[str, Any]] = [
        {"type": "session.start"},
        {
            "type": "approval.prompt",
            "id": "approval-1",
            "prompt": "Allow fetch: example.invalid /data",
            "standing": False,
        },
    ]
    _write_log(log, history)
    live = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    _follow(live, log)
    attached = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    for event in attached.open_prompts_at_attach(log):
        attached.handle(event)
    assert seen == [False, False]


_OPEN_APPROVAL: dict[str, Any] = {
    "type": "approval.prompt",
    "id": "approval-1",
    "prompt": "Allow run_command: ls",
}
_OPEN_QUESTION: dict[str, Any] = {
    "type": "question.prompt",
    "id": "question-1",
    "questions": [{"question": "which port?", "options": []}],
}


@pytest.mark.parametrize(
    ("prompt", "shown", "answer", "slot"),
    [
        (
            _OPEN_APPROVAL,
            b"[y/N/a/d]",
            {"type": "approval.answer", "id": "approval-1", "approved": True, "source": "frontend"},
            "approvals/approval-1.answer",
        ),
        (
            _OPEN_QUESTION,
            b"which port?",
            {
                "type": "question.answer",
                "id": "question-1",
                "answers": ["9090"],
                "source": "frontend",
            },
            "questions/question-1.answer",
        ),
    ],
    ids=["approval", "question"],
)
def test_a_prompt_answered_elsewhere_ends_the_attach_prompt(
    tmp_path: pathlib.Path,
    prompt: dict[str, Any],
    shown: bytes,
    answer: dict[str, Any],
    slot: str,
) -> None:
    """Attach drops a prompt once the journal holds its answer, with nothing typed.

    The wait read the terminal alone: attach stayed on the prompt after another front-end
    answered it, through the end of the run, and Enter then wrote an answer for it.
    """
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    _write_log(log, [{"type": "session.start"}, prompt])
    pid, master = _attach_under_pty(tmp_path)
    try:
        assert shown in _read_pty(master, shown)
        _append_log(log, answer)
        tail_out = _read_pty(master, b"DONE")
        assert _reap(pid) == 0, f"attach stayed on the answered prompt: {tail_out[-300:]!r}"
    finally:
        os.close(master)
    assert tail_out.count(f"[agent6] {ipc.ANSWERED_ELSEWHERE} (source: frontend)".encode()) == 1
    assert not (tmp_path / slot).exists()


def test_an_answer_typed_at_the_attach_prompt_is_written(tmp_path: pathlib.Path) -> None:
    """A line typed at an open attach prompt is the answer the worker reads, with no notice."""
    ipc.write_worker_pid(tmp_path, os.getpid())
    _write_log(tmp_path / "logs.jsonl", [{"type": "session.start"}, _OPEN_APPROVAL])
    pid, master = _attach_under_pty(tmp_path)
    try:
        assert b"[y/N/a/d]" in _read_pty(master, b"[y/N/a/d]")
        os.write(master, b"y\n")
        tail_out = _read_pty(master, b"DONE")
        assert _reap(pid) == 0, tail_out[-300:]
    finally:
        os.close(master)
    assert b"[agent6]" not in tail_out
    assert (ipc.approvals_dir(tmp_path) / "approval-1.answer").read_text(encoding="utf-8") == "yes"


def test_a_worker_that_dies_ends_the_attach_prompt(tmp_path: pathlib.Path) -> None:
    """Attach drops an open prompt once the worker is gone, and writes no answer for it.

    The wait never read the worker pid, so attach stayed on a dead run's prompt.
    """
    worker = subprocess.Popen(["sleep", "60"])
    try:
        ipc.write_worker_pid(tmp_path, worker.pid)
        _write_log(tmp_path / "logs.jsonl", [{"type": "session.start"}, _OPEN_APPROVAL])
        pid, master = _attach_under_pty(tmp_path)
        try:
            assert b"[y/N/a/d]" in _read_pty(master, b"[y/N/a/d]")
            worker.kill()
            worker.wait()
            tail_out = _read_pty(master, b"DONE")
            assert _reap(pid) == 0, f"attach stayed on a dead run's prompt: {tail_out[-300:]!r}"
        finally:
            os.close(master)
    finally:
        worker.kill()
        worker.wait()
    assert tail_out.count(b"[agent6] the run ended before this prompt was answered") == 1
    assert not (ipc.approvals_dir(tmp_path) / "approval-1.answer").exists()


def test_a_line_typed_as_the_prompt_closes_writes_no_answer(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """An answer typed after the journal closed the prompt is not written.

    The worker had consumed the first answer, so the late one landed in the empty slot:
    a `no` on disk for a prompt the journal records as approved.
    """
    log = tmp_path / "logs.jsonl"

    def _late_no(_prompt: str, *, standing: bool = True, until: object = None) -> str:
        _append_log(log, {"type": "approval.answer", "id": "approval-1", "approved": True})
        return "no"

    monkeypatch.setattr(_interact, "default_stdin_approver", _late_no)
    ipc.write_worker_pid(tmp_path, os.getpid())
    _write_log(log, [{"type": "session.start"}, _OPEN_APPROVAL])
    view = _view()
    fe = plan_watch._CliFrontEnd(tmp_path, view)  # pyright: ignore[reportPrivateUsage]
    for event in fe.open_prompts_at_attach(log):
        fe.handle(event)
    assert view.notices == [f"[agent6] {ipc.ANSWERED_ELSEWHERE}"]
    assert not (ipc.approvals_dir(tmp_path) / "approval-1.answer").exists()


def test_a_prompt_whose_execution_ended_takes_no_answer(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A prompt left open by an execution that ended is closed, though a worker is live again.

    A resume reuses the id, so an answer typed at the old prompt landed on a command this
    terminal never showed.
    """
    log = tmp_path / "logs.jsonl"

    def _late_yes(_prompt: str, *, standing: bool = True, until: object = None) -> str:
        _append_log(log, {"type": "session.end", "reason": "steer_abort"})
        _append_log(log, {"type": "loop.resume.start", "iteration": 2})
        _append_log(log, {"type": "approval.prompt", "id": "approval-1", "prompt": "rm -rf x"})
        return "yes"

    monkeypatch.setattr(_interact, "default_stdin_approver", _late_yes)
    ipc.write_worker_pid(tmp_path, os.getpid())
    _write_log(log, [{"type": "session.start"}, _OPEN_APPROVAL])
    view = _view()
    fe = plan_watch._CliFrontEnd(tmp_path, view)  # pyright: ignore[reportPrivateUsage]
    for event in fe.open_prompts_at_attach(log):
        fe.handle(event)
    assert view.notices == ["[agent6] the run ended before this prompt was answered"]
    assert not (ipc.approvals_dir(tmp_path) / "approval-1.answer").exists()


def test_an_older_executions_answer_does_not_close_the_new_prompt(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """An earlier execution's answer to the same id leaves the resumed execution's prompt open.

    Read as this prompt's answer, attach would drop a live prompt and the run would wait on
    an answer no terminal offers.
    """
    asked: list[str] = []
    monkeypatch.setattr(_interact, "default_stdin_approver", _typist("yes", asked))
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    _write_log(
        log,
        [
            {"type": "session.start"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "execution 1 ok?"},
            {"type": "approval.answer", "id": "approval-1", "approved": False, "source": "stdin"},
            {"type": "session.end", "reason": "steer_abort"},
            {"type": "loop.resume.start", "iteration": 2},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "execution 2 ok?"},
        ],
    )
    slot = ipc.approvals_dir(tmp_path) / "approval-1.answer"

    at_attach = _view()
    fe = plan_watch._CliFrontEnd(tmp_path, at_attach)  # pyright: ignore[reportPrivateUsage]
    for event in fe.open_prompts_at_attach(log):
        fe.handle(event)
    assert (asked, at_attach.notices) == (["execution 2 ok?"], [])
    assert slot.read_text(encoding="utf-8") == "yes"

    # The same log met live: execution 1's prompt is closed by its own answer, execution 2's is not.
    slot.unlink()
    asked.clear()
    live = _view()
    _follow(plan_watch._CliFrontEnd(tmp_path, live), log)  # pyright: ignore[reportPrivateUsage]
    assert asked == ["execution 2 ok?"]
    assert live.notices == [f"[agent6] {ipc.ANSWERED_ELSEWHERE} (source: stdin)"]
    assert slot.read_text(encoding="utf-8") == "yes"


def test_two_open_prompts_each_close_on_their_own_answer(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """An answer to one open prompt closes that prompt alone, whichever is asked first."""
    log = tmp_path / "logs.jsonl"
    asked: list[str] = []

    def _approver(
        prompt: str, *, standing: bool = True, until: Callable[[], bool] | None = None
    ) -> str | None:
        if until is not None and until():
            return None
        asked.append(prompt)
        # The second prompt is answered from another front-end while the first is asked here.
        _append_log(log, {"type": "approval.answer", "id": "approval-2", "approved": True})
        return "yes"

    monkeypatch.setattr(_interact, "default_stdin_approver", _approver)
    ipc.write_worker_pid(tmp_path, os.getpid())
    _write_log(
        log,
        [
            {"type": "session.start"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "first?"},
            {"type": "approval.prompt", "id": "approval-2", "prompt": "second?"},
        ],
    )
    view = _view()
    fe = plan_watch._CliFrontEnd(tmp_path, view)  # pyright: ignore[reportPrivateUsage]
    for event in fe.open_prompts_at_attach(log):
        fe.handle(event)
    assert asked == ["first?"]
    assert (ipc.approvals_dir(tmp_path) / "approval-1.answer").read_text(encoding="utf-8") == "yes"
    assert not (ipc.approvals_dir(tmp_path) / "approval-2.answer").exists()
    assert view.notices == [f"[agent6] {ipc.ANSWERED_ELSEWHERE}"]


def test_attach_follows_on_after_a_prompt_answered_elsewhere(tmp_path: pathlib.Path) -> None:
    """The follow loop drops the answered prompt, prints what follows, and asks the next one.

    Attach printed nothing more until a key was pressed. The next prompt reuses the id in a
    resumed execution: read from the wrong journal offset, the first answer would close it.
    """
    ipc.write_worker_pid(tmp_path, os.getpid())
    log = tmp_path / "logs.jsonl"
    _write_log(log, [{"type": "session.start", "user_task": "t"}, _OPEN_APPROVAL])
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        code = 1
        try:
            # The pty, in place of the streams pytest captured.
            sys.stdin = open(0, closefd=False)  # noqa: SIM115  # held until the child exits
            sys.stdout = open(1, "w", closefd=False)  # noqa: SIM115  # held until the child exits
            sys.stderr = open(2, "w", closefd=False)  # noqa: SIM115  # held until the child exits
            code = plan_watch._watch_transcript(tmp_path)  # pyright: ignore[reportPrivateUsage]
            sys.stdout.flush()
        finally:
            os._exit(code)
    try:
        assert b"[y/N/a/d]" in _read_pty(master, b"[y/N/a/d]")
        with log.open("a", encoding="utf-8") as fh:  # one write, so the tail reads one batch
            fh.write(
                "".join(
                    json.dumps(event) + "\n"
                    for event in (
                        {**_OPEN_APPROVAL, "type": "approval.answer", "source": "frontend"},
                        {"type": "session.end", "reason": "steer_abort"},
                        {"type": "loop.resume.start", "iteration": 2},
                        {**_OPEN_APPROVAL, "prompt": "Allow run_command: second"},
                    )
                )
            )
        followed = _read_pty(master, b"[y/N/a/d]")
        assert b"second" in followed, f"the resumed execution's prompt was not asked: {followed!r}"
        assert followed.count(f"[agent6] {ipc.ANSWERED_ELSEWHERE} (source: frontend)".encode()) == 1
        os.write(master, b"y\n")
        slot = ipc.approvals_dir(tmp_path) / "approval-1.answer"
        deadline = time.monotonic() + 10
        while not slot.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert slot.exists(), "the resumed execution's prompt was dropped as answered"
        assert slot.read_text(encoding="utf-8") == "yes"
        _append_log(log, {"type": "session.end", "reason": "finish_session", "all_passed": True})
        _read_pty(master, b"\x00", seconds=5.0)  # drain to the child's exit
        assert _reap(pid) == 0
    finally:
        os.close(master)
