# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A parent steers a running child with `steer_child`, through the child's own steer channel."""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import child_records, directive
from agent6.config import Config
from agent6.harness import (
    child_work,
    loop_conversation,
    loop_state,
    operator_bridge,
    parallel_dispatch,
    run_chain,
    toolset,
)
from agent6.sessions import ipc, journal
from agent6.sessions import layout as sessions_layout
from agent6.tools import child_tools, dispatch, schema, sessions, tool_errors
from agent6.tools import results as tools_results


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _config(*, tool: bool) -> Config:
    return Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {"default": {"model": "p/base"}},
            "harness": {"parallel": {"tool": tool}},
        }
    )


class _Work:
    """The loop's children as the tools reach them, recording each steer."""

    def __init__(self) -> None:
        self.steered: list[tuple[str, str]] = []

    def start(self, task: str, spec: str) -> tools_results.StartedResult:
        raise AssertionError((task, spec))

    def roster(self) -> tuple[child_tools.ChildLine, ...]:
        return ()

    def wait(self, child: str, wait_s: float, interrupted: Any) -> None:
        raise AssertionError((child, wait_s, interrupted))

    def stop(self, child: str) -> None:
        raise AssertionError(child)

    def steer(self, child: str, text: str) -> str:
        self.steered.append((child, text))
        return f"[parallel] child {child} reads this at its next turn boundary."


def _tools(
    tmp_path: pathlib.Path, cfg: Config, *, mode: str = "run", work: Any = None
) -> dispatch.ToolDispatcher:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    d = dispatch.ToolDispatcher(
        root=repo,
        config=cfg,
        isolation="none",
        mode=mode,  # type: ignore[arg-type]
        session_dir=tmp_path / "session",
    )
    d.children = work
    return d


def _names(d: dispatch.ToolDispatcher, mode: str = "run") -> list[str]:
    return [t.name for t in toolset.tool_definitions(d, mode=mode)]  # type: ignore[arg-type]


def test_steer_child_is_offered_where_parallel_is_and_refused_everywhere_else(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool follows `parallel`: the knob, run mode, not a child, a loop that holds children."""
    call = {"id": "run-p1-l1", "text": "also write the test"}
    off = _tools(tmp_path, _config(tool=False), work=_Work())
    assert "steer_child" not in _names(off)
    with pytest.raises(tool_errors.ToolError, match=r"parallel\.tool"):
        off.dispatch("steer_child", call)
    for mode in ("plan", "ask", "machine", "reviewer"):
        other = _tools(tmp_path, _config(tool=True), mode=mode, work=_Work())
        assert "steer_child" not in _names(other, mode)
        with pytest.raises(tool_errors.ToolError, match="not available"):
            other.dispatch("steer_child", call)
    monkeypatch.setenv("AGENT6_CHILD", "1")
    inside = _tools(tmp_path, _config(tool=True), work=_Work())
    assert "steer_child" not in _names(inside)
    with pytest.raises(tool_errors.ToolError, match="inside a child"):
        inside.dispatch("steer_child", call)
    monkeypatch.delenv("AGENT6_CHILD")
    headless = _tools(tmp_path, _config(tool=True), work=None)
    assert "steer_child" not in _names(headless)
    with pytest.raises(tool_errors.ToolError, match="no child dispatcher"):
        headless.dispatch("steer_child", call)
    # A reviewer's contract carries the reviewer mode's tools: the parent cannot hand this down.
    assert "steer_child" not in schema.mode_tools("reviewer").names

    work = _Work()
    on = _tools(tmp_path, _config(tool=True), work=work)
    assert _names(on).index("steer_child") == _names(on).index("parallel") + 1
    (steer,) = [t for t in toolset.tool_definitions(on) if t.name == "steer_child"]
    assert steer.description == (
        "Send an instruction to a running child, by the id `parallel` returned. The child reads"
        " it at its next turn boundary, marked as from its parent, and keeps working; the call"
        " returns at once. `read_background` shows what a child has done since."
    )
    assert steer.input_schema["required"] == ["id", "text"]
    (parallel,) = [t for t in toolset.tool_definitions(on) if t.name == "parallel"]
    assert "`steer_child` sends a running child an instruction;" in parallel.description
    assert "a worker with this run's tools, guards and limits" in parallel.description


def test_steer_child_answers_at_once_and_refuses_a_blank_instruction(
    tmp_path: pathlib.Path,
) -> None:
    """The call hands the text to the loop's children and returns what it was told."""
    work = _Work()
    d = _tools(tmp_path, _config(tool=True), work=work)

    result = d.dispatch("steer_child", {"id": "run-p1-l1", "text": "also write the test"})

    assert work.steered == [("run-p1-l1", "also write the test")]
    assert result.to_wire() == {
        "steered": "run-p1-l1",
        "message": "[parallel] child run-p1-l1 reads this at its next turn boundary.",
    }
    assert result.summary() == "steered run-p1-l1"
    for bad in ({"id": "run-p1-l1", "text": "  \n"}, {"id": "", "text": "t"}, {"text": "t"}):
        with pytest.raises(tool_errors.ToolError):
            d.dispatch("steer_child", bad)
    assert len(work.steered) == 1


class _Running:
    """A group spawner whose children run until the test ends them."""

    def __init__(self) -> None:
        self.results: dict[str, concurrent.futures.Future[child_records.ChildResult]] = {}

    def __call__(
        self, children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        started: list[child_records.StartedChild] = []
        for number, _child in enumerate(children, start=1):
            sid = f"root-{group}-l{number}"
            self.results[sid] = concurrent.futures.Future()
            started.append(child_records.StartedChild(sid, self.results[sid]))
        return started

    def end(self, sid: str, state_dir: pathlib.Path) -> None:
        self.results[sid].set_result(
            child_records.ChildResult(
                spec=child_records.ChildSpec(1, sid, state_dir, None),
                session_dir=state_dir,
                branch=f"agent6/{sid}",
                ok=True,
                error="",
            )
        )


def _dispatcher(tmp_path: pathlib.Path, children: _Running) -> parallel_dispatch.ParallelDispatcher:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    session_dir = tmp_path / "state" / "sessions" / "runs" / "root"
    session_dir.mkdir(parents=True)
    return parallel_dispatch.ParallelDispatcher(
        chain=run_chain.RunChain(
            repo, ref="refs/agent6/root", fallback_parent=_git(repo, "rev-parse", "HEAD")
        ),
        curator=None,
        max_children=4,
        child_spawner=children,
        save_snapshot=lambda *_a, **_k: None,
        log=lambda _m: None,
        emit=lambda *_a: None,
        emit_graph_snapshot=lambda: None,
        session_id="root",
        execution_id="exec-1",
        session_dir=session_dir,
    )


def test_a_steer_reaches_a_running_child_through_its_own_steer_channel(
    tmp_path: pathlib.Path,
) -> None:
    """The parent writes what `agent6 steer <child>` writes: the slot, then the inbox, in order."""
    children = _Running()
    dispatcher = _dispatcher(tmp_path, children)
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)
    dispatcher.dispatch(
        loop_conversation.Conversation(),
        1,
        state,
        [directive.Segment(spec="", task="write a"), directive.Segment(spec="", task="write b")],
    )
    one, two = "root-p1-l1", "root-p2-l1"
    runs = tmp_path / "state" / "sessions" / "runs"

    # A child with no session dir has not started: nothing is written anywhere.
    with pytest.raises(tool_errors.ToolError) as early:
        child_work.steer(dispatcher, one, "also write the test")
    assert str(early.value) == (
        f"child {one} has not started yet; nothing was sent. Send it again once"
        " read_background shows its transcript."
    )
    assert not (runs / one).exists()

    (runs / one).mkdir()
    said = child_work.steer(dispatcher, one, "also write the test")
    assert said == f"[parallel] child {one} reads this at its next turn boundary."
    assert (runs / one / ipc.STEER_ANSWER_FILE).read_text(encoding="utf-8") == (
        "PARENT STEERING (an instruction from the run that started this child):\n"
        "also write the test"
    )
    assert ipc.steer_request_pending(runs / one) and not ipc.steer_interrupt_pending(runs / one)
    # A second steer waits behind the first, as the operator's would.
    child_work.steer(dispatcher, one, "abort")
    (queued,) = ipc.queued_steers(runs / one)
    assert queued.text.endswith("this child):\nabort") and not queued.now
    assert not ipc.steer_answer_is_abort(runs / one)

    # Nothing is sent to a child that has ended, or to an id this run did not start.
    children.end(two, tmp_path / "state")
    (runs / two).mkdir()
    with pytest.raises(tool_errors.ToolError) as ended:
        child_work.steer(dispatcher, two, "one more thing")
    assert str(ended.value) == (
        f"child {two} has ended (ended, not merged); nothing was sent."
        " parallel starts another child."
    )
    assert not (runs / two / ipc.STEER_ANSWER_FILE).exists()
    (runs / "other-run").mkdir()
    for stranger in ("other-run", "root", "root-p9-l1", "bg1"):
        with pytest.raises(tool_errors.ToolError) as unknown:
            child_work.steer(dispatcher, stranger, "do this")
        assert str(unknown.value) == (
            f"no child {stranger} in this run; read_background with no id lists this run's"
            " children."
        )
    assert not (runs / "other-run" / ipc.STEER_ANSWER_FILE).exists()


def _steering(
    told: list[str | None], events: list[journal.Event]
) -> tuple[operator_bridge.Steering, list[str]]:
    """A child's steer reader over scripted steers, with the directives it must not run."""
    ran: list[str] = []

    def parallel() -> Any:
        ran.append("parallel")
        raise AssertionError("a parent's steer dispatched work")

    def decide(state: loop_state.LoopState, question: str, answer: str) -> None:
        del state
        ran.append(f"decision: {question} -> {answer}")

    bridge = operator_bridge.OperatorBridge(
        steer_requested=lambda: bool(told), steer_prompt=lambda: told.pop(0)
    )
    steering = operator_bridge.Steering(
        bridge=bridge,
        parallel=parallel,
        dispatcher=None,  # type: ignore[arg-type]
        record_decision=decide,
        log=lambda _m: None,
        emit=events.append,
    )
    return steering, ran


@pytest.mark.parametrize(
    "text", ["abort", "exit", "/undo", "detach", "/parallel 2 -- go wide", "/pin never rebase"]
)
def test_a_parents_steer_is_an_instruction_whatever_it_says(text: str) -> None:
    """The child injects a parent's steer as text: no verb, no directive, no pin, no ruling."""
    events: list[journal.Event] = []
    steering, ran = _steering([f"{operator_bridge.PARENT_STEERING}\n{text}"], events)
    conversation = loop_conversation.Conversation()
    conversation.notice("TASK:\nwrite a", origin="operator")
    conversation.assistant([{"type": "text", "text": "Should I also write the test?"}])
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)

    assert steering.handle(conversation, 3, state) is None

    wire = json.dumps(conversation.to_wire())
    assert (
        "PARENT STEERING (an instruction from the run that started this child):\\n"
        + json.dumps(text)[1:-1]
        in wire
    )
    assert "OPERATOR STEERING" not in wire
    assert ran == [] and state.pins == []
    kinds = [event.TYPE for event in events]
    assert kinds == ["steer.requested", "steer.parent"]
    parent = events[1]
    assert isinstance(parent, journal.SteerFromParent)
    assert (parent.text, parent.chars) == (text, len(text))


def test_a_childs_transcript_names_its_parent_as_the_speaker(tmp_path: pathlib.Path) -> None:
    """`read_session` and `read_background` show a parent's steer as the parent's words."""
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="run-p1-l1")
    layout.session_dir.mkdir(parents=True)
    rows = [
        {"type": "session.start", "user_task": "write a"},
        {"type": "steer.parent", "chars": 19, "text": "also write the test"},
        {"type": "steer.injected", "chars": 9, "text": "be brief"},
    ]
    layout.logs_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    said = sessions.conversation(layout, max_chars=2000).splitlines()

    assert said == ["user: write a", "", "parent: also write the test", "", "user: be brief"]
