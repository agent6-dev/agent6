"""Exercise compaction at a wait boundary without making a live model call."""

from __future__ import annotations

import dataclasses
import pathlib
import threading
import time
from typing import Any

import pytest

from agent6.app import role_providers
from agent6.config import harness_config
from agent6.config import model as config_model
from agent6.harness import loop_conversation, loop_steer, operator_bridge
from agent6.harness.compaction import driver, window
from agent6.providers import provider_types
from agent6.sessions import journal
from agent6.tools import operator_prompts, schema, waits
from tests.sandbox import test_command_checkin as commands
from tests.unit import test_children_background as children
from tests.unit import test_compaction_writers as writers
from tests.unit import test_harness


def test_missing_or_cli_provider_supplies_no_idle_limits() -> None:
    """An unresolved fixture or a CLI-owned session cannot arm a provider deadline."""
    cfg = config_model.Config.model_validate({"providers": {"cli": {"backend": "claude_code"}}})
    assert role_providers.idle_compaction(cfg, None) is None
    for route in ("missing/model", "cli/model"):
        assert role_providers.idle_compaction(cfg, harness_config.RoleModel(model=route)) is None


def test_worker_clock_uses_the_start_of_the_successful_attempt() -> None:
    """A retry starts a fresh cache lifetime before it generates the reply."""
    starts: list[float] = []

    class Retry:
        def call(self, **_kwargs: Any) -> provider_types.ProviderResponse:
            starts.append(time.monotonic())
            if len(starts) == 1:
                raise provider_types.ProviderError("transient")
            return writers.response()

    caller = test_harness._caller(Retry(), retry_count=1)
    test_harness._call(caller)
    assert starts[0] < caller.counter.request_started <= starts[1]


def test_parked_wait_covers_the_complete_prose_turn() -> None:
    """A parked run summarizes the prose reply before reading the next steer."""
    conv = writers.history()
    conv.assistant([{"type": "text", "text": "work complete"}])
    complete = conv.to_wire()
    provider = writers.Writer(writers.response())
    events: list[Any] = []
    turn = writers.context(
        provider, provider, events, conversation=conv, idle=window.IdleCompaction("p", 0.02, 100)
    )
    deadline = driver.Tiered().idle(turn, time.monotonic())
    assert deadline is not None
    limit = time.monotonic() + 4
    bridge = operator_bridge.OperatorBridge(
        steer_requested=lambda: bool(events) or time.monotonic() >= limit,
        steer_prompt=lambda: "continue with tests",
    )
    harness = test_harness._wf(interactive=True, bridge=bridge)
    with waits.bind(deadline):
        loop_steer.park_for_steer(
            harness, conv, test_harness._state(), turn=3, reason="silent_finish"
        )
    assert deadline.result()
    assert events[-1].kept_messages == 0
    assert provider.calls[0]["messages"][:-1] == complete
    assert "continue with tests" in str(conv.to_wire()[-1])


def _pending(conv: loop_conversation.Conversation) -> loop_conversation.AssistantMessage:
    return conv.assistant(
        [
            {"type": "thinking", "thinking": "choose the command", "signature": "signed"},
            {
                "type": "tool_use",
                "name": "run_command",
                "id": "wait",
                "input": {"argv": ["sleep", "1"]},
            },
        ]
    )


def test_idle_summary_uses_the_cached_request_then_keeps_only_the_pending_reply() -> None:
    conv = writers.history()
    previous = conv.to_wire()
    pending = _pending(conv)
    started = time.monotonic()
    events: list[Any] = []
    accepted = threading.Event()

    def emit(event: journal.Event) -> None:
        events.append(event)
        if isinstance(event, journal.CompactSummarizeDone):
            accepted.set()

    worker = writers.Writer(writers.response())
    turn = writers.context(
        worker,
        writers.Writer(writers.response()),
        events,
        conversation=conv,
        idle=window.IdleCompaction("p", 0.05, 80000),
    )
    compactor = driver.Tiered()
    deadline = compactor.idle(dataclasses.replace(turn, emit=emit), started)
    assert deadline is not None and deadline.at == started + 0.05
    with waits.bind(deadline), waits.during():
        assert accepted.wait(4)
        assert events[-1].reason == "idle"
        assert events[-1].gap_s >= 0.05
        assert conv.messages[-1] is pending
    assert worker.calls[0]["messages"][:-1] == previous
    assert "write the summary now" in worker.calls[0]["messages"][-1]["content"].lower()
    conv.results([loop_conversation.ToolResultItem("wait", "finished", pending.tool_uses[0])])
    assert len(conv) == 3 and "finished" in str(conv.to_wire()[-1])
    assert not compactor.idle(turn, time.monotonic())


@pytest.mark.parametrize("seconds,floor", [(-1, 0), (0.01, 1000000)])
def test_idle_limit_and_size_must_both_apply(seconds: float, floor: int) -> None:
    writer = writers.Writer(writers.response())
    turn = writers.context(writer, writer, [], idle=window.IdleCompaction("p", seconds, floor))
    assert driver.Tiered().idle(turn, time.monotonic()) is None
    assert not writer.calls


def test_idle_rejection_keeps_the_marks_and_the_open_call() -> None:
    worker = writers.Writer(writers.response(""))
    helper = writers.Writer(writers.response(""))
    conv = writers.history()
    _pending(conv)
    before = conv.to_wire()
    turn = writers.context(
        worker, helper, [], conversation=conv, idle=window.IdleCompaction("p", 1, 0)
    )
    deadline = driver.Tiered().idle(turn, time.monotonic() - 2)
    assert deadline is not None
    with waits.bind(deadline):
        for _ in range(2):
            with waits.during():
                time.sleep(0.03)
    assert not deadline.result()
    assert conv.to_wire() == before
    assert len(worker.calls) == len(helper.calls) == 1
    assert "choose the command" not in str(helper.calls[0]["messages"])


@pytest.mark.parametrize("kind", ["command", "background", "child", "approval", "question"])
def test_wait_delivers_its_result_after_the_callback(tmp_path: pathlib.Path, kind: str) -> None:
    fired = threading.Event()
    replied = threading.Event()
    command = "while [ ! -f release ]; do sleep 0.02; done; echo waited"

    def callback() -> bool:
        fired.set()
        if kind in ("command", "background"):
            (tmp_path / "repo" / "release").touch()
        if kind in ("approval", "question"):
            assert replied.wait(2)
        return True

    def answer() -> None:
        assert fired.wait(2), "the wait never called back"
        replied.set()

    def approve(_request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        answer()
        return operator_prompts.ApprovalAnswer(True, "frontend")

    def ask(_request: operator_prompts.QuestionRequest, /) -> operator_prompts.QuestionAnswer:
        answer()
        return operator_prompts.QuestionAnswer(("yes",), "frontend")

    gate = operator_prompts.OperatorPrompts(approver=approve, questioner=ask)
    if kind == "child":
        repo = tmp_path / "repo"
        children._init_repo(repo)
        group = children._Children(repo, "run")
        harness = children._harness(repo, tmp_path / "run", None, group, max_turns=1)
        children._start(harness.parallel, ("", "work"))
        dispatcher = harness.dispatcher
    else:
        dispatcher = commands._dispatcher(tmp_path, checkin=4)
    try:
        if kind == "background":
            dispatcher.dispatch(
                "run_command", {"argv": ["/bin/sh", "-c", command], "background": True}
            )
        deadline = waits.Deadline(time.monotonic() + 0.02, callback)
        with waits.bind(deadline):
            if kind == "command":
                result = commands._run(dispatcher, command)
                assert result["stdout"] == "waited\n" and result["returncode"] == 0
            elif kind == "background":
                result = dispatcher.dispatch(
                    "read_background", {"id": "bg1", "wait_s": 4}
                ).to_wire()
                assert result["returncode"] == 0 and "waited" in result["output"]
            elif kind == "child":
                result = dispatcher.dispatch(
                    "read_background", {"id": "run-p1-l1", "wait_s": 0.3}
                ).to_wire()
                assert result["state"] == "starting"
            elif kind == "approval":
                assert gate.approve("Allow the command?")
            else:
                assert gate.ask((schema.UserQuestion(question="Continue?"),)).answers == ("yes",)
        assert deadline.result()
    finally:
        dispatcher.close()


@pytest.mark.parametrize(
    "writer,cached,expected",
    [
        ("worker", 0, 1),
        ("worker", 599, 1),
        ("worker", 600, 0),
        ("worker", None, 0),
        ("helper", 0, 0),
    ],
)
def test_cold_notice_uses_only_reported_worker_usage(
    writer: str, cached: int | None, expected: int
) -> None:
    response = writers.response(
        input_tokens=500,
        cache_write_tokens=100,
        cache_read_tokens=cached or 0,
        raw={"usage": {"cache_read_input_tokens": cached}},
    )
    provider = writers.Writer(response)
    events: list[Any] = []
    log: list[str] = []
    compactor = driver.Tiered()
    base = writers.context(
        provider, provider, events, summarizer=writer, idle=window.IdleCompaction("p", 2, 0)
    )
    turn = dataclasses.replace(base, log=log.append)
    for _ in range(2):
        turn.conversation.notice("growth" * 30000)
        deadline = compactor.idle(turn, time.monotonic() - 2)
        assert deadline is not None
        with waits.bind(deadline), waits.during():
            time.sleep(0.05)
        assert deadline.result()
    cold = [e for e in events if e.TYPE == "compact.idle.cache_cold"]
    assert len(cold) == expected
    assert sum("may be longer than the cache lasts" in line for line in log) == expected
    if cold:
        assert (cold[0].uncached_tokens, cold[0].cached_tokens) == (600, cached)
        assert cold[0].setting == "providers.p.idle_compact_s" and cold[0].gap_s >= 2
