# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Children run in the background of their parent and land at its turn boundaries.

A scripted provider drives the parent's turns and a fake spawner's children end when the
script says, so the order of ends, notices and landings is the test's own.
"""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from agent6 import budget as agent6_budget
from agent6 import child_records, directive
from agent6.config import Config
from agent6.harness import (
    loop,
    loop_conversation,
    loop_state,
    operator_bridge,
    parallel_dispatch,
    provider_call,
    run_chain,
)
from agent6.providers import ProviderResponse
from agent6.sessions import child_dispatch, event_log
from agent6.tools import dispatch


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


class _Children:
    """A group spawner whose children run until the test ends them, each on a real branch."""

    def __init__(self, repo: pathlib.Path, parent: str) -> None:
        self.repo = repo
        self.parent = parent
        self.base = _git(repo, "rev-parse", "HEAD")
        self.results: dict[str, concurrent.futures.Future[child_records.ChildResult]] = {}
        self.tasks: dict[str, str] = {}
        self.stopped: list[str] = []

    def __call__(
        self, children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        started: list[child_records.StartedChild] = []
        for number, child in enumerate(children, start=1):
            sid = f"{self.parent}-{group}-l{number}"
            self.results[sid] = concurrent.futures.Future()
            self.tasks[sid] = child.task
            started.append(
                child_records.StartedChild(sid, self.results[sid], lambda sid=sid: self.stop(sid))
            )
        return started

    def running(self) -> list[str]:
        return [sid for sid, result in self.results.items() if not result.done()]

    def end(
        self,
        sid: str,
        *,
        path: str = "",
        text: str = "",
        report: str = "",
        stopped_by: str = "",
    ) -> None:
        """End a child: with a commit of `path` on its branch, or with no change."""
        branch = f"agent6/{sid}"
        tip = ""
        if path:
            tree = self.repo.parent / f"wt-{sid}"
            _git(self.repo, "worktree", "add", "-q", "-b", branch, str(tree), self.base)
            (tree / path).write_text(text or f"{sid}\n", encoding="utf-8")
            _git(tree, "add", "-A")
            _git(tree, "commit", "-q", "-m", f"{sid} work")
            _git(self.repo, "worktree", "remove", "--force", str(tree))
            tip = _git(self.repo, "rev-parse", branch)
        number = int(sid.rsplit("-l", 1)[1])
        self.results[sid].set_result(
            child_records.ChildResult(
                spec=child_records.ChildSpec(number, sid, self.repo.parent / sid, None),
                session_dir=self.repo.parent / sid,
                branch=branch,
                ok=True,
                error="",
                imported_ref=branch if tip else "",
                imported_tip=tip,
                has_commits=bool(tip),
                usage=agent6_budget.UsageSettlement(usd=0.25, usd_partial=False, tokens=0),
                report=report,
                stopped_by=stopped_by,
            )
        )

    def stop(self, sid: str) -> None:
        """Stand in for the watch: a stopped child ends with what it had committed."""
        self.stopped.append(sid)
        self.end(sid, path=f"{sid}.txt", stopped_by="stopped by its parent")


def _first_ranked(
    results: Sequence[child_records.ChildResult], *, group: str, base: str
) -> child_records.Ranking:
    del group, base
    order = tuple(res.spec.session_id for res in results)
    return child_records.Ranking(order=order, winner=order[0] if order else "")


def _calls(*uses: tuple[str, dict[str, Any]]) -> ProviderResponse:
    """A model turn that makes these tool calls, in order."""
    blocks = [
        {"type": "tool_use", "id": f"call-{name}-{i}", "name": name, "input": args}
        for i, (name, args) in enumerate(uses)
    ]
    return ProviderResponse(
        text="",
        tool_uses=tuple({k: b[k] for k in ("id", "name", "input")} for b in blocks),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={"content": blocks},
    )


READ = ("read_file", {"path": "README.md"})


class _Script:
    """A provider that answers each call with the next scripted turn, after its step."""

    def __init__(self, *turns: tuple[Callable[[], None] | None, ProviderResponse]) -> None:
        self.turns = list(turns)
        self.seen: list[str] = []

    def call(self, *args: Any, **kwargs: Any) -> ProviderResponse:
        self.seen.append(json.dumps(kwargs.get("messages") or args[1]))
        step, response = self.turns[len(self.seen) - 1]
        if step is not None:
            step()
        return response


def _config() -> Config:
    return Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {"default": {"model": "p/base"}},
            "harness": {"parallel": {"tool": True}, "verify": {"when": "never"}},
            "review": {"when": "never"},
        }
    )


def _harness(
    repo: pathlib.Path,
    session_dir: pathlib.Path,
    provider: Any,
    children: _Children,
    *,
    max_turns: int,
    recoverer: Any = None,
) -> loop.Harness:
    session_dir.mkdir(exist_ok=True)
    cfg = _config()
    events = event_log.EventSink(session_dir / "logs.jsonl")
    return loop.Harness(
        chain=run_chain.RunChain(
            repo, ref="refs/agent6/parent", fallback_parent=_git(repo, "rev-parse", "HEAD")
        ),
        config=cfg,
        provider=provider,
        dispatcher=dispatch.ToolDispatcher(
            root=repo, config=cfg, isolation="none", session_dir=session_dir, events=events
        ),
        logger=lambda _m: None,
        events=events,
        budget=agent6_budget.BudgetTracker(
            max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1
        ),
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(
            child_spawner=children, child_ranker=_first_ranked, child_recoverer=recoverer
        ),
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=max_turns,
    )


def _events(session_dir: pathlib.Path, event_type: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    return [row for row in rows if row["type"] == event_type]


def test_two_tasks_started_in_one_turn_run_while_the_parent_works_and_land_in_order(
    tmp_path: pathlib.Path,
) -> None:
    """The `parallel` tool returns at once; ends are told as they come; landing keeps the order.

    The second task ends first: its end is told at once, and its work waits for the first's.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    one, two = "run-p1-l1", "run-p2-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}), ("parallel", {"task": "write b"}))),
        (None, _calls(READ)),
        (lambda: children.end(two, path="b.txt", report="wrote b"), _calls(READ)),
        (lambda: children.end(one, path="a.txt", report="wrote a"), _calls(READ)),
        (None, _calls(READ)),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=5)

    wf.run("start")

    # Turn 2 read both handles as tool results while both children ran.
    assert children.tasks == {one: "write a", two: "write b"}
    for sid in (one, two):
        assert f"[parallel] started child {sid}." in provider.seen[1]
    assert "[parallel] child" not in provider.seen[2]
    # Turn 4 read the second task's end, and no landing: the first task still ran.
    assert f"[parallel] child {two} finished.\\nsummary: wrote b" in provider.seen[3]
    assert "[parallel] group" not in provider.seen[3]
    # Turn 5 read the first task's end and both landings, the first task's first.
    last = provider.seen[4]
    assert f"[parallel] child {one} finished.\\nsummary: wrote a" in last
    first_landing = last.index("[parallel] group run-p1 complete (1 child(ren))")
    assert first_landing < last.index("[parallel] group run-p2 complete (1 child(ren))")
    assert (repo / "a.txt").exists() and (repo / "b.txt").exists()
    records = child_dispatch.read_records(tmp_path / "run")
    assert [(r.identity, r.state, r.delivered, r.source) for r in records] == [
        ("run-p1", "complete", True, "model"),
        ("run-p2", "complete", True, "model"),
    ]
    first, second = (record.children[0].join for record in records)
    assert first is not None and second is not None
    assert (first.status, second.status) == ("joined", "joined")
    # The second task merged onto the tip the first left, though it ended a turn earlier.
    assert second.target_before == first.target_after == first.sha
    assert [row["group"] for row in _events(tmp_path / "run", "parallel.joined")] == [
        "run-p1",
        "run-p2",
    ]


def test_an_end_stops_the_children_still_running_and_a_resume_lands_their_work(
    tmp_path: pathlib.Path,
) -> None:
    """Children end with the execution that started them; their commits land on the next."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    child = "run-p1-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (None, _calls(READ)),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=2)

    result = wf.run("start")

    assert result.reason == "max_turns"
    assert children.stopped == [child]
    assert result.summary.endswith(
        "; 1 child(ren) ended with the run, their work lands when it resumes"
    )
    (record,) = child_dispatch.read_records(tmp_path / "run")
    # Charged, not landed: the chain holds none of its work.
    assert (record.state, record.delivered, record.children[0].settled) == (
        "settling",
        False,
        True,
    )
    assert not (repo / f"{child}.txt").exists()

    def recoverer(group: str, ids: Sequence[str]) -> list[child_records.ChildResult]:
        assert (group, list(ids)) == ("run-p1", [child])
        return [children.results[child].result()]

    resumed = _Script((None, _calls(READ)), (None, _calls(READ)))
    _harness(
        repo, tmp_path / "run", resumed, _Children(repo, "run"), max_turns=2, recoverer=recoverer
    ).resume()

    assert f"[parallel] child {child} finished (stopped by its parent)." in resumed.seen[0]
    assert "[parallel] group run-p1 complete (1 child(ren))" in resumed.seen[0]
    assert (repo / f"{child}.txt").exists()
    (record,) = child_dispatch.read_records(tmp_path / "run")
    assert (record.state, record.delivered, record.children[0].reported) == ("complete", True, True)


def _dispatcher(
    repo: pathlib.Path, children: _Children, *, per_turn: bool = True
) -> parallel_dispatch.ParallelDispatcher:
    session_dir = repo.parent / "root"
    session_dir.mkdir(exist_ok=True)
    return parallel_dispatch.ParallelDispatcher(
        chain=run_chain.RunChain(
            repo,
            ref="refs/agent6/root",
            fallback_parent=_git(repo, "rev-parse", "HEAD"),
            per_turn=per_turn,
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
        ranker=_first_ranked,
    )


def _start(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    *segments: tuple[str, str],
    state: loop_state.LoopState | None = None,
) -> child_records.Started:
    return dispatcher.dispatch(
        loop_conversation.Conversation(),
        1,
        state or loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0),
        [directive.Segment(spec=spec, task=task) for spec, task in segments],
    )


def test_the_cap_counts_the_children_still_running(tmp_path: pathlib.Path) -> None:
    """`max_children` bounds what runs at once, across dispatches; an ended child frees a place."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)

    assert len(_start(dispatcher, ("3", "attempt it"), state=state).children) == 3
    refused = _start(dispatcher, ("", "one more"), ("", "and another"), state=state)

    assert refused.children == ()
    assert refused.message == (
        "[parallel] this asks for 2 children with 3 running, over"
        " [harness.parallel].max_children = 4; nothing dispatched. Ask for fewer, wait for one"
        " to end, stop one, or raise [harness.parallel].max_children."
    )
    assert len(children.results) == 3
    assert len(child_dispatch.read_records(repo.parent / "root")) == 1
    children.end("root-p1-l2")
    assert _start(dispatcher, ("", "one more"), ("", "and another"), state=state).children == (
        "root-p2-l1",
        "root-p3-l1",
    )


def test_a_tree_that_cannot_be_committed_lands_nothing_and_keeps_the_branch(
    tmp_path: pathlib.Path,
) -> None:
    """With `git.commit_per_turn` off, uncommitted work in the parent's tree holds a merge back."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children, per_turn=False)
    conversation = loop_conversation.Conversation()
    started = _start(dispatcher, ("", "write a"))
    (repo / "README.md").write_text("the parent's own edit\n", encoding="utf-8")
    children.end("root-p1-l1", path="a.txt")

    assert dispatcher.settle(conversation, 2)

    (join,) = dispatcher.joins(started.groups[0])
    assert (join.status, join.branch) == ("failed", "agent6/root-p1-l1")
    assert "uncommitted work that could not be committed" in join.detail
    assert not (repo / "a.txt").exists()
    assert _git(repo, "rev-parse", "--verify", "agent6/root-p1-l1")


def test_a_parents_uncommitted_work_is_committed_before_a_child_lands(
    tmp_path: pathlib.Path,
) -> None:
    """Landing is on a clean tree: the parent's edit becomes a checkpoint, then the merge."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    started = _start(dispatcher, ("", "write a"))
    (repo / "notes.txt").write_text("the parent's own edit\n", encoding="utf-8")
    children.end("root-p1-l1", path="a.txt")

    assert dispatcher.settle(loop_conversation.Conversation(), 2)

    (join,) = dispatcher.joins(started.groups[0])
    assert join.status == "joined"
    subjects = _git(repo, "log", "--format=%s", "refs/agent6/root").splitlines()
    assert subjects[:2] == [
        "merge agent6/root-p1-l1",
        "checkpoint before landing children (turn 2)",
    ]
    assert _git(repo, "show", "refs/agent6/root:notes.txt") == "the parent's own edit"


def test_attempts_are_told_as_each_ends_and_land_once_all_have(tmp_path: pathlib.Path) -> None:
    """A task's attempts are ranked together: one still running holds the task's landing."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    conversation = loop_conversation.Conversation()
    started = _start(dispatcher, ("2", "rewrite it"))
    children.end("root-p1-l2", path="x.txt", text="second\n", report="took the short way")

    assert dispatcher.settle(conversation, 2)
    assert not dispatcher.settle(conversation, 3), "an end is told once"
    assert dispatcher.joins(started.groups[0]) == []
    children.end("root-p1-l1", path="x.txt", text="first\n")
    assert dispatcher.settle(conversation, 4)

    assert [(j.session_id, j.status) for j in dispatcher.joins(started.groups[0])] == [
        ("root-p1-l1", "joined"),
        ("root-p1-l2", "ranked"),
    ]
    assert (repo / "x.txt").read_text(encoding="utf-8") == "first\n"
    said = json.dumps(conversation.to_wire())
    assert said.count("[parallel] child root-p1-l2 finished.") == 1
    assert dispatcher.open == {}


def test_close_stops_what_runs_charges_each_child_once_and_forgets_it(
    tmp_path: pathlib.Path,
) -> None:
    """The execution's end settles what its children used; a second close has nothing left."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    dispatcher.budget = agent6_budget.BudgetTracker(
        max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1
    )
    _start(dispatcher, ("", "write a"), ("", "write b"))
    children.end("root-p2-l1", path="b.txt")

    assert dispatcher.close() == ("root-p1-l1", "root-p2-l1")

    assert children.stopped == ["root-p1-l1"]
    assert dispatcher.budget.estimate_usd()[0] == pytest.approx(0.5)
    assert dispatcher.close() == ()
    assert dispatcher.budget.estimate_usd()[0] == pytest.approx(0.5)
    assert [r.state for r in child_dispatch.read_records(repo.parent / "root")] == ["settling"] * 2


def test_the_model_reads_a_running_child_waits_for_it_and_stops_another(
    tmp_path: pathlib.Path,
) -> None:
    """`read_background` and `stop_background` take a child's session id beside a `bg<N>`."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    one, two = "run-p1-l1", "run-p2-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a\nand test it"}), ("parallel", {"task": "b"}))),
        (None, _calls(("read_background", {"id": one, "wait_s": 0}))),
        (None, _calls(("stop_background", {"id": two}))),
        (
            lambda: children.end(one, path="a.txt", report="wrote a"),
            _calls(("read_background", {"wait_s": 30})),
        ),
        (None, _calls(READ)),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=5)

    wf.run("start")

    def answers(seen: str) -> list[dict[str, Any]]:
        return [
            json.loads(block["content"])
            for message in json.loads(seen)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]

    # The read of a running child: both rosters, its line, and no transcript yet.
    read = answers(provider.seen[2])[-1]
    assert read["children"] == [f"[{one}] running: write a", f"[{two}] running: b"]
    assert read["output"] == f"[{one}] running: write a\n\n(no transcript yet)"
    # The stop answers at once; the child ends in the background and lands like any other.
    assert children.stopped == [two]
    stopped = answers(provider.seen[3])[-1]
    assert stopped["children"][1] in (f"[{two}] stopping: b", f"[{two}] ended, not landed: b")
    assert f"[parallel] child {two} finished (stopped by its parent)." in provider.seen[3]
    # The wait for the next child to end returned with the first task's child ended.
    waited = answers(provider.seen[4])[-1]
    assert waited["children"][0] == f"[{one}] ended, not landed: write a"
    last = provider.seen[4]
    assert last.index("[parallel] group run-p1 complete") < last.index(
        "[parallel] group run-p2 complete"
    )
    assert f"{two} (agent6/{two}): joined at" in last and "(stopped by its parent)" in last
    roster = wf.parallel.roster()
    assert [(line.id, line.running) for line in roster] == [(one, False), (two, False)]
    assert all(line.state.startswith("joined at ") for line in roster)


def test_wait_for_returns_when_the_child_ends_the_wait_runs_out_or_a_stop_cuts_it(
    tmp_path: pathlib.Path,
) -> None:
    """A read's wait ends with the child, at `wait_s`, or when the operator wants out."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    _start(dispatcher, ("", "write a"))

    started = time.monotonic()
    dispatcher.wait_for("root-p1-l1", 0.2, lambda: False)
    assert 0.15 < time.monotonic() - started < 2.0
    started = time.monotonic()
    dispatcher.wait_for("", 30.0, lambda: True)
    assert time.monotonic() - started < 2.0
    timer = threading.Timer(0.1, lambda: children.end("root-p1-l1"))
    timer.start()
    started = time.monotonic()
    dispatcher.wait_for("", 30.0, lambda: False)
    timer.join()
    assert time.monotonic() - started < 5.0
    assert [(line.state, line.running) for line in dispatcher.roster()] == [
        ("ended, not landed", False)
    ]
    dispatcher.stop("root-p1-l1")
    assert children.stopped == [], "an ended child is left as it is"
