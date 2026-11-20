# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Children run in the background of their parent and merge at its turn boundaries.

A scripted provider drives the parent's turns and a fake spawner's children end when the
script says, so the order of ends, notices and merges is the test's own.
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
from agent6.graph import curator as graph_curator
from agent6.harness import (
    child_work,
    loop,
    loop_conversation,
    loop_state,
    operator_bridge,
    parallel_dispatch,
    parallel_recovery,
    provider_call,
    run_chain,
)
from agent6.providers import ProviderResponse
from agent6.sessions import child_dispatch, event_log, ipc
from agent6.sessions import layout as sessions_layout
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
                commits=1 if tip else 0,
                usage=agent6_budget.UsageSettlement(usd=0.25, usd_partial=False, tokens=0),
                report=report,
                stopped_by=stopped_by,
            )
        )

    def stop(self, sid: str, *, path: str | None = None) -> None:
        """Stand in for the watch: a stopped child ends with what it had committed."""
        self.stopped.append(sid)
        self.end(
            sid, path=f"{sid}.txt" if path is None else path, stopped_by="stopped by its parent"
        )


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


def _config(judges: Sequence[str] = ()) -> Config:
    return Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {"default": {"model": "p/base"}},
            "harness": {
                "parallel": {"tool": True, "judges": list(judges)},
                "verify": {"when": "never"},
            },
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
    curator: graph_curator.GraphCurator | None = None,
    judges: Sequence[str] = (),
) -> loop.Harness:
    session_dir.mkdir(exist_ok=True)
    cfg = _config(judges)
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
        curator=curator,
        budget=agent6_budget.BudgetTracker(
            max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1
        ),
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(child_spawner=children, child_recoverer=recoverer),
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=max_turns,
    )


def _events(session_dir: pathlib.Path, event_type: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    return [row for row in rows if row["type"] == event_type]


def test_a_finished_task_merges_at_the_next_boundary_while_an_earlier_one_still_runs(
    tmp_path: pathlib.Path,
) -> None:
    """The `parallel` tool returns at once; each task merges at the boundary after its child ends.

    A finished task waited for every task started before it: a slow child held the work of
    every later one.
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
    # Turn 4 read the second task's end and its merge, while the first task still ran.
    assert f"[parallel] child {two} finished.\\nsummary: wrote b" in provider.seen[3]
    assert "[parallel] group run-p2 complete (1 child(ren))" in provider.seen[3]
    assert "[parallel] group run-p1" not in provider.seen[3]
    # Turn 5 read the first task's end and its merge.
    last = provider.seen[4]
    assert f"[parallel] child {one} finished.\\nsummary: wrote a" in last
    assert "[parallel] group run-p1 complete (1 child(ren))" in last
    assert (repo / "a.txt").exists() and (repo / "b.txt").exists()
    records = child_dispatch.read_records(tmp_path / "run")
    assert [(r.identity, r.state, r.delivered, r.source) for r in records] == [
        ("run-p1", "complete", True, "model"),
        ("run-p2", "complete", True, "model"),
    ]
    first, second = (record.children[0].join for record in records)
    assert first is not None and second is not None
    assert (first.status, second.status) == ("joined", "joined")
    # The first task merged onto the tip the second left: it merged a boundary later.
    assert first.target_before == second.target_after == second.sha
    assert [row["group"] for row in _events(tmp_path / "run", "parallel.joined")] == [
        "run-p2",
        "run-p1",
    ]
    # A merge's rows carry the turn they are written at: the views read the turn off any row.
    turns = [row["turn"] for row in _events(tmp_path / "run", "budget.turn")]
    assert turns == sorted(turns) and turns[-1] == 5


def test_tasks_that_end_before_one_boundary_merge_in_the_order_written(
    tmp_path: pathlib.Path,
) -> None:
    """Several finished tasks merge at one boundary oldest first; the later gets a conflict."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    conversation = loop_conversation.Conversation()
    started = _start(dispatcher, ("", "slow"), ("", "write x"), ("", "write x too"))
    children.end("root-p3-l1", path="x.txt", text="third\n")
    children.end("root-p2-l1", path="x.txt", text="second\n")

    assert dispatcher.settle(conversation, 2)

    assert dispatcher.joins(started.groups[0]) == []
    (second,) = dispatcher.joins(started.groups[1])
    (third,) = dispatcher.joins(started.groups[2])
    assert (second.status, third.status) == ("joined", "conflict")
    assert third.detail == "merge conflict in x.txt"
    assert (repo / "x.txt").read_text(encoding="utf-8") == "second\n"
    assert list(dispatcher.open) == ["root-p1"]


def test_an_end_stops_the_children_still_running_and_a_resume_merges_nothing_of_theirs(
    tmp_path: pathlib.Path,
) -> None:
    """Children end with the execution that started them; a stopped child never merges.

    The resume merged the commits of a child the run's end had stopped before its first call.
    """
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
        "; 1 child(ren) stopped with the run, their branches hold their work"
    )
    (record,) = child_dispatch.read_records(tmp_path / "run")
    # Charged, not merged: the chain holds none of its work.
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

    # The resumed model reads what it resumed into: the child, its branch, its commits.
    assert (
        f"[parallel] child {child} stopped by its parent, not merged;"
        f" agent6/{child} holds 1 commit(s); conversation at last call: unknown; last ran: unknown."
        in resumed.seen[0]
    )
    assert (
        "[parallel] group run-p1 complete (1 child(ren)):\\n"
        f"  - {child} (agent6/{child}): stopped by its parent, not merged; the branch holds"
        " 1 commit(s); conversation at last call: unknown; last ran: unknown."
        "\\nEvery child stopped: nothing merged." in resumed.seen[0]
    )
    assert not (repo / f"{child}.txt").exists()
    assert _git(repo, "rev-parse", "--verify", f"agent6/{child}")
    (record,) = child_dispatch.read_records(tmp_path / "run")
    assert (record.state, record.delivered, record.children[0].reported) == ("complete", True, True)
    join = record.children[0].join
    assert join is not None
    assert (join.status, join.detail, join.commits, join.branch, join.sha) == (
        "stopped",
        "stopped by its parent",
        1,
        f"agent6/{child}",
        "",
    )
    assert join.target_before == join.target_after


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


def test_a_tree_that_cannot_be_committed_merges_nothing_and_keeps_the_branch(
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


def test_a_parents_uncommitted_work_is_committed_before_a_child_merges(
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
        "checkpoint before merging children (turn 2)",
    ]
    assert _git(repo, "show", "refs/agent6/root:notes.txt") == "the parent's own edit"


def test_attempts_are_told_as_each_ends_and_merge_once_all_have(tmp_path: pathlib.Path) -> None:
    """A task's attempts are ranked together: one still running holds the task's merge."""
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


def test_a_stopped_attempt_is_no_candidate_and_the_others_are_ranked(
    tmp_path: pathlib.Path,
) -> None:
    """Among attempts a stopped child is not ranked: the first of the others merges."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    conversation = loop_conversation.Conversation()
    started = _start(dispatcher, ("3", "rewrite it"))
    child_work.stop(dispatcher, "root-p1-l1")
    children.end("root-p1-l2", path="x.txt", text="second\n")
    children.end("root-p1-l3", path="x.txt", text="third\n")

    assert dispatcher.settle(conversation, 2)

    joins = dispatcher.joins(started.groups[0])
    assert [(j.session_id, j.status, j.detail) for j in joins] == [
        ("root-p1-l1", "stopped", "stopped by its parent"),
        ("root-p1-l2", "joined", ""),
        ("root-p1-l3", "ranked", "2 of 2"),
    ]
    assert (repo / "x.txt").read_text(encoding="utf-8") == "second\n"
    assert not (repo / "root-p1-l1.txt").exists()
    (record,) = child_dispatch.read_records(repo.parent / "root")
    assert record.ranking is not None
    assert record.ranking.order == ("root-p1-l2", "root-p1-l3")
    said = json.dumps(conversation.to_wire())
    assert "Review what merged and continue." in said


def test_a_task_whose_every_child_stopped_merges_nothing_and_says_so(
    tmp_path: pathlib.Path,
) -> None:
    """Every attempt stopped: no ranking, no merge, and one notice that nothing merged."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    conversation = loop_conversation.Conversation()
    started = _start(dispatcher, ("2", "rewrite it"))
    child_work.stop(dispatcher, "root-p1-l1")
    children.stop("root-p1-l2", path="")
    before = _git(repo, "rev-parse", "HEAD")

    assert dispatcher.settle(conversation, 2)

    assert [(j.status, j.commits) for j in dispatcher.joins(started.groups[0])] == [
        ("stopped", 1),
        ("stopped", 0),
    ]
    (record,) = child_dispatch.read_records(repo.parent / "root")
    assert record.ranking is not None and record.ranking.order == ()
    assert record.message == (
        "[parallel] group root-p1 complete (2 child(ren)):\n"
        "  - root-p1-l1 (agent6/root-p1-l1): stopped by its parent, not merged; the branch"
        " holds 1 commit(s); conversation at last call: unknown; last ran: unknown.\n"
        "  - root-p1-l2 (agent6/root-p1-l2): stopped by its parent, not merged; the branch"
        " holds 0 commit(s); conversation at last call: unknown; last ran: unknown.\n"
        "Every child stopped: nothing merged."
    )
    assert _git(repo, "rev-parse", "HEAD") == before and dispatcher.open == {}
    said = json.dumps(conversation.to_wire())
    assert (
        "[parallel] child root-p1-l2 stopped by its parent, not merged; agent6/root-p1-l2 holds"
        " 0 commit(s); conversation at last call: unknown; last ran: unknown." in said
    )


def test_a_resume_merges_the_task_of_a_child_that_ended_by_itself(tmp_path: pathlib.Path) -> None:
    """A finished child merges at resume if its execution ended before the boundary."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "root")
    dispatcher = _dispatcher(repo, children)
    _start(dispatcher, ("", "write a"), ("", "write b"))
    children.end("root-p1-l1", path="a.txt", report="wrote a")

    # The execution ends before the boundary: the second task's child is stopped with it.
    assert dispatcher.close(3) == (("root-p2-l1",), ("root-p1-l1",))

    ended = children.results

    def recoverer(group: str, children: Sequence[str]) -> list[child_records.ChildResult]:
        del group
        return [ended[sid].result() for sid in children]

    resumed = _dispatcher(repo, _Children(repo, "root"))
    resumed.child_recoverer = recoverer
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="t", execution_id="exec-2", tool_calls=0)

    parallel_recovery.reconcile(resumed, state, conversation, turn=3)

    first, second = child_dispatch.read_records(repo.parent / "root")
    assert first.children[0].join is not None and second.children[0].join is not None
    assert (first.children[0].join.status, second.children[0].join.status) == ("joined", "stopped")
    assert (repo / "a.txt").exists() and not (repo / "root-p2-l1.txt").exists()
    assert (first.state, second.state, first.delivered, second.delivered) == (
        "complete",
        "complete",
        True,
        True,
    )
    assert resumed.open == {}


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

    # The first task's child was stopped; the second's had ended and merges at a resume.
    assert dispatcher.close(3) == (("root-p1-l1",), ("root-p2-l1",))

    assert children.stopped == ["root-p1-l1"]
    assert dispatcher.budget.estimate_usd()[0] == pytest.approx(0.5)
    assert dispatcher.close(3) == ((), ())
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
    # The stop answers at once; the child ends in the background and never merges.
    assert children.stopped == [two]
    stopped = answers(provider.seen[3])[-1]
    words = (
        f"stopped by its parent, not merged; agent6/{two} holds 1 commit(s);"
        " conversation at last call: unknown; last ran: unknown"
    )
    assert stopped["children"][1] in (f"[{two}] stopping: b", f"[{two}] {words}: b")
    assert f"[parallel] child {two} {words}." in provider.seen[3]
    assert "Every child stopped: nothing merged." in provider.seen[3]
    # The wait for the next child to end returned with the first task's child ended.
    waited = answers(provider.seen[4])[-1]
    assert waited["children"] == [
        f"[{one}] ended, not merged: write a",
        f"[{two}] {words}: b",
    ]
    assert "[parallel] group run-p1 complete" in provider.seen[4]
    assert (repo / "a.txt").exists() and not (repo / f"{two}.txt").exists()
    assert _git(repo, "rev-parse", "--verify", f"agent6/{two}")
    roster = child_work.roster(wf.parallel)
    assert [(line.id, line.running) for line in roster] == [(one, False), (two, False)]
    assert roster[0].state.startswith("joined at ") and roster[1].state == words
    (failed,) = [[], *(_events(tmp_path / "run", "parallel.failed"))][-1:]
    assert failed == [], "a stopped child is no failed join"
    rows = _events(tmp_path / "run", "parallel.joined")
    assert rows[0]["children"][0]["status"] == "stopped" and rows[0]["children"][0]["commits"] == 1


def test_a_stopped_child_does_not_hold_the_end(tmp_path: pathlib.Path) -> None:
    """A finish in the turn that stopped the last child goes through: its task will not merge."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    child = "run-p1-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (
            None,
            _calls(("stop_background", {"id": child}), ("finish_session", {"summary": "done"})),
        ),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=4)

    result = wf.run("start")

    assert (result.reason, result.turns) == ("finish_session", 2)
    assert _events(tmp_path / "run", "finish.refused.children") == []
    assert not (repo / f"{child}.txt").exists()


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
    child_work.wait_for(dispatcher, "root-p1-l1", 0.2, lambda: False)
    assert 0.15 < time.monotonic() - started < 2.0
    started = time.monotonic()
    child_work.wait_for(dispatcher, "", 30.0, lambda: True)
    assert time.monotonic() - started < 2.0
    timer = threading.Timer(0.1, lambda: children.end("root-p1-l1"))
    timer.start()
    started = time.monotonic()
    child_work.wait_for(dispatcher, "", 30.0, lambda: False)
    timer.join()
    assert time.monotonic() - started < 5.0
    assert [(line.state, line.running) for line in child_work.roster(dispatcher)] == [
        ("ended, not merged", False)
    ]
    child_work.stop(dispatcher, "root-p1-l1")
    assert children.stopped == [], "an ended child is left as it is"


@pytest.mark.parametrize("now", [False, True])
@pytest.mark.parametrize("child_id", ["run-p1-l1", ""])
def test_a_pending_steer_ends_a_child_wait_without_consuming_it(
    tmp_path: pathlib.Path, now: bool, child_id: str
) -> None:
    """Plain and immediate steers end child waits without stopping the child or losing the steer."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run"
    children = _Children(repo, "run")
    started = 0.0

    def steer() -> None:
        nonlocal started
        ipc.submit_steer(session_dir, "inspect the output", now=now)
        started = time.monotonic()

    def check_wait() -> None:
        assert time.monotonic() - started < 2.0, "the pending steer did not end the 4s wait"
        assert ipc.steer_request_pending(session_dir)
        assert ipc.steer_interrupt_pending(session_dir) is now
        assert children.running() == ["run-p1-l1"] and children.stopped == []

    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (steer, _calls(("read_background", {"id": child_id, "wait_s": 4}))),
        (check_wait, _calls(READ)),
    )
    wf = _harness(repo, session_dir, provider, children, max_turns=3)

    wf.run("start")

    assert "[run-p1-l1] running: write a" in provider.seen[2]
    assert ipc.read_steer_answer(session_dir) == "inspect the output"


def _say(text: str) -> ProviderResponse:
    """A model turn of prose, with no tool call."""
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={"content": [{"type": "text", "text": text}]},
    )


def test_finish_session_is_refused_until_every_task_has_merged(tmp_path: pathlib.Path) -> None:
    """A finish while a child runs comes back naming the child and what to do with it."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    child = "run-p1-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (None, _calls(("finish_session", {"summary": "done"}))),
        (lambda: children.end(child, path="a.txt"), _calls(("read_background", {"wait_s": 30}))),
        (None, _calls(("finish_session", {"summary": "done, with a.txt"}))),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=6)

    result = wf.run("start")

    assert (result.reason, result.turns) == ("finish_session", 4)
    refusal = (
        "[harness] finish_session refused: 1 child(ren) have pending work:\\n"
        f"- {child}: running\\n"
        "read_background with a child's id reads it, read_background with wait_s and no id"
        " waits for the next to end, and stop_background stops one, whose work then does not"
        " merge. Tasks complete at a turn boundary after their children end, and attempts"
        " once their judges have ranked them."
    )
    assert refusal in provider.seen[2]
    assert "[parallel] group run-p1 complete" in provider.seen[3]
    assert (repo / "a.txt").exists() and children.stopped == []
    (refused,) = _events(tmp_path / "run", "finish.refused.children")
    assert (refused["turn"], refused["children"], refused["trigger"]) == (
        2,
        [child],
        "finish_session",
    )
    assert result.summary == "done, with a.txt"


def test_a_finish_in_the_turn_a_child_ended_waits_for_its_work_to_merge(
    tmp_path: pathlib.Path,
) -> None:
    """An ended child's work merges at the boundary: the finish before it comes back once."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    child = "run-p1-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (
            lambda: children.end(child, path="a.txt"),
            _calls(("read_background", {"id": child}), ("finish_session", {"summary": "early"})),
        ),
        (None, _calls(("finish_session", {"summary": "after the merge"}))),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=5)

    result = wf.run("start")

    assert (result.reason, result.summary) == ("finish_session", "after the merge")
    assert f"- {child}: ended, not merged" in provider.seen[2]
    assert "[parallel] group run-p1 complete" in provider.seen[2]


def test_a_prose_only_turn_does_not_end_a_run_whose_children_run(tmp_path: pathlib.Path) -> None:
    """A turn with no tool call is an end too: it is held like `finish_session`."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    child = "run-p1-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (None, _calls(READ)),
        (None, _calls(READ)),
        (None, _say("Waiting for the child to finish.")),
        (lambda: children.end(child, path="a.txt"), _calls(("read_background", {"wait_s": 30}))),
        (None, _calls(("finish_session", {"summary": "done"}))),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=8)

    result = wf.run("start")

    assert (result.reason, result.turns) == ("finish_session", 6)
    assert "[harness] the run cannot end yet: 1 child(ren) have pending work:" in provider.seen[4]
    assert f"- {child}: running" in provider.seen[4]
    (refused,) = _events(tmp_path / "run", "finish.refused.children")
    assert (refused["turn"], refused["trigger"]) == (4, "silent_finish")


def test_a_dispatched_task_joins_the_task_graph_when_its_work_merges(
    tmp_path: pathlib.Path,
) -> None:
    """While a child works a task the graph holds no open node for it.

    An open node read as the parent's own next work: the focus banner named the child's task,
    and a second task dispatched in the same turn hung under the first.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    layout = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="run")
    curator = graph_curator.GraphCurator(layout)
    children = _Children(repo, "run")

    def end_both() -> None:
        children.end("run-p1-l1", path="a.txt")
        children.end("run-p2-l1", path="b.txt")

    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}), ("parallel", {"task": "write b"}))),
        (None, _calls(READ)),
        (end_both, _calls(READ)),
        (None, _calls(("finish_session", {"summary": "done"}))),
    )
    wf = _harness(repo, layout.session_dir, provider, children, max_turns=5, curator=curator)

    result = wf.run("start")

    assert result.reason == "finish_session"
    assert not any("[harness focus] Current task" in seen for seen in provider.seen)
    nodes = curator.nodes()
    (root,) = [node for node in nodes.values() if node.parent_id is None]
    tasks = {node.title: node for node in nodes.values() if node.parent_id is not None}
    assert sorted(tasks) == ["write a", "write b"]
    assert all(
        (node.status, node.parent_id, bool(node.commit_sha)) == ("passed", root.id, True)
        for node in tasks.values()
    )
    records = child_dispatch.read_records(layout.session_dir)
    assert [record.node_id for record in records] == [tasks["write a"].id, tasks["write b"].id]
    assert [record.parent_node for record in records] == [root.id, root.id]


def test_a_merge_at_the_runs_last_boundary_is_in_the_snapshot_a_resume_reads(
    tmp_path: pathlib.Path,
) -> None:
    """A notice told at a boundary is in the resume snapshot, as its record says.

    The record said delivered while the snapshot predated the notice: a run that ended at
    that boundary resumed with a model that never heard its child's work had merged.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    child = "run-p1-l1"
    provider = _Script(
        (None, _calls(("parallel", {"task": "write a"}))),
        (lambda: children.end(child, path="a.txt", report="wrote a"), _calls(READ)),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=2)

    result = wf.run("start")

    assert result.reason == "max_turns"
    (record,) = child_dispatch.read_records(tmp_path / "run")
    assert (record.state, record.delivered, record.children[0].reported) == ("complete", True, True)
    resumed = _Script((None, _calls(("finish_session", {"summary": "done"}))))
    _harness(repo, tmp_path / "run", resumed, _Children(repo, "run"), max_turns=2).resume()
    assert f"[parallel] child {child} finished.\\nsummary: wrote a" in resumed.seen[0]
    assert resumed.seen[0].count("[parallel] group run-p1 complete") == 1


def test_an_end_is_held_while_attempts_are_ranked_and_a_wait_waits_for_their_judge(
    tmp_path: pathlib.Path,
) -> None:
    """The judges run while the parent takes turns; the task merges the boundary after they end.

    The parent's turn waited inside the merge for one model call that ranked the attempts.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    children = _Children(repo, "run")
    one, two, judge = "run-p1-l1", "run-p1-l2", "run-rp1-l1"

    def end_attempts() -> None:
        children.end(one, path="x.txt", text="one\n")
        children.end(two, path="x.txt", text="two\n")

    def end_judge_soon() -> None:
        threading.Timer(0.3, lambda: children.end(judge)).start()

    provider = _Script(
        (None, _calls(("parallel", {"task": "rewrite it", "spec": "2"}))),
        (end_attempts, _calls(READ)),
        (None, _calls(("finish_session", {"summary": "too early"}))),
        (end_judge_soon, _calls(("read_background", {"wait_s": 30}))),
        (None, _calls(("finish_session", {"summary": "ranked and merged"}))),
    )
    wf = _harness(repo, tmp_path / "run", provider, children, max_turns=6, judges=("default",))

    started = time.monotonic()
    result = wf.run("start")

    assert (result.reason, result.turns, result.summary) == (
        "finish_session",
        5,
        "ranked and merged",
    )
    assert time.monotonic() - started < 20.0, "the wait ended with the judge, not at its 30 seconds"
    # The judge is a reviewer child the run started once both attempts had ended.
    assert children.tasks[judge] == "Rank 2 attempts"
    assert "[parallel] group" not in provider.seen[2]
    # The finish in the turn after came back: both attempts were being ranked.
    refusal = provider.seen[3]
    assert "[harness] finish_session refused: 2 child(ren) have pending work:" in refusal
    assert f"- {one}: being ranked" in refusal and f"- {two}: being ranked" in refusal
    assert "and attempts once their judges have ranked them." in refusal
    # The wait for the next to end waited for the judge; its end ranked and merged the task.
    last = provider.seen[4]
    assert "[parallel] group run-p1 complete (2 child(ren))" in last
    assert (
        "Ranked by verify result, then cost: no judge gave a usable order (default: it left"
        " no verdict)." in last
    )
    assert (repo / "x.txt").read_text(encoding="utf-8") == "one\n"
    states = {
        record.identity: record.state for record in child_dispatch.read_records(tmp_path / "run")
    }
    assert states == {"run-p1": "complete", "run-rp1": "complete"}
