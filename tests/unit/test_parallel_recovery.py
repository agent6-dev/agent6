# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An interrupted dispatch is finished or diagnosed by the next execution, exactly once.

Real temp repos and the real record files; crashes are injected at each boundary the record
marks, and a second harness resumes from the snapshot the first wrote.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import child_records, paths
from agent6.app import child_watch, children
from agent6.config import Config
from agent6.harness import (
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
from agent6.sessions import budget_events, child_dispatch, event_log, journal
from agent6.tools import results as tools_results
from tests import child_env


class _Crash(BaseException):
    """The process dying: nothing in the loop catches it."""


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


def _head(repo: pathlib.Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _make_branch(repo: pathlib.Path, branch: str, base: str, fname: str, wt: pathlib.Path) -> str:
    _git(repo, "worktree", "add", "-b", branch, str(wt), base)
    (wt / fname).write_text(f"{fname}\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", f"{branch} work")
    _git(repo, "worktree", "remove", "--force", str(wt))
    return _git(repo, "rev-parse", branch)


def _resp_tool(name: str, args: dict[str, Any], tu_id: str) -> ProviderResponse:
    return ProviderResponse(
        text="",
        tool_uses=({"id": tu_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def _provider() -> mock.MagicMock:
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    return provider


class _OneShotSteer:
    def __init__(self, text: str) -> None:
        self.text = text
        self._fired = False

    def requested(self) -> bool:
        return not self._fired

    def prompt(self) -> str:
        return self.text

    def clear(self) -> None:
        self._fired = True


class _Events:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: journal.Event) -> None:
        self.emitted.append((event.TYPE, event.model_dump(mode="json", exclude_unset=True)))

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [f for t, f in self.emitted if t == event_type]


def _silent(_msg: str) -> None:
    return None


def _child_result(
    spec_id: str,
    workdir: pathlib.Path,
    *,
    branch: str = "",
    tip: str = "",
    ok: bool = True,
    error: str = "",
    usd: float = 0.0,
) -> child_records.ChildResult:
    child = int(spec_id.rsplit("-l", 1)[1])
    return child_records.ChildResult(
        spec=child_records.ChildSpec(child, spec_id, workdir, None),
        session_dir=workdir,
        branch=branch or f"agent6/{spec_id}",
        ok=ok,
        error=error,
        imported_ref=branch if tip else "",
        imported_tip=tip,
        has_commits=bool(tip),
        usage=agent6_budget.UsageSettlement(usd=usd, usd_partial=False, tokens=0) if ok else None,
    )


def _build_wf(
    repo: pathlib.Path,
    session_dir: pathlib.Path,
    provider: mock.MagicMock,
    *,
    steer_text: str = "",
    child_spawner: Any = None,
    child_recoverer: Any = None,
    budget: agent6_budget.BudgetTracker | None = None,
) -> loop.Harness:
    session_dir.mkdir(parents=True, exist_ok=True)
    steer = _OneShotSteer(steer_text)
    events: Any = _Events(session_dir / "logs.jsonl")
    disp = mock.MagicMock(execution_id="execution", last_verification=None, verify_withheld=False)
    disp.dispatch.return_value = tools_results.RawResult({"content": "hi\n"})
    disp.command_policy.return_value = "allow"
    return loop.Harness(
        chain=run_chain.RunChain(repo, ref="refs/agent6/parent", fallback_parent=_head(repo)),
        config=Config.model_validate(
            {"harness": {"parallel": {"judges": []}, "verify": {"at_start": True}}}
        ),
        provider=provider,
        dispatcher=disp,
        logger=_silent,
        events=events,
        budget=budget,
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(
            child_spawner=child_spawner,
            child_recoverer=child_recoverer,
            steer_requested=steer.requested,
            steer_prompt=steer.prompt,
            steer_clear=steer.clear,
        ),
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=2,
    )


def _first_call_texts(provider: mock.MagicMock) -> list[str]:
    messages = (
        provider.call.call_args_list[0].kwargs.get("messages")
        or provider.call.call_args_list[0].args[0]
    )
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block["text"])
    return out


def _records(session_dir: pathlib.Path) -> tuple[child_dispatch.DispatchRecord, ...]:
    return child_dispatch.read_records(session_dir)


def _no_children(group: str, children: Sequence[str]) -> list[child_records.ChildResult]:
    del group, children
    return []


def _tracker() -> agent6_budget.BudgetTracker:
    return agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)


def _resume(parent: pathlib.Path, execution: str) -> None:
    event_log.EventSink(parent / "logs.jsonl").emit(
        journal.SessionResume(
            session_id=parent.name,
            user_task="t",
            steer="",
            mode="run",
            turn=1,
            messages=0,
            execution_id=execution,
        )
    )


def _balance(meter: agent6_budget.BudgetTracker, at: int, dollars: int) -> None:
    meter.record_plan_preflight(
        "p",
        agent6_budget.PlanUsage.single(0, 300, 0, observed_ns=at, credits_balance=f"${dollars}"),
    )


def _running_panel(repo: pathlib.Path, parent: pathlib.Path, group: str) -> pathlib.Path:
    child = child_env.create_record(repo, f"{group}-l1", parent=parent.name, mode="reviewer")
    child_dispatch.write_record(
        parent,
        child_dispatch.DispatchRecord(
            identity=group,
            source="panel",
            merge_policy="record_only",
            owner="first",
            turn=1,
            base="",
            state="running",
            children=(child_dispatch.ChildRecord(session_id=child.name, number=1, task="review"),),
        ),
        now=1,
    )
    return child


def _reconcile(
    repo: pathlib.Path,
    parent: pathlib.Path,
    meter: agent6_budget.BudgetTracker,
    *,
    stop_at: str = "",
) -> None:
    cfg = child_env.config()
    dispatcher = child_env.dispatcher(repo, parent, cfg, meter)
    recover = children.build_child_recoverer(cfg, repo, paths.state_dir(repo), mode="run")
    assert recover is not None

    def recover_until(group: str, children: Sequence[str]) -> list[child_records.ChildResult]:
        if group == stop_at:
            raise _Crash
        return recover(group, children)

    dispatcher.child_recoverer = recover_until
    parallel_recovery.reconcile(
        dispatcher,
        loop_state.LoopState(original_task="t", tool_calls=0),
        loop_conversation.Conversation(),
        turn=1,
    )


@pytest.mark.parametrize(
    ("split_baseline", "extra_reading", "new_charge"),
    [(False, None, 0), (False, (30, 7), 1), (False, (15, 12), 2), (True, None, 2)],
)
def test_resume_settles_only_credit_movement_missing_from_the_parent_journal(
    short_repo: pathlib.Path,
    split_baseline: bool,
    extra_reading: tuple[int, int] | None,
    new_charge: int,
) -> None:
    """A watch can journal a decline before the parent settles the child that observed it."""
    repo = short_repo / "repo"
    child_env.init_repo(repo)
    parent = child_env.create_record(repo, "parent")
    parent_sink = event_log.EventSink(parent / "logs.jsonl")
    if split_baseline:
        before = _tracker()
        _balance(before, 10, 10)
        budget_events.emit_budget_update(before, parent_sink.emit)
        _resume(parent, "first")
    child = _running_panel(repo, parent, "parent-r1")
    meter = _tracker()
    child_sink = event_log.EventSink(child / "logs.jsonl")

    def balance(at: int, dollars: int) -> None:
        _balance(meter, at, dollars)
        budget_events.emit_budget_update(meter, child_sink.emit)

    balance(10, 10)
    balance(20, 8)
    first = _tracker()
    first.record_plan_preflight(
        "other", agent6_budget.PlanUsage.single(0, 300, 0, credits_balance="$5")
    )
    if split_baseline:
        _balance(first, 20, 8)
    else:
        first.note_running(child.name, child_watch.running_usage(child))
    budget_events.emit_budget_update(first, parent_sink.emit)
    prior_charge = first.estimate_usd()[0]
    assert prior_charge == (0 if split_baseline else 2)
    if extra_reading is not None:
        balance(*extra_reading)
    direct = children.child_usage(child)
    assert direct is not None
    first.settle_child(child.name, direct)
    _resume(parent, "next")
    fresh = _tracker()
    _reconcile(repo, parent, fresh)
    usage = children.child_usage(parent)
    assert usage is not None
    assert (usage.usd, fresh.available().usd) == (prior_charge + new_charge, 10 - new_charge)
    assert usage.usd == first.estimate_usd()[0]
    rows = [json.loads(line) for line in parent_sink.path.read_text().splitlines()]
    settled = next(row for row in rows if child.name in row.get("settled_children", ()))
    if not split_baseline:
        assert [(r["observed_ns"], r["balance_usd"]) for r in settled["credit_readings"]] == (
            [extra_reading] if extra_reading else []
        )
    fresh.record_plan_preflight(
        "other", agent6_budget.PlanUsage.single(0, 300, 0, credits_balance="$1")
    )
    assert fresh.available().usd == 10 - new_charge


def test_interrupted_recovery_counts_only_what_the_remaining_child_adds(
    short_repo: pathlib.Path,
) -> None:
    """One recovery settles $10 to $8. The next child adds $8 to $7 in the same series."""
    repo = short_repo / "repo"
    child_env.init_repo(repo)
    parent = child_env.create_record(repo, "parent")
    _resume(parent, "first")
    first = _tracker()
    _balance(first, 10, 10)
    budget_events.emit_budget_update(first, event_log.EventSink(parent / "logs.jsonl").emit)
    for group, final in [("parent-r1", 8), ("parent-r2", 7)]:
        child = _running_panel(repo, parent, group)
        meter = _tracker()
        _balance(meter, 10, 10)
        _balance(meter, 20, 8)
        _balance(meter, 30, final)
        budget_events.emit_budget_update(meter, event_log.EventSink(child / "logs.jsonl").emit)

    _resume(parent, "second")
    second = _tracker()
    with pytest.raises(_Crash):
        _reconcile(repo, parent, second, stop_at="parent-r2")
    assert second.available().usd == 8
    _resume(parent, "third")
    fresh = _tracker()
    _reconcile(repo, parent, fresh)
    usage = children.child_usage(parent)
    assert usage is not None
    assert (usage.usd, fresh.available().usd) == (3, 9)


def test_a_failed_record_write_launches_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is the dispatch's precondition: an unwritable one refuses before any clone."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    calls: list[str] = []

    @child_env.blocking
    def spawner(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        calls.append(group)
        return []

    def refuse(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(child_dispatch, "write_record", refuse)
    provider = _provider()
    budget = _tracker()
    wf = _build_wf(
        repo,
        tmp_path / "run-abc",
        provider,
        steer_text="/parallel a",
        child_spawner=spawner,
        budget=budget,
    )
    wf.run("start")
    assert calls == []
    assert budget.available().usd == 10.0
    texts = [t for call in provider.call.call_args_list for t in _texts(call)]
    assert any("could not persist the dispatch record" in t and "disk full" in t for t in texts)


def _texts(call: Any) -> list[str]:
    messages = call.kwargs.get("messages") or call.args[0]
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block["text"])
    return out


def test_a_crash_before_launch_resumes_to_nothing_ran(tmp_path: pathlib.Path) -> None:
    """A record written before the launch survives the crash; the resume closes it: nothing ran."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"

    @child_env.blocking
    def crashing(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        raise _Crash

    budget = _tracker()
    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel 2 -- a",
            child_spawner=crashing,
            budget=budget,
        ).run("start")
    (record,) = _records(session_dir)
    assert record.state == "running"  # written before the launch was attempted
    assert [c.session_id for c in record.children] == ["run-abc-p1-l1", "run-abc-p1-l2"]
    assert budget.available().usd == 10.0  # no child holds a share of it

    seen: list[tuple[str, Sequence[str]]] = []

    def recoverer(group: str, ids: Sequence[str]) -> list[child_records.ChildResult]:
        seen.append((group, tuple(ids)))
        return []

    provider2 = _provider()
    budget2 = _tracker()
    wf2 = _build_wf(repo, session_dir, provider2, child_recoverer=recoverer, budget=budget2)
    wf2.resume()
    assert seen == [("run-abc-p1", ("run-abc-p1-l1", "run-abc-p1-l2"))]
    assert isinstance(wf2.events, _Events)
    assert wf2.events.of("parallel.returned") == [
        {"group": "run-abc-p1", "source": "operator", "children": []}
    ]
    (record,) = _records(session_dir)
    assert record.state == "complete" and record.delivered
    assert "nothing ran" in record.message
    assert budget2.available().usd == 10.0
    assert any("nothing ran" in t for t in _first_call_texts(provider2))
    # A third execution has nothing left to reconcile and delivers nothing again.
    provider3 = _provider()
    _build_wf(repo, session_dir, provider3, child_recoverer=recoverer, budget=_tracker()).resume()
    assert len(seen) == 1
    assert sum("nothing ran" in t for t in _first_call_texts(provider3)) == 1  # history, once


def test_a_crash_after_one_child_started_recovers_it_once(tmp_path: pathlib.Path) -> None:
    """The child that finished merges once and is settled once; the one that never ran reads so."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)
    tip = _make_branch(repo, "agent6/run-abc-p1-l1", base, "child1.txt", tmp_path / "wt1")

    @child_env.blocking
    def crashing(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        raise _Crash

    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel 2 -- a",
            child_spawner=crashing,
            budget=_tracker(),
        ).run("start")

    def recoverer(group: str, ids: Sequence[str]) -> list[child_records.ChildResult]:
        return [
            _child_result(ids[0], tmp_path / "w1", branch="agent6/run-abc-p1-l1", tip=tip, usd=1.5),
            _child_result(ids[1], tmp_path / "w2", ok=False, error="never started"),
        ]

    provider2 = _provider()
    budget2 = _tracker()
    _build_wf(repo, session_dir, provider2, child_recoverer=recoverer, budget=budget2).resume()
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert log.count("run-abc-p1-l1 work") == 1, log
    (record,) = _records(session_dir)
    assert record.state == "complete"
    l1, l2 = record.children
    assert l1.merge is not None and l1.merge.status == "merged" and l1.settled
    assert l2.merge is not None and l2.merge.status == "failed" and not l2.settled
    assert budget2.estimate_usd()[0] == pytest.approx(1.5)
    assert budget2.available().usd == pytest.approx(8.5)  # the unknown child holds nothing
    texts = _first_call_texts(provider2)
    assert any("merged into parent chain at" in t and "never started" in t for t in texts)


def test_a_merge_committed_before_the_record_is_not_merged_twice(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the merge and the record's write: the resume finds the merge, records it."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)
    tips: dict[str, str] = {}

    @child_env.blocking
    def spawner(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        out = []
        for i in range(1, len(children) + 1):
            sid = f"run-abc-{group}-l{i}"
            tips[sid] = _make_branch(
                repo, f"agent6/{sid}", base, f"child{i}.txt", tmp_path / f"wt{i}"
            )
            out.append(
                _child_result(
                    sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tips[sid], usd=1.0
                )
            )
        return out

    real_write = child_dispatch.write_record
    writes: list[str] = []

    def crash_after_first_merge(session_dir: pathlib.Path, record: Any, *, now: float) -> None:
        writes.append(record.state)
        if record.state == "merging" and record.children[0].merge is not None:
            raise _Crash  # the merge is on the chain, the record is not
        real_write(session_dir, record, now=now)

    monkeypatch.setattr(child_dispatch, "write_record", crash_after_first_merge)
    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel a",
            child_spawner=spawner,
            budget=_tracker(),
        ).run("start")
    monkeypatch.setattr(child_dispatch, "write_record", real_write)
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert log.count("run-abc-p1-l1 work") == 1, log

    def recoverer(group: str, ids: Sequence[str]) -> list[child_records.ChildResult]:
        return [
            _child_result(sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tips[sid], usd=1.0)
            for i, sid in enumerate(ids, start=1)
        ]

    budget2 = _tracker()
    _build_wf(repo, session_dir, _provider(), child_recoverer=recoverer, budget=budget2).resume()
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert log.count("run-abc-p1-l1 work") == 1, log
    (record,) = _records(session_dir)
    assert record.state == "complete"
    assert all(
        c.merge is not None and c.merge.status == "merged" and c.settled for c in record.children
    )
    # The first execution settled the child; settled usage stays with it.
    assert budget2.estimate_usd()[0] == pytest.approx(0.0)
    assert budget2.available().usd == 10.0


def test_a_result_completed_but_undelivered_is_delivered_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)

    @child_env.blocking
    def spawner(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        sid = f"run-abc-{group}-l1"
        tip = _make_branch(repo, f"agent6/{sid}", base, "child1.txt", tmp_path / "wt1")
        return [_child_result(sid, tmp_path / "w1", branch=f"agent6/{sid}", tip=tip, usd=1.0)]

    real_deliver = parallel_dispatch.ParallelDispatcher.deliver

    def crash_before_delivery(self: Any, *a: Any, **k: Any) -> None:
        raise _Crash

    monkeypatch.setattr(parallel_dispatch.ParallelDispatcher, "deliver", crash_before_delivery)
    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel a",
            child_spawner=spawner,
            budget=_tracker(),
        ).run("start")
    monkeypatch.setattr(parallel_dispatch.ParallelDispatcher, "deliver", real_deliver)
    (record,) = _records(session_dir)
    assert record.state == "complete" and not record.delivered
    provider2 = _provider()
    _build_wf(
        repo, session_dir, provider2, child_recoverer=_no_children, budget=_tracker()
    ).resume()
    assert sum("merged into parent chain at" in t for t in _first_call_texts(provider2)) == 1
    (record,) = _records(session_dir)
    assert record.delivered
    assert _git(repo, "log", "--oneline", "refs/agent6/parent").count("run-abc-p1-l1 work") == 1


def test_child_spend_exhausts_the_parent_before_its_next_call(tmp_path: pathlib.Path) -> None:
    """A child that spent past the parent's cap stops the parent at its next provider call."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)
    tip = _make_branch(repo, "agent6/run-abc-p1-l1", base, "child1.txt", tmp_path / "wt1")

    @child_env.blocking
    def crashing(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        raise _Crash

    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel a",
            child_spawner=crashing,
            budget=_tracker(),
        ).run("start")

    def recoverer(group: str, ids: Sequence[str]) -> list[child_records.ChildResult]:
        return [
            _child_result(ids[0], tmp_path / "w1", branch="agent6/run-abc-p1-l1", tip=tip, usd=50.0)
        ]

    budget2 = _tracker()
    result = _build_wf(
        repo, session_dir, _provider(), child_recoverer=recoverer, budget=budget2
    ).resume()
    assert result.reason == "budget_exhausted"
    assert budget2.estimate_usd()[0] == pytest.approx(50.0)


def test_a_fresh_dispatch_records_every_boundary(tmp_path: pathlib.Path) -> None:
    """An uninterrupted dispatch leaves a complete, delivered record with merges, usage and refs."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)

    ceilings: list[float] = []

    @child_env.blocking
    def spawner(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        out = []
        ceilings.extend(child.allowance.usd for child in children)
        for i in range(1, len(children) + 1):
            sid = f"run-abc-{group}-l{i}"
            tip = _make_branch(repo, f"agent6/{sid}", base, f"child{i}.txt", tmp_path / f"wt{i}")
            out.append(
                _child_result(sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tip, usd=0.5)
            )
        return out

    budget = _tracker()
    events_wf = _build_wf(
        repo,
        session_dir,
        _provider(),
        steer_text="/parallel a\n/parallel b",
        child_spawner=spawner,
        budget=budget,
    )
    events_wf.run("start")
    first, second = _records(session_dir)
    assert ceilings == [10.0, 10.0]  # each child starts under all the run has left
    for record in (first, second):
        assert record.state == "complete" and record.delivered
        assert record.owner and record.source == "operator" and record.base == base
        (entry,) = record.children
        assert entry.merge is not None and entry.merge.status == "merged" and entry.reported
        assert entry.merge.target_before and entry.merge.target_after == entry.merge.sha
        assert entry.usage == child_dispatch.UsageRecord(usd=0.5) and entry.settled
    # The second task merged onto the first's merge: in the order written.
    assert first.children[0].merge is not None and second.children[0].merge is not None
    assert second.children[0].merge.target_before == first.children[0].merge.target_after
    assert budget.estimate_usd()[0] == pytest.approx(1.0)
    assert budget.available().usd == pytest.approx(9.0)
    stored = json.loads(child_dispatch.record_path(session_dir, "run-abc-p1").read_text())
    assert stored["version"] == 15
    assert stored["children"][0]["merge"]["status"] == "merged"
    assert stored["children"][0]["merge"]["sha"] == first.children[0].merge.sha
    assert (stored["merge_policy"], stored["spec"], stored["node_id"]) == ("merge", "", None)
    assert "segments" not in stored
