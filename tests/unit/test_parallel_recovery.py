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
import types
from collections.abc import Sequence
from typing import Any
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import lane_records
from agent6.harness import loop, operator_bridge, parallel_dispatch, provider_call, run_chain
from agent6.providers import ProviderResponse
from agent6.sessions import journal, lane_dispatch
from agent6.tools import results as tools_results


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


def _lane_result(
    spec_id: str,
    workdir: pathlib.Path,
    *,
    branch: str = "",
    tip: str = "",
    ok: bool = True,
    error: str = "",
    usd: float = 0.0,
) -> lane_records.LaneResult:
    lane = int(spec_id.rsplit("-l", 1)[1])
    return lane_records.LaneResult(
        spec=lane_records.LaneSpec(lane, spec_id, workdir, None),
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
    lane_spawner: Any = None,
    lane_recoverer: Any = None,
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
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            git=types.SimpleNamespace(
                control="agent6",
                commit_per_turn=True,
                commit=types.SimpleNamespace(
                    checkpoint_message="agent6",
                    name="",
                    email="",
                    trailer="",
                ),
            ),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                settled_stop_after=12,
                stagnation_notice_after_s=300.0,
                verify=mock.MagicMock(command=(), when="never", at_start=True, retries=2),
                prompt=mock.MagicMock(system_prompt_file=""),
                parallel=types.SimpleNamespace(max_lanes=4),
            ),
        ),
        provider=provider,
        dispatcher=disp,
        logger=_silent,
        events=events,
        budget=budget,
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(
            lane_spawner=lane_spawner,
            lane_recoverer=lane_recoverer,
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


def _records(session_dir: pathlib.Path) -> tuple[lane_dispatch.DispatchRecord, ...]:
    return lane_dispatch.read_records(session_dir)


def _no_lanes(group: str, children: Sequence[str]) -> list[lane_records.LaneResult]:
    del group, children
    return []


def _tracker() -> agent6_budget.BudgetTracker:
    return agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)


def test_a_failed_record_write_launches_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is the dispatch's precondition: an unwritable one refuses before any clone."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    calls: list[str] = []

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        calls.append(group)
        return []

    def refuse(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(lane_dispatch, "write_record", refuse)
    provider = _provider()
    budget = _tracker()
    wf = _build_wf(
        repo,
        tmp_path / "run-abc",
        provider,
        steer_text="/parallel a",
        lane_spawner=spawner,
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

    def crashing(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        raise _Crash

    budget = _tracker()
    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel 2 -- a",
            lane_spawner=crashing,
            budget=budget,
        ).run("start")
    (record,) = _records(session_dir)
    assert record.state == "running"  # written before the launch was attempted
    assert [c.session_id for c in record.children] == ["run-abc-p1-l1", "run-abc-p1-l2"]
    assert budget.available().usd == 10.0  # no lane holds a share of it

    seen: list[tuple[str, Sequence[str]]] = []

    def recoverer(group: str, children: Sequence[str]) -> list[lane_records.LaneResult]:
        seen.append((group, tuple(children)))
        return []

    provider2 = _provider()
    budget2 = _tracker()
    wf2 = _build_wf(repo, session_dir, provider2, lane_recoverer=recoverer, budget=budget2)
    wf2.resume()
    assert seen == [("run-abc-p1", ("run-abc-p1-l1", "run-abc-p1-l2"))]
    (record,) = _records(session_dir)
    assert record.state == "complete" and record.delivered
    assert "nothing ran" in record.message
    assert budget2.available().usd == 10.0
    assert any("nothing ran" in t for t in _first_call_texts(provider2))
    # A third execution has nothing left to reconcile and delivers nothing again.
    provider3 = _provider()
    _build_wf(repo, session_dir, provider3, lane_recoverer=recoverer, budget=_tracker()).resume()
    assert len(seen) == 1
    assert sum("nothing ran" in t for t in _first_call_texts(provider3)) == 1  # history, once


def test_a_crash_after_one_child_started_recovers_it_once(tmp_path: pathlib.Path) -> None:
    """The lane that finished joins once and is settled once; the one that never ran reads so."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)
    tip = _make_branch(repo, "agent6/run-abc-p1-l1", base, "lane1.txt", tmp_path / "wt1")

    def crashing(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        raise _Crash

    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel 2 -- a",
            lane_spawner=crashing,
            budget=_tracker(),
        ).run("start")

    def recoverer(group: str, children: Sequence[str]) -> list[lane_records.LaneResult]:
        return [
            _lane_result(
                children[0], tmp_path / "w1", branch="agent6/run-abc-p1-l1", tip=tip, usd=1.5
            ),
            _lane_result(children[1], tmp_path / "w2", ok=False, error="never started"),
        ]

    provider2 = _provider()
    budget2 = _tracker()
    _build_wf(repo, session_dir, provider2, lane_recoverer=recoverer, budget=budget2).resume()
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert log.count("run-abc-p1-l1 work") == 1, log
    (record,) = _records(session_dir)
    assert record.state == "complete"
    l1, l2 = record.children
    assert l1.join is not None and l1.join.status == "joined" and l1.settled
    assert l2.join is not None and l2.join.status == "failed" and not l2.settled
    assert budget2.estimate_usd()[0] == pytest.approx(1.5)
    assert budget2.available().usd == pytest.approx(8.5)  # the unknown lane holds nothing
    texts = _first_call_texts(provider2)
    assert any("joined at" in t and "never started" in t for t in texts)


def test_a_merge_landed_before_the_record_is_not_merged_twice(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the merge and the record's write: the resume finds the merge, records it."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)
    tips: dict[str, str] = {}

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        out = []
        for i in range(1, len(lanes) + 1):
            sid = f"run-abc-{group}-l{i}"
            tips[sid] = _make_branch(
                repo, f"agent6/{sid}", base, f"lane{i}.txt", tmp_path / f"wt{i}"
            )
            out.append(
                _lane_result(
                    sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tips[sid], usd=1.0
                )
            )
        return out

    real_write = lane_dispatch.write_record
    writes: list[str] = []

    def crash_after_first_join(session_dir: pathlib.Path, record: Any, *, now: float) -> None:
        writes.append(record.state)
        if record.state == "joining" and record.children[0].join is not None:
            raise _Crash  # the merge is on the chain, the record is not
        real_write(session_dir, record, now=now)

    monkeypatch.setattr(lane_dispatch, "write_record", crash_after_first_join)
    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel 2 -- a",
            lane_spawner=spawner,
            budget=_tracker(),
        ).run("start")
    monkeypatch.setattr(lane_dispatch, "write_record", real_write)
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert log.count("run-abc-p1-l1 work") == 1 and "run-abc-p1-l2" not in log, log

    def recoverer(group: str, children: Sequence[str]) -> list[lane_records.LaneResult]:
        return [
            _lane_result(sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tips[sid], usd=1.0)
            for i, sid in enumerate(children, start=1)
        ]

    budget2 = _tracker()
    _build_wf(repo, session_dir, _provider(), lane_recoverer=recoverer, budget=budget2).resume()
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert log.count("run-abc-p1-l1 work") == 1 and log.count("run-abc-p1-l2 work") == 1, log
    (record,) = _records(session_dir)
    assert record.state == "complete"
    assert all(
        c.join is not None and c.join.status == "joined" and c.settled for c in record.children
    )
    # The first execution settled both lanes; settled usage stays with it.
    assert budget2.estimate_usd()[0] == pytest.approx(0.0)
    assert budget2.available().usd == 10.0


def test_a_result_completed_but_undelivered_is_delivered_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        sid = f"run-abc-{group}-l1"
        tip = _make_branch(repo, f"agent6/{sid}", base, "lane1.txt", tmp_path / "wt1")
        return [_lane_result(sid, tmp_path / "w1", branch=f"agent6/{sid}", tip=tip, usd=1.0)]

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
            lane_spawner=spawner,
            budget=_tracker(),
        ).run("start")
    monkeypatch.setattr(parallel_dispatch.ParallelDispatcher, "deliver", real_deliver)
    (record,) = _records(session_dir)
    assert record.state == "complete" and not record.delivered
    provider2 = _provider()
    _build_wf(repo, session_dir, provider2, lane_recoverer=_no_lanes, budget=_tracker()).resume()
    assert sum("joined at" in t for t in _first_call_texts(provider2)) == 1
    (record,) = _records(session_dir)
    assert record.delivered
    assert _git(repo, "log", "--oneline", "refs/agent6/parent").count("run-abc-p1-l1 work") == 1


def test_child_spend_exhausts_the_parent_before_its_next_call(tmp_path: pathlib.Path) -> None:
    """A lane that spent past the parent's cap stops the parent at its next provider call."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)
    tip = _make_branch(repo, "agent6/run-abc-p1-l1", base, "lane1.txt", tmp_path / "wt1")

    def crashing(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        raise _Crash

    with pytest.raises(_Crash):
        _build_wf(
            repo,
            session_dir,
            _provider(),
            steer_text="/parallel a",
            lane_spawner=crashing,
            budget=_tracker(),
        ).run("start")

    def recoverer(group: str, children: Sequence[str]) -> list[lane_records.LaneResult]:
        return [
            _lane_result(
                children[0], tmp_path / "w1", branch="agent6/run-abc-p1-l1", tip=tip, usd=50.0
            )
        ]

    budget2 = _tracker()
    result = _build_wf(
        repo, session_dir, _provider(), lane_recoverer=recoverer, budget=budget2
    ).resume()
    assert result.reason == "budget_exhausted"
    assert budget2.estimate_usd()[0] == pytest.approx(50.0)


def test_a_fresh_dispatch_records_every_boundary(tmp_path: pathlib.Path) -> None:
    """An uninterrupted dispatch leaves a complete, delivered record with joins, usage and refs."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    session_dir = tmp_path / "run-abc"
    base = _head(repo)

    ceilings: list[float] = []

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        out = []
        ceilings.extend(lane.allowance.usd for lane in lanes)
        for i in range(1, len(lanes) + 1):
            sid = f"run-abc-{group}-l{i}"
            tip = _make_branch(repo, f"agent6/{sid}", base, f"lane{i}.txt", tmp_path / f"wt{i}")
            out.append(
                _lane_result(sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tip, usd=0.5)
            )
        return out

    budget = _tracker()
    events_wf = _build_wf(
        repo,
        session_dir,
        _provider(),
        steer_text="/parallel 2 -- a",
        lane_spawner=spawner,
        budget=budget,
    )
    events_wf.run("start")
    (record,) = _records(session_dir)
    assert record.state == "complete" and record.delivered
    assert record.owner and record.source == "operator" and record.base == _head(repo)
    assert ceilings == [10.0, 10.0]  # each lane starts under all the run has left
    for child in record.children:
        assert child.join is not None and child.join.status == "joined"
        assert child.join.target_before and child.join.target_after == child.join.sha
        assert child.usage == lane_dispatch.UsageRecord(usd=0.5) and child.settled
    assert budget.estimate_usd()[0] == pytest.approx(1.0)
    assert budget.available().usd == pytest.approx(9.0)
    stored = json.loads(lane_dispatch.record_path(session_dir, "run-abc-p1").read_text())
    assert stored["version"] == 5
    assert stored["join_policy"] == "merge"
