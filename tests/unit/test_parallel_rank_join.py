# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A dispatch joined by rank: the injected ranker orders the lanes and one lane lands."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any, cast

import pytest

from agent6 import budget, directive, lane_records, paths
from agent6.app import lanes as app_lanes
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import (
    loop_conversation,
    loop_state,
    parallel_dispatch,
    parallel_join,
    run_chain,
)
from agent6.providers import Provider, ProviderResponse
from agent6.sessions import event_log, journal, lane_dispatch
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repository whose three lane branches each rewrite the same file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    for lane in (1, 2, 3):
        _git(root, "checkout", "-q", "-b", f"agent6/root-p1-l{lane}", "main")
        (root / "a.txt").write_text(f"lane {lane}\n", encoding="utf-8")
        _git(root, "commit", "-qam", f"lane {lane}")
    _git(root, "checkout", "-q", "main")
    return root


class _Lanes:
    """A group spawner that returns the prepared branches, and a ranker that counts its calls."""

    def __init__(
        self, repo: pathlib.Path, order: Sequence[int], *, failed: Sequence[int] = ()
    ) -> None:
        self.repo = repo
        self.order = tuple(f"root-p1-l{lane}" for lane in order)
        self.failed = set(failed)
        self.ranked: list[list[str]] = []
        self.results: list[lane_records.LaneResult] = []

    def spawn(
        self, lanes: list[lane_records.LaneTask], group: str, *, at: str | None = None
    ) -> list[lane_records.LaneResult]:
        self.results = [self.result(i) for i in range(1, len(lanes) + 1)]
        return self.results

    def result(self, lane: int, *, has_commits: bool = True) -> lane_records.LaneResult:
        branch = f"agent6/root-p1-l{lane}"
        ok = lane not in self.failed
        return lane_records.LaneResult(
            spec=lane_records.LaneSpec(lane, f"root-p1-l{lane}", pathlib.Path("/nowhere"), None),
            session_dir=pathlib.Path("/nowhere"),
            branch=branch,
            ok=ok,
            error="" if ok else "no result (failed · crashed)",
            imported_ref=branch if ok and has_commits else "",
            imported_tip=_git(self.repo, "rev-parse", branch) if ok and has_commits else "",
            has_commits=ok and has_commits,
            usage=budget.UsageSettlement(usd=0.25, usd_partial=False, tokens=0),
            report=f"lane {lane} finished",
        )

    def rank(
        self, results: Sequence[lane_records.LaneResult], *, group: str, base: str
    ) -> lane_records.Ranking:
        assert group == "root-p1"
        self.ranked.append([res.spec.session_id for res in results])
        return lane_records.Ranking(
            order=tuple(rid for rid in self.order if rid in self.ranked[-1]),
            ranked_by="judge",
            rationale="two reads best",
        )


def _dispatcher(
    repo: pathlib.Path, lanes: _Lanes, events: list[Any], *, ranker: bool = True
) -> parallel_dispatch.ParallelDispatcher:
    session_dir = repo.parent / "root"
    session_dir.mkdir(exist_ok=True)
    return parallel_dispatch.ParallelDispatcher(
        chain=run_chain.RunChain(
            repo,
            ref=git_read.chain_ref_for("root"),
            fallback_parent=_git(repo, "rev-parse", "main"),
        ),
        curator=None,
        max_lanes=4,
        lane_spawner=lanes.spawn,
        save_snapshot=lambda *_a, **_k: None,
        log=lambda _m: None,
        emit=events.append,
        emit_graph_snapshot=lambda: None,
        session_id="root",
        execution_id="exec-1",
        session_dir=session_dir,
        ranker=lanes.rank if ranker else None,
    )


def _dispatch(
    dispatcher: parallel_dispatch.ParallelDispatcher, spec: str = "3"
) -> lane_records.DispatchResult:
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)
    return dispatcher.dispatch(
        loop_conversation.Conversation(),
        0,
        state,
        [directive.Segment(task="rewrite a.txt", spec=spec)],
        join="rank",
    )


def _record(repo: pathlib.Path) -> lane_dispatch.DispatchRecord:
    (record,) = lane_dispatch.read_records(repo.parent / "root")
    return record


def test_a_rank_join_lands_the_first_ranked_lane_alone(repo: pathlib.Path) -> None:
    """The three branches conflict with each other; merging more than the winner would fail."""
    lanes = _Lanes(repo, order=(2, 3, 1))
    events: list[Any] = []

    result = _dispatch(_dispatcher(repo, lanes, events))

    assert [(j.session_id, j.status, j.detail) for j in result.lanes] == [
        ("root-p1-l1", "ranked", "3 of 3"),
        ("root-p1-l2", "joined", ""),
        ("root-p1-l3", "ranked", "2 of 3"),
    ]
    chain = git_read.chain_ref_for("root")
    assert _git(repo, "rev-parse", f"{chain}^{{tree}}") == _git(
        repo, "rev-parse", "agent6/root-p1-l2^{tree}"
    )
    assert _git(repo, "rev-parse", "main") == _git(repo, "rev-parse", f"{chain}^1")
    assert lanes.ranked == [["root-p1-l1", "root-p1-l2", "root-p1-l3"]]
    record = _record(repo)
    assert (record.join_policy, record.state) == ("rank", "complete")
    assert record.ranking == lane_dispatch.RankRecord(
        order=("root-p1-l2", "root-p1-l3", "root-p1-l1"),
        ranked_by="judge",
        rationale="two reads best",
    )
    assert "  - root-p1-l3 (agent6/root-p1-l3): ranked 2 of 3; not joined" in result.message
    assert not [e for e in events if isinstance(e, journal.ParallelFailed)]
    (joined,) = [e for e in events if isinstance(e, journal.ParallelJoined)]
    assert [lane["status"] for lane in joined.lanes] == ["ranked", "joined", "ranked"]


def test_a_resumed_rank_join_neither_ranks_nor_merges_again(repo: pathlib.Path) -> None:
    """The order is on the record before the first join, and a landed merge is recognised."""
    lanes = _Lanes(repo, order=(2, 3, 1))
    dispatcher = _dispatcher(repo, lanes, [])
    _dispatch(dispatcher)
    tip = _git(repo, "rev-parse", git_read.chain_ref_for("root"))
    record = _record(repo)
    interrupted = record.model_copy(
        update={
            "state": "joining",
            "children": tuple(c.model_copy(update={"join": None}) for c in record.children),
        }
    )

    result = dispatcher.land(interrupted, lanes.results)

    assert len(lanes.ranked) == 1
    assert _git(repo, "rev-parse", git_read.chain_ref_for("root")) == tip
    assert [j.status for j in result.lanes] == ["ranked", "joined", "ranked"]


def test_a_lane_without_a_result_is_not_ranked(repo: pathlib.Path) -> None:
    """A failed lane keeps its failure; the ranker sees only lanes that ended with a result."""
    lanes = _Lanes(repo, order=(3, 2, 1), failed=(3,))
    events: list[Any] = []

    result = _dispatch(_dispatcher(repo, lanes, events))

    assert lanes.ranked == [["root-p1-l1", "root-p1-l2"]]
    assert [(j.status, j.detail) for j in result.lanes] == [
        ("ranked", "2 of 2"),
        ("joined", ""),
        ("failed", "no result (failed · crashed)"),
    ]
    (failed,) = [e for e in events if isinstance(e, journal.ParallelFailed)]
    assert [lane["session_id"] for lane in failed.lanes or []] == ["root-p1-l3"]


def test_no_lane_with_a_result_ranks_nothing(repo: pathlib.Path) -> None:
    """With nothing to rank the ranker is not asked, and the chain stays unborn."""
    lanes = _Lanes(repo, order=(1, 2), failed=(1, 2))

    result = _dispatch(_dispatcher(repo, lanes, []), spec="2")

    assert lanes.ranked == []
    assert [j.status for j in result.lanes] == ["failed", "failed"]
    assert _record(repo).ranking == lane_dispatch.RankRecord(order=())
    assert git_read.chain_tip(repo, git_read.chain_ref_for("root")) is None


def test_a_ranker_that_leaves_a_lane_out_still_places_it(repo: pathlib.Path) -> None:
    """Every lane with a result has a place: an order that omits one puts it last."""
    lanes = _Lanes(repo, order=(3,))

    result = _dispatch(_dispatcher(repo, lanes, []))

    assert _record(repo).ranking is not None
    assert [(j.status, j.detail) for j in result.lanes] == [
        ("ranked", "2 of 3"),
        ("ranked", "3 of 3"),
        ("joined", ""),
    ]


def test_a_rank_join_without_a_ranker_is_refused_before_any_clone(repo: pathlib.Path) -> None:
    """Only an execution built with a ranker can join by rank."""
    lanes = _Lanes(repo, order=(1,))

    result = _dispatch(_dispatcher(repo, lanes, [], ranker=False))

    assert (result.dispatched, lanes.results) == (False, [])
    assert "no ranker for a rank join" in result.message
    assert lane_dispatch.read_records(repo.parent / "root") == ()


def test_a_ranked_lane_reads_as_its_place_in_the_task_graph_note() -> None:
    """The DAG note says where the lane placed, not that it failed."""
    join = lane_records.LaneJoin("root-p1-l1", "agent6/root-p1-l1", "ranked", "", "3 of 3")

    assert parallel_join.lane_note(join) == "root-p1-l1 ranked 3 of 3; not joined"


class _Judge:
    """A judge that keeps what it was shown and puts lane 2 first."""

    def __init__(self) -> None:
        self.shown: list[str] = []

    def call(self, **kw: Any) -> ProviderResponse:
        self.shown.append(kw["messages"][0]["content"])
        reply = {"ranking": ["root-p1-l2", "root-p1-l1"], "rationale": "two is tidier"}
        return ProviderResponse(
            text=json.dumps(reply),
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw={"content": [{"type": "text", "text": json.dumps(reply)}]},
        )


def _lane_record(repo: pathlib.Path, sid: str, verification: str) -> pathlib.Path:
    """Write an imported lane's record: a manifest and a journal that ends with a verdict."""
    dest = sessions_layout.bucket_dir(paths.state_dir(repo), "runs") / sid
    dest.mkdir(parents=True)
    manifest = {"version": 2, "session_id": sid, "mode": "run", "user_task": "rewrite a.txt"}
    (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "rewrite a.txt"},
        {"type": "session.stop", "reason": "finish_session", "verification": verification},
    ]
    (dest / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return dest


def test_the_apps_ranker_judges_the_imported_work_and_records_the_outcome(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge reads each lane's diff from its base; the stamps and the journal keep the order."""
    monkeypatch.setenv("XDG_STATE_HOME", str(repo.parent / "state"))
    origin_state = paths.state_dir(repo)
    root = _lane_record(repo, "root", "not_applicable")
    lanes = _Lanes(repo, order=())
    results = [
        dataclasses.replace(res, session_dir=_lane_record(repo, res.spec.session_id, "passed"))
        for res in (lanes.result(1), lanes.result(2))
    ]
    judge = _Judge()

    def seat(*_a: object) -> Provider:
        return cast(Provider, judge)

    said: list[str] = []
    cfg = Config.model_validate(
        {
            "providers": {"o": {"backend": "openai", "base_url": "https://x/v1"}},
            "models": {"default": {"model": "o/worker-1"}},
        }
    )
    ranker = app_lanes.build_lane_ranker(
        cfg,
        repo,
        origin_state,
        coordinator_dir=root,
        base_sha=_git(repo, "rev-parse", "main"),
        build_provider=seat,
        judging_status=contextlib.nullcontext,
        budget=budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20),
        events=event_log.EventSink(root / "logs.jsonl"),
        reporter=app_reporter.Reporter(out=said.append, err=said.append),
    )

    ranking = ranker(results, group="root-p1", base="")

    assert ranking == lane_records.Ranking(
        order=("root-p1-l2", "root-p1-l1"), ranked_by="judge", rationale="two is tidier"
    )
    (shown,) = judge.shown
    assert "--- CANDIDATE root-p1-l1 ---" in shown and "+lane 1" in shown and "+lane 2" in shown
    placements = {
        sid: sessions_manifest.read_manifest(origin_state / "sessions" / "runs" / sid).placement
        for sid in ("root-p1-l1", "root-p1-l2")
    }
    assert placements["root-p1-l2"] == sessions_manifest.PlacementStamp(
        rank=1, of=2, winner=True, group="root-p1"
    )
    assert placements["root-p1-l1"] == sessions_manifest.PlacementStamp(
        rank=2, of=2, winner=False, group="root-p1"
    )
    comparison = sessions_manifest.read_manifest(root).comparison
    assert comparison is not None and comparison.ranking == ("root-p1-l2", "root-p1-l1")
    journaled = [json.loads(line) for line in (root / "logs.jsonl").read_text().splitlines()]
    (compared,) = [row for row in journaled if row["type"] == "parallel.compared"]
    assert [lane["session_id"] for lane in compared["ranking"]] == ["root-p1-l2", "root-p1-l1"]
    assert said[0] == "ranked candidates (best first):"
    assert said[1].startswith("  1. root-p1-l2  passed  ")
