# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A dispatch joined by rank: the injected ranker orders the children and one child lands."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any, cast

import pytest

from agent6 import budget, child_records, directive, kinds, paths
from agent6.app import children as app_children
from agent6.app import parallel_results
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import judge as harness_judge
from agent6.harness import (
    loop_conversation,
    loop_state,
    parallel_dispatch,
    parallel_join,
    run_chain,
)
from agent6.providers import Provider, ProviderResponse
from agent6.sessions import child_dispatch, event_log, journal
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repository whose three child branches each rewrite the same file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    for child in (1, 2, 3):
        _git(root, "checkout", "-q", "-b", f"agent6/root-p1-l{child}", "main")
        (root / "a.txt").write_text(f"child {child}\n", encoding="utf-8")
        _git(root, "commit", "-qam", f"child {child}")
    _git(root, "checkout", "-q", "main")
    return root


class _Children:
    """A group spawner that returns the prepared branches, and a ranker that counts its calls."""

    def __init__(
        self,
        repo: pathlib.Path,
        order: Sequence[int],
        *,
        failed: Sequence[int] = (),
        crowns: bool = True,
    ) -> None:
        self.repo = repo
        self.order = tuple(f"root-p1-l{child}" for child in order)
        self.failed = set(failed)
        self.crowns = crowns
        self.ranked: list[list[str]] = []
        self.results: list[child_records.ChildResult] = []

    def spawn(
        self, children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.ChildResult]:
        self.results = [self.result(i) for i in range(1, len(children) + 1)]
        return self.results

    def result(self, child: int, *, has_commits: bool = True) -> child_records.ChildResult:
        branch = f"agent6/root-p1-l{child}"
        ok = child not in self.failed
        return child_records.ChildResult(
            spec=child_records.ChildSpec(
                child, f"root-p1-l{child}", pathlib.Path("/nowhere"), None
            ),
            session_dir=pathlib.Path("/nowhere"),
            branch=branch,
            ok=ok,
            error="" if ok else "no result (failed · crashed)",
            imported_ref=branch if ok and has_commits else "",
            imported_tip=_git(self.repo, "rev-parse", branch) if ok and has_commits else "",
            has_commits=ok and has_commits,
            usage=budget.UsageSettlement(usd=0.25, usd_partial=False, tokens=0),
            report=f"child {child} finished",
        )

    def rank(
        self, results: Sequence[child_records.ChildResult], *, group: str, base: str
    ) -> child_records.Ranking:
        assert group == "root-p1"
        self.ranked.append([res.spec.session_id for res in results])
        order = tuple(rid for rid in self.order if rid in self.ranked[-1])
        return child_records.Ranking(
            order=order,
            ranked_by="judge",
            rationale="two reads best",
            winner=order[0] if order and self.crowns else "",
        )


def _dispatcher(
    repo: pathlib.Path, children: _Children, events: list[Any], *, ranker: bool = True
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
        max_children=4,
        child_spawner=children.spawn,
        save_snapshot=lambda *_a, **_k: None,
        log=lambda _m: None,
        emit=events.append,
        emit_graph_snapshot=lambda: None,
        session_id="root",
        execution_id="exec-1",
        session_dir=session_dir,
        ranker=children.rank if ranker else None,
    )


def _dispatch(
    dispatcher: parallel_dispatch.ParallelDispatcher, spec: str = "3"
) -> child_records.DispatchResult:
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)
    return dispatcher.dispatch(
        loop_conversation.Conversation(),
        0,
        state,
        [directive.Segment(task="rewrite a.txt", spec=spec)],
        join="rank",
    )


def _record(repo: pathlib.Path) -> child_dispatch.DispatchRecord:
    (record,) = child_dispatch.read_records(repo.parent / "root")
    return record


def test_a_rank_join_lands_the_first_ranked_child_alone(repo: pathlib.Path) -> None:
    """The three branches conflict with each other; merging more than the winner would fail."""
    children = _Children(repo, order=(2, 3, 1))
    events: list[Any] = []

    result = _dispatch(_dispatcher(repo, children, events))

    assert [(j.session_id, j.status, j.detail) for j in result.children] == [
        ("root-p1-l1", "ranked", "3 of 3"),
        ("root-p1-l2", "joined", ""),
        ("root-p1-l3", "ranked", "2 of 3"),
    ]
    chain = git_read.chain_ref_for("root")
    assert _git(repo, "rev-parse", f"{chain}^{{tree}}") == _git(
        repo, "rev-parse", "agent6/root-p1-l2^{tree}"
    )
    assert _git(repo, "rev-parse", "main") == _git(repo, "rev-parse", f"{chain}^1")
    assert children.ranked == [["root-p1-l1", "root-p1-l2", "root-p1-l3"]]
    record = _record(repo)
    assert (record.join_policy, record.state) == ("rank", "complete")
    assert record.ranking == child_dispatch.RankRecord(
        order=("root-p1-l2", "root-p1-l3", "root-p1-l1"),
        ranked_by="judge",
        rationale="two reads best",
        winner="root-p1-l2",
    )
    assert "  - root-p1-l3 (agent6/root-p1-l3): ranked 2 of 3; not joined" in result.message
    assert not [e for e in events if isinstance(e, journal.ParallelFailed)]
    (joined,) = [e for e in events if isinstance(e, journal.ParallelJoined)]
    assert [child["status"] for child in joined.children] == ["ranked", "joined", "ranked"]


def test_a_resumed_rank_join_neither_ranks_nor_merges_again(repo: pathlib.Path) -> None:
    """The order is on the record before the first join, and a landed merge is recognised."""
    children = _Children(repo, order=(2, 3, 1))
    dispatcher = _dispatcher(repo, children, [])
    _dispatch(dispatcher)
    tip = _git(repo, "rev-parse", git_read.chain_ref_for("root"))
    record = _record(repo)
    interrupted = record.model_copy(
        update={
            "state": "joining",
            "children": tuple(c.model_copy(update={"join": None}) for c in record.children),
        }
    )

    result = dispatcher.land(interrupted, children.results)

    assert len(children.ranked) == 1
    assert _git(repo, "rev-parse", git_read.chain_ref_for("root")) == tip
    assert [j.status for j in result.children] == ["ranked", "joined", "ranked"]


def test_a_child_without_a_result_is_not_ranked(repo: pathlib.Path) -> None:
    """A failed child keeps its failure; the ranker sees only children that ended with a result."""
    children = _Children(repo, order=(3, 2, 1), failed=(3,))
    events: list[Any] = []

    result = _dispatch(_dispatcher(repo, children, events))

    assert children.ranked == [["root-p1-l1", "root-p1-l2"]]
    assert [(j.status, j.detail) for j in result.children] == [
        ("ranked", "2 of 2"),
        ("joined", ""),
        ("failed", "no result (failed · crashed)"),
    ]
    (failed,) = [e for e in events if isinstance(e, journal.ParallelFailed)]
    assert [child["session_id"] for child in failed.children or []] == ["root-p1-l3"]


def test_no_child_with_a_result_ranks_nothing(repo: pathlib.Path) -> None:
    """With nothing to rank the ranker is not asked, and the chain stays unborn."""
    children = _Children(repo, order=(1, 2), failed=(1, 2))

    result = _dispatch(_dispatcher(repo, children, []), spec="2")

    assert children.ranked == []
    assert [j.status for j in result.children] == ["failed", "failed"]
    assert _record(repo).ranking == child_dispatch.RankRecord(order=())
    assert git_read.chain_tip(repo, git_read.chain_ref_for("root")) is None


def test_a_ranker_that_leaves_a_child_out_still_places_it(repo: pathlib.Path) -> None:
    """Every child with a result has a place: an order that omits one puts it last."""
    children = _Children(repo, order=(3,))

    result = _dispatch(_dispatcher(repo, children, []))

    assert _record(repo).ranking is not None
    assert [(j.status, j.detail) for j in result.children] == [
        ("ranked", "2 of 3"),
        ("ranked", "3 of 3"),
        ("joined", ""),
    ]


def test_a_ranking_that_crowns_no_child_joins_none(repo: pathlib.Path) -> None:
    """The order is a judge's; a ranking with no winner places every child and merges nothing."""
    children = _Children(repo, order=(2, 1, 3), crowns=False)

    result = _dispatch(_dispatcher(repo, children, []))

    assert [(j.status, j.detail) for j in result.children] == [
        ("ranked", "2 of 3"),
        ("ranked", "1 of 3"),
        ("ranked", "3 of 3"),
    ]
    assert git_read.chain_tip(repo, git_read.chain_ref_for("root")) is None
    record = _record(repo)
    assert record.ranking is not None and record.ranking.winner == ""


@pytest.mark.parametrize(
    ("gates", "ranking", "harness_gate", "winner"),
    [
        (("passed", "passed"), ("a", "b"), True, "a"),
        (("not_applicable", "not_applicable"), ("b", "a"), True, "b"),
        # A child that did not verify, ranked over one that did, is not crowned.
        (("failed", "passed"), ("a", "b"), True, ""),
        (("unverified", "passed"), ("a", "b"), True, ""),
        (("failed", "passed"), ("b", "a"), True, "b"),
        # Every gate ran red: nobody wins.
        (("failed", "failed"), ("a", "b"), True, ""),
        (("unverified", "unverified"), ("a", "b"), True, ""),
        # `harness.verify.when = "never"`: an unverified finish is no failure, as for one run.
        (("unverified", "unverified"), ("a", "b"), False, "a"),
        (("failed", "unverified"), ("b", "a"), False, ""),
    ],
)
def test_the_winner_is_a_verified_child_or_nobody(
    gates: tuple[kinds.Verification, kinds.Verification],
    ranking: tuple[str, ...],
    harness_gate: bool,
    winner: str,
) -> None:
    """The judge's order is model output: it cannot crown a red child over a green one."""
    candidates = [
        harness_judge.CandidateBrief(
            session_id=sid, task="t", diff="", verification=gate, cost_usd=0.0
        )
        for sid, gate in zip(("a", "b"), gates, strict=True)
    ]

    crowned = parallel_results.ranked_winner(candidates, ranking, harness_gate=harness_gate)
    assert crowned == winner
    assert parallel_results.ranked_winner(candidates, (), harness_gate=harness_gate) == ""


def test_a_rank_join_without_a_ranker_is_refused_before_any_clone(repo: pathlib.Path) -> None:
    """Only an execution built with a ranker can join by rank."""
    children = _Children(repo, order=(1,))

    result = _dispatch(_dispatcher(repo, children, [], ranker=False))

    assert (result.dispatched, children.results) == (False, [])
    assert "no ranker for a rank join" in result.message
    assert child_dispatch.read_records(repo.parent / "root") == ()


def test_a_ranked_child_reads_as_its_place_in_the_task_graph_note() -> None:
    """The DAG note says where the child placed, not that it failed."""
    join = child_records.ChildJoin("root-p1-l1", "agent6/root-p1-l1", "ranked", "", "3 of 3")

    assert parallel_join.child_note(join) == "root-p1-l1 ranked 3 of 3; not joined"


class _Judge:
    """A judge that keeps what it was shown and puts child 2 first."""

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


def _child_record(repo: pathlib.Path, sid: str, verification: str) -> pathlib.Path:
    """Write an imported child's record: a manifest and a journal that ends with a verdict."""
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
    """The judge reads each child's diff from its base; the stamps and journal keep the order."""
    monkeypatch.setenv("XDG_STATE_HOME", str(repo.parent / "state"))
    origin_state = paths.state_dir(repo)
    root = _child_record(repo, "root", "not_applicable")
    children = _Children(repo, order=())
    results = [
        dataclasses.replace(res, session_dir=_child_record(repo, res.spec.session_id, "passed"))
        for res in (children.result(1), children.result(2))
    ]
    judge = _Judge()

    def reviewer(*_a: object) -> Provider:
        return cast(Provider, judge)

    said: list[str] = []
    cfg = Config.model_validate(
        {
            "providers": {"o": {"backend": "openai", "base_url": "https://x/v1"}},
            "models": {"default": {"model": "o/worker-1"}},
        }
    )
    ranker = app_children.build_child_ranker(
        cfg,
        repo,
        origin_state,
        parent_dir=root,
        base_sha=_git(repo, "rev-parse", "main"),
        build_provider=reviewer,
        judging_status=contextlib.nullcontext,
        budget=budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20),
        events=event_log.EventSink(root / "logs.jsonl"),
        reporter=app_reporter.Reporter(out=said.append, err=said.append),
    )

    ranking = ranker(results, group="root-p1", base="")

    assert ranking == child_records.Ranking(
        order=("root-p1-l2", "root-p1-l1"),
        ranked_by="judge",
        rationale="two is tidier",
        winner="root-p1-l2",
    )
    (shown,) = judge.shown
    assert "--- CANDIDATE root-p1-l1 ---" in shown and "+child 1" in shown and "+child 2" in shown
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
    assert [child["session_id"] for child in compared["ranking"]] == ["root-p1-l2", "root-p1-l1"]
    assert said[0] == "ranked candidates (best first):"
    assert said[1].startswith("  1. root-p1-l2  passed  ")


def test_the_rank_summary_states_a_failed_children_reason_once() -> None:
    """A failed child's line read `no result: no result (...)`."""
    text = parallel_join.rank_summary(
        [
            child_records.ChildJoin(
                "root-p1-l1",
                "agent6/root-p1-l1",
                "failed",
                "",
                "no result (failed · crashed); no branch (the run made no commit)",
            )
        ]
    )

    assert text.splitlines() == [
        "0 of 1 child(ren) ranked:",
        "  root-p1-l1: no result (failed · crashed); no branch (the run made no commit)",
    ]
