# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Attempts at one task are ranked by reviewer children in the background, or mechanically."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from agent6 import budget, child_records, directive, kinds, paths
from agent6.app import children as app_children
from agent6.app import compare as app_compare
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError
from agent6.harness import (
    child_work,
    loop_conversation,
    loop_state,
    parallel_dispatch,
    parallel_join,
    parallel_rank,
    parallel_recovery,
    run_chain,
)
from agent6.sessions import child_dispatch, completion, event_log
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read

ONE, TWO, THREE = "root-p1-l1", "root-p1-l2", "root-p1-l3"
JUDGE = "root-rp1-l1"


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repository whose three attempt branches each rewrite the same file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    for child in (1, 2, 3):
        _git(root, "checkout", "-q", "-b", f"agent6/root-p1-l{child}", "main")
        (root / "a.txt").write_text(f"child {child}\n", encoding="utf-8")
        _git(root, "commit", "-qam", f"child {child}")
    _git(root, "checkout", "-q", "main")
    return root


def _brief(
    sid: str, verification: kinds.Verification | None = "passed", cost: float = 0.0, **kw: Any
) -> child_records.CandidateBrief:
    kw.setdefault("diff", "")
    return child_records.CandidateBrief(
        session_id=sid, task="rewrite a.txt", verification=verification, cost_usd=cost, **kw
    )


def _said(text: str, **kw: Any) -> completion.CompletionRecord:
    return completion.CompletionRecord(
        text=text, stop_reason="answered", output_tokens=1, calls=1, **kw
    )


# ---- the pure half: the mechanical order, a judge's reply, the fold, the winner ------------------


def test_the_mechanical_order_is_verify_pass_first_then_the_cheaper_stable_in_ties() -> None:
    briefs = [
        _brief("failed-cheap", "failed", 0.01),
        _brief("passed-dear", "passed", 0.50),
        _brief("unknown", None, 0.02),
        _brief("passed-cheap", "passed", 0.10),
    ]
    assert parallel_rank.mechanical_ranking(briefs) == (
        "passed-cheap",
        "passed-dear",
        "failed-cheap",
        "unknown",
    )
    tied = [_brief("first"), _brief("second"), _brief("third")]
    assert parallel_rank.mechanical_ranking(tied) == ("first", "second", "third")


def test_a_judges_reply_is_an_order_of_every_attempt_with_its_reason() -> None:
    """The verdict is JSON with `order` and `summary`; prose or a fence around it is read past."""
    ids = ("a", "b")
    plain = _said('{"order": ["b", "a"], "summary": "b is  tidier\\nand passes"}')
    assert parallel_rank.reply_of("reviewer", plain, ids, max_turns=20) == parallel_rank.JudgeReply(
        "reviewer", ("b", "a"), "b is tidier and passes"
    )
    fenced = _said('I compared them.\n```json\n{"order": ["a", "b"], "summary": "a"}\n```')
    assert parallel_rank.reply_of("reviewer", fenced, ids, max_turns=20).order == ("a", "b")


@pytest.mark.parametrize(
    ("record", "why"),
    [
        (None, "it left no verdict"),
        (_said('{"order": ["a", "ghost"], "summary": "x"}'), "did not name every attempt"),
        (_said('{"order": ["a"], "summary": "x"}'), "did not name every attempt"),
        (_said('{"order": ["a", "a"], "summary": "x"}'), "did not name every attempt"),
        (_said("still reading", asked_for_tools=True), "no verdict within 20 replies"),
        (_said("I prefer the second one."), "unparseable reviewer output"),
        (
            _said("").model_copy(
                update={"stop_reason": "provider_error", "diagnostics": ("boom",)}
            ),
            "boom",
        ),
    ],
)
def test_a_reply_with_no_usable_order_says_why(
    record: completion.CompletionRecord | None, why: str
) -> None:
    reply = parallel_rank.reply_of("reviewer", record, ("a", "b"), max_turns=20)
    assert reply.order is None and why in reply.said
    failed = parallel_rank.reply_of("reviewer", record, ("a", "b"), max_turns=20, error="it died")
    assert (failed.order, failed.said) == (None, "it died")


def test_one_order_ranks_several_are_summed_and_a_tie_goes_to_the_mechanical_order() -> None:
    briefs = [_brief("a", cost=0.3), _brief("b", cost=0.2), _brief("c", cost=0.1)]
    reply = parallel_rank.JudgeReply
    single = parallel_rank.combined(briefs, [reply("reviewer", ("b", "c", "a"), "b reads best")])
    assert (single.order, single.ranked_by, single.rationale, single.fallback) == (
        ("b", "c", "a"),
        "judge",
        "b reads best",
        "",
    )
    summed = parallel_rank.combined(
        briefs, [reply("x", ("a", "b", "c"), "a"), reply("y", ("b", "a", "c"), "b")]
    )
    # a and b tie at one place each: the cheaper, b, is first.
    assert summed.order == ("b", "a", "c")
    assert summed.rationale == "lowest sum of ranks over 2 judges\n  x: a\n  y: b"
    partly = parallel_rank.combined(
        briefs, [reply("x", None, "no verdict within 20 replies"), reply("y", ("a", "b", "c"), "a")]
    )
    assert partly.order == ("a", "b", "c") and partly.ranked_by == "judge"
    assert partly.rationale == "a", "the judges' words are the judges' that ranked"


def test_with_no_usable_order_the_mechanical_order_ranks_and_says_why() -> None:
    briefs = [_brief("a", cost=0.3), _brief("b", cost=0.2)]
    none = parallel_rank.combined(
        briefs, [parallel_rank.JudgeReply("reviewer", None, "no verdict within 20 replies")]
    )
    assert (none.order, none.ranked_by, none.rationale) == (("b", "a"), "mechanical", "")
    assert none.fallback == (
        "no judge gave a usable order (reviewer: no verdict within 20 replies)"
    )
    unasked = parallel_rank.combined(briefs, [], unasked="no judge is configured")
    assert (unasked.order, unasked.fallback) == (("b", "a"), "no judge is configured")


@pytest.mark.parametrize(
    ("gates", "order", "harness_gate", "crowned"),
    [
        (("passed", "failed"), ("a", "b"), True, "a"),
        (("passed", "failed"), ("b", "a"), True, ""),
        (("failed", "failed"), ("a", "b"), True, ""),
        (("unverified", "unverified"), ("a", "b"), True, ""),
        (("unverified", "unverified"), ("a", "b"), False, "a"),
        (("not_applicable", "not_applicable"), ("b", "a"), True, "b"),
        (("failed", "unverified"), ("b", "a"), False, ""),
    ],
)
def test_the_winner_is_a_verified_attempt_or_nobody(
    gates: tuple[kinds.Verification, kinds.Verification],
    order: tuple[str, ...],
    harness_gate: bool,
    crowned: str,
) -> None:
    """A judge's order is model output: it cannot crown a red attempt over a green one."""
    briefs = [_brief(sid, gate) for sid, gate in zip(("a", "b"), gates, strict=True)]
    assert parallel_rank.winner(briefs, order, harness_gate=harness_gate) == crowned
    assert parallel_rank.winner(briefs, (), harness_gate=harness_gate) == ""


def test_a_judges_opening_lists_each_attempt_with_its_verify_result_cost_and_diff() -> None:
    briefs = [
        _brief(
            "run-p1-l1",
            "passed",
            0.0312,
            ref="agent6/run-p1-l1",
            tip="1" * 40,
            summary="rewrote it\nin one pass",
            diff="+one\n",
        ),
        _brief(
            "run-p1-l2", None, 0.5, ref="refs/agent6/run-p1-l2", tip="2" * 40, diff="x" * 70_000
        ),
    ]
    text = parallel_rank.opening(briefs, base="b" * 40)
    assert text.startswith("TASK:\nrewrite a.txt\n\nBASE: bbbbbbbbbbbb (your checkout)\n\n")
    assert (
        "ATTEMPT run-p1-l1\n"
        "branch: origin/agent6/run-p1-l1 at 111111111111\n"
        "verify: passed\n"
        "cost: $0.0312\n"
        "summary: rewrote it\n  in one pass\n"
        "DIFF:\n+one\n"
    ) in text
    # A ref that is no branch is not in a clone by name: the tip alone names it.
    assert (
        "ATTEMPT run-p1-l2\nbranch: 222222222222\nverify: unknown\ncost: $0.5000\nDIFF:\n" in text
    )
    assert text.endswith(
        "\n[diff cut at 60,000 characters; `git diff bbbbbbbbbbbb 222222222222` shows all of it]"
    )
    assert "x" * parallel_rank.DIFF_CAP + "\n[diff cut" in text
    # Runs picked by hand may not share a task or a base: each then gives its own.
    mixed = [_brief("a"), dataclasses.replace(_brief("b"), task="another task")]
    alone = parallel_rank.opening(mixed, base="")
    assert not alone.startswith("TASK:") and "BASE" not in alone
    assert "ATTEMPT b\ntask: another task\nverify: passed" in alone


def test_a_judge_is_a_reviewer_child_with_the_tools_a_reviewer_is_handed() -> None:
    """One contract per judge: the rank prompt, the reply cap, the reviewer's share of the tools."""
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {
                "default": {"model": "p/base"},
                "reviewer": {"model": "p/critic", "temperature": 0.2},
            },
            "harness": {"parallel": {"judges": ["2", "p/second", "helper"]}},
        }
    )
    judges = parallel_rank.configured_judges(cfg, log=lambda _m: None)
    assert [(j.selector, j.settings.route.model) for j in judges] == [
        ("reviewer", "critic"),
        ("reviewer", "critic"),
        ("p/second", "second"),
        ("helper", "base"),
    ]
    offered = (
        "apply_edit",
        "finish_session",
        "parallel",
        "read_file",
        "run_command",
        "steer_child",
    )
    tasks = parallel_rank.judge_tasks(
        judges, [_brief("a"), _brief("b")], base="", max_turns=7, tools=offered
    )
    assert [(t.model, t.settings) for t in tasks] == [(j.selector, j.settings) for j in judges]
    contract = tasks[0].contract
    assert contract is not None and contract.review_binding is None
    assert (contract.max_turns, contract.temperature) == (7, 0.2)
    assert contract.allowed_tools == ("read_file", "run_command")
    assert contract.opening == parallel_rank.opening([_brief("a"), _brief("b")], base="")
    assert "ranking attempts at the SAME task" in contract.system
    assert "You have at most 7 replies" in contract.system
    assert '{"order": ["<attempt id>", "..."]' in contract.system
    assert "`git diff <base> <tip>`" in contract.system
    quiet = parallel_rank.judge_tasks(judges[:1], [_brief("a")], base="", max_turns=7, tools=())
    assert quiet[0].contract is not None and "git diff" not in quiet[0].contract.system
    # A host with no model of its own hands down the whole reviewer surface.
    whole = parallel_rank.judge_tasks(judges[:1], [_brief("a")], base="", max_turns=7, tools=None)
    assert whole[0].contract is not None
    assert "run_command" in whole[0].contract.allowed_tools
    assert "apply_edit" not in whole[0].contract.allowed_tools
    # A role with no route, and no judge at all, rank nothing.
    bare = Config.model_validate({"harness": {"parallel": {"judges": []}}})
    assert parallel_rank.configured_judges(bare, log=lambda _m: None) == ()
    assert parallel_rank.configured_judges(Config(), log=lambda _m: None) == ()


def test_scheduling_checks_the_whole_roster_against_the_current_limit() -> None:
    """A config copy with a lower limit must not schedule a previously valid judge roster."""
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai"}},
            "models": {"default": {"model": "p/base"}},
            "harness": {"parallel": {"judges": ["2", "2"]}},
        }
    )
    parallel = cfg.harness.parallel.model_copy(update={"max_children": 3})
    narrowed = cfg.model_copy(
        update={"harness": cfg.harness.model_copy(update={"parallel": parallel})}
    )
    with pytest.raises(ConfigError, match=r"4 children.*max_children = 3"):
        parallel_rank.configured_judges(narrowed, log=lambda _m: None)


# ---- the dispatcher: judges in the background of the run -----------------------------------------


class _Run:
    """A group spawner over the fixture's branches: attempts end at once, judges when told."""

    def __init__(self, repo: pathlib.Path, state_dir: pathlib.Path) -> None:
        self.repo = repo
        self.state_dir = state_dir
        self.calls: list[tuple[str, list[child_records.ChildTask]]] = []
        self.judges: dict[str, concurrent.futures.Future[child_records.ChildResult]] = {}
        self.stopped: list[str] = []
        self.gates: dict[int, kinds.Verification | None] = {}
        self.stopped_by: dict[int, str] = {}

    def __call__(
        self, children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        self.calls.append((group, list(children)))
        started: list[child_records.StartedChild] = []
        for number, _child in enumerate(children, start=1):
            sid = f"root-{group}-l{number}"
            result: concurrent.futures.Future[child_records.ChildResult] = (
                concurrent.futures.Future()
            )
            if group.startswith("r"):
                self.judges[sid] = result
            else:
                result.set_result(self.attempt(number))
            started.append(child_records.StartedChild(sid, result, lambda sid=sid: self.stop(sid)))
        return started

    def attempt(self, number: int) -> child_records.ChildResult:
        sid = f"root-p1-l{number}"
        branch = f"agent6/{sid}"
        return child_records.ChildResult(
            spec=child_records.ChildSpec(number, sid, pathlib.Path("/nowhere"), None),
            session_dir=pathlib.Path("/nowhere"),
            branch=branch,
            ok=True,
            error="",
            imported_ref=branch,
            imported_tip=_git(self.repo, "rev-parse", branch),
            has_commits=True,
            commits=1,
            verification=self.gates.get(number, "passed"),
            usage=budget.UsageSettlement(usd=0.25, usd_partial=False, tokens=0),
            report=f"child {number} finished",
            stopped_by=self.stopped_by.get(number, ""),
        )

    def judge_ends(self, sid: str, text: str, *, asked_for_tools: bool = False) -> None:
        """End a judge with this last reply, as its session would record it."""
        session_dir = sessions_layout.bucket_dir(self.state_dir, "reviewers") / sid
        session_dir.mkdir(parents=True)
        completion.write_completion(
            session_dir,
            completion.CompletionRecord(
                text=text,
                stop_reason="max_turns" if asked_for_tools else "answered",
                output_tokens=1,
                calls=1,
                session_id=sid,
                asked_for_tools=asked_for_tools,
            ),
        )
        self.judges[sid].set_result(
            child_records.ChildResult(
                spec=child_records.ChildSpec(1, sid, session_dir, None, mode="reviewer"),
                session_dir=session_dir,
                branch="",
                ok=not asked_for_tools,
                error="no result (max turns)" if asked_for_tools else "",
                usage=budget.UsageSettlement(usd=0.1, usd_partial=False, tokens=0),
            )
        )

    def stop(self, sid: str) -> None:
        self.stopped.append(sid)
        if sid in self.judges and not self.judges[sid].done():
            self.judges[sid].set_result(
                child_records.ChildResult(
                    spec=child_records.ChildSpec(1, sid, pathlib.Path("/nowhere"), None),
                    session_dir=pathlib.Path("/nowhere"),
                    branch="",
                    ok=False,
                    error="no result (stopped)",
                    usage=budget.UsageSettlement(usd=0.05, usd_partial=False, tokens=0),
                )
            )


def _judge(selector: str = "reviewer") -> parallel_rank.Judge:
    return parallel_rank.Judge(selector, kinds.ModelSettings(kinds.ModelRoute("p", "critic")))


def _dispatcher(
    repo: pathlib.Path,
    run: _Run,
    events: list[Any],
    *,
    judges: Sequence[parallel_rank.Judge] = (),
    recorder: child_records.RankRecorder | None = None,
) -> parallel_dispatch.ParallelDispatcher:
    session_dir = sessions_layout.bucket_dir(run.state_dir, "runs") / "root"
    session_dir.mkdir(parents=True, exist_ok=True)
    return parallel_dispatch.ParallelDispatcher(
        chain=run_chain.RunChain(
            repo,
            ref=git_read.chain_ref_for("root"),
            fallback_parent=_git(repo, "rev-parse", "main"),
        ),
        curator=None,
        max_children=4,
        child_spawner=run,
        save_snapshot=lambda *_a, **_k: None,
        log=lambda _m: None,
        emit=events.append,
        emit_graph_snapshot=lambda: None,
        session_id="root",
        execution_id="exec-1",
        session_dir=session_dir,
        budget=budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1),
        judges=judges,
        reviewer_turns=20,
        reviewer_tools=lambda: ("read_file", "run_command", "apply_edit"),
        rank_recorder=recorder,
    )


def _start(
    dispatcher: parallel_dispatch.ParallelDispatcher, spec: str = "3"
) -> loop_conversation.Conversation:
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)
    dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(spec=spec, task="rewrite a.txt")]
    )
    return conversation


def _records(run: _Run) -> dict[str, child_dispatch.DispatchRecord]:
    session_dir = sessions_layout.bucket_dir(run.state_dir, "runs") / "root"
    return {r.identity: r for r in child_dispatch.read_records(session_dir)}


def test_attempts_are_ranked_by_a_reviewer_in_the_background_and_merge_after_it_ends(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The judges start when the attempts have ended; the task merges a boundary after they end.

    The ranking was one model call made inside the merge, with no tools, and the parent's
    turn waited for it.
    """
    run = _Run(repo, tmp_path / "state")
    events: list[Any] = []
    recorded: list[tuple[list[str], child_records.Ranking, str]] = []

    def recorder(
        candidates: Sequence[child_records.CandidateBrief],
        results: Sequence[child_records.ChildResult],
        ranking: child_records.Ranking,
        *,
        group: str,
    ) -> None:
        del results
        recorded.append(([c.session_id for c in candidates], ranking, group))

    dispatcher = _dispatcher(repo, run, events, judges=(_judge(),), recorder=recorder)
    conversation = _start(dispatcher)

    assert dispatcher.settle(conversation, 2), "the attempts' ends are told"

    # The judges are reviewer children of the run, cut at the attempts' base, in a group of theirs.
    (attempts, (group, (judge,))) = run.calls
    assert (attempts[0], group) == ("p1", "rp1")
    assert judge.contract is not None and judge.contract.allowed_tools == (
        "read_file",
        "run_command",
    )
    assert judge.limits is not None and judge.limits.max_turns == 20
    for sid in (ONE, TWO, THREE):
        assert f"ATTEMPT {sid}\nbranch: origin/agent6/{sid} at " in judge.contract.opening
    assert "+child 2" in judge.contract.opening and "verify: passed" in judge.contract.opening
    # Meanwhile nothing has merged, the task reads as being ranked, and it holds the end.
    records = _records(run)
    assert (records["root-p1"].state, records["root-p1"].ranking) == ("ranking", None)
    assert (records["root-rp1"].source, records["root-rp1"].delivered) == ("panel", True)
    roster = child_work.roster(dispatcher)
    assert [(line.id, line.state, line.running) for line in roster] == [
        (sid, "being ranked", True) for sid in (ONE, TWO, THREE)
    ]
    assert child_work.unmerged(dispatcher) == tuple(
        (sid, "being ranked") for sid in (ONE, TWO, THREE)
    )
    assert dispatcher.running() == 1 and git_read.chain_tip(repo, "refs/agent6/root") is None
    assert not dispatcher.settle(conversation, 3), "a boundary while the judge runs changes nothing"
    assert len(run.calls) == 2, "the judges are started once"

    run.judge_ends(JUDGE, json.dumps({"order": [TWO, THREE, ONE], "summary": "two is tidier"}))
    assert dispatcher.settle(conversation, 4)

    assert [(j.session_id, j.status, j.detail) for j in dispatcher.joins("root-p1")] == [
        (ONE, "ranked", "3 of 3"),
        (TWO, "joined", ""),
        (THREE, "ranked", "2 of 3"),
    ]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "child 2\n"
    records = _records(run)
    assert records["root-p1"].ranking == child_dispatch.RankRecord(
        order=(TWO, THREE, ONE), ranked_by="judge", rationale="two is tidier", winner=TWO
    )
    assert (records["root-rp1"].state, records["root-rp1"].children[0].settled) == (
        "complete",
        True,
    )
    said = json.dumps(conversation.to_wire())
    assert "Ranked by the judges: two is tidier\\nReview what merged and continue." in said
    assert "root-rp1" not in said, "a judge's own record is never a notice"
    assert dispatcher.open == {} and dispatcher.running() == 0
    # The judge's spend is the run's: three attempts and one reviewer.
    assert dispatcher.budget is not None
    assert dispatcher.budget.estimate_usd()[0] == pytest.approx(0.85)
    ((briefed, ranking, group),) = recorded
    assert (briefed, group) == ([ONE, TWO, THREE], "root-p1")
    assert (ranking.winner, ranking.judge_cost_usd, ranking.judge_cost_partial) == (TWO, 0.1, False)


def test_a_judge_with_no_usable_order_leaves_the_mechanical_order_and_the_notice_says_so(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A judge that spends its last reply on a tool call ranks nothing: verify, then cost."""
    run = _Run(repo, tmp_path / "state")
    run.gates = {1: "failed"}
    dispatcher = _dispatcher(repo, run, [], judges=(_judge(),))
    conversation = _start(dispatcher)
    dispatcher.settle(conversation, 2)
    run.judge_ends(JUDGE, "let me read one more file", asked_for_tools=True)

    assert dispatcher.settle(conversation, 3)

    ranking = _records(run)["root-p1"].ranking
    assert ranking is not None
    assert (ranking.order, ranking.ranked_by, ranking.winner) == (
        (TWO, THREE, ONE),
        "mechanical",
        TWO,
    )
    assert ranking.fallback == (
        "no judge gave a usable order (reviewer: no verdict within 20 replies)"
    )
    assert (
        "Ranked by verify result, then cost: no judge gave a usable order (reviewer: no verdict"
        " within 20 replies).\\nReview what merged and continue."
    ) in json.dumps(conversation.to_wire())
    assert (repo / "a.txt").read_text(encoding="utf-8") == "child 2\n"


def test_without_a_judge_or_with_one_candidate_the_mechanical_order_ranks_at_once(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """No reviewer is started where there is nothing for one to decide."""
    run = _Run(repo, tmp_path / "state")
    dispatcher = _dispatcher(repo, run, [])
    conversation = _start(dispatcher, "2")

    assert dispatcher.settle(conversation, 2)

    assert [group for group, _children in run.calls] == ["p1"]
    ranking = _records(run)["root-p1"].ranking
    assert ranking is not None
    assert (ranking.order, ranking.fallback) == ((ONE, TWO), "no judge is configured")
    assert "Ranked by verify result, then cost: no judge is configured." in json.dumps(
        conversation.to_wire()
    )

    alone = _Run(repo, tmp_path / "other")
    alone.stopped_by = {2: "stopped by its parent", 3: "stopped"}
    judged = _dispatcher(repo, alone, [], judges=(_judge(),))
    judged.chain = run_chain.RunChain(
        repo, ref="refs/agent6/other", fallback_parent=_git(repo, "rev-parse", "main")
    )
    conversation = _start(judged)
    assert judged.settle(conversation, 2)
    assert [group for group, _children in alone.calls] == ["p1"], "one candidate needs no judge"
    ranking = _records(alone)["root-p1"].ranking
    assert ranking is not None and (ranking.order, ranking.fallback) == ((ONE,), "one candidate")


def test_the_executions_end_stops_its_judges_and_a_resume_ranks_from_what_they_left(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Judges end with the execution that started them and are never started twice."""
    run = _Run(repo, tmp_path / "state")
    dispatcher = _dispatcher(repo, run, [], judges=(_judge(), _judge("p/second")))
    conversation = _start(dispatcher, "2")
    dispatcher.settle(conversation, 2)
    first, second = "root-rp1-l1", "root-rp1-l2"
    run.judge_ends(first, json.dumps({"order": [TWO, ONE], "summary": "two"}))

    stopped, ended = dispatcher.close(3)

    assert (stopped, ended) == ((), (ONE, TWO)) and run.stopped == [first, second]
    assert dispatcher.budget is not None
    assert dispatcher.budget.estimate_usd()[0] == pytest.approx(0.5 + 0.1 + 0.05)
    records = _records(run)
    # Both are charged and neither is complete: no judge runs, so the task is not being ranked.
    assert (records["root-p1"].state, records["root-rp1"].state) == ("settling", "settling")

    attempts = {res.spec.session_id: res for res in (run.attempt(1), run.attempt(2))}
    judged = {sid: run.judges[sid].result() for sid in (first, second)}

    def recoverer(group: str, children: Sequence[str]) -> list[child_records.ChildResult]:
        del group
        return [{**attempts, **judged}[sid] for sid in children]

    again = _Run(repo, tmp_path / "state")
    resumed = _dispatcher(repo, again, [], judges=(_judge(), _judge("p/second")))
    resumed.child_recoverer = recoverer
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="t", execution_id="exec-2", tool_calls=0)

    parallel_recovery.reconcile(resumed, state, conversation, turn=3)

    assert again.calls == [], "no judge is started for a task an earlier execution's judges ranked"
    records = _records(again)
    ranking = records["root-p1"].ranking
    assert ranking is not None and (ranking.order, ranking.ranked_by) == ((TWO, ONE), "judge")
    assert ranking.rationale == "two", "the judge that was stopped ranked nothing"
    assert (records["root-p1"].state, records["root-rp1"].state) == ("complete", "complete")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "child 2\n"
    said = json.dumps(conversation.to_wire())
    assert "[parallel] group root-p1 complete" in said and "root-rp1" not in said


def test_a_task_with_one_child_merges_without_a_ranking(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    run = _Run(repo, tmp_path / "state")
    dispatcher = _dispatcher(repo, run, [], judges=(_judge(),))
    conversation = _start(dispatcher, "")

    assert dispatcher.settle(conversation, 2)

    record = _records(run)["root-p1"]
    assert (record.join_policy, record.ranking, record.state) == ("merge", None, "complete")
    assert "Ranked by" not in json.dumps(conversation.to_wire())


def test_a_ranked_child_reads_as_its_place_in_the_task_graph_note() -> None:
    """The DAG note says where the child placed, not that it failed."""
    join = child_records.ChildJoin(ONE, f"agent6/{ONE}", "ranked", "", "3 of 3")
    assert parallel_join.child_note(join) == f"{ONE} ranked 3 of 3; not joined"


def test_the_rank_summary_states_a_failed_childs_reason_once() -> None:
    """A failed child's line read `no result: no result (...)`."""
    reason = "no result (failed · crashed); no branch (the run made no commit)"
    text = parallel_join.rank_summary(
        [child_records.ChildJoin(ONE, f"agent6/{ONE}", "failed", "", reason)]
    )
    assert text.splitlines() == ["0 of 1 child(ren) ranked:", f"  {ONE}: {reason}"]


# ---- the app's side: the ranking's record outside the dispatch record ----------------------------


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


def test_the_apps_recorder_stamps_journals_and_prints_a_ranking_the_harness_made(
    repo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order reaches the manifests, the journal and the operator's table, once each."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    origin_state = paths.state_dir(repo)
    root = _child_record(repo, "root", "not_applicable")
    run = _Run(repo, origin_state)
    results = [
        dataclasses.replace(res, session_dir=_child_record(repo, res.spec.session_id, "passed"))
        for res in (run.attempt(1), run.attempt(2))
    ]
    said: list[str] = []
    recorder = app_children.build_rank_recorder(
        repo,
        origin_state,
        parent_dir=root,
        events=event_log.EventSink(root / "logs.jsonl"),
        reporter=app_reporter.Reporter(out=said.append, err=said.append),
    )
    briefs = parallel_rank.candidates(
        repo, results, base=_git(repo, "rev-parse", "main"), task="rewrite a.txt"
    )
    assert "+child 1" in briefs[0].diff and briefs[1].verification == "passed"
    ranking = child_records.Ranking(
        order=(TWO, ONE),
        ranked_by="judge",
        rationale="two is tidier",
        winner=TWO,
        judge_cost_usd=0.1,
    )

    recorder(briefs, results, ranking, group="root-p1")

    placements = {
        sid: sessions_manifest.read_manifest(origin_state / "sessions" / "runs" / sid).placement
        for sid in (ONE, TWO)
    }
    assert placements[TWO] == sessions_manifest.PlacementStamp(
        rank=1, of=2, winner=True, group="root-p1"
    )
    assert placements[ONE] == sessions_manifest.PlacementStamp(
        rank=2, of=2, winner=False, group="root-p1"
    )
    comparison = sessions_manifest.read_manifest(root).comparison
    assert comparison is not None and comparison.ranking == (TWO, ONE)
    assert (comparison.ranked_by, comparison.judge_cost_usd) == ("judge", 0.1)
    journaled = [json.loads(line) for line in (root / "logs.jsonl").read_text().splitlines()]
    (compared,) = [row for row in journaled if row["type"] == "parallel.compared"]
    assert [child["session_id"] for child in compared["ranking"]] == [TWO, ONE]
    assert (compared["rationale"], compared["judge_cost_usd"]) == ("two is tidier", 0.1)
    assert said[0] == "ranked candidates (best first):"
    assert said[1].startswith(f"  1. {TWO}  passed  ")
    assert said[-1] == "\njudge: two is tidier"
    assert app_compare.outcome_of(ranking).ranking == (TWO, ONE)
