from __future__ import annotations

import pathlib
from unittest import mock

from agent6 import review_records
from agent6.harness import loop_state, review_gate, run_chain
from agent6.sessions import event_log
from agent6.vcs import git_read
from tests import review_helpers
from tests.unit import test_review_gate


def test_panel_records_the_chain_tree_without_mutating_git(tmp_path: pathlib.Path) -> None:
    """Review the chain commit even when the workspace changes during a seat call."""
    git = review_helpers.git
    git(tmp_path, "init", "-q")
    (tmp_path / "foo.py").write_text("x = 1\n")
    git(tmp_path, "add", "foo.py")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "operator.txt").write_text("not the run's file")
    chain = run_chain.RunChain(
        tmp_path,
        ref="refs/agent6/test/head",
        fallback_parent=base,
        base_sha=base,
        untracked_at_start=frozenset({"operator.txt"}),
    )
    (tmp_path / "foo.py").write_text("x = 2\n")
    tip = chain.commit("edit")
    index = (tmp_path / ".git" / "index").read_bytes()
    sink = event_log.EventSink(tmp_path / "journal" / "logs.jsonl")
    rows: list[dict[str, object]] = []
    sink.subscribe(rows.append)
    provider = mock.MagicMock()

    def seat(**_kwargs: object) -> object:
        (tmp_path / "foo.py").write_text("x = 3\n")
        return test_review_gate._resp('{"verdict":"pass","summary":"ok","findings":[]}')

    provider.call.side_effect = seat
    reviewer = review_gate.Reviewer(
        settings=review_gate.ReviewSettings(reviewers=[review_gate.ReviewSeat("s", "m", provider)]),
        chain=chain,
        hide=(),
        review_tools=lambda: ([], mock.MagicMock()),
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    reviewer.critique(state, trigger="periodic", turn=1)
    panel = next(row for row in rows if row["type"] == "loop.review.panel")
    target = panel.get("target")
    assert isinstance(target, dict)
    assert target["snapshot_commit"] == tip
    assert target["base_tree"] == git_read.tree_sha(tmp_path, base)
    assert target["head_tree"] == git_read.tree_sha(tmp_path, tip)
    assert git_read.read_file_at(tmp_path, target["head_tree"], "foo.py") == "x = 2\n"
    assert not git_read.tree_entry(tmp_path, target["head_tree"], "operator.txt")
    assert chain.tip() == tip
    assert git(tmp_path, "rev-parse", "HEAD").strip() == base
    assert (tmp_path / ".git" / "index").read_bytes() == index
    start = next(row for row in rows if row["type"] == "loop.review.start")
    seat_row = next(row for row in rows if row["type"] == "loop.review.seat")
    assert start["panel_id"] == panel["panel_id"] == seat_row["panel_id"]
    assert seat_row["summary"] == "ok"
    assert seat_row["findings"] == []
    assert seat_row["seat_index"] == 0
    assert panel["completion"] == "complete"


def test_a_panel_over_unchanged_trees_is_reused_not_sat_again(tmp_path: pathlib.Path) -> None:
    """The same trees, paths and seats reuse the complete panel; a changed tree sits a new one."""
    git = review_helpers.git
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "foo.py").write_text("x = 1\n")
    git(repo, "add", "foo.py")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD").strip()
    chain = run_chain.RunChain(
        repo, ref="refs/agent6/reuse/head", fallback_parent=base, base_sha=base
    )
    (repo / "foo.py").write_text("x = 2\n")
    chain.commit("edit")
    rows: list[dict[str, object]] = []
    sink = event_log.EventSink(tmp_path / "logs.jsonl")  # outside the repo: the tree stays put
    sink.subscribe(rows.append)
    provider = mock.MagicMock()
    provider.call.return_value = test_review_gate._resp(
        '{"verdict":"block","summary":"leak","findings":[{"category":"security",'
        '"severity":"block","title":"leak","detail":"a secret","location":{"path":"foo.py",'
        '"side":"head","start_line":1,"end_line":1}}]}'
    )
    reviewer = review_gate.Reviewer(
        settings=review_gate.ReviewSettings(
            reviewers=[review_gate.ReviewSeat("s", "m", provider)], block_on="any"
        ),
        chain=chain,
        hide=(),
        review_tools=lambda: ([], mock.MagicMock()),
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    first = reviewer.critique(state, trigger="periodic", turn=1)
    assert first is not None and provider.call.call_count == 1
    (panel,) = [row for row in rows if row["type"] == "loop.review.panel"]
    assert panel["roster"] == [
        dict(seat_index=0, model_route="m", focus="s", scope="diff", temperature=0.0)
    ]
    again = reviewer.critique(state, trigger="periodic", turn=2)
    assert again is not None and provider.call.call_count == 1
    (reused,) = [row for row in rows if row["type"] == "loop.review.reused"]
    assert reused["reused_panel_id"] == panel["panel_id"] != reused["panel_id"]
    assert reused["turn"] == 2 and reused["findings"] == 1
    assert again.text == (
        "Review panel reused: the reviewed trees are unchanged since panel"
        f" {str(panel['panel_id'])[:12]} sat at turn 1; its 1 recorded finding stands, and"
        " nothing new was raised."
    )
    assert "leak" not in again.text  # a finding reaches the worker once
    assert len([row for row in rows if row["type"] == "loop.review.start"]) == 1
    # An end on the same trees: a fresh panel's repeat would neither reject nor inject.
    ended = reviewer.critique(state, trigger="before_finish", turn=3)
    assert ended is not None and ended.satisfied and provider.call.call_count == 1
    assert len([row for row in rows if row["type"] == "loop.review.reused"]) == 2
    # A changed tree sits a new panel.
    (repo / "foo.py").write_text("x = 3\n")
    chain.commit("edit again")
    assert reviewer.critique(state, trigger="periodic", turn=4) is not None
    assert provider.call.call_count == 2
    # A record written before rosters were kept is never reused.
    legacy = state.review_panels[0].model_copy(update={"roster": ()})
    target = chain.review_target()
    assert target is not None
    roster = (review_records.SeatAssignment(0, "m", "s", "diff", 0.0),)
    assert review_gate.reusable_panel((legacy,), target, roster) is None


def test_unborn_panel_records_empty_base_and_no_base_rules(tmp_path: pathlib.Path) -> None:
    git = review_helpers.git
    git(tmp_path, "init", "-q")
    (tmp_path / "AGENTS.md").write_text("head-only rule")
    chain = run_chain.RunChain(
        tmp_path, ref="refs/agent6/unborn/head", fallback_parent=None, base_sha=""
    )
    chain.commit("first change")
    provider = mock.Mock()
    provider.call.return_value = test_review_gate._resp('{"verdict":"pass","findings":[]}')
    rows: list[dict[str, object]] = []
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.subscribe(rows.append)
    reviewer = review_gate.Reviewer(
        review_gate.ReviewSettings(reviewers=[review_gate.ReviewSeat("s", "m", provider)]),
        chain,
        (),
        lambda: ([], mock.Mock()),
        lambda: None,
        lambda _s: None,
        sink.emit,
    )
    reviewer.critique(
        loop_state.LoopState(original_task="t", tool_calls=0), trigger="periodic", turn=1
    )
    starts = [row for row in rows if row["type"] == "loop.review.start"]
    assert len(starts) == 1
    target = starts[0]["target"]
    assert isinstance(target, dict)
    assert target["base_commit"] is None
    assert target["base_tree"] == git_read.EMPTY_TREE
    assert starts[0]["rules_complete"] is True and starts[0]["rule_sources"] == []
    message = provider.call.call_args.kwargs["messages"][0]["content"]
    assert "REVIEW RULES (base tree):\nNo base rules exist." in message
