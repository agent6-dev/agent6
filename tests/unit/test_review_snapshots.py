from __future__ import annotations

import pathlib
from unittest import mock

from agent6 import event_log
from agent6.harness import _chain, _loop_state, _reviewer
from agent6.vcs import git_read
from tests.unit import test_review_diff_collect, test_review_gate


def test_panel_records_the_chain_tree_without_mutating_git(tmp_path: pathlib.Path) -> None:
    """Review the chain commit even when the workspace changes during a seat call."""
    git = test_review_diff_collect._git
    git(tmp_path, "init", "-q")
    (tmp_path / "foo.py").write_text("x = 1\n")
    git(tmp_path, "add", "foo.py")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "operator.txt").write_text("not the run's file")
    chain = _chain.RunChain(
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
    reviewer = _reviewer.Reviewer(
        settings=_reviewer.ReviewSettings(reviewers=[_reviewer.ReviewSeat("s", "m", provider)]),
        chain=chain,
        hide=(),
        review_tools=lambda: ([], mock.MagicMock()),
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
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


def test_unborn_panel_records_empty_base_and_no_base_rules(tmp_path: pathlib.Path) -> None:
    git = test_review_diff_collect._git
    git(tmp_path, "init", "-q")
    (tmp_path / "AGENTS.md").write_text("head-only rule")
    chain = _chain.RunChain(
        tmp_path, ref="refs/agent6/unborn/head", fallback_parent=None, base_sha=""
    )
    chain.commit("first change")
    provider = mock.Mock()
    provider.call.return_value = test_review_gate._resp('{"verdict":"pass","findings":[]}')
    rows: list[dict[str, object]] = []
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.subscribe(rows.append)
    reviewer = _reviewer.Reviewer(
        _reviewer.ReviewSettings(reviewers=[_reviewer.ReviewSeat("s", "m", provider)]),
        chain,
        (),
        lambda: ([], mock.Mock()),
        lambda: None,
        lambda _s: None,
        sink.emit,
    )
    reviewer.critique(
        _loop_state.LoopState(original_task="t", tool_calls=0), trigger="periodic", turn=1
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
