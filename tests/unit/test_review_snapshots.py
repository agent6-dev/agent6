from __future__ import annotations

import dataclasses
import pathlib
from unittest import mock

import pytest

from agent6 import paths, review_records
from agent6.harness import loop_state, review_gate, run_chain
from agent6.sessions import event_log, layout
from agent6.vcs import git_ops, git_read, review
from tests import review_helpers, review_lane_env
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
        settings=review_gate.ReviewSettings(reviewers=[review_lane_env.seat("s", "m", provider)]),
        chain=chain,
        hide=(),
        executor=mock.MagicMock,
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    reviewer.critique(state, trigger="periodic", turn=1)
    panel = next(row for row in rows if row["type"] == "review.panel")
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
    start = next(row for row in rows if row["type"] == "review.start")
    seat_row = next(row for row in rows if row["type"] == "review.seat")
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
            reviewers=[review_lane_env.seat("s", "m", provider)], block_on="any"
        ),
        chain=chain,
        hide=(),
        executor=mock.MagicMock,
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    first = reviewer.critique(state, trigger="periodic", turn=1)
    assert first is not None and provider.call.call_count == 1
    (panel,) = [row for row in rows if row["type"] == "review.panel"]
    assert panel["roster"] == [
        dict(seat_index=0, model_route="m", focus="s", scope="diff", temperature=0.0)
    ]
    again = reviewer.critique(state, trigger="periodic", turn=2)
    assert again is not None and provider.call.call_count == 1
    (reused,) = [row for row in rows if row["type"] == "review.reused"]
    assert reused["reused_panel_id"] == panel["panel_id"] != reused["panel_id"]
    assert reused["turn"] == 2 and reused["findings"] == 1
    assert again.text == (
        "Review panel reused: the reviewed trees are unchanged since panel"
        f" {str(panel['panel_id'])[:12]} sat at turn 1; its 1 recorded finding stands, and"
        " nothing new was raised."
    )
    assert "leak" not in again.text  # a finding reaches the worker once
    assert len([row for row in rows if row["type"] == "review.start"]) == 1
    # An end on the same trees: a fresh panel's repeat would neither reject nor inject.
    ended = reviewer.critique(state, trigger="finish", turn=3)
    assert ended is not None and ended.satisfied and provider.call.call_count == 1
    assert len([row for row in rows if row["type"] == "review.reused"]) == 2
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
        review_gate.ReviewSettings(reviewers=[review_lane_env.seat("s", "m", provider)]),
        chain,
        (),
        mock.MagicMock,
        lambda: None,
        lambda _s: None,
        sink.emit,
    )
    reviewer.critique(
        loop_state.LoopState(original_task="t", tool_calls=0), trigger="periodic", turn=1
    )
    starts = [row for row in rows if row["type"] == "review.start"]
    assert len(starts) == 1
    target = starts[0]["target"]
    assert isinstance(target, dict)
    assert target["base_commit"] is None
    assert target["base_tree"] == git_read.EMPTY_TREE
    assert starts[0]["rules_complete"] is True and starts[0]["rule_sources"] == []
    message = provider.call.call_args.kwargs["messages"][0]["content"][0]["text"]
    assert "REVIEW RULES (base tree):\nNo base rules exist." in message


@pytest.mark.parametrize("checkpoint", ["none", "earlier", "current"])
def test_session_panel_retains_cloneable_bytes_without_checkpointing(
    tmp_path: pathlib.Path, checkpoint: str
) -> None:
    """Failed-verify work survives gc without changing the index, HEAD or ordinary chain."""
    git = review_helpers.git
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for name in ("foo.py", "deleted.py", "old.py"):
        (repo / name).write_text("before\n")
    (repo / "AGENTS.md").write_text("base rule\n")
    (repo / "hidden").mkdir()
    (repo / "hidden/modified").write_text("historical modified\n")
    (repo / "hidden/deleted").write_text("historical deleted\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD").strip()
    (repo / "operator.txt").write_text("operator only\n")
    (repo / "hidden/key").write_text("hidden only\n")
    (repo / "hidden/modified").write_text("hidden worktree change\n")
    (repo / "hidden/deleted").unlink()
    chain = run_chain.RunChain(
        repo,
        ref=git_read.chain_ref_for("snapshot"),
        fallback_parent=base,
        base_sha=base,
        per_turn=checkpoint != "none",
        untracked_at_start=frozenset({"operator.txt"}),
        hidden=frozenset({"hidden"}),
        branch="agent6/snapshot",
    )
    if checkpoint == "earlier":
        (repo / "foo.py").write_text("earlier checkpoint\n")
        assert chain.commit("successful turn")
    (repo / "foo.py").write_text("reviewed\n")
    (repo / "new.py").write_text("model-created\n")
    (repo / "deleted.py").unlink()
    (repo / "old.py").rename(repo / "renamed.py")
    (repo / "AGENTS.md").write_text("changed rule\n")
    if checkpoint == "current":
        assert chain.commit("edit")
    index = (repo / ".git/index").read_bytes()
    branches = git(repo, "for-each-ref", "refs/heads/")
    tip = chain.tip()
    saved = layout.SessionLayout(paths.state_dir(repo), "snapshot")
    saved.ensure()
    sink = event_log.EventSink(saved.logs_path)
    provider = mock.Mock()

    def seat(**_kwargs: object) -> object:
        (repo / "foo.py").write_text("later parent\n")
        return test_review_gate._resp('{"verdict":"pass","summary":"ok","findings":[]}')

    provider.call.side_effect = seat
    harness = test_review_gate._wf(
        repo,
        chain=chain,
        events=sink,
        review=review_gate.ReviewSettings(reviewers=[review_lane_env.seat("s", "m", provider)]),
    )
    reviewer = dataclasses.replace(harness.reviewer, hide=(str(repo / "hidden"),))
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.last_ok = False
    state.verify.last_tail = "test failed on reviewed bytes"
    assert reviewer.critique(state, trigger="verify_fail", turn=1) is not None
    panel = state.review_panels[0]
    target = panel.target
    assert target.snapshot_commit
    assert target.snapshot_ref.startswith("refs/agent6/snapshot/review/")
    assert git_read.chain_tip(repo, target.snapshot_ref) == target.snapshot_commit
    assert target.hidden_paths == ("hidden",)
    assert (target.head_commit is not None) is (checkpoint == "current")
    if checkpoint == "earlier":
        assert git_read.read_file_at(repo, tip, "foo.py") == "earlier checkpoint\n"
        assert target.head_tree != git_read.tree_sha(repo, tip)
    assert target.base_tree == git_read.tree_sha(repo, base)
    assert chain.tip() == tip
    assert git(repo, "rev-parse", "HEAD").strip() == base
    assert git(repo, "for-each-ref", "refs/heads/") == branches
    assert (repo / ".git/index").read_bytes() == index
    later = chain.review_target()
    assert later is not None
    assert review_gate.reusable_panel((panel,), later, panel.roster) is None
    assert "base rule\n" in provider.call.call_args.kwargs["messages"][0]["content"][0]["text"]
    assert (
        "test failed on reviewed bytes"
        in provider.call.call_args.kwargs["messages"][0]["content"][0]["text"]
    )
    git(repo, "gc", "--prune=now")
    clone = tmp_path / "clone"
    git_ops.clone_repo(repo, clone)
    git_ops.fetch_branch(clone, repo, target.snapshot_ref)
    git_ops.checkout_detached(clone, target.snapshot_commit)
    assert (clone / "foo.py").read_text() == "reviewed\n"
    assert (clone / "new.py").read_text() == "model-created\n"
    assert (clone / "renamed.py").read_text() == "before\n"
    assert (clone / "AGENTS.md").read_text() == "changed rule\n"
    assert all(
        not (clone / p).exists() for p in ("deleted.py", "old.py", "operator.txt", "hidden/key")
    )
    assert (clone / "hidden/modified").read_text() == "historical modified\n"
    assert (clone / "hidden/deleted").read_text() == "historical deleted\n"
    assert git_read.read_file_at(clone, target.base_tree, "AGENTS.md") == "base rule\n"
    (repo / "foo.py").write_text("reviewed\n")
    assert reviewer.critique(state, trigger="periodic", turn=2) is not None
    assert len(state.review_panels) == 1
    assert git_read.session_refs(repo, "snapshot") == (
        ((chain.ref,) if checkpoint != "none" else ()) + (target.snapshot_ref,)
    )


@pytest.mark.parametrize(
    "unavailable",
    ["", "base-tree", "head-tree", "base-blob", "head-blob", "base-subtree", "head-subtree"],
    ids=[
        "unreachable",
        "base-tree",
        "head-tree",
        "base-blob",
        "head-blob",
        "base-subtree",
        "head-subtree",
    ],
)
@pytest.mark.parametrize("reuse", [False, True], ids=["create", "reuse"])
def test_review_session_retains_its_supplied_target_not_the_checkout(
    tmp_path: pathlib.Path, unavailable: str, reuse: bool
) -> None:
    """A revision target needs its own anchor even when neither side has a live ref."""
    git = review_helpers.git
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "foo.py").write_text("checkout\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "checkout")
    commits: list[str] = []
    (repo / "nested").mkdir()
    for text in ("base\n", "head\n"):
        (repo / "foo.py").write_text(text)
        (repo / "nested/file").write_text(f"nested {text}")
        tree = git_read.worktree_tree(repo, None, ())
        commits.append(git(repo, "commit-tree", tree, "-m", "loose").strip())
    compared = review.comparison(repo, base="..".join(commits), head="HEAD", paths=())
    (repo / "foo.py").write_text("checkout\n")
    saved = layout.SessionLayout(paths.state_dir(repo), "revision")
    saved.ensure()
    rows: list[dict[str, object]] = []
    sink = event_log.EventSink(saved.logs_path)
    sink.subscribe(rows.append)
    provider = mock.Mock()
    provider.call.return_value = test_review_gate._resp('{"verdict":"pass","findings":[]}')
    harness = test_review_gate._wf(
        repo,
        events=sink,
        review=review_gate.ReviewSettings(reviewers=[review_lane_env.seat("s", "m", provider)]),
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    if reuse:
        harness.reviewer.prepare(state, compared.target, compared.diff, panel_id="retained")
    if unavailable:
        side, kind = unavailable.split("-")
        tree = compared.target.base_tree if side == "base" else compared.target.head_tree
        suffix = {"tree": "", "blob": ":foo.py", "subtree": ":nested"}[kind]
        oid = git(repo, "rev-parse", tree + suffix).strip()
        (repo / ".git/objects" / oid[:2] / oid[2:]).unlink()
    result = harness.reviewer.review_target(state, compared.target, compared.diff)
    if unavailable:
        assert result is None and not state.review_panels
        assert any(r.get("reason") == "target_unreadable" for r in rows)
        provider.call.assert_not_called()
        return
    assert result is not None
    target = state.review_panels[0].target
    assert target.snapshot_ref
    git(repo, "gc", "--prune=now")
    clone = tmp_path / "clone"
    git_ops.clone_repo(repo, clone)
    git_ops.fetch_branch(clone, repo, target.snapshot_ref)
    git_ops.checkout_detached(clone, target.snapshot_commit)
    assert (clone / "foo.py").read_text() == "head\n"
    assert git_read.read_file_at(clone, target.base_tree, "foo.py") == "base\n"


@pytest.fixture(autouse=True)
def panel_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_gate, "run_panel", review_lane_env.run_panel)
