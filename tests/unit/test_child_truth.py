# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read verification, task completion and merges from a real detached child's run."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import paths
from agent6.config import layer
from agent6.config import model as config_model
from agent6.graph import storage
from agent6.harness.verify import child as child_verify
from agent6.sessions import child_dispatch, layout
from agent6.ui.cli import entry
from tests import launch_env
from tests.unit import test_children_cli as children_cli


def _run(
    tmp: pathlib.Path, monkeypatch: pytest.MonkeyPatch, control: str, *, when: str = "never"
) -> pathlib.Path:
    repo = tmp / "repo"
    repo.mkdir()
    git = children_cli._git
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-qm", "base")
    monkeypatch.chdir(repo)

    def reply(body: dict[str, Any]) -> launch_env.Reply:
        turn = children_cli._turn(body)
        if turn == 0:
            return [("apply_edit", {"path": "answer", "old_string": "", "new_string": "done\n"})]
        if control == "model" and turn == 1:
            return children_cli._commit()
        if turn == (2 if control == "model" else 1):
            return [("run_verify_command", {})]
        return [("finish_session", {"summary": "wrote answer"})]

    with launch_env.scripted_provider(reply_for=reply) as (base_url, _calls):
        cfg = config_model.Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": control, "worktree": {"base_dir": str(tmp / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"verify": {"command": ["test", "-f", "answer"], "when": when}},
                "review": {"when": "never"},
            }
        )
        config = tmp / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        entry.cli_main(
            ["--config", str(config), "run", "--auto-approve", "--parallel", "1", "write answer"]
        )
    runs = paths.state_dir(repo) / "sessions" / "runs"
    return next(p for p in runs.iterdir() if (p / "dispatch").is_dir())


def test_parent_names_the_child_that_verified_its_exact_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The parent ended unverified despite its child's full pass on the same content."""
    parent = _run(tmp_path, monkeypatch, "agent6")
    (record,) = child_dispatch.read_records(parent)
    child = record.children[0].session_id
    capsys.readouterr()
    assert entry.cli_main(["sessions", "show", parent.name, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verification"] == "passed"
    assert entry.cli_main(["sessions", "show", parent.name]) == 0
    assert f"verified by child {child}" in capsys.readouterr().out
    assert entry.cli_main(["sessions", "receipt", parent.name]) == 0
    receipt = capsys.readouterr().out
    assert "Final tree passed" in receipt and f"verified by child {child}" in receipt
    assert "No verification recorded" not in receipt
    assert entry.cli_main(["sessions", "receipt", parent.name, "--json"]) == 0
    parent_receipt = json.loads(capsys.readouterr().out)
    assert not parent_receipt["verifications"] and parent_receipt["final_verify_id"] is None
    proof = parent_receipt["child_verification"]
    assert (proof["session_id"], proof["tree"]) == (child, parent_receipt["end_tree"])
    assert entry.cli_main(["sessions", "receipt", child, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["final_verify_id"] == proof["verify_id"]

    # Even with the same imported ref and end word, only the actual full gate can certify.
    command, tree = tuple(proof["command"]), proof["tree"]
    assert child_verify.covering_pass(parent, tree, ("another-gate",)) is None
    assert child_verify.covering_pass(parent, "another-tree", command) is None
    logs = parent.parent / child / "logs.jsonl"
    events = [json.loads(line) for line in logs.read_text().splitlines()]
    changes: dict[str, Any]
    for changes in ({"scoped": True}, {"tree_before": "changed"}, {"verify_id": None}):
        logs.write_text(
            "".join(
                json.dumps(row | changes if row["type"] == "verify.end" else row) + "\n"
                for row in events
            )
        )
        assert child_verify.covering_pass(parent, tree, command) is None, changes


def test_a_child_pass_does_not_cancel_the_parents_finish_gate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A child's result cannot stand in for a gate the parent must run."""
    parent = _run(tmp_path, monkeypatch, "agent6", when="finish")
    capsys.readouterr()
    assert entry.cli_main(["sessions", "receipt", parent.name, "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["final_verify_id"]
    assert receipt["child_verification"] is None


def test_model_git_task_waits_for_a_mark_with_its_retained_ref(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A retained ref supplies work to inspect, not a task verdict."""
    parent = _run(tmp_path, monkeypatch, "model")
    (record,) = child_dispatch.read_records(parent)
    merge = record.children[0].merge
    assert (
        merge and children_cli._git(pathlib.Path.cwd(), "show", f"{merge.branch}:answer") == "done"
    )
    node = storage.load_graph(layout.layout_of(parent))[record.node_id or ""]
    assert node.status == "pending" and merge.branch in node.notes and not node.commit_sha
    assert entry.cli_main(["sessions", "show", parent.name]) == 0
    out = capsys.readouterr().out
    assert "no_child_result" not in out and f"retained on {merge.branch}" in out
    assert "1 open task(s): write answer" in out
    assert not (pathlib.Path.cwd() / "answer").exists()
    assert entry.cli_main(["sessions", "show", parent.name, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["stop_reason"] == "children_ranked"
    assert shown["verification"] == "unverified" and shown["child_verification"] is None
    assert entry.cli_main(["sessions", "show", record.children[0].session_id]) == 0
    child = capsys.readouterr().out
    assert merge.branch in child and " · unmerged" not in child


def test_show_distinguishes_a_parent_chain_merge_from_a_checkout_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A merge into the parent chain alone must not read as a merge into the checkout."""
    parent = _run(tmp_path, monkeypatch, "agent6")
    (record,) = child_dispatch.read_records(parent)
    child = record.children[0].session_id
    assert children_cli._git(pathlib.Path.cwd(), "ls-tree", "--name-only", "HEAD") == "README.md"
    merge = record.children[0].merge
    assert merge and children_cli._git(pathlib.Path.cwd(), "show", f"{merge.sha}:answer") == "done"
    capsys.readouterr()
    pages: list[str] = []
    for sid in (parent.name, child):
        assert entry.cli_main(["sessions", "show", sid]) == 0
        pages.append(capsys.readouterr().out)
    assert all(f"merged into {parent.name}'s chain at {merge.sha[:12]}" in page for page in pages)
    assert (
        "checkout:   no merge recorded" in pages[0] and f"sessions merge {parent.name}" in pages[0]
    )
    assert f"parent merge: merged into {parent.name}'s chain at {merge.sha[:12]}" in pages[1]
    assert (
        "checkout:   no direct merge recorded" in pages[1]
        and f"sessions show {parent.name}" in pages[1]
    )
    assert f"sessions merge {child}" not in pages[1] and " · unmerged" not in pages[1]
    assert entry.cli_main(["sessions", "show", child, "--json"]) == 0
    own = json.loads(capsys.readouterr().out)
    assert (own["merged_into_parent"], own["parent_merge_sha"], own["merged_into_checkout"]) == (
        parent.name,
        merge.sha,
        None,
    )

    label = f"passed · merged into {parent.name}'s chain"
    assert own["status_label"] == label
    assert entry.cli_main(["attach", child, "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["status_label"] == label
    assert snapshot["unmerged"] is False
    assert (snapshot["merged_into_parent"], snapshot["parent_merge_sha"]) == (
        parent.name,
        merge.sha,
    )
    assert entry.cli_main(["sessions", "merge", parent.name]) == 0
    assert "unmerged runs that changed" not in capsys.readouterr().out

    # A legacy checkout stamp can name the destination without recording a tip.
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["merged"].pop("tip")
    manifest_path.write_text(json.dumps(manifest))
    assert entry.cli_main(["attach", parent.name, "--json"]) == 0
    legacy = json.loads(capsys.readouterr().out)
    assert legacy["merged_into_checkout"] == "main" and legacy["unmerged"] is False

    # A commit on the imported branch is not part of the earlier parent merge.
    repo = pathlib.Path.cwd()
    git = children_cli._git
    git(repo, "switch", merge.branch)
    git(repo, "commit", "--allow-empty", "-qm", "Extend answer")
    git(repo, "switch", "main")
    assert entry.cli_main(["attach", child, "--json"]) == 0
    advanced = json.loads(capsys.readouterr().out)
    assert advanced["unmerged"] is True and advanced["status_label"] == "passed · unmerged"
    assert advanced["parent_merge_sha"] == merge.sha
    assert entry.cli_main(["sessions", "show", child]) == 0
    assert f"merge with: agent6 sessions merge {child}" in capsys.readouterr().out


def test_show_keeps_the_parent_chain_and_checkout_destinations_for_one_child(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Merging a child directly into main does not overwrite its parent-chain merge."""
    parent = _run(tmp_path, monkeypatch, "agent6")
    (record,) = child_dispatch.read_records(parent)
    child = record.children[0].session_id
    repo = pathlib.Path.cwd()
    parent_tip = children_cli._git(repo, "rev-parse", f"refs/agent6/{parent.name}/head")
    first = record.children[0].merge
    assert first
    git = children_cli._git
    checkout = tmp_path / "child-branch"
    git(repo, "worktree", "add", str(checkout), first.branch)
    (checkout / "answer").write_text("extended\n")
    git(checkout, "commit", "-qam", "Extend answer")
    assert entry.cli_main(["sessions", "merge", child, "--into", "main", "--method", "squash"]) == 0
    assert children_cli._git(repo, "show", "main:answer") == "extended"
    capsys.readouterr()
    assert entry.cli_main(["sessions", "show", child, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["merged_into_parent"] == parent.name
    assert shown["parent_merge_sha"] == parent_tip
    assert shown["merged_into_checkout"] == "main"
    assert shown["status_label"] == "passed · merged into main"
    assert entry.cli_main(["attach", child, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status_label"] == shown["status_label"]
