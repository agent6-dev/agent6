# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read verification, task completion and landing from a real detached child's run."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import paths
from agent6.config import Config, layer
from agent6.graph import storage
from agent6.harness.verify import child as child_verify
from agent6.sessions import child_dispatch, layout
from agent6.ui.cli import cli_main
from tests import launch_env
from tests.unit import test_children_cli as children_cli


def _run(tmp: pathlib.Path, monkeypatch: pytest.MonkeyPatch, control: str) -> pathlib.Path:
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
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": control, "worktree": {"base_dir": str(tmp / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"verify": {"command": ["test", "-f", "answer"], "when": "never"}},
                "review": {"when": "never"},
            }
        )
        config = tmp / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        cli_main(
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
    assert cli_main(["sessions", "show", parent.name, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verification"] == "passed"
    assert cli_main(["sessions", "show", parent.name]) == 0
    assert f"verified by child {child}" in capsys.readouterr().out
    assert cli_main(["sessions", "receipt", parent.name]) == 0
    receipt = capsys.readouterr().out
    assert "Final tree passed" in receipt and f"verified by child {child}" in receipt
    assert "No verification recorded" not in receipt
    assert cli_main(["sessions", "receipt", parent.name, "--json"]) == 0
    parent_receipt = json.loads(capsys.readouterr().out)
    assert not parent_receipt["verifications"] and parent_receipt["final_verify_id"] is None
    proof = parent_receipt["child_verification"]
    assert (proof["session_id"], proof["tree"]) == (child, parent_receipt["end_tree"])
    assert cli_main(["sessions", "receipt", child, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["final_verify_id"] == proof["verify_id"]

    # Even with the same imported ref and end word, only the actual full gate can certify.
    command, tree = tuple(proof["command"]), proof["tree"]
    assert child_verify.covering_pass(parent, tree, ("another-gate",)) is None
    assert child_verify.covering_pass(parent, "another-tree", command) is None
    logs = parent.parent / child / "logs.jsonl"
    events = [json.loads(line) for line in logs.read_text().splitlines()]
    for changes in ({"scoped": True}, {"tree_before": "changed"}, {"verify_id": None}):
        logs.write_text(
            "".join(
                json.dumps(row | changes if row["type"] == "verify.end" else row) + "\n"
                for row in events
            )
        )
        assert child_verify.covering_pass(parent, tree, command) is None, changes


def test_model_git_completed_task_passes_with_its_retained_ref(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The DAG failed a finished task solely because there was no parent chain."""
    parent = _run(tmp_path, monkeypatch, "model")
    (record,) = child_dispatch.read_records(parent)
    join = record.children[0].join
    assert join and children_cli._git(pathlib.Path.cwd(), "show", f"{join.branch}:answer") == "done"
    node = storage.load_graph(layout.layout_of(parent))[record.node_id or ""]
    assert node.status == "passed"
    assert cli_main(["sessions", "show", parent.name]) == 0
    out = capsys.readouterr().out
    assert "no_child_result" not in out and f"retained on {join.branch}" in out
    assert not (pathlib.Path.cwd() / "answer").exists()
    assert cli_main(["sessions", "show", parent.name, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["stop_reason"] == "children_ranked"
    assert shown["verification"] == "unverified" and shown["child_verification"] is None
    assert cli_main(["sessions", "show", record.children[0].session_id]) == 0
    child = capsys.readouterr().out
    assert join.branch in child and " · unmerged" not in child


def test_show_tells_checkout_from_child_join_even_on_the_child_itself(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parent join was labelled merged while the child's own page offered to merge it."""
    parent = _run(tmp_path, monkeypatch, "agent6")
    (record,) = child_dispatch.read_records(parent)
    child = record.children[0].session_id
    assert children_cli._git(pathlib.Path.cwd(), "ls-tree", "--name-only", "HEAD") == "README.md"
    join = record.children[0].join
    assert join and children_cli._git(pathlib.Path.cwd(), "show", f"{join.sha}:answer") == "done"
    capsys.readouterr()
    pages: list[str] = []
    for sid in (parent.name, child):
        assert cli_main(["sessions", "show", sid]) == 0
        pages.append(capsys.readouterr().out)
    assert all(f"joined {parent.name}'s chain at {join.sha[:12]}" in page for page in pages)
    assert (
        "checkout:   no merge recorded" in pages[0] and f"sessions merge {parent.name}" in pages[0]
    )
    assert (
        "checkout:   no direct merge recorded" in pages[1]
        and f"sessions show {parent.name}" in pages[1]
    )
    assert f"sessions merge {child}" not in pages[1] and " · unmerged" not in pages[1]
    assert cli_main(["sessions", "show", child, "--json"]) == 0
    own = json.loads(capsys.readouterr().out)
    assert (own["joined_to"], own["join_sha"], own["merged_into"]) == (parent.name, join.sha, None)
