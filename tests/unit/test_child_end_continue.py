"""A parent sends an ended child back with notes through the real tool and provider."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import signal
import subprocess
import sys
from typing import Any

import pytest

from agent6 import paths
from agent6.app.parallel import launch as child_launch
from agent6.config import layer
from agent6.config import model as config_model
from agent6.sessions import child_dispatch
from agent6.ui.cli import entry
from tests import launch_env
from tests.unit import test_children_background as background
from tests.unit import test_children_cli as cli


def _run_parent(args: list[str], *, child_budget: bool) -> None:
    """Cap only the first child's launch at two tokens in the budget case."""
    task = ["--auto-approve", "PARENT: send work"]
    if not child_budget:
        subprocess.run([*args, "run", *task], check=True, timeout=120)
        return
    write = child_launch.write

    def write_launch(directory: pathlib.Path, launch: child_launch.ChildLaunch) -> pathlib.Path:
        assert launch.allowance is not None
        allowance = dataclasses.replace(launch.allowance, tokens=2)
        return write(directory, launch.model_copy(update={"allowance": allowance}))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(child_launch, "write", write_launch)
        assert entry.cli_main([*args[1:], "run", "--max-tokens-fallback", "6", *task]) == 3


def _remove_evidence(ending: str, repo: pathlib.Path, record: pathlib.Path, ref: str) -> None:
    sid = record.name
    if ending == "missing manifest":
        (record / "manifest.json").unlink()
    if ending == "missing snapshot":
        (record / "loop_state.json").unlink()
    if ending == "rewound ref":
        cli._git(repo, "update-ref", ref, cli._git(repo, "rev-parse", f"{ref}^"))
    if ending == "missing ref":
        cli._git(repo, "update-ref", "-d", ref)
    if ending in ("missing tree", "missing index"):
        key = "tree" if ending == "missing tree" else "index_tree"
        stop = next(
            r
            for r in map(json.loads, (record / "logs.jsonl").read_text().splitlines())
            if r["type"] == "session.stop"
        )
        pins = cli._git(
            repo, "for-each-ref", "--format=%(refname)", f"refs/agent6/{sid}/"
        ).splitlines()
        for pin in pins:
            if cli._git(repo, "rev-parse", f"{pin}^{{tree}}") == stop[key]:
                cli._git(repo, "update-ref", "-d", pin)


def _first_execution(turn: int, control: str, ending: str, runs: pathlib.Path) -> launch_env.Reply:
    if turn == 0:
        return "apply_edit", {"path": "first", "new_string": "first commit"}
    if turn == 1 and control == "model":
        return "run_command", {
            "argv": [
                "sh",
                "-c",
                "git add first && git -c user.name=t -c user.email=t@t commit -m first"
                " && echo staged >draft && git add draft && echo working >draft",
            ]
        }
    if ending == "turn limit":
        return "read_file", {"path": "first"}
    if ending == "ranked" and turn == 1:
        return "run_command", {"argv": ["sh", "-c", 'echo attempt > "attempt-${PWD##*-}"']}
    if ending == "provider error":
        return 404  # a status no retry recovers, so the child ends at once
    if ending == "killed":
        (worker,) = runs.glob("*-p1-l1/worker.pid")
        os.kill(int(worker.read_text().split()[0]), signal.SIGKILL)
    return launch_env.FINISH


def _check_draft(control: str, ending: str, read: str, notice: str, clones: pathlib.Path) -> None:
    if control == "model" and ending == "killed":
        assert "earlier clone kept at" in notice
        (archive,) = clones.glob("*/kept-*/*")
        assert (archive / "draft").read_text().strip() == "working"
        assert cli._git(archive, "show", ":draft") == "staged"
        assert "No such file" in read
    elif control == "model":
        assert "working" in read and "staged" in read


@pytest.mark.parametrize(
    ("control", "ending"),
    [
        ("agent6", "provider error"),
        ("model", "provider error"),
        ("model", "finished"),
        ("model", "parent merged"),
        ("agent6", "budget"),
        ("agent6", "turn limit"),
        ("agent6", "uncommitted"),
        ("agent6", "ranked"),
        ("agent6", "merged"),
        ("model", "missing snapshot"),
        ("model", "missing ref"),
        ("model", "rewound ref"),
        ("model", "missing tree"),
        ("model", "missing index"),
        ("model", "missing manifest"),
        ("model", "killed"),
        ("agent6", "killed"),
    ],
)
def test_parent_resumes_then_steers_its_ended_child(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, control: str, ending: str
) -> None:
    """The next execution needs the same conversation, commit, index and working files."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    monkeypatch.chdir(repo)
    runs = paths.state_dir(repo) / "sessions" / "runs"
    clones = tmp_path / "clones"
    setup_log = tmp_path / "setup.log"
    child_calls: list[dict[str, Any]] = []
    parent_calls: list[dict[str, Any]] = []
    resumed = False
    turns = 0

    def parent_reply(body: dict[str, Any]) -> launch_env.Reply:
        nonlocal turns
        parent_calls.append(body)
        ids = cli._started(body)
        sid = selected if resumed else ids[0] if ids else ""
        if resumed:
            turns += 1
            if turns == 1:
                return "steer_child", {"id": sid, "text": "REVISIT the saved work"}
            if turns == 2:
                return "read_background", {"id": sid, "wait_s": 30}
            if turns == 3 and control == "agent6" and ending != "merged":
                assert not (repo / "second").exists()
                return "merge_child", {"id": sid}
            return launch_env.FINISH
        if not ids:
            start: tuple[str, dict[str, Any]] = (
                "parallel",
                {
                    "task": "CHILD: make a first commit",
                    "spec": "2" if ending == "ranked" else "1",
                },
            )
            return [start, ("read_background", {"wait_s": 120})] if ending == "budget" else start
        complete = "[parallel] group" in json.dumps(body["messages"])
        after_end = launch_env.FINISH
        if ending == "budget":
            # The parent spends its last two tokens only after the child's result arrives.
            assert complete
            after_end = ("read_file", {"path": "README.md"})
        return after_end if complete else ("read_background", {"id": sid, "wait_s": 30})

    def reply(body: dict[str, Any]) -> launch_env.Reply:
        if "PARENT:" in cli._first_task(body):
            return parent_reply(body)
        if resumed:
            child_calls.append(body)
            if len(child_calls) == 1:
                return "run_command", {
                    "argv": ["sh", "-c", "cat first; cat draft; git show :draft; cat environment"]
                }
            if len(child_calls) == 2:
                return "apply_edit", {"path": "second", "new_string": "done"}
            return launch_env.FINISH
        return _first_execution(cli._turn(body), control, ending, runs)

    with launch_env.scripted_provider(reply_for=reply) as (url, _):
        cfg = config_model.Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": url}},
                "models": {"default": {"model": "p/m"}},
                "git": {
                    "control": control,
                    "commit_per_turn": ending != "uncommitted",
                    "dirty_tree": "include",
                    "worktree": {
                        "base_dir": str(clones),
                        "setup_command": [
                            "sh",
                            "-c",
                            f"echo ready >environment; echo setup >>{setup_log}",
                        ],
                    },
                },
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {
                    "parallel": {"tool": True, "judges": []},
                    "verify": {"infer": False},
                    # Room for the parent: its finish waits while the child's task is unmarked.
                    "max_turns": 8 if ending == "turn limit" else -1,
                },
                "review": {"when": "never"},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        args = [str(pathlib.Path(sys.executable).with_name("agent6")), "--config", str(config)]
        _run_parent(args, child_budget=ending == "budget")
        (parent,) = [d for d in runs.iterdir() if not cli._CHILD_ID.search(d.name)]
        sid = parent.name + "-p1-l1"
        (original,) = child_dispatch.read_records(parent)
        if ending == "ranked":
            (sid,) = [
                c.session_id for c in original.children if c.merge and c.merge.status == "ranked"
            ]
        selected = sid
        record = runs / sid
        ref = f"refs/agent6/{sid}/head" if control == "model" else f"agent6/{sid}"
        if ending in ("budget", "turn limit", "provider error"):
            stops = [
                row["reason"]
                for row in map(json.loads, (record / "logs.jsonl").read_text().splitlines())
                if row["type"] == "session.stop"
            ]
            assert stops == [
                {
                    "budget": "budget_exhausted",
                    "turn limit": "max_turns",
                    "provider error": "provider_error",
                }[ending]
            ]
        if ending == "budget":
            merge = original.child(sid).merge
            assert merge is not None and merge.status == "failed"
            assert merge.detail == f"no result (failed · budget exhausted); {ref} holds 1 commit"
        unmeasured_dirt = control == "model" and ending == "killed"
        assert bool(list(clones.rglob("child-*"))) is unmeasured_dirt
        if ending == "provider error":
            assert f"{ref} holds 1 commit" in json.dumps(parent_calls[-1]["messages"])
        if ending == "parent merged":
            cli._git(repo, "merge", "--ff-only", ref)
        _remove_evidence(ending, repo, record, ref)
        cli._git(repo, "gc", "--prune=now")
        parent_calls.clear()
        resumed = True
        resume_caps = ["--max-tokens-fallback", "100"] if ending == "budget" else []
        subprocess.run(
            [*args, "resume", parent.name, *resume_caps, "--steer", "send notes"],
            check=True,
            timeout=120,
        )
    answer = cli._text(
        next(m for m in reversed(parent_calls[1]["messages"]) if m["role"] == "tool")
    )
    resumed_notice = "\n".join(
        cli._text(m)
        for m in parent_calls[0]["messages"]
        if m["role"] == "user" and "[parallel] children this run can continue" in cli._text(m)
    )
    if ending in ("merged", "rewound ref") or ending.startswith("missing"):
        assert not child_calls
        reason = {
            "merged": "parent's tree",
            "missing snapshot": "snapshot",
            "missing ref": "ref",
            "rewound ref": "commits",
            "missing manifest": "manifest",
            "missing index": "index tree",
            "missing tree": "end tree",
        }[ending]
        assert reason in answer, answer
        assert f"  - {sid}:" not in resumed_notice
        return
    assert f"  - {sid}:" in resumed_notice
    assert "fresh clone" in answer
    assert (
        "uncommitted files could not be restored" if ending == "killed" else "end tree"
    ) in answer
    assert len(child_calls) == 3 and "REVISIT" in json.dumps(child_calls[0]["messages"])
    assert cli._turn(child_calls[0]) > 0
    read = cli._text(next(m for m in reversed(child_calls[1]["messages"]) if m["role"] == "tool"))
    assert "first commit" in read and "ready" in read
    _check_draft(control, ending, read, answer, clones)
    assert setup_log.read_text().splitlines() == ["setup"] * (3 if ending == "ranked" else 2)
    assert not list(clones.rglob("child-*"))
    (group,) = child_dispatch.read_records(parent)
    assert group.child(sid).continuation is not None
    assert (repo / "second").exists() is (control == "agent6" and ending != "uncommitted")
    if ending == "ranked":
        assert group.ranking == original.ranking
        assert [c for c in group.children if c.session_id != sid] == [
            c for c in original.children if c.session_id != sid
        ]
