"""The same child conversation and clone continue only under their owning parent."""

from __future__ import annotations

import json
import pathlib
import threading
from typing import Any

import pytest

from agent6 import paths
from agent6.app import child_launch, child_resume
from agent6.config import Config, layer
from agent6.sessions import child_dispatch
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import cli_main
from tests import launch_env
from tests.unit import test_children_background as background
from tests.unit import test_children_cli as children_cli


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_the_cli_continues_the_same_child_and_refuses_direct_resume(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    control: str,
) -> None:
    """A parent restarts its stopped child without losing its conversation or committed work."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    ready = threading.Event()
    seen: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []

    def reply(body: dict[str, Any]) -> launch_env.Reply:  # noqa: PLR0911  # one reply per scripted turn
        task = children_cli._first_task(body)
        turn = children_cli._turn(body)
        if "PARENT:" in task:
            parents.append(body)
            ids = children_cli._started(body)
            sid = ids[0] if ids else ""
            if turn == 1:
                assert ready.wait(120)
            return [
                [("parallel", {"task": "CHILD: start the work"})],
                [("stop_background", {"id": sid})],
                [("read_background", {"id": sid, "wait_s": 120})],
                [("steer_child", {"id": sid, "text": "CONTINUE-OWNED: finish"})],
                [("read_background", {"id": sid, "wait_s": 120})],
                [("finish_session", {"summary": "child continued"})],
            ][min(turn, 5)]
        seen.append(body)
        text = json.dumps(body["messages"])
        if "CONTINUE-OWNED" in text:
            if "continued.txt" not in text:
                return [
                    ("read_file", {"path": "child.txt"}),
                    ("apply_edit", {"path": "continued.txt", "new_string": "continued\n"}),
                ]
            if control == "model" and "commit -m continuation" not in text:
                return [
                    (
                        "run_command",
                        {
                            "argv": [
                                "sh",
                                "-c",
                                "git add continued.txt && git -c user.name=test"
                                " -c user.email=test@example.com commit -m continuation",
                            ]
                        },
                    )
                ]
            return [("finish_session", {"summary": "same child finished"})]
        if turn == 0:
            return [("apply_edit", {"path": "child.txt", "new_string": "original work\n"})]
        if turn == 1 and control == "model":
            return [
                (
                    "run_command",
                    {
                        "argv": [
                            "sh",
                            "-c",
                            "git add child.txt && git -c user.name=test"
                            " -c user.email=test@example.com commit -m child",
                        ]
                    },
                )
            ]
        ready.set()
        return [("run_command", {"argv": ["sleep", "60"]})]

    with launch_env.scripted_provider(reply_for=reply) as (url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": control, "worktree": {"base_dir": str(tmp_path / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"parallel": {"tool": True}, "verify": {"infer": False}},
                "review": {"when": "never"},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        monkeypatch.chdir(repo)
        code = cli_main(
            ["--config", str(config), "run", "--auto-approve", "PARENT: continue child"]
        )
    output = capsys.readouterr()
    assert code == 0, output.out[-3000:] + output.err[-3000:]
    runs = paths.state_dir(repo) / "sessions" / "runs"
    (parent,) = [d for d in runs.iterdir() if not children_cli._CHILD_ID.search(d.name)]
    (record,) = child_dispatch.read_records(parent)
    (child,) = record.children
    assert child.continuation is not None and child.continuation.sequence == 1
    assert child.join is not None
    assert child.join.status == ("joined" if control == "agent6" else "unmerged")
    continued = [b for b in seen if "CONTINUE-OWNED" in json.dumps(b["messages"])]
    assert continued
    stopped = json.dumps(parents[3]["messages"])
    assert "conversation at last call: 1 tokens; last ran" in stopped
    assert "last ran: unknown" not in stopped
    assert children_cli._turn(continued[0]) > 0
    assert "original work" in json.dumps(continued[-1])
    assert (repo / "continued.txt").exists() is (control == "agent6")
    assert "continued" in background._git(repo, "show", f"{child.join.branch}:continued.txt")
    assert "parallel.child_continued" in (parent / "logs.jsonl").read_text()
    assert cli_main(["--config", str(config), "resume", child.session_id]) == 2
    output = capsys.readouterr()
    assert "steer_child" in output.out + output.err


@pytest.mark.parametrize("foreign", ["parent", "group", "number", "session", "clone", "reviewer"])
def test_a_private_launch_cannot_continue_another_child(
    tmp_path: pathlib.Path, foreign: str
) -> None:
    """A launch must match the record's complete child identity and its execution's clone."""
    launch = child_launch.ChildLaunch(
        worktree=tmp_path,
        record_state=tmp_path / "state",
        session_id="root-p1-l1",
        task="work",
        config=tmp_path / "config.toml",
        parent="root",
        group="root-p1",
        number=1,
        continuation="finish",
    )
    record = sessions_manifest.SessionManifest.model_validate(
        {
            "session_id": launch.session_id,
            "task": "work",
            "mode": "run",
            "parallel": {"parent": "root", "group": "root-p1", "number": 1},
        }
    )
    assert child_resume.authorized(launch, record, session_id=launch.session_id, root=tmp_path)
    edits: dict[str, Any] = {
        "parent": {"parent": "other"},
        "group": {"group": "other-p1"},
        "number": {"number": 2},
        "session": {"session_id": "root-p1-l2"},
        "clone": {"worktree": tmp_path / "other"},
        "reviewer": {"contract": object()},
    }
    assert not child_resume.authorized(
        launch.model_copy(update=edits[foreign]),
        record,
        session_id=launch.session_id,
        root=tmp_path,
    )
