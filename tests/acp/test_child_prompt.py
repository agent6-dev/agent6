# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Child work reaches the editor on the parent's ACP connection."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

from agent6 import paths
from tests import launch_env
from tests.acp import test_runner


def _repo(tmp_path: pathlib.Path, url: str, *, tool: bool) -> pathlib.Path:
    repo = test_runner._repo(tmp_path / "repo")
    subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-qm", "seed"], check=True)
    config = paths.repo_config_path(repo)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f'[providers.local]\nbackend="openai"\nauth_style="none"\nbase_url="{url}"\n'
        '[models.default]\nmodel="local/m"\n'
        '[sandbox]\nisolation="none"\nrun_commands="ask"\n'
        f"[harness.parallel]\ntool={str(tool).lower()}\n[harness.verify]\ninfer=false\n"
        f'[review]\nwhen="never"\n[git.worktree]\nbase_dir="{tmp_path / "clones"}"\n'
    )
    return repo


def test_parallel_prompt_streams_the_child_before_its_permission_is_answered(
    tmp_path: pathlib.Path,
) -> None:
    """A child blocked on a command shows its tool and takes the editor's denial."""
    with launch_env.scripted_provider(
        [("run_command", {"argv": ["touch", "not-approved"]}), launch_env.FINISH]
    ) as (url, calls):
        repo = _repo(tmp_path, url, tool=True)
        wire = test_runner._Wire()
        try:
            parent = wire.new_session(repo)
            wire.prompt(parent, "/parallel inspect the command")
            child = f"{parent}-p1-l1"
            texts: list[str] = []
            permission: dict[str, Any] = {}
            while not (permission and any(child in t and "run_command" in t for t in texts)):
                message = wire.recv(timeout=60)
                assert message.get("id") != 3, texts
                params = message["params"]
                assert params["sessionId"] == parent
                if message["method"] == "session/request_permission":
                    assert child in params["toolCall"]["title"], params
                    permission = message
                else:
                    content = params["update"].get("content")
                    if isinstance(content, dict):
                        texts.append(content["text"])
            deny = next(o for o in permission["params"]["options"] if o["kind"] == "reject_once")
            wire.send(
                id=permission["id"],
                result={"outcome": {"outcome": "selected", "optionId": deny["optionId"]}},
            )
            while (message := wire.recv(timeout=60)).get("id") != 3:
                assert message["method"] == "session/update"
                assert message["params"]["sessionId"] == parent
                content = message["params"]["update"].get("content")
                if isinstance(content, dict):
                    texts.append(content["text"])
            assert message["result"]["stopReason"] == "end_turn", texts
            assert sum(child in t and "Local child completed." in t for t in texts) == 1, texts
            assert len(calls) == 2
            tools = {t["function"]["name"] for t in calls[0]["tools"]}
            assert not tools.intersection({"parallel", "steer_child", "merge_child"})
            assert not (repo / "not-approved").exists()
            assert "run_command not approved" in calls[-1]["messages"][-1]["content"]
        finally:
            wire.close()


def test_a_first_prompt_dispatches_each_task_in_the_same_session(tmp_path: pathlib.Path) -> None:
    """Neither task in an opening batch becomes a model instruction to the parent."""
    with launch_env.scripted_provider() as (url, calls):
        repo = _repo(tmp_path, url, tool=False)
        wire = test_runner._Wire()
        try:
            parent = wire.new_session(repo)
            wire.prompt(parent, "/parallel FIRST_SUBTASK /parallel 1 -- SECOND_SUBTASK")
            seen: list[dict[str, Any]] = []
            while (message := wire.recv(timeout=60)).get("id") != 3:
                assert message["params"]["sessionId"] == parent
                seen.append(message["params"]["update"])
            assert message["result"]["stopReason"] == "end_turn", seen
            tasks = [
                json.dumps(next(m for m in b["messages"] if m["role"] == "user")) for b in calls
            ]
            assert len(tasks) == 2
            assert all(
                sum(word in t for t in tasks) == 1 for word in ("FIRST_SUBTASK", "SECOND_SUBTASK")
            )
            assert any(
                c["name"] == "parallel" for u in seen for c in u.get("availableCommands", [])
            )
        finally:
            wire.close()
