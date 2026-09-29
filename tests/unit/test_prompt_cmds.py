# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt show` assembles the real system prompt for the current repo."""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from agent6 import paths
from agent6.ui.cli import prompt_cmds  # pyright: ignore[reportPrivateUsage]


def _git_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "repo"
    p.mkdir()
    (p / "f.py").write_text("x = 1\n", encoding="utf-8")
    (p / "AGENTS.md").write_text("# conventions\n- be terse here\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
        "PATH": os.environ.get("PATH", ""),
    }
    subprocess.run(["git", "init", "-q"], cwd=p, check=True)
    subprocess.run(["git", "add", "-A"], cwd=p, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, env=env, check=True)
    return p


def _isolate(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, repo: pathlib.Path) -> None:
    monkeypatch.chdir(repo)
    # isolate from the developer's real global config / state
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def test_prompt_show_run_mode_injects_agents_md(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    rc = prompt_cmds._cmd_prompt_show(None, mode="run")
    out = capsys.readouterr().out
    assert rc == 0
    # The run-mode base and the per-repo summary block.
    assert "<agent6>" in out and "<repo-priors>" in out
    # the repo's AGENTS.md is injected verbatim into the prompt
    assert "be terse here" in out


def test_prompt_show_plan_mode_differs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    rc = prompt_cmds._cmd_prompt_show(None, mode="plan")
    out = capsys.readouterr().out
    assert rc == 0 and "PLAN mode" in out


def test_prompt_show_includes_recorded_memories(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`prompt show` includes the recorded memories.

    It claims to print what the worker receives, so it passes the memories (and skills) the run loop
    injects; an operator checking whether a recorded memory would reach future runs must not see
    '(none recorded yet)' while the real prompt carries it.
    """
    from agent6 import memory

    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    state = paths.state_dir(repo)
    state.mkdir(parents=True, exist_ok=True)
    memory.add(state, "facts", "the deploy script needs sudo")

    assert prompt_cmds._cmd_prompt_show(None, mode="run") == 0
    out = capsys.readouterr().out
    assert "the deploy script needs sudo" in out


def test_prompt_show_prints_the_tools_and_the_first_message(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The system prompt is half of what the model receives.

    The tool definitions travel in the API's `tools` field and the task rides a first-message
    header; the system prompt alone shows no run_command guidance, so an operator reading it judges
    the model blind. `prompt show` prints all three, and `--json` the same as one object, with the
    tool list this config actually exposes.
    """
    import json

    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    assert prompt_cmds._cmd_prompt_show(None, mode="run") == 0
    out = capsys.readouterr().out
    assert "=== tools (" in out and "--- run_command" in out and "--- finish_session" in out
    assert "=== first user message ===" in out and "`finish_session` requests the run end" in out

    assert prompt_cmds._cmd_prompt_show(None, mode="ask", as_json=True) == 0
    exchange = json.loads(capsys.readouterr().out)
    names = [t["name"] for t in exchange["tools"]]
    assert "read_file" in names and "agent6_docs" in names
    assert "apply_edit" not in names and "finish_session" not in names  # ask has no edit/finish
    assert set(exchange) == {"mode", "system", "tools", "first_message", "mcp_tools_pending"}
    assert all("input_schema" in t and "description" in t for t in exchange["tools"])


def test_prompt_show_infers_the_gate_a_run_would_infer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`prompt show` infers the gate a run would infer.

    A run infers its gate before assembling the prompt, and the gate decides the `<verify-command>`
    block, the commit rule and whether `run_verify_command` is offered; skipping it prints "this run
    has no verify command" for every repo whose gate is inferred.
    """
    repo = _git_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "# agents\n\nbe terse here\n\n## Verify command\n\n```bash\npytest -q\n```\n",
        encoding="utf-8",
    )
    _isolate(tmp_path, monkeypatch, repo)

    assert prompt_cmds._cmd_prompt_show(None, mode="run") == 0

    out = capsys.readouterr().out
    assert "pytest -q" in out
    assert "no verify command" not in out
    assert "run_verify_command" in out


def test_a_withheld_tool_gets_no_block_and_no_offer(tmp_path: pathlib.Path) -> None:
    """A withheld tool gets no block and no offer.

    `run_commands = "no"` withholds every command tool, and a metric with no `[harness.metric]` can
    only error; a prompt block describing a tool the model does not have is one it cannot act on,
    and the metric tool must not be offered while `run_verify_command` is hidden.
    """
    import tempfile

    from agent6.config import Config
    from agent6.harness import (
        _toolset,  # pyright: ignore[reportPrivateUsage]
        model_exchange_for,
    )
    from agent6.tools import dispatch

    withheld = Config.model_validate(
        {
            "sandbox": {"run_commands": "no"},
            "harness": {
                "verify_command": ["pytest", "-q"],
                "metric": {"command": ["m"], "pattern": r"x:(\d+)", "goal": "minimize"},
            },
        }
    )
    exchange = model_exchange_for(withheld, tmp_path, "run", state_dir=tmp_path)
    assert "<verify-command>" not in exchange.system
    assert "<metric-command>" not in exchange.system
    assert "<no-verify-command>" in exchange.system

    with tempfile.TemporaryDirectory() as td:
        plain = dispatch.ToolDispatcher(root=pathlib.Path(td), config=Config())
        names = [t.name for t in _toolset.tool_definitions(plain, mode="run")]
    assert "run_metric_command" not in names, "offered with no [harness.metric]"


def test_plan_mode_does_not_name_a_gate_it_says_is_absent(tmp_path: pathlib.Path) -> None:
    """Plan mode does not name a gate it says is absent.

    One plan prompt must not carry both "run_verify_command runs the operator's gate" and
    "`run_verify_command` is not available", forty lines apart.
    """
    from agent6.config import Config
    from agent6.harness import model_exchange_for

    gateless = model_exchange_for(Config(), tmp_path, "plan", state_dir=tmp_path).system
    assert "<no-verify-command>" in gateless
    assert "run_verify_command runs the operator's gate" not in gateless

    gated = model_exchange_for(
        Config.model_validate({"harness": {"verify_command": ["pytest", "-q"]}}),
        tmp_path,
        "plan",
        state_dir=tmp_path,
    ).system
    assert "run_verify_command runs the operator's gate" in gated
