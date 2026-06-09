# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""plan-mode unit tests covering schema, dispatcher, system prompt,
tool-filter, and the Workflow's plan-output side effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config, load_config
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.schema import (
    PLAN_EXTRA_TOOLS,
    ApplyEditInput,
    ApplyPatchInput,
    DagAddTaskInput,
    FinishPlanningInput,
    FinishRunInput,
    ReadFileInput,
    RunCommandInput,
)
from agent6.types import RepoSummary
from agent6.workflows import loop as loopmod
from agent6.workflows.loop import Workflow

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
profile = "auto"
agent_network = "open"
run_commands = "no"
protect_git = true
protect_agent6 = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _silent(_msg: str) -> None:
    return None


def _config(tmp_path: Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


# --- schema -------------------------------------------------------------


def test_finish_planning_requires_nonempty_fields() -> None:
    with pytest.raises(ValueError):
        FinishPlanningInput(summary="", plan_markdown="x")
    with pytest.raises(ValueError):
        FinishPlanningInput(summary="x", plan_markdown="")


def test_finish_planning_tool_name() -> None:
    assert FinishPlanningInput.TOOL_NAME == "finish_planning"


def test_plan_extra_tools_includes_finish_planning_excludes_finish_run() -> None:
    names = {t.TOOL_NAME for t in PLAN_EXTRA_TOOLS}
    assert FinishPlanningInput.TOOL_NAME in names
    assert FinishRunInput.TOOL_NAME not in names


# --- dispatcher ---------------------------------------------------------


def test_dispatch_finish_planning_returns_ack(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "finish_planning",
        {"summary": "looks good", "plan_markdown": "# Plan\n\n## Tasks\n- t1\n"},
    )
    assert out["acknowledged"] is True
    assert out["summary"] == "looks good"
    assert out["plan_bytes"] == len(b"# Plan\n\n## Tasks\n- t1\n")


def test_dispatch_finish_planning_rejects_empty(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError):
        d.dispatch("finish_planning", {"summary": "", "plan_markdown": "x"})


def test_dispatch_finish_run_echoes_structured_result(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("finish_run", {"summary": "done", "result": {"approved": True}})
    assert out == {"acknowledged": True, "summary": "done", "result": {"approved": True}}


def test_dispatch_finish_run_result_defaults_none(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("finish_run", {"summary": "done"})
    assert out == {"acknowledged": True, "summary": "done", "result": None}


# --- system prompt & tool definitions -----------------------------------


def test_build_system_prompt_plan_mode_mentions_plan(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    text = loopmod._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="plan"
    )
    assert "PLAN mode" in text or "plan mode" in text.lower()


def test_build_system_prompt_warns_against_git_checkout_revert(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    text = loopmod._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="run"
    )
    assert "git checkout" in text
    assert ".git/" in text
    assert "git show HEAD:path/to/file" in text


def test_build_system_prompt_describes_auto_metric_feedback(tmp_path: Path) -> None:
    p = tmp_path / "agent6.toml"
    p.write_text(
        _VALID_TOML
        + '\n[workflow.metric]\ncommand = ["python3", "bench.py"]\n'
        + 'pattern = "CYCLES: (\\\\d+)"\ngoal = "minimize"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    text = loopmod._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="run"
    )
    assert "automatically runs this" in text
    assert "[harness metric]" in text
    assert "flat/worse" in text


def test_tool_definitions_plan_mode_filters_edit_tools(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    defs = loopmod._tool_definitions(d, mode="plan")  # pyright: ignore[reportPrivateUsage]
    names = {t.name for t in defs}
    assert ApplyEditInput.TOOL_NAME not in names
    assert ApplyPatchInput.TOOL_NAME not in names
    assert FinishRunInput.TOOL_NAME not in names
    assert FinishPlanningInput.TOOL_NAME in names


def test_tool_definitions_run_mode_includes_edit_tools(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    defs = loopmod._tool_definitions(d, mode="run")  # pyright: ignore[reportPrivateUsage]
    names = {t.name for t in defs}
    assert ApplyEditInput.TOOL_NAME in names
    assert ApplyPatchInput.TOOL_NAME in names
    assert FinishRunInput.TOOL_NAME in names
    assert FinishPlanningInput.TOOL_NAME not in names


def test_tool_definitions_ask_mode_is_read_only_with_commands(tmp_path: Path) -> None:
    # ask: read tools + run_command (when the config allows it), but NO edits and
    # NO control tools (no finish_run/finish_planning/DAG) -- it silent-finishes.
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"'), "utf-8")
    cfg = load_config(p)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = {t.name for t in loopmod._tool_definitions(d, mode="ask")}  # pyright: ignore[reportPrivateUsage]
    assert ReadFileInput.TOOL_NAME in names  # can read
    assert RunCommandInput.TOOL_NAME in names  # can run commands to investigate
    assert ApplyEditInput.TOOL_NAME not in names  # but not edit
    assert ApplyPatchInput.TOOL_NAME not in names
    assert FinishRunInput.TOOL_NAME not in names
    assert FinishPlanningInput.TOOL_NAME not in names
    assert DagAddTaskInput.TOOL_NAME not in names


def test_dispatcher_refuses_mutations_in_ask_mode(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, mode="ask")
    with pytest.raises(ToolError, match="ask mode"):
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": "create", "old_string": "", "new_string": "x\n"}]},
        )
    with pytest.raises(ToolError, match="ask mode"):
        d.dispatch("apply_patch", {"patch": "--- a\n+++ b\n"})


# --- Workflow plan-mode validation --------------------------------------


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
        "provider_retry_delay_s": 0.01,
    }
    defaults.update(kw)
    return Workflow(**defaults)


def test_workflow_plan_mode_without_output_path_raises() -> None:
    wf = _wf(mode="plan", plan_output_path=None)
    with pytest.raises(ValueError, match="plan_output_path"):
        wf.run("anything")
