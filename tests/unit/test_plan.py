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
api_format = "anthropic"
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
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
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


def test_finish_planning_fields_are_documented_in_the_schema() -> None:
    # Both fields must carry a description in the emitted JSON schema, so the
    # model disambiguates plan_markdown (the deliverable) from summary at the
    # exact surface it fills -- without it, models dumped the whole plan into
    # `summary` and left a degenerate plan.md.
    props = FinishPlanningInput.model_json_schema()["properties"]
    assert "plan_markdown" in props["plan_markdown"]["description"] or "plan.md" in (
        props["plan_markdown"]["description"]
    )
    assert "NOT the plan" in props["summary"]["description"]


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


def test_system_prompt_file_override_replaces_run_base_keeps_blocks(tmp_path: Path) -> None:
    custom = tmp_path / "prompt.txt"
    custom.write_text("<role>CUSTOM WORKER. apply_edit + finish_run.</role>", encoding="utf-8")
    cfg = Config.model_validate({"prompt": {"system_prompt_file": str(custom)}})
    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    run = loopmod._build_system_prompt(config=cfg, repo=repo, mode="run")  # pyright: ignore[reportPrivateUsage]
    plan = loopmod._build_system_prompt(config=cfg, repo=repo, mode="plan")  # pyright: ignore[reportPrivateUsage]
    # override replaces the run base...
    assert "CUSTOM WORKER" in run and "<edit-rules>" not in run
    # ...but the dynamic blocks (budget, repo-priors) still append
    assert "<budget-awareness>" in run and "<repo-priors>" in run
    # other modes are unaffected (scoped to run)
    assert "CUSTOM WORKER" not in plan


def test_decompose_swaps_dag_rules_block(tmp_path: Path) -> None:
    """[prompt].decompose swaps the run-mode 'DAG optional' block for the
    'decompose first' directive; default keeps the optional block. The sentinel
    is always filled (never leaks) and only run mode is affected."""
    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    off = Config.model_validate({"prompt": {"decompose": "off"}})
    on = Config.model_validate({"prompt": {"decompose": "on"}})
    auto = Config()  # unresolved "auto" reaching the engine renders like off
    run_off = loopmod._build_system_prompt(config=off, repo=repo, mode="run")  # pyright: ignore[reportPrivateUsage]
    run_on = loopmod._build_system_prompt(config=on, repo=repo, mode="run")  # pyright: ignore[reportPrivateUsage]
    run_auto = loopmod._build_system_prompt(config=auto, repo=repo, mode="run")  # pyright: ignore[reportPrivateUsage]
    assert "__DAG_RULES_BLOCK__" not in run_off and "__DAG_RULES_BLOCK__" not in run_on
    assert "<dag-rules>" in run_off and "<decompose-first>" not in run_off
    assert "<decompose-first>" in run_on and "<dag-rules>" not in run_on
    assert "<dag-rules>" in run_auto and "<decompose-first>" not in run_auto
    # decompose is a run-mode worker feature: other modes never carry either block
    # or a leaked sentinel.
    for mode in ("plan", "ask", "machine", "agent"):
        text = loopmod._build_system_prompt(config=on, repo=repo, mode=mode)  # pyright: ignore[reportPrivateUsage]
        assert "__DAG_RULES_BLOCK__" not in text and "<decompose-first>" not in text


def test_decompose_defaults_auto(tmp_path: Path) -> None:
    assert Config().prompt.decompose == "auto"


def test_decompose_hint_is_run_mode_only() -> None:
    """The decompose-first user-message hint must NOT leak into plan/ask, which
    also wire a curator (root_id non-None): it references the run-only
    <decompose-first> block and tells the worker to edit."""
    hint = loopmod._initial_dag_hint  # pyright: ignore[reportPrivateUsage]
    rid = "01" + "A" * 24
    run_dec = hint(rid, "run", True)
    assert "<decompose-first>" in run_dec and "Do not edit" in run_dec
    for mode in ("plan", "ask", "machine", "agent"):
        h = hint(rid, mode, True)
        assert "<decompose-first>" not in h and "Do not edit before the plan" not in h
        assert "optional" in h  # falls back to the plain optional-DAG hint
    # decompose off, or no curator, never emits the directive
    assert "<decompose-first>" not in hint(rid, "run", False)
    assert hint(None, "run", True) == ""


def test_system_prompt_file_validator_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a readable file"):
        Config.model_validate({"prompt": {"system_prompt_file": str(tmp_path / "nope.txt")}})


def testwarn_if_prompt_override_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli._preflight import warn_if_prompt_override_incomplete

    good = tmp_path / "good.txt"
    good.write_text("use apply_edit and call finish_run when done", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("just go do stuff", encoding="utf-8")
    # complete override -> silent
    warn_if_prompt_override_incomplete(
        Config.model_validate({"prompt": {"system_prompt_file": str(good)}})
    )
    assert capsys.readouterr().err == ""
    # missing both contracts -> warns about each
    warn_if_prompt_override_incomplete(
        Config.model_validate({"prompt": {"system_prompt_file": str(bad)}})
    )
    err = capsys.readouterr().err
    assert "finish_run" in err and "apply_edit/apply_patch" in err
    # no override -> silent
    warn_if_prompt_override_incomplete(Config())
    assert capsys.readouterr().err == ""


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
    # Run-mode only: plan/ask do not expose `run_metric_command`, and the
    # auto-metric-after-verify behaviour the block describes is the run loop's.
    for mode in ("plan", "ask"):
        other = loopmod._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
            config=cfg, repo=repo, mode=mode
        )
        assert "<metric-command>" not in other, mode
        assert "run_metric_command" not in other, mode


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


def test_tool_definitions_machine_and_agent_modes_are_read_only_finish(tmp_path: Path) -> None:
    # machine authoring + machine agent-state: read-only navigation + finish_run,
    # NO edit/patch/verify/run_command/DAG (the deliverable is a finish_run result).
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"'), "utf-8")
    cfg = load_config(p)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    for mode in ("machine", "agent"):
        names = {t.name for t in loopmod._tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert ReadFileInput.TOOL_NAME in names, mode
        assert FinishRunInput.TOOL_NAME in names, mode
        assert ApplyEditInput.TOOL_NAME not in names, mode
        assert ApplyPatchInput.TOOL_NAME not in names, mode
        assert RunCommandInput.TOOL_NAME not in names, mode
        assert DagAddTaskInput.TOOL_NAME not in names, mode
        assert FinishPlanningInput.TOOL_NAME not in names, mode


def test_build_system_prompt_machine_and_agent_modes(tmp_path: Path) -> None:
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
    machine = loopmod._build_system_prompt(config=cfg, repo=repo, mode="machine")  # pyright: ignore[reportPrivateUsage]
    assert "MACHINE-AUTHORING" in machine
    assert "run_verify_command" not in machine  # no verify block in authoring mode
    agent = loopmod._build_system_prompt(config=cfg, repo=repo, mode="agent")  # pyright: ignore[reportPrivateUsage]
    assert "state of a state machine" in agent
    assert "run_verify_command" not in agent


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
    assert "agent6_docs" in names  # self-help is available in ask mode


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
        "config": MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), require_verify_to_finish=False),
        ),
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
