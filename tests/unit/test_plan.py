# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Plan mode: the schema, the dispatcher, the system prompt, the tool filter and the plan output."""

from __future__ import annotations

import json
import pathlib
import subprocess
import types
from typing import Any, cast
from unittest import mock

import pytest

from agent6 import kinds
from agent6.config import Config, load_config
from agent6.harness import _chain, _loop_state, _provider_call
from agent6.harness import loop as loopmod
from agent6.providers import ProviderResponse
from agent6.tools import dispatch, errors, mcp_client, results, schema

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
isolation = "auto"
run_commands = "no"
protect_git = true
[git]
dirty_tree = "ask"
branch_per_run = true
[harness]
verify_command = ["true"]
[budget]
max_tokens_fallback = 2000000
"""


def _silent(_msg: str) -> None:
    return None


def _config(tmp_path: pathlib.Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


# --- schema -------------------------------------------------------------


def test_finish_planning_requires_nonempty_fields() -> None:
    with pytest.raises(ValueError):
        schema.FinishPlanningInput(summary="", plan_markdown="x")
    with pytest.raises(ValueError):
        schema.FinishPlanningInput(summary="x", plan_markdown="")


def test_finish_planning_tool_name() -> None:
    assert schema.FinishPlanningInput.TOOL_NAME == "finish_planning"


def test_finish_planning_fields_are_documented_in_the_schema() -> None:
    # Both fields must carry a description in the emitted JSON schema, so the
    # model disambiguates plan_markdown (the deliverable) from summary at the
    # exact surface it fills -- without it, models dumped the whole plan into
    # `summary` and left a degenerate plan.md.
    props = schema.FinishPlanningInput.model_json_schema()["properties"]
    assert (
        "plan_markdown" in props["plan_markdown"]["description"]
        or "plan.md" in (props["plan_markdown"]["description"])
    )
    assert "NOT the plan" in props["summary"]["description"]


def test_plan_extra_tools_includes_finish_planning_excludes_finish_session() -> None:
    names = {t.TOOL_NAME for t in schema.PLAN_EXTRA_TOOLS}
    assert schema.FinishPlanningInput.TOOL_NAME in names
    assert schema.FinishSessionInput.TOOL_NAME not in names


# --- dispatcher ---------------------------------------------------------


def test_dispatch_finish_planning_returns_ack(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg, mode="plan")
    out = d.dispatch(
        "finish_planning",
        {"summary": "looks good", "plan_markdown": "# Plan\n\n## Tasks\n- t1\n"},
    ).to_wire()
    assert out["acknowledged"] is True
    assert out["summary"] == "looks good"
    assert out["plan_bytes"] == len(b"# Plan\n\n## Tasks\n- t1\n")


def test_dispatch_finish_planning_rejects_empty(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg, mode="plan")
    with pytest.raises(errors.ToolError, match="summary"):
        d.dispatch("finish_planning", {"summary": "", "plan_markdown": "x"})


def test_dispatch_finish_session_echoes_structured_result(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("finish_session", {"summary": "done", "result": {"approved": True}}).to_wire()
    assert out == {"acknowledged": True, "summary": "done", "result": {"approved": True}}


def test_dispatch_finish_session_result_defaults_none(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("finish_session", {"summary": "done"}).to_wire()
    assert out == {"acknowledged": True, "summary": "done", "result": None}


# --- system prompt & tool definitions -----------------------------------


def test_build_system_prompt_plan_mode_mentions_plan(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    text = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="plan", skills=None
    )
    assert "PLAN mode" in text or "plan mode" in text.lower()


def test_system_prompt_file_override_replaces_run_base_keeps_blocks(tmp_path: pathlib.Path) -> None:
    custom = tmp_path / "prompt.txt"
    custom.write_text("<role>CUSTOM WORKER. apply_edit + finish_session.</role>", encoding="utf-8")
    cfg = Config.model_validate({"prompt": {"system_prompt_file": str(custom)}})
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    run = loopmod.build_system_prompt(config=cfg, repo=repo, mode="run", skills=None)  # pyright: ignore[reportPrivateUsage]
    plan = loopmod.build_system_prompt(config=cfg, repo=repo, mode="plan", skills=None)  # pyright: ignore[reportPrivateUsage]
    # override replaces the run base...
    assert "CUSTOM WORKER" in run and "<agent6>" not in run
    # ...but the dynamic blocks (budget, repo-priors) still append
    assert "<budget-awareness>" in run and "<repo-priors>" in run
    # other modes are unaffected (scoped to run)
    assert "CUSTOM WORKER" not in plan


def test_decompose_swaps_dag_rules_block(tmp_path: pathlib.Path) -> None:
    """`[prompt].decompose` swaps the run-mode 'DAG optional' block for the 'decompose first' one.

    The default keeps the optional block; the sentinel is always filled (never leaks), and only run
    mode is affected.
    """
    repo = kinds.RepoSummary(
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
    run_off = loopmod.build_system_prompt(config=off, repo=repo, mode="run", skills=None)  # pyright: ignore[reportPrivateUsage]
    run_on = loopmod.build_system_prompt(config=on, repo=repo, mode="run", skills=None)  # pyright: ignore[reportPrivateUsage]
    run_auto = loopmod.build_system_prompt(config=auto, repo=repo, mode="run", skills=None)  # pyright: ignore[reportPrivateUsage]
    assert "__DAG_RULES_BLOCK__" not in run_off and "__DAG_RULES_BLOCK__" not in run_on
    assert "<dag-rules>" in run_off and "<decompose-first>" not in run_off
    assert "<decompose-first>" in run_on and "<dag-rules>" not in run_on
    assert "<dag-rules>" in run_auto and "<decompose-first>" not in run_auto
    # decompose is a run-mode worker feature: other modes never carry either block
    # or a leaked sentinel.
    for mode in ("plan", "ask", "agent"):
        text = loopmod.build_system_prompt(config=on, repo=repo, mode=mode, skills=None)  # pyright: ignore[reportPrivateUsage]
        assert "__DAG_RULES_BLOCK__" not in text and "<decompose-first>" not in text


def test_decompose_defaults_auto(tmp_path: pathlib.Path) -> None:
    assert Config().prompt.decompose == "auto"


def test_dag_hint_renders_only_where_the_dag_tools_exist() -> None:
    """The DAG hint renders only for modes whose tool surface has `add_task`.

    The decompose-first directive is run-only (it references the run-only `<decompose-first>` block
    and tells the worker to edit); ask wires a curator too but exposes no DAG tools, so a hint there
    names a tool the model cannot call.
    """
    hint = loopmod.initial_dag_hint  # pyright: ignore[reportPrivateUsage]
    rid = "01" + "A" * 24
    run_dec = hint(rid, "run", True)
    assert "<decompose-first>" in run_dec and "Do not edit" in run_dec
    plan_hint = hint(rid, "plan", True)
    assert "<decompose-first>" not in plan_hint and "optional" in plan_hint
    for mode in ("ask", "agent"):
        assert hint(rid, mode, True) == ""
    # decompose off, or no curator, never emits the directive
    assert "<decompose-first>" not in hint(rid, "run", False)
    assert hint(None, "run", True) == ""


def test_system_prompt_file_validator_rejects_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="not a readable file"):
        Config.model_validate({"prompt": {"system_prompt_file": str(tmp_path / "nope.txt")}})


def testwarn_if_prompt_override_incomplete(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.app import preflight

    good = tmp_path / "good.txt"
    good.write_text("use apply_edit and call finish_session when done", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("just go do stuff", encoding="utf-8")
    # complete override -> silent
    preflight.warn_if_prompt_override_incomplete(
        Config.model_validate({"prompt": {"system_prompt_file": str(good)}})
    )
    assert capsys.readouterr().err == ""
    # missing both contracts -> warns about each
    preflight.warn_if_prompt_override_incomplete(
        Config.model_validate({"prompt": {"system_prompt_file": str(bad)}})
    )
    err = capsys.readouterr().err
    assert "finish_session" in err and "apply_edit/apply_patch" in err
    # no override -> silent
    preflight.warn_if_prompt_override_incomplete(Config())
    assert capsys.readouterr().err == ""


def test_build_system_prompt_warns_against_git_checkout_revert(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    text = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="run", skills=None
    )
    assert "git checkout" in text
    assert ".git/" in text
    assert "git show HEAD:path" in text


def test_build_system_prompt_describes_auto_metric_feedback(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "agent6.toml"
    # `run_commands = "no"` withholds every command tool, the metric included,
    # and the block describing it goes with the tool.
    p.write_text(
        _VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"')
        + '\n[harness.metric]\ncommand = ["python3", "bench.py"]\n'
        + 'pattern = "CYCLES: (\\\\d+)"\ngoal = "minimize"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    text = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="run", skills=None
    )
    assert "the harness runs the metric and" in text
    assert "[harness metric]" in text
    # Run-mode only: plan/ask do not expose `run_metric_command`, and the
    # auto-metric-after-verify behaviour the block describes is the run loop's.
    for mode in ("plan", "ask"):
        other = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
            config=cfg, repo=repo, mode=mode, skills=None
        )
        assert "<metric-command>" not in other, mode
        assert "run_metric_command" not in other, mode


def test_run_commands_no_withholds_the_command_tools_and_every_rule_about_them(
    tmp_path: pathlib.Path,
) -> None:
    """`run_commands = "no"` withholds the command tools and every rule about them.

    One answer per run: the gate does not exist, and the tool list, the verify block, the metric
    block and the auto-commit rule all say so. The metric tool is an extra outside ALL_TOOLS, so the
    offer side must read the policy rather than hand the model a tool with only a refusal behind it
    while the commit rule names a verify the same prompt calls absent.
    """
    p = tmp_path / "agent6.toml"
    p.write_text(
        _VALID_TOML.replace(  # carries run_commands = "no" and a verify_command
            'verify_command = ["true"]', 'verify_command = ["true"]\nverify_when = "step"'
        )
        + '\n[harness.metric]\ncommand = ["python3", "bench.py"]\n'
        + 'pattern = "CYCLES: (\\\\d+)"\ngoal = "minimize"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )

    text = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=repo, mode="run", skills=None
    )
    assert "<no-verify-command>" in text
    assert "<metric-command>" not in text
    assert "after each passing verify" not in text
    assert "commits each editing turn" in text

    names = {
        t.name for t in loopmod.tool_definitions(dispatch.ToolDispatcher(root=tmp_path, config=cfg))
    }  # pyright: ignore[reportPrivateUsage]
    assert schema.RunMetricInput.TOOL_NAME not in names
    assert schema.RunCommandInput.TOOL_NAME not in names


def test_no_commands_removes_run_command_from_read_only_mode_prompts(
    tmp_path: pathlib.Path,
) -> None:
    """Plan and ask do not advertise probes when run_command is withheld from the tools."""
    cfg = _config(tmp_path)  # run_commands = "no"
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    dispatcher = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    for mode in ("plan", "ask"):
        prompt = loopmod.build_system_prompt(config=cfg, repo=repo, mode=mode, skills=None)  # pyright: ignore[reportPrivateUsage]
        names = {tool.name for tool in loopmod.tool_definitions(dispatcher, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "run_command" not in names
        assert "run_command" not in prompt, mode
        assert "probe's writes" not in prompt, mode


def test_read_only_mode_prompts_splice_the_command_note_cleanly(tmp_path: pathlib.Path) -> None:
    """The read-only mode prompts splice the command note cleanly.

    Plan's rule block opens on a bullet with no blank line, and ask's prose carries the command note
    as a sentence, not a bullet.
    """
    from agent6.config import Config

    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    for commands in ("ask", "no"):
        cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
        plan = loopmod.build_system_prompt(config=cfg, repo=repo, mode="plan", skills=None)  # pyright: ignore[reportPrivateUsage]
        rules = plan.split("<tool-use-rules>\n", 1)[1].split("</tool-use-rules>", 1)[0]
        assert rules.startswith("- ") and "\n\n" not in rules, rules
        ask = loopmod.build_system_prompt(config=cfg, repo=repo, mode="ask", skills=None)  # pyright: ignore[reportPrivateUsage]
        assert "\n- run_command" not in ask
        assert ("not exposed. run_command runs jailed" in ask) == (commands == "ask")


def test_tool_definitions_plan_mode_filters_edit_tools(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    defs = loopmod.tool_definitions(d, mode="plan")  # pyright: ignore[reportPrivateUsage]
    names = {t.name for t in defs}
    assert schema.ApplyEditInput.TOOL_NAME not in names
    assert schema.ApplyPatchInput.TOOL_NAME not in names
    assert schema.FinishSessionInput.TOOL_NAME not in names
    assert schema.FinishPlanningInput.TOOL_NAME in names


def test_tool_definitions_run_mode_includes_edit_tools(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    defs = loopmod.tool_definitions(d, mode="run")  # pyright: ignore[reportPrivateUsage]
    names = {t.name for t in defs}
    assert schema.ApplyEditInput.TOOL_NAME in names
    assert schema.ApplyPatchInput.TOOL_NAME in names
    assert schema.FinishSessionInput.TOOL_NAME in names
    assert schema.FinishPlanningInput.TOOL_NAME not in names


def test_tool_definitions_machine_and_agent_modes_are_read_only_finish(
    tmp_path: pathlib.Path,
) -> None:
    # a read-only machine agent state: navigation + finish_session, NO
    # edit/patch/verify/run_command/DAG (the deliverable is a finish_session result).
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"'), "utf-8")
    cfg = load_config(p)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    for mode in ("machine", "agent"):
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert schema.ReadFileInput.TOOL_NAME in names, mode
        assert schema.FinishSessionInput.TOOL_NAME in names, mode
        assert schema.ApplyEditInput.TOOL_NAME not in names, mode
        assert schema.ApplyPatchInput.TOOL_NAME not in names, mode
        assert schema.RunCommandInput.TOOL_NAME not in names, mode
        assert schema.DagAddTaskInput.TOOL_NAME not in names, mode
        assert schema.FinishPlanningInput.TOOL_NAME not in names, mode


def test_mcp_tools_are_run_mode_only(tmp_path: pathlib.Path) -> None:
    """MCP tools are arbitrary external capabilities agent6 cannot classify as read-only.

    Appended to the tool list in every mode, or routed by the dispatcher before its mode guards, a
    "read-only" plan/ask (or a machine-authoring loop told not to touch anything) could call a
    mutating filesystem/GitHub MCP tool. Both layers gate on run mode: the list omits them, and the
    dispatcher refuses them.
    """
    cfg = _config(tmp_path)
    fake_mgr = types.SimpleNamespace(
        descriptors=lambda: [
            types.SimpleNamespace(
                qualified_name="mcp__fs__write_file",
                server_name="fs",
                tool_name="write_file",
                description="write a file",
                input_schema={"type": "object"},
            )
        ]
    )

    d_run = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    d_run._mcp_manager = cast("mcp_client.MCPManager", fake_mgr)  # pyright: ignore[reportPrivateUsage]
    run_names = {t.name for t in loopmod.tool_definitions(d_run, mode="run")}
    assert "mcp__fs__write_file" in run_names

    for mode in ("plan", "ask", "machine", "agent"):
        d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
        d._mcp_manager = cast("mcp_client.MCPManager", fake_mgr)  # pyright: ignore[reportPrivateUsage]
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "mcp__fs__write_file" not in names, mode

    # The dispatcher backstop: a read-only-mode dispatcher refuses mcp__* even
    # if a tool-list regression re-exposed it.
    d_plan = dispatch.ToolDispatcher(root=tmp_path, config=cfg, mode="plan")
    d_plan._mcp_manager = cast("mcp_client.MCPManager", fake_mgr)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(errors.ToolError, match="not available in plan mode"):
        d_plan._dispatch_inner("mcp__fs__write_file", {})  # pyright: ignore[reportPrivateUsage]


def test_build_system_prompt_machine_and_agent_modes(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    agent = loopmod.build_system_prompt(config=cfg, repo=repo, mode="agent", skills=None)  # pyright: ignore[reportPrivateUsage]
    assert "state of a state machine" in agent
    assert "run_verify_command" not in agent


def test_tool_definitions_ask_mode_is_read_only_with_commands(tmp_path: pathlib.Path) -> None:
    # ask: read tools + run_command (when the config allows it), but NO edits and
    # NO control tools (no finish_session/finish_planning/DAG) -- it silent-finishes.
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"'), "utf-8")
    cfg = load_config(p)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg)
    names = {t.name for t in loopmod.tool_definitions(d, mode="ask")}  # pyright: ignore[reportPrivateUsage]
    assert schema.ReadFileInput.TOOL_NAME in names  # can read
    assert schema.RunCommandInput.TOOL_NAME in names  # can run commands to investigate
    assert schema.ApplyEditInput.TOOL_NAME not in names  # but not edit
    assert schema.ApplyPatchInput.TOOL_NAME not in names
    assert schema.FinishSessionInput.TOOL_NAME not in names
    assert schema.FinishPlanningInput.TOOL_NAME not in names
    assert schema.DagAddTaskInput.TOOL_NAME not in names
    assert "agent6_docs" in names  # self-help is available in ask mode


def test_dispatcher_refuses_mutations_in_ask_mode(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg, mode="ask")
    with pytest.raises(errors.ToolError, match="ask mode"):
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": "create", "old_string": "", "new_string": "x\n"}]},
        )
    with pytest.raises(errors.ToolError, match="ask mode"):
        d.dispatch("apply_patch", {"patch": "--- a\n+++ b\n"})


# --- Harness plan-mode validation --------------------------------------


def _wf(
    root: pathlib.Path | None = None,
    *,
    ref: str | None = None,
    fallback_parent: str | None = None,
    branch: str | None = None,
    per_step: bool = True,
    base_sha: str = "",
    **kw: Any,
) -> loopmod.Harness:
    defaults: dict[str, Any] = {
        "chain": _chain.RunChain(
            root or pathlib.Path("/tmp"),
            ref=ref,
            branch=branch,
            fallback_parent=fallback_parent,
            per_step=per_step,
            base_sha=base_sha,
        ),
        "config": mock.MagicMock(
            prompt=mock.MagicMock(system_prompt_file=""),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_retries=2,
            ),
        ),
        "provider": mock.MagicMock(),
        "dispatcher": mock.MagicMock(),
        "logger": _silent,
        "call": _provider_call.CallSettings(retry_delay_s=0.01),
    }
    defaults.update(kw)
    return loopmod.Harness(**defaults)


def test_workflow_plan_mode_without_output_path_raises() -> None:
    wf = _wf(mode="plan", plan_output_path=None)
    with pytest.raises(ValueError, match="plan_output_path"):
        wf.run("anything")


# --- plan.md on disk is the source the planner reads --------------------


_STALE_PLAN = "# Plan: X\n\n## Open questions\n> **Q:** which store?\n> **A:**\n"
_ANSWERED_PLAN = "# Plan: X\n\n## Open questions\n> **Q:** which store?\n> **A:** postgres\n"


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "x.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _tool_use(name: str, args: dict[str, Any], tu_id: str = "tu1") -> ProviderResponse:
    block = {"type": "tool_use", "id": tu_id, "name": name, "input": args}
    return ProviderResponse(
        text="",
        tool_uses=({"id": tu_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [block]},
    )


def _plan_wf(
    repo: pathlib.Path, provider: Any, plan_path: pathlib.Path, state_path: pathlib.Path
) -> loopmod.Harness:
    return loopmod.Harness(
        chain=_chain.RunChain(repo),
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=mock.MagicMock(system_prompt_file="", decompose="off"),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_retries=2,
            ),
        ),
        provider=provider,
        dispatcher=mock.MagicMock(
            dispatch=mock.MagicMock(return_value=results.RawResult({"acknowledged": True}))
        ),
        logger=_silent,
        call=_provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_iterations=5,
        mode="plan",
        plan_output_path=plan_path,
        resume_state_path=state_path,
    )


def test_an_operator_edit_to_plan_md_survives_the_next_finish_planning(
    tmp_path: pathlib.Path,
) -> None:
    """An operator edit to plan.md survives the next finish_planning.

    `agent6 plan edit` writes plan.md, then `agent6 resume --steer` continues the planner; plan.md
    on disk is the plan and the conversation holds only a copy, so the resumed planner is shown the
    file, not its snapshot.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    plan_path, state_path = tmp_path / "plan.md", tmp_path / "loop_state.json"

    first = mock.MagicMock()
    first.call.return_value = _tool_use(
        "finish_planning", {"summary": "s", "plan_markdown": _STALE_PLAN}
    )
    _plan_wf(repo, first, plan_path, state_path).run("plan it")
    assert plan_path.read_text(encoding="utf-8") == _STALE_PLAN

    plan_path.write_text(_ANSWERED_PLAN, encoding="utf-8")  # agent6 plan edit

    class Planner:
        """A planner shown the current plan.md carries its answers forward."""

        def __init__(self) -> None:
            self.seen = ""

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.seen = json.dumps(kwargs["messages"])
            plan = _ANSWERED_PLAN if "postgres" in self.seen else _STALE_PLAN
            return _tool_use("finish_planning", {"summary": "s", "plan_markdown": plan})

    planner = Planner()
    _plan_wf(repo, planner, plan_path, state_path).resume()

    assert "postgres" in planner.seen  # the harness put the file in front of it
    assert plan_path.read_text(encoding="utf-8") == _ANSWERED_PLAN  # the edit survived


def test_an_unchanged_plan_md_is_not_injected_twice(tmp_path: pathlib.Path) -> None:
    """Re-read every turn, inject only on change.

    An untouched plan.md costs tokens once, not once per turn.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    plan_path, state_path = tmp_path / "plan.md", tmp_path / "loop_state.json"

    first = mock.MagicMock()
    first.call.return_value = _tool_use(
        "finish_planning", {"summary": "s", "plan_markdown": _STALE_PLAN}
    )
    _plan_wf(repo, first, plan_path, state_path).run("plan it")

    resumed = mock.MagicMock()
    resumed.call.side_effect = [
        _tool_use("read_file", {"path": "x.txt"}, tu_id="t1"),
        _tool_use("read_file", {"path": "x.txt"}, tu_id="t2"),
        _tool_use("finish_planning", {"summary": "s", "plan_markdown": _STALE_PLAN}, tu_id="t3"),
    ]
    _plan_wf(repo, resumed, plan_path, state_path).resume()

    final = json.dumps(resumed.call.call_args_list[-1].kwargs["messages"])
    assert final.count("which store?") == 2  # the finish_planning arg, plus ONE injection


def test_an_unreadable_plan_parks_the_execution(tmp_path: pathlib.Path) -> None:
    """An unreadable plan parks the execution.

    Continuing on the planner's own copy burns budget on direction the operator may have superseded;
    the execution ends with the remedy instead.
    """
    from agent6.harness import _snapshot

    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    plan.chmod(0o000)
    try:
        wf = loopmod.Harness(
            chain=_chain.RunChain(tmp_path),
            config=mock.MagicMock(),
            provider=mock.MagicMock(),
            dispatcher=mock.MagicMock(),
            mode="plan",
            plan_output_path=plan,
            logger=lambda _m: None,
        )
        got = wf._maybe_inject_plan(  # pyright: ignore[reportPrivateUsage]
            mock.MagicMock(), _loop_state.LoopState(original_task="t", tool_calls=4), iteration=3
        )
    finally:
        plan.chmod(0o600)
    assert isinstance(got, _snapshot.SessionResult)
    assert got.reason == "plan_unreadable" and got.completed is False
    assert "plan.md unreadable" in got.summary
    assert "agent6 resume" in got.summary  # the remedy is in hand
    assert got.iterations == 3 and got.tool_calls == 4


def test_the_decisions_block_renders_when_rulings_exist(tmp_path: pathlib.Path) -> None:
    """The decisions block renders when rulings exist.

    The operator's rulings ride the system prompt in every mode, after the memory block; nothing
    renders when none are recorded.
    """
    from agent6.config import Config
    from agent6.harness import _prompt_blocks  # pyright: ignore[reportPrivateUsage]

    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="",
        head_sha="",
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
        is_git=False,
    )
    for mode in ("run", "plan", "ask"):
        text = _prompt_blocks.build_system_prompt(
            config=Config(),
            repo=repo,
            mode=mode,
            skills=None,
            decisions="- 2026-08-23 00:00Z [s] Q: Modal?\n  A: No.",
            decisions_path="/m/DECISIONS.md",
        )
        assert "<decisions>" in text and "A: No." in text and "/m/DECISIONS.md" in text
        assert "recorded rulings" in text and "each ask_user answer" not in text
    assert "<decisions>" not in _prompt_blocks.build_system_prompt(
        config=Config(), repo=repo, mode="run", skills=None
    )
