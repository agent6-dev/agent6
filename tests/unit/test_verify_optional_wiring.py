# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Gateless wiring: with no verify_command, the verify tool is hidden and the
system prompt swaps the verify block for the no-verify block."""

from __future__ import annotations

from pathlib import Path

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import RepoSummary
from agent6.workflows._prompt_blocks import build_system_prompt


def _cfg(*, verify: bool) -> Config:
    data = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Config.model_validate(data)


def _repo(root: Path) -> RepoSummary:
    return RepoSummary(
        root=root,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def test_verify_tool_hidden_when_command_unset(tmp_path: Path) -> None:
    with_verify = ToolDispatcher(root=tmp_path, config=_cfg(verify=True))
    gateless = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert "run_verify_command" in with_verify.available_tool_names()
    assert "run_verify_command" not in gateless.available_tool_names()


def test_adopt_verify_command_probes_the_jail_path(tmp_path: Path) -> None:
    """Mid-run adoption refuses a bare runner the jail PATH cannot resolve
    (adopting it would turn an honest settle into an unexecutable-verify
    abort) and accepts a resolvable one, which also unhides the verify tool.
    Path-form commands pass through: they resolve against the mounted cwd."""
    d = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert d.adopt_verify_command(("no-such-binary-zq9", "test")) is False
    assert "run_verify_command" not in d.available_tool_names()
    assert d.adopt_verify_command(("sh", "-c", "true")) is True
    assert "run_verify_command" in d.available_tool_names()
    d2 = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert d2.adopt_verify_command(("./scripts/check.sh",)) is True


def test_system_prompt_switches_verify_block(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with_verify = build_system_prompt(config=_cfg(verify=True), repo=repo, mode="run")
    gateless = build_system_prompt(config=_cfg(verify=False), repo=repo, mode="run")
    assert "<verify-command>" in with_verify and "<no-verify-command>" not in with_verify
    assert "<no-verify-command>" in gateless and "<verify-command>" not in gateless


def test_no_verify_block_wording_matches_the_mode(tmp_path: Path) -> None:
    """The gateless block must name the mode's real terminal tool and must not
    claim auto-commits in the read-only modes: plan finishes via
    `finish_planning`, ask has no terminal tool at all, and neither can edit."""
    repo = _repo(tmp_path)
    cfg = _cfg(verify=False)
    run = build_system_prompt(config=cfg, repo=repo, mode="run")
    plan = build_system_prompt(config=cfg, repo=repo, mode="plan")
    ask = build_system_prompt(config=cfg, repo=repo, mode="ask")

    def block(text: str) -> str:
        start = text.index("<no-verify-command>")
        return text[start : text.index("</no-verify-command>", start)]

    run_block, plan_block, ask_block = block(run), block(plan), block(ask)
    assert "finish_run" in run_block and "commits each editing step" in run_block
    assert "finish_planning" in plan_block
    assert "finish_run" not in plan_block and "commits" not in plan_block
    assert "finish_run" not in ask_block and "finish_planning" not in ask_block
    assert "commits" not in ask_block
    # All three still disarm stray instructions to call the absent verify tool.
    for b in (run_block, plan_block, ask_block):
        assert "Ignore any" in b and "run_verify_command" in b


def test_a_gate_the_run_may_never_execute_is_dropped_at_the_start(tmp_path: Path) -> None:
    """`run_commands = "no"` withholds every command tool, the gate included.
    Keeping the gate made the run unwinnable: nothing could go green, so nothing
    committed, it finished red, and the prompt named a tool it did not have."""
    from agent6.app.preflight import infer_verify_if_unset
    from agent6.budget import BudgetTracker
    from agent6.events import EventSink
    from agent6.providers import TranscriptSink

    cfg = Config.model_validate(
        {"workflow": {"verify_command": ["true"]}, "sandbox": {"run_commands": "no"}}
    )
    got = infer_verify_if_unset(
        cfg,
        tmp_path,
        mode="run",
        events=EventSink(tmp_path / "logs.jsonl"),
        transcript_sink=TranscriptSink(tmp_path / "transcript.md"),
        budget=BudgetTracker(),
    )
    assert got.workflow.verify_command == ()
    d = ToolDispatcher(root=tmp_path, config=got)
    assert "run_verify_command" not in d.available_tool_names()
    assert "no verify command" in build_system_prompt(config=got, repo=_repo(tmp_path)).lower()


def test_a_run_that_cannot_run_commands_is_gateless_wherever_it_starts(tmp_path: Path) -> None:
    """The rule lived only in preflight's fresh-run path, so a RESUMED leg was
    re-gated with every command tool withheld: nothing could go green, the leg
    committed nothing, and the manifest was re-pinned to claim a gate that
    never judged anything."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent6.workflows.loop import Workflow

    wf = Workflow.__new__(Workflow)
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest", "-q"))
    )
    wf.dispatcher = MagicMock()
    wf.dispatcher.command_policy.return_value = "no"
    assert wf._gate_argv() == ()  # pyright: ignore[reportPrivateUsage]
    wf.dispatcher.command_policy.return_value = "ask"
    assert wf._gate_argv() == ("pytest", "-q")  # pyright: ignore[reportPrivateUsage]


def test_a_deny_mid_run_takes_the_gate_with_it(tmp_path: Path) -> None:
    """`deny for the rest of the run` and an away-mode of deny both flip the
    EFFECTIVE policy to "no" while the config still names a gate. The leg kept
    the gate, lost the tool, and ended red."""
    from agent6.config import Config
    from agent6.runs.ipc import set_away_mode
    from agent6.tools.dispatch import ToolDispatcher

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = Config.model_validate({"workflow": {"verify_command": ["true"]}})
    d = ToolDispatcher(root=tmp_path, config=cfg, run_dir=run_dir)
    assert "run_verify_command" in d.available_tool_names()
    set_away_mode(run_dir, "deny")
    assert d.command_policy() == "no"
    assert "run_verify_command" not in d.available_tool_names()


def test_a_gate_is_never_adopted_when_the_worker_cannot_run_one(tmp_path: Path) -> None:
    """Adoption checked the jail PATH but not the policy, so a --no-commands
    run re-acquired a gate mid-run and undid the preflight drop."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = Config.model_validate({"sandbox": {"run_commands": "no"}})
    d = ToolDispatcher(root=tmp_path, config=cfg, run_dir=run_dir)
    assert d.adopt_verify_command(("/bin/true",)) is False
    assert d._config.workflow.verify_command == ()  # pyright: ignore[reportPrivateUsage]


def test_the_worker_gets_the_tool_for_a_gate_adopted_mid_run(tmp_path: Path) -> None:
    """The tool list was built once per leg. A gateless run that adopted a gate
    was TOLD to run run_verify_command while that tool was absent from every
    remaining call: commits stopped, the finish was graded failed, exit 4."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher
    from agent6.workflows._toolset import tool_definitions

    d = ToolDispatcher(root=tmp_path, config=Config())
    before = {t.name for t in tool_definitions(d, mode="run")}
    assert "run_verify_command" not in before

    assert d.adopt_verify_command(("/bin/true",)) is True
    after = {t.name for t in tool_definitions(d, mode="run")}
    assert "run_verify_command" in after, "the adopted gate has no tool to run it"
