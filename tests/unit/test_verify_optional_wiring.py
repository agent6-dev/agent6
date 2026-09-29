# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Gateless wiring: no verify_command hides the verify tool and swaps in the no-verify block."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess as sp
import time
from collections.abc import Callable
from unittest import mock

import pytest

from agent6 import git_ops, kinds, paths
from agent6.app import _execution as app__execution
from agent6.app import _session as app__session
from agent6.app import _setup as app__setup
from agent6.app import confine, preflight, stamps
from agent6.app import providers as app_providers
from agent6.config import Config, layer
from agent6.harness import _chain, _prompt_blocks, _snapshot, _verify_verdict
from agent6.sandbox import detect
from agent6.tools import dispatch


def _cfg(*, verify: bool) -> Config:
    data = {"harness": {"verify_command": ["true"]}} if verify else {}
    return Config.model_validate(data)


def _repo(root: pathlib.Path) -> kinds.RepoSummary:
    return kinds.RepoSummary(
        root=root,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def test_verify_tool_hidden_when_command_unset(tmp_path: pathlib.Path) -> None:
    with_verify = dispatch.ToolDispatcher(root=tmp_path, config=_cfg(verify=True))
    gateless = dispatch.ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert "run_verify_command" in with_verify.available_tool_names()
    assert "run_verify_command" not in gateless.available_tool_names()


def test_adopt_verify_command_probes_the_jail_path(tmp_path: pathlib.Path) -> None:
    """Mid-run adoption refuses a bare runner the jail PATH cannot resolve and accepts one it can.

    Adopting an unresolvable runner would turn an honest settle into an unexecutable-verify
    abort; a path-form command resolves against the mounted cwd.
    """
    d = dispatch.ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert d.adopt_verify_command(("no-such-binary-zq9", "test")) is False
    assert "run_verify_command" not in d.available_tool_names()
    assert d.adopt_verify_command(("sh", "-c", "true")) is True
    assert "run_verify_command" in d.available_tool_names()
    d2 = dispatch.ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert d2.adopt_verify_command(("./scripts/check.sh",)) is True


def test_system_prompt_switches_verify_block(tmp_path: pathlib.Path) -> None:
    repo = _repo(tmp_path)
    with_verify = _prompt_blocks.build_system_prompt(
        config=_cfg(verify=True), repo=repo, mode="run", skills=None
    )
    gateless = _prompt_blocks.build_system_prompt(
        config=_cfg(verify=False), repo=repo, mode="run", skills=None
    )
    assert "<verify-command>" in with_verify and "<no-verify-command>" not in with_verify
    assert "<no-verify-command>" in gateless and "<verify-command>" not in gateless
    # Every verify rule lives INSIDE the conditional block: the base leaked
    # gate prose ("run project tests only through...", the stale_gate rule,
    # "after each passing verify") into gateless prompts, which then needed
    # an "Ignore any other instruction" patch-line to disarm it.
    assert gateless.count("run_verify_command") == 1  # the block's own initial absence
    assert "stale_gate" not in gateless and "passing verify" not in gateless
    assert "commits each editing turn" in gateless  # the gate-aware commit rule
    assert "run project tests only through" not in gateless.lower()
    assert "stale_gate" in with_verify and "commits each editing turn" in with_verify
    # The per-step commit rule belongs to a gate that judges each step.
    never = Config.model_validate({"harness": {"verify_command": ["true"], "verify_when": "never"}})
    assert "pending changes automatically after each passing" in _prompt_blocks.build_system_prompt(
        config=never, repo=repo, mode="run", skills=None
    )


def test_no_verify_block_wording_matches_the_mode(tmp_path: pathlib.Path) -> None:
    """The gateless block states the gate's absence and nothing else, in every mode.

    The terminal tool is each base prompt's fact; ask has none.
    """
    repo = _repo(tmp_path)
    cfg = _cfg(verify=False)
    run = _prompt_blocks.build_system_prompt(config=cfg, repo=repo, mode="run", skills=None)
    plan = _prompt_blocks.build_system_prompt(config=cfg, repo=repo, mode="plan", skills=None)
    ask = _prompt_blocks.build_system_prompt(config=cfg, repo=repo, mode="ask", skills=None)

    def block(text: str) -> str:
        start = text.index("<no-verify-command>")
        return text[start : text.index("</no-verify-command>", start)]

    run_block, plan_block, ask_block = block(run), block(plan), block(ask)
    assert "finish_session" not in run_block and "finish_session requests the run end" in run
    assert "finish_planning" not in plan_block and "`finish_planning` ends the pass" in plan
    assert "finish_session" not in plan_block and "commits" not in plan_block
    assert "finish_session" not in ask_block and "finish_planning" not in ask_block
    assert "commits" not in ask_block
    # The commit claim is the base sentinel's, one owner (run mode only);
    # the block no longer needs an "Ignore any other instruction" patch-line
    # because no verify prose leaks outside the verify block.
    assert "commits" not in run_block and "commits each editing turn" in run
    for b in (run_block, plan_block, ask_block):
        assert "Ignore any" not in b


def test_a_execution_that_cannot_run_commands_is_gateless_wherever_it_starts(
    tmp_path: pathlib.Path,
) -> None:
    """Both lifecycles decide gatedness once, at execution start, with the frozen system prompt.

    The rule lived only in preflight's fresh-run path, so a resumed execution was re-gated with
    every command tool withheld and re-pinned to claim a gate that never judged anything.
    """
    from agent6.app import reporter as app_reporter
    from agent6.sessions import ipc

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    gated = Config.model_validate({"harness": {"verify_command": ["pytest", "-q"]}})

    assert preflight.drop_gate_if_unrunnable(
        gated, session_dir=session_dir, reporter=reporter
    ).harness.verify_command == (
        "pytest",
        "-q",
    )
    withheld = Config.model_validate(
        {"harness": {"verify_command": ["pytest", "-q"]}, "sandbox": {"run_commands": "no"}}
    )
    assert (
        preflight.drop_gate_if_unrunnable(
            withheld, session_dir=session_dir, reporter=reporter
        ).harness.verify_command
        == ()
    )
    assert any("running gateless" in line for line in said)

    # An away-mode of deny reaches the same answer: the EFFECTIVE policy, not
    # just the configured knob.
    ipc.set_away_mode(session_dir, "deny")
    assert (
        preflight.drop_gate_if_unrunnable(
            gated, session_dir=session_dir, reporter=reporter
        ).harness.verify_command
        == ()
    )


def test_a_deny_after_a_red_gate_does_not_turn_the_run_green(tmp_path: pathlib.Path) -> None:
    """Gatedness is frozen at execution start; a later deny withdraws the tools, never the verdict.

    Reading the live policy let a mid-run deny flip a failed gate to not_applicable, the exit
    code to 0, and `git.auto_merge` merged the red branch.
    """
    import types

    from agent6.harness import _loop_state, loop

    wf = loop.Harness.__new__(loop.Harness)
    wf.chain = _chain.RunChain(tmp_path)
    wf.config = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        harness=types.SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=("pytest", "-q"),
            verify_when="finish",
            verify_retries=2,
            verify_timeout_s=60.0,
            verify_infer=True,
        )
    )
    wf.dispatcher = mock.MagicMock()
    wf.dispatcher.command_policy.return_value = "no"  # denied mid-run
    state = mock.MagicMock(spec=_loop_state.LoopState)
    state.verify = _verify_verdict.VerifyVerdict(last_ok=False, edited_since=False)

    assert wf.gate.tree_green(state.verify) is False


def test_a_deny_mid_run_takes_the_gate_with_it(tmp_path: pathlib.Path) -> None:
    """An effective policy of "no" under a configured gate keeps the gate, loses the tool, ends red.

    `deny for the rest of the run` and an away-mode of deny both flip it.
    """
    from agent6.config import Config
    from agent6.sessions import ipc

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    cfg = Config.model_validate({"harness": {"verify_command": ["true"]}})
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg, session_dir=session_dir)
    assert "run_verify_command" in d.available_tool_names()
    ipc.set_away_mode(session_dir, "deny")
    assert d.command_policy() == "no"
    assert "run_verify_command" not in d.available_tool_names()


def test_a_gate_is_never_adopted_when_the_worker_cannot_run_one(tmp_path: pathlib.Path) -> None:
    """Adoption checks the policy too, so a --no-commands run never re-acquires a gate mid-run."""
    from agent6.config import Config

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    cfg = Config.model_validate({"sandbox": {"run_commands": "no"}})
    d = dispatch.ToolDispatcher(root=tmp_path, config=cfg, session_dir=session_dir)
    assert d.adopt_verify_command(("/bin/true",)) is False
    assert d._config.harness.verify_command == ()  # pyright: ignore[reportPrivateUsage]


def test_the_worker_gets_the_tool_for_a_gate_adopted_mid_run(tmp_path: pathlib.Path) -> None:
    """The tool list was built once per execution.

    A gateless run that adopted a gate was TOLD to run run_verify_command while that tool was absent
    from every remaining call: commits stopped, the finish was graded failed, exit 4.
    """
    from agent6.config import Config
    from agent6.harness import _toolset

    d = dispatch.ToolDispatcher(root=tmp_path, config=Config())
    before = {t.name for t in _toolset.tool_definitions(d, mode="run")}
    assert "run_verify_command" not in before

    assert d.adopt_verify_command(("/bin/true",)) is True
    after = {t.name for t in _toolset.tool_definitions(d, mode="run")}
    assert "run_verify_command" in after, "the adopted gate has no tool to run it"


class _Stop(Exception):  # noqa: N818  # a signal, not an error  # a signal, not an error
    """Sentinel: the lifecycle reached pin_gate with this execution's final gate."""


def _git_repo(path: pathlib.Path) -> None:
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "-A"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _role_cfg(extra: dict[str, object]) -> Config:
    # Callers setenv("K", ...) so provider construction finds the key.
    return Config.model_validate(
        {
            "providers": {"anthropic": {"api_format": "anthropic", "api_key_env": "K"}},
            "models": {
                "worker": {"provider": "anthropic", "model": "m"},
                "reviewer": {"provider": "anthropic", "model": "m"},
            },
            **extra,
        }
    )


def _capture_pin(pinned: list[tuple[tuple[str, ...], str]]) -> Callable[..., None]:
    def _pin(_dir: pathlib.Path, argv: object, origin: str, **_k: object) -> None:
        pinned.append((tuple(argv), origin))  # pyright: ignore[reportArgumentType]
        raise _Stop()

    return _pin


@pytest.mark.parametrize(
    ("snapshot_gate", "manifest_gate", "origin"),
    [
        ((), ("true",), "adopted"),
        (("true",), (), "unadopted"),
    ],
)
def test_resume_uses_the_gate_pin_newer_than_a_crash_snapshot(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_gate: tuple[str, ...],
    manifest_gate: tuple[str, ...],
    origin: str,
) -> None:
    """Adoption re-pins the manifest before the after-tools snapshot advances.

    A crash in that window leaves the snapshot's gate stale in either direction; resume must keep
    the newer pin rather than undoing adoption or un-adoption.
    """
    from agent6.app import _execution as app__execution
    from agent6.app import resume

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "crashed-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "crashed-AAAA11",
                "mode": "run",
                "user_task": "t",
                "harness": {"verify_command": manifest_gate, "verify_origin": origin},
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": _snapshot.SNAPSHOT_VERSION,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 2,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": snapshot_gate,
            }
        ),
        encoding="utf-8",
    )
    cfg = _role_cfg({"sandbox": {"run_commands": "yes"}})
    effective = layer.EffectiveConfig(config=cfg, sources={}, layers=())

    def _effective(*_a: object, **_k: object) -> layer.EffectiveConfig:
        return effective

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _none(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(layer, "load_effective", _effective)
    monkeypatch.setattr(app__session, "select_isolation", _strict)
    monkeypatch.setattr(app__setup, "check_provider_keys", _none)
    monkeypatch.setattr(git_ops, "verify_git_identity", _none)
    used: list[tuple[str, ...]] = []

    def _execution(
        _cfg: Config, _layout: object, inputs: app__execution.ExecutionInputs, **_kw: object
    ) -> app__execution.ExecutionEnd:
        used.append(inputs.gate(_cfg, mock.MagicMock()).harness.verify_command)
        return app__execution.ExecutionEnd(0)

    monkeypatch.setattr(app__execution, "run_execution", _execution)
    assert (
        resume.resume_task(
            None, "crashed-AAAA11", started_at=time.time(), frontend=mock.MagicMock(), force=False
        )
        == 0
    )
    assert used == [manifest_gate]
    persisted = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert tuple(persisted["harness"]["verify_command"]) == manifest_gate


def test_a_withheld_resumed_execution_is_not_regated_by_the_snapshot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drop must have the last word at execution start.

    With commands withheld, the snapshot-reuse block ran AFTER drop_gate_if_unrunnable and handed
    the dropped gate straight back: the execution resumed gated-but-unwinnable, printed two
    contradictory preamble lines, committed nothing all execution, and exited 4 over a gate that
    never ran.
    """
    from agent6.app import reporter as app_reporter
    from agent6.app import resume
    from agent6.ui.cli import run as cli_run

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "withheld-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {"version": 3, "session_id": "withheld-AAAA11", "mode": "run", "user_task": "t"}
        ),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": _snapshot.SNAPSHOT_VERSION,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": ["pytest", "-q"],
            }
        ),
        encoding="utf-8",
    )
    cfg = _role_cfg({"sandbox": {"run_commands": "no"}})
    monkeypatch.setenv("K", "test-key")

    def _none(*_a: object, **_k: object) -> None:
        return None

    # The real type: preflight reads `explicit_leaves` to tell a default this
    # host cannot honour (degrade) from a value the operator set (refuse).
    def _load(*_a: object, **_k: object) -> layer.EffectiveConfig:
        return layer.EffectiveConfig(config=cfg, sources={}, layers=())

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _provider(*_a: object, **_k: object) -> mock.MagicMock:
        return mock.MagicMock()

    def _yes(*_a: object) -> bool:
        return True

    pinned: list[tuple[tuple[str, ...], str]] = []
    monkeypatch.setattr(layer, "load_effective", _load)
    monkeypatch.setattr(app__setup, "detect_env", object)
    monkeypatch.setattr(detect, "resolve_isolation", _strict)
    monkeypatch.setattr(confine, "warn_sandbox_gaps", _none)
    monkeypatch.setattr(confine, "check_network_support", _none)
    monkeypatch.setattr(preflight, "budget_preflight", _none)
    monkeypatch.setattr(app_providers, "build_role_provider", _provider)
    monkeypatch.setattr(app__setup, "check_provider_keys", _none)
    monkeypatch.setattr(git_ops, "verify_git_identity", _none)
    monkeypatch.setattr(stamps, "pin_gate", _capture_pin(pinned))

    said: list[str] = []
    frontend = dataclasses.replace(cli_run.session_frontend(), confirm_unconfined_autorun=_yes)
    with pytest.raises(_Stop):
        resume.resume_task(
            None,
            "withheld-AAAA11",
            started_at=time.time(),
            frontend=frontend,
            force=False,
            reporter=app_reporter.Reporter(out=said.append, err=said.append),
        )
    assert pinned == [((), "")], f"the withheld execution was re-gated: {pinned}"
    assert any("running gateless" in line for line in said)


def test_a_withheld_fresh_execution_is_not_regated_by_inference(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With commands withheld, inference never re-gates the execution from AGENTS.md.

    The pin labelled the inferred command "configured".
    """
    from agent6.app import preflight
    from agent6.app import reporter as app_reporter
    from agent6.app import run as run_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Verify: make check\n", encoding="utf-8")
    _git_repo(repo)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("K", "test-key")
    cfg = _role_cfg(
        {
            "sandbox": {"run_commands": "no"},
            "harness": {"verify_command": ["pytest", "-q"]},
            "git": {"branch_per_run": False},
        }
    )

    def _none(*_a: object, **_k: object) -> None:
        return None

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _provider(*_a: object, **_k: object) -> mock.MagicMock:
        return mock.MagicMock()

    pinned: list[tuple[tuple[str, ...], str]] = []
    monkeypatch.setattr(app__setup, "detect_env", object)
    monkeypatch.setattr(detect, "resolve_isolation", _strict)
    monkeypatch.setattr(confine, "warn_sandbox_gaps", _none)
    monkeypatch.setattr(confine, "check_network_support", _none)
    monkeypatch.setattr(preflight, "budget_preflight", _none)
    monkeypatch.setattr(app_providers, "build_role_provider", _provider)
    monkeypatch.setattr(git_ops, "verify_git_identity", _none)
    monkeypatch.setattr(stamps, "pin_gate", _capture_pin(pinned))

    said: list[str] = []
    frontend = mock.MagicMock()
    frontend.should_spawn_tui.return_value = False
    frontend.stream_modes.return_value = (False, False)
    with pytest.raises(_Stop):
        run_mod.run_task(
            cfg,
            "t",
            started_at=time.time(),
            frontend=frontend,
            mode="run",
            reporter=app_reporter.Reporter(out=said.append, err=said.append),
        )
    assert pinned == [((), "")], f"the withheld execution was re-gated: {pinned}"
    assert any("running gateless" in line for line in said)


def test_hardened_fs_rule_renders_only_under_hardened(tmp_path: pathlib.Path) -> None:
    """The hardened create-a-top-level-entry workaround renders only under hardened."""
    repo = _repo(tmp_path)
    cfg = _cfg(verify=True)
    strict = _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="strict", protected_paths=True
    )
    hardened = _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="hardened", protected_paths=True
    )
    assert "Under hardened isolation" not in strict
    assert "__HARDENED_FS_RULE__" not in strict
    assert "Under hardened isolation" in hardened


def test_patch_only_prompt_names_only_the_offered_edit_tool(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The patch-only experiment removes apply_edit's contract from the prompt with the tool."""
    monkeypatch.setenv("AGENT6_DISABLE_APPLY_EDIT", "1")
    cfg = _cfg(verify=True)
    prompt = _prompt_blocks.build_system_prompt(
        config=cfg, repo=_repo(tmp_path), mode="run", skills=None
    )
    from agent6.harness import _toolset

    names = {
        tool.name
        for tool in _toolset.tool_definitions(dispatch.ToolDispatcher(root=tmp_path, config=cfg))
    }
    assert "apply_edit" not in names
    assert "apply_edit" not in prompt
    assert "apply_patch" in names and "apply_patch" in prompt


def test_git_protect_rule_renders_only_when_the_bind_exists(tmp_path: pathlib.Path) -> None:
    """The '.git/ is protected' line renders only under strict with protect_git on.

    Every unjailed run was told it while nothing protected it.
    """
    repo = _repo(tmp_path)
    on = _cfg(verify=True)
    off = Config.model_validate(
        {"harness": {"verify_command": ["true"]}, "sandbox": {"protect_git": False}}
    )
    marker = ".git/` is read-only inside the jail"
    for isolation, cfg, expect in (
        ("strict", on, True),
        ("strict", off, False),
        ("hardened", on, False),
        ("none", on, False),
    ):
        out = _prompt_blocks.build_system_prompt(
            config=cfg,
            repo=repo,
            mode="run",
            skills=None,
            isolation=isolation,  # pyright: ignore[reportArgumentType]
        )
        assert (marker in out) is expect, (isolation, expect)
        assert "__GIT_PROTECT_RULE__" not in out


def test_agents_md_section_absent_when_repo_has_none(tmp_path: pathlib.Path) -> None:
    """A repo without AGENTS.md gets no empty conventions header."""
    repo = _repo(tmp_path)
    out = _prompt_blocks.build_system_prompt(
        config=_cfg(verify=True), repo=repo, mode="run", skills=None
    )
    assert "AGENTS.md (project conventions):" not in out
    assert "(empty)" not in out


def test_prompt_git_rules_match_git_control(tmp_path: pathlib.Path) -> None:
    """Under [git].control = "model" the prompt never claims the harness commits automatically."""
    repo = _repo(tmp_path)
    agent6_cfg = Config.model_validate({"harness": {"verify_command": ["true"]}})
    model_cfg = Config.model_validate(
        {
            "harness": {"verify_command": ["true"]},
            "git": {"control": "model"},
            "sandbox": {"protect_git": False},
        }
    )
    agent6_prompt = _prompt_blocks.build_system_prompt(
        config=agent6_cfg, repo=repo, mode="run", skills=None
    )
    model_prompt = _prompt_blocks.build_system_prompt(
        config=model_cfg, repo=repo, mode="run", skills=None
    )
    assert "The harness commits" in agent6_prompt
    assert "You own git" not in agent6_prompt
    assert "The harness commits" not in model_prompt
    assert "You own git" in model_prompt

    gateless_cfg = Config.model_validate(
        {"git": {"control": "model"}, "sandbox": {"protect_git": False}}
    )
    gateless = _prompt_blocks.build_system_prompt(
        config=gateless_cfg,
        repo=repo,
        mode="run",
        skills=None,
    )
    start = gateless.index("<no-verify-command>")
    block = gateless[start : gateless.index("</no-verify-command>", start)]
    assert "finish_session" not in block and "finish_session requests the run end" in gateless
    assert "commits each editing turn" not in block


def test_model_git_rule_does_not_offer_a_withheld_run_command(tmp_path: pathlib.Path) -> None:
    """Model-controlled git never tells the worker to commit through a withheld run_command."""
    cfg = Config.model_validate(
        {
            "git": {"control": "model"},
            "sandbox": {"run_commands": "no", "protect_git": False},
        }
    )
    prompt = _prompt_blocks.build_system_prompt(
        config=cfg, repo=_repo(tmp_path), mode="run", skills=None
    )
    from agent6.harness import _toolset

    names = {
        tool.name
        for tool in _toolset.tool_definitions(
            dispatch.ToolDispatcher(root=tmp_path, config=cfg), mode="run"
        )
    }
    assert "run_command" not in names
    assert "run_command" not in prompt
    assert "uncommitted" in prompt


def test_budget_block_names_the_plan_meter_for_subscription_runs(tmp_path: pathlib.Path) -> None:
    """A subscription run's budget block names the plan meter, not USD caps that never bind it.

    The plan line renders exactly when a configured role rides a chatgpt provider.
    """
    repo = _repo(tmp_path)
    sub = Config.model_validate(
        {
            "providers": {"gpt": {"api_format": "chatgpt"}},
            "models": {
                "worker": {"provider": "gpt", "model": "gpt-5.6-sol"},
                "reviewer": {"provider": "gpt", "model": "gpt-5.6-sol"},
            },
            "budget": {"max_percent": 3},
        }
    )
    prompt = _prompt_blocks.build_system_prompt(config=sub, repo=repo, mode="run", skills=None)
    assert "meter in plan percent (max_percent 3 points per run)" in prompt
    plain = _prompt_blocks.build_system_prompt(config=Config(), repo=repo, mode="run", skills=None)
    assert "plan percent" not in plain


def test_verify_infer_false_pins_gatelessness_at_preflight(tmp_path: pathlib.Path) -> None:
    """verify_infer = false skips every inference tier, so a run can be gateless on purpose."""
    import json

    from agent6 import budget as agent6_budget
    from agent6 import event_log
    from agent6.config import Config

    (tmp_path / "AGENTS.md").write_text(
        "## Verify command\n\n```bash\ntrue\n```\n", encoding="utf-8"
    )
    budget = agent6_budget.BudgetTracker(max_usd=-1.0, max_tokens_fallback=-1, max_percent=-1.0)

    cfg_on = preflight.infer_verify_if_unset(
        Config(),
        tmp_path,
        mode="run",
        events=event_log.EventSink(tmp_path / "on.jsonl"),
        transcript_sink=mock.MagicMock(),
        budget=budget,
    )
    assert cfg_on.harness.verify_command, "the fence must infer when the knob is on"

    off_log = tmp_path / "off.jsonl"
    cfg_off = preflight.infer_verify_if_unset(
        Config.model_validate({"harness": {"verify_infer": False}}),
        tmp_path,
        mode="run",
        events=event_log.EventSink(off_log),
        transcript_sink=mock.MagicMock(),
        budget=budget,
    )
    assert cfg_off.harness.verify_command == ()
    rows = [json.loads(line) for line in off_log.read_text(encoding="utf-8").splitlines()]
    assert any(r["type"] == "loop.verify_inferred" and r["source"] == "disabled" for r in rows)


def test_verify_infer_false_pins_gatelessness_at_adoption(tmp_path: pathlib.Path) -> None:
    """The same knob turns mid-run adoption off.

    Inside a container whose python3 lacks pytest the adopted gate was an always-red no-op.
    """
    from agent6.config import Config
    from agent6.harness import _loop_state, loop

    (tmp_path / "AGENTS.md").write_text(
        "## Verify command\n\n```bash\ntrue\n```\n", encoding="utf-8"
    )
    dispatcher = mock.MagicMock()
    wf = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=Config.model_validate({"harness": {"verify_infer": False}}),
        provider=mock.MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _line: None,
    )
    turn = _loop_state.TurnState(iteration=1, resp=mock.MagicMock(), assistant=mock.MagicMock())
    wf.gate.maybe_adopt(mock.MagicMock(), turn)
    dispatcher.adopt_verify_command.assert_not_called()
    assert wf.gate.command(_verify_verdict.VerifyVerdict()) == ()


def test_prompt_says_nothing_commits_under_commit_per_step_off(tmp_path: pathlib.Path) -> None:
    """With `[git].commit_per_step = false` the prompt never promises a commit after a green."""
    repo = _repo(tmp_path)
    cfg = Config.model_validate(
        {"harness": {"verify_command": ["true"]}, "git": {"commit_per_step": False}}
    )
    out = _prompt_blocks.build_system_prompt(config=cfg, repo=repo, mode="run", skills=None)
    assert "Nothing commits automatically" in out
    assert "The harness commits" not in out and "You own git" not in out


def test_hardened_rule_renders_only_where_the_jail_carries_protect_paths(
    tmp_path: pathlib.Path,
) -> None:
    """The placeholder rule renders only when Landlock carves around protect paths.

    An ordinary hardened run has none and was told to `apply_edit` placeholders it never needed.
    """
    repo = _repo(tmp_path)
    cfg = Config.model_validate({"harness": {"verify_command": ["true"]}})
    plain = _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="hardened"
    )
    assert "cannot CREATE new" not in plain
    carved = _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="hardened", protected_paths=True
    )
    assert "cannot CREATE new" in carved


def test_the_dag_block_and_tools_follow_the_curator(tmp_path: pathlib.Path) -> None:
    """A run built without a curator is neither taught nor offered the task tools.

    Every call errored "DAG curator not available", and the block's first instruction was
    unsatisfiable.
    """
    from agent6.harness import _toolset

    repo = _repo(tmp_path)
    cfg = Config.model_validate({"prompt": {"decompose": "on"}})
    with_dag = _prompt_blocks.build_system_prompt(config=cfg, repo=repo, mode="run", skills=None)
    without = _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, dag_available=False
    )
    assert "<decompose-first>" in with_dag and "add_task" in with_dag
    assert "<decompose-first>" not in without and "add_task" not in without
    names = {
        t.name
        for t in _toolset.tool_definitions(
            dispatch.ToolDispatcher(root=tmp_path, config=cfg), mode="run"
        )
    }
    assert not {"add_task", "update_task", "list_tasks"} & names


def test_the_stale_gate_sentence_is_run_mode_only(tmp_path: pathlib.Path) -> None:
    """The verify block names `finish_session`'s stale_gate field in run prompts only."""
    repo = _repo(tmp_path)
    cfg = _cfg(verify=True)
    assert "stale_gate" in _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None
    )
    assert "stale_gate" not in _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="plan", skills=None
    )
    assert "stale_gate" not in _prompt_blocks.build_system_prompt(
        config=cfg, repo=repo, mode="ask", skills=None
    )


def test_a_resumes_key_check_precedes_isolation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume runs the fresh run's preflight in the same place, pricing the model too."""
    from agent6.app import resume

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "order-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": "order-AAAA11", "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": _snapshot.SNAPSHOT_VERSION,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 2,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": [],
            }
        ),
        encoding="utf-8",
    )
    effective = layer.EffectiveConfig(config=_role_cfg({}), sources={}, layers=())
    seen: list[str] = []

    class _Stop(Exception):  # noqa: N818  # a signal, not an error  # a signal, not an error
        pass

    def _effective(*_a: object, **_k: object) -> layer.EffectiveConfig:
        return effective

    def _route(*_a: object, **_k: object) -> bool:
        seen.append("route_preflight")
        return True

    def _isolation(*_a: object, **_k: object) -> str:
        seen.append("select_isolation")
        raise _Stop

    monkeypatch.setattr(layer, "load_effective", _effective)
    monkeypatch.setattr(preflight, "route_preflight", _route)
    monkeypatch.setattr(app__session, "select_isolation", _isolation)
    with pytest.raises(_Stop):
        resume.resume_task(
            None, "order-AAAA11", started_at=time.time(), frontend=mock.MagicMock(), force=False
        )
    assert seen == ["route_preflight", "select_isolation"]


@pytest.mark.parametrize("configured", [True, False])
def test_a_gate_withheld_on_resume_is_one_clipped_line(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    configured: bool,
) -> None:
    """A resume under withheld commands reports the gate change once, with the argv clipped.

    One cause was reported up to three times, kilobytes of argv each time.
    """
    from agent6.app import _execution as app__execution
    from agent6.app import resume

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    gate = ("pytest", "-q", *(f"--deselect=tests/test_{i}.py::test_case" for i in range(40)))
    assert len(" ".join(gate)) > 4 * preflight.GATE_TEXT_WIDTH
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "withheld-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "withheld-AAAA11",
                "mode": "run",
                "user_task": "t",
                "harness": {"verify_command": list(gate), "verify_origin": "configured"},
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": _snapshot.SNAPSHOT_VERSION,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 2,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": list(gate),
            }
        ),
        encoding="utf-8",
    )
    harness = {"verify_command": list(gate)} if configured else {}
    cfg = _role_cfg({"sandbox": {"run_commands": "no"}, "harness": harness})
    effective = layer.EffectiveConfig(config=cfg, sources={}, layers=())

    def _effective(*_a: object, **_k: object) -> layer.EffectiveConfig:
        return effective

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _none(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(layer, "load_effective", _effective)
    monkeypatch.setattr(app__session, "select_isolation", _strict)
    monkeypatch.setattr(app__setup, "check_provider_keys", _none)
    monkeypatch.setattr(git_ops, "verify_git_identity", _none)

    def _execution(
        _cfg: Config, _layout: object, inputs: app__execution.ExecutionInputs, **_kw: object
    ) -> app__execution.ExecutionEnd:
        inputs.gate(_cfg, mock.MagicMock())
        return app__execution.ExecutionEnd(0)

    monkeypatch.setattr(app__execution, "run_execution", _execution)
    rc = resume.resume_task(
        None, "withheld-AAAA11", started_at=time.time(), frontend=mock.MagicMock(), force=False
    )
    assert rc == 0

    err = capsys.readouterr().err
    gate_lines = [ln for ln in err.splitlines() if "verify gate" in ln or "verify command" in ln]
    assert len(gate_lines) == 1, gate_lines
    (line,) = gate_lines
    assert "commands are withheld" in line and "(pytest -q --deselect" in line
    argv_text = line[line.index("(") + 1 : line.rindex("):")]
    assert argv_text.endswith("\u2026") and len(argv_text) == preflight.GATE_TEXT_WIDTH
