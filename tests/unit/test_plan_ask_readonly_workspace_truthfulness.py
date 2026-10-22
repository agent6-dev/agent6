# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Truthfulness pin: plan/ask do not promise a read-only WORKSPACE.

Both modes withhold the structured edit tools (apply_edit/apply_patch), commit
tools, and the task DAG, but a jailed `run_command` can still write the
workspace as a side effect. So an absolute "you CANNOT change anything" or a
"read-only guarantee" is a lie; the prompts state the truth: commands run
jailed, a probe's writes land in the workspace, and nothing carries them
forward.

plan is clamped like ask: a standing run_commands="allow" becomes "ask", so a
write during planning is operator-approved, never auto-run.
"""

from __future__ import annotations

import pathlib
from typing import Literal

from agent6 import kinds
from agent6.config import Config, load_config
from agent6.harness import _prompt_blocks  # pyright: ignore[reportPrivateUsage]

_INTERACTIVE: tuple[Literal["plan", "ask"], ...] = ("plan", "ask")

_CONFIG_TOML = "\n".join(
    (
        "[agent6]",
        "config_version = 1",
        "[providers.anthropic]",
        'api_format = "anthropic"',
        'api_key_env = "ANTHROPIC_API_KEY"',
        "[models.worker]",
        'provider = "anthropic"',
        'model = "x"',
    )
)


def _prompt(tmp_path: pathlib.Path, mode: Literal["plan", "ask"]) -> str:
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(_CONFIG_TOML, encoding="utf-8")
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="",
        head_sha="",
        file_count=0,
        top_level=("notes.txt",),
        agents_md="",
        recent_log="",
        is_git=False,
    )
    return _prompt_blocks.build_system_prompt(
        config=load_config(cfg_path), repo=repo, mode=mode, skills=None
    )


def _norm(tmp_path: pathlib.Path, mode: Literal["plan", "ask"]) -> str:
    """Return the prompt lower-cased with runs of whitespace collapsed.

    A substring check then does not depend on where the prose happens to wrap.
    """
    return " ".join(_prompt(tmp_path, mode).lower().split())


def test_plan_and_ask_prompts_do_not_promise_a_read_only_workspace(tmp_path: pathlib.Path) -> None:
    for mode in _INTERACTIVE:
        low = _norm(tmp_path, mode)
        # A jailed run_command can write the workspace, so no false absolute.
        assert "cannot change anything" not in low, mode
        assert "cannot edit files" not in low, mode
        assert "read-only guarantee" not in low, mode
        # The truth is stated: commands run jailed and their writes land in
        # the workspace.
        assert "jailed" in low, mode
        assert "land in the workspace" in low, mode


def test_plan_and_ask_prompts_state_that_probe_writes_go_nowhere(tmp_path: pathlib.Path) -> None:
    """Withhold the false guarantee and state the consequence.

    Nothing carries a probe's writes forward, so an edit the answer or plan needs is described or
    recorded, never applied.
    """
    for mode in _INTERACTIVE:
        low = _norm(tmp_path, mode)
        assert "nothing carries them forward" in low, mode


def test_plan_clamps_run_commands_like_ask(tmp_path: pathlib.Path) -> None:
    """Plan clamps run_commands like ask.

    Plan runs with the operator present, so a standing `run_commands="allow"` is clamped to "ask"
    exactly as in ask: a write during planning is approved per call, never auto-run. The clamp only
    tightens (a configured "deny" stays "deny"). An allow-all session answer upgrades the
    clamped "ask" back to "allow".
    """
    from agent6.app import _setup
    from agent6.sessions import ipc

    allowed = Config.model_validate({"sandbox": {"run_commands": "allow"}})
    # Both interactive modes clamp allow->ask; run keeps the operator's allow.
    assert _setup.session_config(allowed, "plan").sandbox.run_commands == "ask"
    assert _setup.session_config(allowed, "ask").sandbox.run_commands == "ask"
    assert _setup.session_config(allowed, "run").sandbox.run_commands == "allow"
    # Only tightens: a withheld "deny" is never loosened by the clamp.
    denied = Config.model_validate({"sandbox": {"run_commands": "deny"}})
    assert _setup.session_config(denied, "plan").sandbox.run_commands == "deny"
    # The clamped "ask" still upgrades to "allow" for the session on allow-all.
    assert ipc.effective_run_commands("ask", tmp_path) == "ask"
    ipc.set_session_allow(tmp_path, ipc.COMMAND_SCOPE)
    assert ipc.effective_run_commands("ask", tmp_path) == "allow"
