# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 --config FILE machine run` honours the flag layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

_MACHINE = """\
machine = "tiny"
version = 1
initial = "done"

[budget]
max_transitions = 5

[states.done]
kind   = "terminal"
status = "ok"
reason = "nothing to do"
"""

_AGENT_MACHINE = """\
machine = "agent-config"
version = 1
initial = "judge"

[budget]
max_transitions = 5

[schemas.verdict]
ok = "bool"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_secs = 5
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


def test_machine_run_reads_the_explicit_config_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/config.md presents --config as a general layer; machine run
    resolved without it, so the file the operator named was ignored."""
    from agent6.app.machine.run import run_machine

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[machine]\nsnapshot_keep = 41\n", encoding="utf-8")
    mfile = tmp_path / "tiny.asm.toml"
    mfile.write_text(_MACHINE, encoding="utf-8")

    seen: list[int] = []
    from agent6.app.machine import run as run_mod

    real = run_mod.load_effective_with_overlay

    def spy(repo_root: Path, overlay: dict[str, object], **kw: object):
        eff = real(repo_root, overlay, **kw)  # pyright: ignore[reportArgumentType]
        seen.append(eff.config.machine.snapshot_keep)
        return eff

    monkeypatch.setattr(run_mod, "load_effective_with_overlay", spy)
    frontend = MagicMock()
    frontend.reporter = MagicMock()
    run_machine(mfile, frontend, config_path=explicit)
    assert seen and seen[0] == 41, "the --config layer never reached machine run"


def test_explicit_config_reaches_each_agent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The supervisor validated `--config`, but passed only the machine-file
    overlay to child agent runs, so their effective config silently lost it."""
    from agent6.app.machine import run as run_mod
    from agent6.machine import AgentExecResult

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path.parent / f"{tmp_path.name}-config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path.parent / f"{tmp_path.name}-state"))
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(
        """\
[providers.local]
api_format = "openai"
base_url = "http://127.0.0.1:9"

[models.worker]
provider = "local"
model = "test-model"

[workflow]
max_iterations = 17
""",
        encoding="utf-8",
    )
    mfile = tmp_path / "agent.asm.toml"
    mfile.write_text(_AGENT_MACHINE, encoding="utf-8")
    child_overlays: list[dict[str, object]] = []

    def fake_build(overlay: dict[str, object], *_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        child_overlays.append(overlay)

        def run(*_args: object, **_kwargs: object) -> AgentExecResult:
            return AgentExecResult(reason="finish_session", payload={"ok": True})

        return run

    monkeypatch.setattr(run_mod, "build_machine_agent_runner", fake_build)
    frontend = MagicMock()
    frontend.reporter = MagicMock()

    code = run_mod.run_machine(mfile, frontend, config_path=explicit)
    assert code == 0, frontend.reporter.mock_calls
    assert child_overlays[0]["workflow"]["max_iterations"] == 17  # type: ignore[index]
