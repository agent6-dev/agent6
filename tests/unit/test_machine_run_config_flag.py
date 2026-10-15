# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 --config FILE machine run` honours the flag layer."""

from __future__ import annotations

import pathlib
from unittest import mock

import pytest

from agent6.app import machine_agent
from agent6.app import reporter as app_reporter
from agent6.config import layer

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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`machine run` resolves the top-level `--config` layer, as docs/config.md presents it."""
    from agent6.app.machine import run as machine_run

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[machine]\nsnapshot_keep = 41\n", encoding="utf-8")
    mfile = tmp_path / "tiny.asm.toml"
    mfile.write_text(_MACHINE, encoding="utf-8")

    seen: list[int] = []

    real = layer.load_effective_with_overlay

    def spy(repo_root: pathlib.Path, overlay: dict[str, object], **kw: object):
        eff = real(repo_root, overlay, **kw)  # pyright: ignore[reportArgumentType]
        seen.append(eff.config.machine.snapshot_keep)
        return eff

    monkeypatch.setattr(layer, "load_effective_with_overlay", spy)
    machine_run.run_machine(mfile, mock.MagicMock(), config_paths=(explicit,))
    assert seen and seen[0] == 41, "the --config layer never reached machine run"


def test_explicit_config_reaches_each_agent_state(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Child agent runs receive `--config` along with the machine-file overlay."""
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

[harness]
max_iterations = 17
""",
        encoding="utf-8",
    )
    mfile = tmp_path / "agent.asm.toml"
    mfile.write_text(_AGENT_MACHINE, encoding="utf-8")
    child_overlays: list[dict[str, object]] = []

    def fake_build(
        overlays: dict[str, machine_agent.StateOverlay], *_args: object, **_kwargs: object
    ):  # type: ignore[no-untyped-def]
        child_overlays.extend(entry.overlay for entry in overlays.values())

        def run(*_args: object, **_kwargs: object) -> AgentExecResult:
            return AgentExecResult(reason="finish_session", payload={"ok": True})

        return run

    monkeypatch.setattr(machine_agent, "build_machine_agent_runner", fake_build)
    reporter = mock.MagicMock()

    code = run_mod.run_machine(mfile, reporter, config_paths=(explicit,))
    assert code == 0, reporter.mock_calls
    assert child_overlays[0]["harness"]["max_iterations"] == 17  # type: ignore[index]


def test_a_states_own_budget_is_judged_before_it_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state's `config` sets the fallback cap the machine's config leaves at 0.

    Judged against the machine's config, the run was refused for a cap the state does not run
    under.
    """
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

[budget]
max_tokens_fallback = 0
""",
        encoding="utf-8",
    )
    mfile = tmp_path / "agent.asm.toml"
    mfile.write_text(
        _AGENT_MACHINE.replace(
            "timeout_secs = 5\n",
            "timeout_secs = 5\nconfig = { budget = { max_tokens_fallback = 100000 } }\n",
        ),
        encoding="utf-8",
    )

    def fake_build(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        def run(*_a: object, **_k: object) -> AgentExecResult:
            return AgentExecResult(reason="finish_session", payload={"ok": True})

        return run

    monkeypatch.setattr(machine_agent, "build_machine_agent_runner", fake_build)
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)

    assert run_mod.run_machine(mfile, reporter, config_paths=(explicit,)) == 0, said
    assert not [line for line in said if "max_tokens_fallback" in line], said
