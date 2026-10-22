# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A machine `tool` state receives an operator secret only through two declarations.

Its own `pass_env` names the variable, the operator's `[machine].pass_env` allows it (never the
machine overlay), and the run refuses at startup when the two disagree.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6.app.machine import StateConfig, machine_pass_env_refusal, run
from agent6.config import Config
from agent6.machine import MachineError, ToolState, load_machine
from agent6.ui.cli import machine_check  # pyright: ignore[reportPrivateUsage]

MACHINE = """
machine = "secretdemo"
version = 1
initial = "fetch"

[budget]
max_usd = 1.0
max_transitions = 100

[states.fetch]
kind = "tool"
command = ["fetch"]
timeout_s = 5
pass_env = ["X_TOKEN", "X_REGION"]
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def _write(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "secret.asm.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_tool_state_declares_the_variables_it_wants(tmp_path: pathlib.Path) -> None:
    spec = load_machine(_write(tmp_path, MACHINE))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    assert fetch.pass_env == ("X_TOKEN", "X_REGION")
    with pytest.raises(MachineError, match="invalid environment variable name"):
        load_machine(_write(tmp_path, MACHINE.replace('"X_REGION"', '"BAD NAME"')))


def test_the_operator_allowlist_is_never_a_machine_overlay_or_a_provider_key(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(MachineError, match=r"machine\.pass_env"):
        load_machine(_write(tmp_path, MACHINE + '\n[config.machine]\npass_env = ["X_TOKEN"]\n'))
    with pytest.raises(ValueError, match="never passes a provider key to a machine's tool"):
        Config.model_validate(
            {
                "providers": {"anthropic": {"api_format": "anthropic", "api_key_env": "MY_KEY"}},
                "machine": {"pass_env": ["MY_KEY"]},
            }
        )


def test_a_variable_the_operator_has_not_allowed_refuses_the_run(tmp_path: pathlib.Path) -> None:
    spec = load_machine(_write(tmp_path, MACHINE))
    refusal = machine_pass_env_refusal(Config(), spec.states)
    assert refusal is not None and "[states.fetch] asks for X_TOKEN, X_REGION" in refusal
    assert "[machine].pass_env does not allow" in refusal
    allowed = Config.model_validate({"machine": {"pass_env": ["X_TOKEN", "X_REGION"]}})
    assert machine_pass_env_refusal(allowed, spec.states) is None


def test_a_pass_env_refusal_ends_the_run_at_startup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state asking for a variable the operator has not allowed refuses before anything runs."""
    from unittest import mock

    monkeypatch.chdir(tmp_path)
    mfile = _write(tmp_path, MACHINE)
    reporter = mock.MagicMock()
    assert run.run_machine(mfile, reporter) == 2
    said = " ".join(str(c.args[0]) for c in reporter.refuse.call_args_list)
    assert "[states.fetch] asks for X_TOKEN, X_REGION" in said


def test_the_jail_gets_only_the_declared_variables_that_are_set(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("X_TOKEN", "t0k3n")
    monkeypatch.delenv("X_REGION", raising=False)
    monkeypatch.setenv("OTHER_SECRET", "no")
    factory = run.machine_tool_policy_factory(
        {"fetch": StateConfig(Config(), "full")},
        tmp_path,
        protect_paths=(),
        data_dir=None,
        shared_writes=(),
    )
    env = dict(factory("fetch", ("fetch",), 5.0, ("X_TOKEN", "X_REGION")).env)
    assert env.get("X_TOKEN") == "t0k3n"
    assert "X_REGION" not in env and "OTHER_SECRET" not in env
    assert "X_TOKEN" not in dict(factory("fetch", ("fetch",), 5.0, ()).env)


def test_machine_check_names_every_declared_variable(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert machine_check._cmd_machine_check(_write(tmp_path, MACHINE)) == 0
    out = capsys.readouterr().out
    assert "[states.fetch] receives the environment variable(s) X_TOKEN, X_REGION" in out
