# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for Phase 4 machine ergonomics: status, poke, run --exit-on-wait."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.cli import main
from agent6.config_layer import resolved_state_dir
from agent6.machine import MachineJournal

WAITER_DELAYED = """
machine = "waiter_delayed"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def _write_machine(tmp_path: Path) -> Path:
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER_DELAYED, encoding="utf-8")
    return f


def test_run_exit_on_wait_yields_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    out = capsys.readouterr().out
    assert "WAITING" in out
    # The wait was armed and persisted.
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    pending = MachineJournal(root).read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"


def test_status_reports_waiting_state_and_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "waiter_delayed" in out
    assert "next wake:" in out
    assert "spend: $0.0000" in out


def test_status_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "status", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err


def test_poke_drops_signal_for_waiting_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "waiter_delayed"])
    assert code == 0
    assert "poked" in capsys.readouterr().out
    # The signal is now pending for the next take_signal().
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() is True


def test_poke_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "poke", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err


AGENT_MACHINE_HARD = """
machine = "hard-usd"
version = 1
initial = "judge"

[budget]
max_usd = 1.0
max_transitions = 10

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


def _hard_usd_cfg(model: str):
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"p": {"kind": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": model}},
        }
    )


def _load_spec(tmp_path: Path, body: str):
    from agent6.machine import load_machine

    f = tmp_path / "m.asm.toml"
    f.write_text(body, encoding="utf-8")
    return load_machine(f)


def test_hard_usd_preflight_refuses_unpriced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [budget].max_usd is a hard promise; an unpriced inherited worker model
    # cannot honor it, so machine run must refuse up front.
    from agent6.cli.machine_cmds import (
        _hard_usd_preflight_error,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    spec = _load_spec(tmp_path, AGENT_MACHINE_HARD)
    err = _hard_usd_preflight_error(spec, _hard_usd_cfg("nobody/unpriced"))
    assert err is not None
    assert "max_usd" in err and "nobody/unpriced" in err and "judge" in err


def test_hard_usd_preflight_passes_best_effort_and_priced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json as _json

    from agent6.cli.machine_cmds import (
        _hard_usd_preflight_error,  # pyright: ignore[reportPrivateUsage]
    )

    # best_effort_usd_limit never refuses, priced or not.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    soft = AGENT_MACHINE_HARD.replace("max_usd = 1.0", "best_effort_usd_limit = 1.0")
    spec = _load_spec(tmp_path, soft)
    assert _hard_usd_preflight_error(spec, _hard_usd_cfg("nobody/unpriced")) is None

    # max_usd passes once the model has price data.
    cache = tmp_path / "cache"
    (cache / "models").mkdir(parents=True)
    (cache / "models" / "p.json").write_text(
        _json.dumps({"models": ["priced/m"], "pricing": {"priced/m": [1.0, 2.0]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))
    spec_hard = _load_spec(tmp_path, AGENT_MACHINE_HARD)
    assert _hard_usd_preflight_error(spec_hard, _hard_usd_cfg("priced/m")) is None


def test_hard_usd_preflight_checks_per_state_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A soft machine budget with a hard per-state max_usd still requires price
    # data for that state's model.
    from agent6.cli.machine_cmds import (
        _hard_usd_preflight_error,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    body = AGENT_MACHINE_HARD.replace("max_usd = 1.0", "best_effort_usd_limit = 1.0").replace(
        'kind = "agent"', 'kind = "agent"\nmax_usd = 0.5\nmodel = "pinned/unpriced"', 1
    )
    spec = _load_spec(tmp_path, body)
    err = _hard_usd_preflight_error(spec, _hard_usd_cfg("priced/m"))
    assert err is not None and "pinned/unpriced" in err
