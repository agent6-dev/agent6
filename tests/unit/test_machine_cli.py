# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for Phase 4 machine ergonomics: status, poke, run --exit-on-wait."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.cli import main
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
    root = tmp_path / ".agent6" / "machines" / "waiter_delayed"
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
    root = tmp_path / ".agent6" / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() is True


def test_poke_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "poke", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err
