# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The startup consent gate for the one dangerous combination: the sandbox is
disabled AND run_command is auto-approved (no confinement, no prompt)."""

from __future__ import annotations

import pytest

from agent6.cli.run import _confirm_unconfined_autorun  # pyright: ignore[reportPrivateUsage]
from agent6.config import Config


def _cfg(run_commands: str) -> Config:
    return Config.model_validate({"sandbox": {"run_commands": run_commands}})


def test_no_gate_when_sandboxed() -> None:
    # Auto-approve while jailed is normal, safe operation: no confirm.
    assert _confirm_unconfined_autorun("strict", _cfg("yes")) is True
    assert _confirm_unconfined_autorun("hardened", _cfg("yes")) is True


def test_no_gate_when_unconfined_but_still_prompting() -> None:
    # Sandbox off but run_command still asks per command: the per-command prompt
    # is the checkpoint, so no extra startup gate.
    assert _confirm_unconfined_autorun("none", _cfg("ask")) is True
    assert _confirm_unconfined_autorun("none", _cfg("no")) is True


def test_non_interactive_combo_proceeds_with_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No TTY (CI, machine run): the explicit opt-outs are the consent; proceed
    # with a loud warning rather than blocking automation.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _confirm_unconfined_autorun("none", _cfg("yes")) is True
    err = capsys.readouterr().err
    assert "DANGER" in err
    assert "non-interactive" in err


def _fixed_prompt(answer: str | None):  # type: ignore[no-untyped-def]
    def prompt(_text: str) -> str | None:
        return answer

    return prompt


def test_interactive_combo_requires_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("agent6.cli.run._tty_prompt", _fixed_prompt("y"))
    assert _confirm_unconfined_autorun("none", _cfg("yes")) is True


def test_interactive_combo_defaults_to_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # Empty answer (bare Enter) / None (no tty read) both abort.
    monkeypatch.setattr("agent6.cli.run._tty_prompt", _fixed_prompt(""))
    assert _confirm_unconfined_autorun("none", _cfg("yes")) is False
    monkeypatch.setattr("agent6.cli.run._tty_prompt", _fixed_prompt(None))
    assert _confirm_unconfined_autorun("none", _cfg("yes")) is False
