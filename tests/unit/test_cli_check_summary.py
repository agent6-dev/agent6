# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` summary keeps advisory statuses distinct from PASS."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.ui.cli.check_cmds import _doctor_check_config  # pyright: ignore[reportPrivateUsage]


def test_no_providers_is_info_not_pass(capsys: pytest.CaptureFixture[str]) -> None:
    # A fresh setup (zero providers) is unusable until `agent6 connect`; the
    # check must not render that instruction as a PASS.
    checks = _doctor_check_config(Config())
    by_name = {c.name: c for c in checks}
    assert by_name["config.provider_keys"].status == "INFO"
    assert "agent6 connect" in by_name["config.provider_keys"].detail
    assert by_name["config.git_policy"].status == "PASS"
    assert "[INFO] config.provider_keys" in capsys.readouterr().out


def test_check_summary_carries_info_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `check verify` on a default config: verify_command is unset, an advisory.
    # The summary line must say INFO (previously coerced to PASS) and exit 0.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["check", "verify"])
    assert rc == 0
    out = capsys.readouterr().out
    summary = out.split("== summary ==", 1)[1]
    assert "[INFO] verify.argv" in summary
    assert "[PASS]" not in summary
    assert "—" not in out
