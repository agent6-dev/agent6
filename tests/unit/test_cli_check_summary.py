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
    # Sections state facts; the one summary states every verdict.
    assert "[INFO]" not in capsys.readouterr().out


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


def test_check_verify_uses_the_jail_path_not_the_ambient_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host-only PATH entry cannot make a verify command executable inside the jail."""
    from agent6.ui.cli import check_cmds

    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    executable = host_bin / "host-only"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(host_bin))
    cfg = Config.model_validate({"workflow": {"verify_command": ["host-only"]}})

    checks = check_cmds._doctor_check_verify(cfg)  # pyright: ignore[reportPrivateUsage]

    assert [(check.name, check.status) for check in checks] == [("verify.head", "FAIL")]


def test_check_verify_says_what_this_repo_infers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With verify_command unset, `check verify` names the command a run here
    would infer (the deterministic tiers) rather than "inferred per run", and
    says when there is nothing to infer from."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert "unset; nothing here to infer from" in capsys.readouterr().out
    (tmp_path / "verify.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "verify.sh").chmod(0o755)
    assert main(["check", "verify"]) == 0
    assert "unset; a run here infers ./verify.sh (from verify.sh)" in capsys.readouterr().out


def test_boundaries_reports_spawned_mcp_as_unconfined_without_a_jail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An isolation-none policy does not enforce the path grants printed for a jailed server."""
    from agent6.ui.cli import check_cmds

    cfg = Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"notes": {"command": ["notes-mcp"]}}}}
    )

    check_cmds._boundaries_mcp(cfg, tmp_path, "none")  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr().out
    assert "notes: spawned UNCONFINED" in out
    assert "spawned in the jail" not in out
    assert "paths ro+" not in out


def test_boundaries_fails_when_a_run_would_refuse_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resolved host network is not an effective boundary when an explicit session is refused."""
    from agent6.ui.cli import check_cmds

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[sandbox]\nisolation = "hardened"\nnetwork = "session"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _hardened(_requested: str, _env: object) -> str:
        return "hardened"

    def _no_degradation(_env: object) -> None:
        return None

    monkeypatch.setattr(check_cmds, "resolve_isolation", _hardened)
    monkeypatch.setattr(check_cmds, "degrade_reason", _no_degradation)

    rc = check_cmds._cmd_check(  # pyright: ignore[reportPrivateUsage]
        config_path, section="boundaries"
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] boundaries: a run would refuse:" in out.split("== summary ==", 1)[1]
    assert "sandbox.network = 'session' requires the strict isolation" in out


def test_boundaries_alone_prints_no_empty_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The boundaries section reports facts and reaches no verdict, so its own
    invocation ended on a bare `== summary ==` that read "nothing ran"."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["check", "boundaries"]) == 0
    out = capsys.readouterr().out
    assert "== boundaries ==" in out
    file_tools = out.split("in-process file tools (", 1)[1].splitlines()[0]
    for name in (
        "read_file",
        "list_dir",
        "outline",
        "find_definition",
        "find_references",
        "apply_edit",
        "apply_patch",
    ):
        assert name in file_tools
    assert "== summary ==" not in out
