# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` summary keeps advisory statuses distinct from PASS."""

from __future__ import annotations

import pathlib
import shlex
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import event_log
from agent6.app import _setup, preflight
from agent6.config import Config
from agent6.sandbox import detect
from agent6.ui.cli import check_cmds as cli_check_cmds  # pyright: ignore[reportPrivateUsage]


def test_no_providers_is_info_not_pass(capsys: pytest.CaptureFixture[str]) -> None:
    # A fresh setup is unusable until `agent6 connect`; that instruction is not a PASS.
    checks = cli_check_cmds._doctor_check_config(Config())
    by_name = {c.name: c for c in checks}
    assert by_name["config.provider_keys"].status == "INFO"
    assert "agent6 connect" in by_name["config.provider_keys"].detail
    assert by_name["config.git_policy"].status == "PASS"
    # Sections state facts; the one summary states every verdict.
    assert "[INFO]" not in capsys.readouterr().out


def test_check_summary_carries_info_through(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `check verify` on a default config: verify_command is unset, an advisory INFO at exit 0.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host-only PATH entry cannot make a verify command executable inside the jail."""
    from agent6.ui.cli import check_cmds

    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    executable = host_bin / "host-only"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(host_bin))
    cfg = Config.model_validate({"harness": {"verify_command": ["host-only"]}})

    checks = check_cmds._doctor_check_verify(cfg)  # pyright: ignore[reportPrivateUsage]

    assert [(check.name, check.status) for check in checks] == [("verify.head", "FAIL")]


def test_check_verify_says_what_this_repo_infers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With verify_command unset, `check verify` names the command a run here would infer.

    It says when there is nothing to infer from.
    """
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert "unset; nothing here to infer from" in capsys.readouterr().out
    (tmp_path / "verify.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "verify.sh").chmod(0o755)
    assert main(["check", "verify"]) == 0
    assert "unset; a run here infers ./verify.sh from verify.sh" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("fence", "command"),
    [
        ("RUSTUP_HOME=/x cargo test", "sh -c 'RUSTUP_HOME=/x cargo test'"),
        ("go test ./... # all packages", "go test ./..."),
        ("node --test\ngo test ./...", "sh -c 'node --test && go test ./...'"),
        ("$ go test ./...", "go test ./..."),
        ("pytest 'a #b' &&\n  ruff check", "sh -c 'pytest '\"'\"'a #b'\"'\"' && ruff check'"),
    ],
)
def test_check_verify_reads_the_agents_md_fence_as_shell(
    fence: str,
    command: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An assignment, a comment, a second line or a `$ ` prompt in the fence is not an argument."""
    from agent6.ui.cli import main

    (tmp_path / "AGENTS.md").write_text(
        f"## Verify command\n\n```bash\n{fence}\n```\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert f"unset; a run here infers {command} from AGENTS.md\n" in capsys.readouterr().out


def test_run_header_prints_the_inferred_gate_as_check_verify_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run header prints an inferred pipeline shell-quoted, the form `check verify` prints.

    Joined on bare spaces, the header's `sh -c` line pastes as a different command.
    """
    from agent6.ui.cli import main

    (tmp_path / "AGENTS.md").write_text(
        "## Verify command\n\n```bash\npytest -q &&\n  mypy --strict pkg\n```\n", encoding="utf-8"
    )
    command = "sh -c 'pytest -q && mypy --strict pkg'"
    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert f"unset; a run here infers {command} from AGENTS.md\n" in capsys.readouterr().out

    cfg = preflight.infer_verify_if_unset(
        Config(),
        tmp_path,
        mode="run",
        events=event_log.EventSink(tmp_path / "logs.jsonl"),
        transcript_sink=mock.MagicMock(),
        budget=agent6_budget.BudgetTracker(max_usd=-1.0, max_tokens_fallback=-1, max_percent=-1.0),
    )
    header = capsys.readouterr().err
    assert f"verify_command not set; inferred from agents_md: {command}\n" in header
    assert shlex.split(command) == list(cfg.harness.verify_command)


def test_boundaries_reports_spawned_mcp_as_unconfined_without_a_jail(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resolved host network is not an effective boundary when an explicit session is refused."""
    from agent6.ui.cli import check_cmds

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[sandbox]\nisolation = "hardened"\nnetwork = "session"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_setup, "detect_env", object)

    def _hardened(_requested: str, _env: object) -> str:
        return "hardened"

    def _no_degradation(_env: object) -> None:
        return None

    monkeypatch.setattr(detect, "resolve_isolation", _hardened)
    monkeypatch.setattr(detect, "degrade_reason", _no_degradation)

    rc = check_cmds._cmd_check(  # pyright: ignore[reportPrivateUsage]
        config_path, section="boundaries"
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] boundaries: a run would refuse:" in out.split("== summary ==", 1)[1]
    assert "sandbox.network = 'session' requires the strict isolation" in out


def test_boundaries_alone_prints_no_empty_summary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The boundaries section reaches no verdict, so its own invocation prints no empty summary."""
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
