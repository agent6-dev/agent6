# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` summary keeps advisory statuses distinct from PASS."""

from __future__ import annotations

import base64
import dataclasses
import json
import pathlib
import shlex
import time
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import event_log, paths, secret_store
from agent6.app import _setup, preflight
from agent6.config import ChatGPTProviderEntry, Config, load_config
from agent6.providers import _oauth_protocol, _transport, oauth, types
from agent6.sandbox import detect
from agent6.ui.cli import check_cmds as cli_check_cmds  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli import main


@pytest.mark.parametrize("name", ["chatgpt", "work"])
@pytest.mark.parametrize("state", ["missing", "legacy", "rebound", "contradiction"])
def test_chatgpt_diagnosis_is_shared_by_check_and_model(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    state: str,
) -> None:
    """Every model surface preserves its output while naming the credential check's cause."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f'[providers.{name}]\napi_format = "chatgpt"\n'
        f'[models.worker]\nprovider = "{name}"\nmodel = "m1"\n'
    )
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    claim = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "claim"}})
    access = "e30." + base64.urlsafe_b64encode(claim.encode()).decode().rstrip("=") + ".sig"
    if state == "legacy":
        secret_store.save_secret(
            name,
            "",
            extra={
                "oauth_access_token": access,
                "oauth_refresh_token": "refresh-secret",
                "oauth_expires_at": "9999999999",
            },
        )
    elif state != "missing":
        grant = oauth.stored_grant_for(
            _oauth_protocol.TokenGrant(access, "refresh-secret", 3600), entry, now=time.time()
        )
        secret_store.save_oauth_grant(
            name,
            dataclasses.replace(
                grant,
                binding="other" if state == "rebound" else grant.binding,
                account_id="contradicting" if state == "contradiction" else "claim",
            ),
        )
    with pytest.raises(types.ProviderError) as refused:
        oauth.OAuthCredential(name, entry).token()
    cause = str(refused.value)
    assert f"agent6 connect {name}`" in cause
    assert {
        "missing": "No ChatGPT sign-in",
        "legacy": "predates credential binding",
        "rebound": "other provider settings",
        "contradiction": "does not match its own token",
    }[state] in cause
    assert _setup.check_provider_keys(load_config(config)) == cause
    with mock.patch.object(_transport, "http_request") as request:
        assert main(["check", "config"]) == 1
        checked = capsys.readouterr()
        assert f"[FAIL] config.provider_keys: {cause}" in checked.out
        for command in (["model"], ["model", "worker", name], ["model", "worker", f"{name}/m2"]):
            assert main(command) == 0
            output = capsys.readouterr()
            assert cause in output.err
            assert output.out
            assert access not in output.out + output.err
            assert "refresh-secret" not in output.out + output.err
        request.assert_not_called()
    worker = load_config(config).models.worker
    assert worker is not None and worker.model == "m2"


def test_healthy_local_sign_in_check_does_not_claim_remote_acceptance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A listing failure cannot turn a local credential check into remote authentication."""
    monkeypatch.chdir(tmp_path)
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.chatgpt]\napi_format = "chatgpt"\n'
        '[models.worker]\nprovider = "chatgpt"\nmodel = "m1"\n'
    )
    grant = oauth.stored_grant_for(
        _oauth_protocol.TokenGrant("AT", "RT", 3600),
        ChatGPTProviderEntry(api_format="chatgpt"),
        now=time.time(),
    )
    secret_store.save_oauth_grant("chatgpt", dataclasses.replace(grant, account_id="account"))
    with mock.patch.object(_transport, "http_request", side_effect=types.ProviderError("offline")):
        assert main(["check", "config"]) == 0
    out = capsys.readouterr().out
    assert "[PASS] config.provider_keys" in out
    assert "resolve locally" in out and "remote authentication is not verified" in out


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


@pytest.mark.usefixtures("jail_bin")
@pytest.mark.parametrize(
    ("fence", "command"),
    [
        ("RUSTUP_HOME=/x cargo test", "sh -c 'RUSTUP_HOME=/x cargo test'"),
        ("go test ./... # all packages", "go test ./..."),
        ("node --test\ngo test ./...", "sh -c 'node --test && go test ./...'"),
        ("$ go test ./...", "go test ./..."),
        ("$ go test ./...\n$", "go test ./..."),
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
    """An assignment, a comment, a second line or a `$` prompt in the fence is not an argument."""
    from agent6.ui.cli import main

    (tmp_path / "AGENTS.md").write_text(
        f"## Verify command\n\n```bash\n{fence}\n```\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert f"unset; a run here infers {command} from AGENTS.md\n" in capsys.readouterr().out


@pytest.mark.usefixtures("jail_bin")
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
        budget=agent6_budget.BudgetTracker(
            max_usd=-1.0, max_tokens_fallback=-1, max_plan_points=-1.0
        ),
    )
    header = capsys.readouterr().err
    assert f"verify_command not set; inferred from AGENTS.md: {command}\n" in header
    assert shlex.split(command) == list(cfg.harness.verify_command)


def test_check_verify_and_the_run_header_say_why_a_manifest_was_passed_over(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    jail_bin: pathlib.Path,
) -> None:
    """A manifest whose runner the sandbox lacks is named with the reason, never as the gate.

    Offered unchecked, `npm test` aborted the run at its first gate as unexecutable.
    """
    from agent6.ui.cli import main

    (jail_bin / "npm").unlink()
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    why = "passed over: package.json names `npm test --silent`, and the sandbox has no `npm`"
    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    out = capsys.readouterr().out
    assert "infers" not in out and "nothing here to infer from" not in out
    assert "unset; nothing here a run can use (a run asks the reviewer model" in out
    assert f"else goes gateless); {why}\n" in out

    (tmp_path / "Makefile").write_text("test:\n\t./run\n", encoding="utf-8")
    assert main(["check", "verify"]) == 0
    assert f"unset; a run here infers make test from Makefile:test; {why}\n" in (
        capsys.readouterr().out
    )

    log = tmp_path / "logs.jsonl"
    cfg = preflight.infer_verify_if_unset(
        Config(),
        tmp_path,
        mode="run",
        events=event_log.EventSink(log),
        transcript_sink=mock.MagicMock(),
        budget=agent6_budget.BudgetTracker(
            max_usd=-1.0, max_tokens_fallback=-1, max_plan_points=-1.0
        ),
    )
    header = capsys.readouterr().err
    assert cfg.harness.verify_command == ("make", "test")
    assert "verify_command not set; inferred from Makefile:test: make test\n" in header
    assert f"\n         {why}\n" in header


@pytest.mark.usefixtures("jail_bin")
def test_check_verify_and_the_run_header_name_the_manifests_the_gate_leaves_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Beside the manifest the gate came from, the other ecosystems' manifests are named.

    Unnamed, a Go regression passed a gate that ran only the Node tests, with no word of it.
    """
    from agent6.ui.cli import main

    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    left_out = "passed over: go.mod names `go test ./...`, which this gate does not run"
    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert f"unset; a run here infers npm test --silent from package.json; {left_out}\n" in (
        capsys.readouterr().out
    )

    cfg = preflight.infer_verify_if_unset(
        Config(),
        tmp_path,
        mode="run",
        events=event_log.EventSink(tmp_path / "logs.jsonl"),
        transcript_sink=mock.MagicMock(),
        budget=agent6_budget.BudgetTracker(
            max_usd=-1.0, max_tokens_fallback=-1, max_plan_points=-1.0
        ),
    )
    header = capsys.readouterr().err
    assert cfg.harness.verify_command == ("npm", "test", "--silent")
    assert "verify_command not set; inferred from package.json: npm test --silent\n" in header
    assert f"\n         {left_out}\n" in header


def test_sandbox_reports_spawned_mcp_as_unconfined_without_a_jail(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An isolation-none policy does not enforce the path grants printed for a jailed server."""
    from agent6.ui.cli import _check_sandbox

    cfg = Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"notes": {"command": ["notes-mcp"]}}}}
    )

    _check_sandbox._mcp_lines(cfg, tmp_path, "none")  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr().out
    assert "notes: spawned UNCONFINED" in out
    assert "spawned in the jail" not in out
    assert "paths ro+" not in out


_ENV = detect.Environment(
    in_container=False,
    container_signals=(),
    kernel=detect.KernelInfo(raw="6.8", major=6, minor=8),
    userns_supported=True,
    landlock_abi=5,
    seccomp_arch_supported=True,
    sandbox_available=True,
)


def _stub_probes(monkeypatch: pytest.MonkeyPatch, isolation: str) -> None:
    """Pin the host at one level and make the jail probes pass without a launcher."""
    from agent6 import kinds
    from agent6.ui.cli import _check_sandbox

    monkeypatch.setattr(_setup, "detect_env", lambda: _ENV)

    def _select(_requested: str, _env: object) -> str:
        return isolation

    def _no_degradation(_env: object) -> None:
        return None

    def _run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        rc = 2 if policy.argv[0].endswith("getent") else 0  # the DNS probe reads blocked
        return kinds.CommandResult(
            argv=policy.argv, returncode=rc, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(detect, "resolve_isolation", _select)
    monkeypatch.setattr(detect, "degrade_reason", _no_degradation)
    monkeypatch.setattr(_check_sandbox, "landlock_abi", lambda: 5)
    monkeypatch.setattr(_check_sandbox, "run_in_jail", _run)


def test_sandbox_fails_when_a_run_would_refuse_the_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resolved host network is not an effective boundary when an explicit session is refused."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[sandbox]\nisolation = "partial"\nnetwork = "session"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _stub_probes(monkeypatch, "partial")

    rc = cli_check_cmds._cmd_check(  # pyright: ignore[reportPrivateUsage]
        (config_path,), section="sandbox"
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] sandbox.refusal: a run would refuse:" in out.split("== summary ==", 1)[1]
    assert "sandbox.network = 'session' requires sandbox.isolation = 'full'" in out


def test_sandbox_alone_prints_its_summary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`check sandbox` on its own ends with the three rows it owns."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    _stub_probes(monkeypatch, "full")
    assert main(["check", "sandbox"]) == 0
    out = capsys.readouterr().out
    assert "== sandbox ==" in out
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
    summary = out.split("== summary ==", 1)[1]
    assert "[PASS] sandbox.isolation: selected 'full'" in summary
    assert "[PASS] sandbox.refusal: every explicit setting works on 'full'" in summary
    assert "[PASS] sandbox: all jail probes passed" in summary
