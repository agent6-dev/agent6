# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for Phase 5 `agent6 machine create`: authoring prompts and CLI flow."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent6.cli import machine_cmds as cli  # create + its preflight helpers live here now
from agent6.cli import main
from agent6.machine import (
    TOML_PAYLOAD_KEY,
    AgentExecResult,
    AgentRequest,
    build_authoring_prompt,
    extract_toml,
)

VALID_MACHINE = """\
machine = "greeter"
version = 1
initial = "say"

[budget]
max_usd = 1.0
max_transitions = 10

[states.say]
kind = "tool"
command = ["echo", "hi"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "greeted"

[states.fail]
kind = "terminal"
status = "failed"
reason = "echo failed"
"""

INVALID_MACHINE = """\
machine = "greeter"
version = 1
initial = "nowhere"

[budget]
max_usd = 1.0
max_transitions = 10

[states.done]
kind = "terminal"
status = "ok"
reason = "x"
"""


# --- pure pieces -----------------------------------------------------------


def test_extract_toml_returns_string() -> None:
    assert extract_toml({TOML_PAYLOAD_KEY: "machine = 'x'"}) == "machine = 'x'"


@pytest.mark.parametrize(
    "payload",
    [None, {}, {TOML_PAYLOAD_KEY: ""}, {TOML_PAYLOAD_KEY: "   "}, {TOML_PAYLOAD_KEY: 7}],
)
def test_extract_toml_returns_none(payload: dict[str, object] | None) -> None:
    assert extract_toml(payload) is None  # type: ignore[arg-type]


def test_build_authoring_prompt_first_attempt() -> None:
    prompt = build_authoring_prompt("Poll a queue", attempt=1)
    assert "authoring guide" in prompt
    assert "Poll a queue" in prompt
    assert "finish_run" in prompt
    assert "fix the previous draft" not in prompt


def test_build_authoring_prompt_retry_includes_diagnostics() -> None:
    prompt = build_authoring_prompt(
        "Poll a queue",
        attempt=2,
        prior_toml="machine = 'bad'",
        diagnostics=["initial 'nowhere' names no state"],
    )
    assert "Attempt 2: fix the previous draft" in prompt
    assert "initial 'nowhere' names no state" in prompt
    assert "machine = 'bad'" in prompt


# --- CLI flow --------------------------------------------------------------


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def _require_runnable(*_a: object, **_k: object) -> None:
        return None

    def _load(_root: object, _explicit: object = None) -> object:
        cfg = SimpleNamespace(
            sandbox=SimpleNamespace(profile="none"),
            require_runnable=_require_runnable,
        )
        return SimpleNamespace(config=cfg)

    def _keys_ok(_cfg: object) -> str | None:
        return None

    def _profile(_profile_name: object, _env: object) -> object:
        return object()

    monkeypatch.setattr(cli, "load_effective", _load)
    monkeypatch.setattr(cli, "_check_provider_keys", _keys_ok)
    monkeypatch.setattr(cli, "select_profile", _profile)


def _stub_runner(monkeypatch: pytest.MonkeyPatch, results: Iterable[AgentExecResult]) -> None:
    seq = iter(results)

    def fake_build(
        cfg: object, root: Path, profile: object, transcript_dir: Path
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(_request: AgentRequest) -> AgentExecResult:
            return next(seq)

        return run

    monkeypatch.setattr(cli, "_build_machine_agent_runner", fake_build)


def test_create_inherits_worker_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The authoring agent must INHERIT the worker model (model=None), not get
    an empty-string override. `model=""` overwrote the worker model with "" and
    failed min_length validation, making every `machine create` attempt error
    out -- a path the request-ignoring stub runner never exercised."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    captured: list[AgentRequest] = []

    def fake_build(
        cfg: object, root: Path, profile: object, transcript_dir: Path
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest) -> AgentExecResult:
            captured.append(request)
            return AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.0
            )

        return run

    monkeypatch.setattr(cli, "_build_machine_agent_runner", fake_build)
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    assert captured, "runner was never invoked"
    assert captured[0].model is None  # inherit, not "" (which would fail to validate)


def test_create_writes_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [AgentExecResult(reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.02)],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    out = capsys.readouterr()
    written = tmp_path / "greeter.asm.toml"
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith('machine = "greeter"')
    assert "wrote draft" in out.err
    assert "spent ~$0.0200" in out.err


def test_create_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: INVALID_MACHINE}, usd=0.01
            ),
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.03
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    out = capsys.readouterr()
    assert (tmp_path / "greeter.asm.toml").exists()
    # both attempts' spend summed
    assert "spent ~$0.0400" in out.err
    assert "attempt 2/3" in out.err


def test_create_refuses_to_overwrite_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "greeter.asm.toml"
    existing.write_text("# do not clobber\n", encoding="utf-8")
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [AgentExecResult(reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.0)],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 1
    out = capsys.readouterr()
    # untouched
    assert existing.read_text(encoding="utf-8") == "# do not clobber\n"
    assert "REFUSING to overwrite" in out.err
    # validated draft dumped to stdout
    assert out.out.startswith('machine = "greeter"')


def test_create_output_flag_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "custom.asm.toml"
    target.write_text("# old\n", encoding="utf-8")
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [AgentExecResult(reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.0)],
    )
    code = main(["machine", "create", "Greet the user", "-o", str(target)])
    assert code == 0
    assert target.read_text(encoding="utf-8").startswith('machine = "greeter"')


def test_create_never_valid_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: INVALID_MACHINE}, usd=0.01
            ),
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: INVALID_MACHINE}, usd=0.01
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user", "--max-attempts", "2"])
    assert code == 1
    out = capsys.readouterr()
    assert "no valid machine after 2 attempt(s)" in out.err
    assert "Last diagnostics:" in out.err
    # last invalid draft echoed on stdout for reference
    assert out.out.startswith('machine = "greeter"')
    assert not (tmp_path / "greeter.asm.toml").exists()


def test_create_no_payload_gives_diagnostic_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            AgentExecResult(reason="max_iterations", payload=None, usd=0.0),
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.01
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    assert (tmp_path / "greeter.asm.toml").exists()


def test_create_rejects_bad_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "create", "Greet the user", "--max-attempts", "0"])
    assert code == 2
    assert "--max-attempts must be >= 1" in capsys.readouterr().err
