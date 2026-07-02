# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for Phase 5 `agent6 machine create`: authoring prompts and CLI flow."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent6.cli import machine_cmds as cli  # create + its preflight helpers live here now
from agent6.cli import main
from agent6.machine import (
    SCRIPTS_PAYLOAD_KEY,
    TOML_PAYLOAD_KEY,
    AgentExecResult,
    AgentRequest,
    build_authoring_prompt,
    extract_scripts,
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


def test_build_authoring_prompt_unpriced_steers_to_best_effort() -> None:
    # An unpriced worker model would make a drafted `max_usd` machine refuse to
    # run, so the prompt must steer the draft to `best_effort_usd_limit`.
    priced = build_authoring_prompt("Poll a queue", attempt=1, worker_unpriced=False)
    assert "## Budget" not in priced
    unpriced = build_authoring_prompt("Poll a queue", attempt=1, worker_unpriced=True)
    assert "best_effort_usd_limit" in unpriced
    assert "NO price data" in unpriced


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


def test_build_authoring_prompt_retry_includes_prior_scripts() -> None:
    # A retry must show the failing scripts, not just the toml. Without them
    # the model regenerates every file blind to fix a one-line lint error.
    prompt = build_authoring_prompt(
        "Poll a queue",
        attempt=2,
        prior_toml="machine = 'x'",
        diagnostics=["ruff (lint) found problems: scripts/run.py:1:1: F401"],
        prior_scripts={"scripts/run.py": "import os\nprint(1)", "scripts/go.sh": "echo hi"},
    )
    assert "`scripts/run.py`:" in prompt
    assert "import os" in prompt
    assert "`scripts/go.sh`:" in prompt
    assert "Change ONLY what the diagnostics name" in prompt
    # .py gets a python fence, others a plain fence
    assert "```python\nimport os" in prompt
    assert "```\necho hi" in prompt


# --- CLI flow --------------------------------------------------------------


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def _require_runnable(*_a: object, **_k: object) -> None:
        return None

    def _resolve(_role: object) -> object:
        return SimpleNamespace(model="test-model")

    def _load(_root: object, _explicit: object = None) -> object:
        cfg = SimpleNamespace(
            sandbox=SimpleNamespace(profile="none"),
            require_runnable=_require_runnable,
            models=SimpleNamespace(resolve=_resolve),
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
        cfg: object, root: Path, profile: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(_request: AgentRequest, _events_log: object = None) -> AgentExecResult:
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
        cfg: object, root: Path, profile: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
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
    # mode="machine" -> authoring system prompt + read-only tools. If the
    # plumbing dropped it the authoring agent would silently fall back to the
    # 29k coding prompt with no test catching it.
    assert captured[0].mode == "machine"


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


def test_create_writes_watchable_event_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """machine create writes a logs.jsonl in the draft dir (run.start carrying the
    NL task + run.end) and points the agent runner at that same path, so the TUI
    can open the dashboard on the draft and follow the authoring live, like a run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    captured_log: list[object] = []

    def fake_build(
        cfg: object, root: Path, profile: object, transcript_dir: Path, **kw: object
    ) -> Callable[[AgentRequest, object], AgentExecResult]:
        def run(_request: AgentRequest, events_log: object = None) -> AgentExecResult:
            captured_log.append(events_log)  # events_log is now per CALL, not per build
            return AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.0
            )

        return run

    monkeypatch.setattr(cli, "_build_machine_agent_runner", fake_build)
    assert main(["machine", "create", "Greet the user"]) == 0

    logs = list((tmp_path / "state").glob("**/machine-drafts/*/logs.jsonl"))
    assert len(logs) == 1
    # The runner was pointed at that same log (so the subprocess appends to it).
    assert captured_log and str(captured_log[0]) == str(logs[0])
    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "run.start"
    assert events[0]["user_task"] == "Greet the user"  # the dashboard header
    assert any(e["type"] == "run.end" for e in events)


def test_create_saves_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The natural-language task is saved to the draft dir as prompt.txt, so the
    draft is self-describing (otherwise the task only survives embedded inside the
    authoring transcript)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [AgentExecResult(reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.0)],
    )
    code = main(["machine", "create", "Greet the user warmly"])
    assert code == 0
    prompts = list((tmp_path / "state").glob("**/machine-drafts/*/prompt.txt"))
    assert len(prompts) == 1
    assert prompts[0].read_text(encoding="utf-8") == "Greet the user warmly"


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


def test_create_output_flag_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # -o into a directory that does not exist yet must not crash.
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [AgentExecResult(reason="finish_run", payload={TOML_PAYLOAD_KEY: VALID_MACHINE}, usd=0.0)],
    )
    target = tmp_path / "new" / "deep" / "m.asm.toml"
    code = main(["machine", "create", "Greet the user", "-o", str(target)])
    assert code == 0
    assert target.read_text(encoding="utf-8").startswith('machine = "greeter"')


def test_create_retry_prompt_carries_prior_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When a draft fails the script gate, the NEXT attempt's prompt must show
    # the prior script source so the model patches instead of regenerating.
    from agent6.cli import scriptcheck

    if "ruff" not in scriptcheck.available_tools():
        pytest.skip("ruff not installed")
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    prompts: list[str] = []
    bad = "print(undefined_name)\n"  # F821
    good = "import json\nprint(json.dumps({}))\n"
    responses = iter(
        [
            AgentExecResult(
                reason="finish_run",
                payload={
                    TOML_PAYLOAD_KEY: SCRIPT_MACHINE,
                    SCRIPTS_PAYLOAD_KEY: {"scripts/run.py": bad},
                },
                usd=0.01,
            ),
            AgentExecResult(
                reason="finish_run",
                payload={
                    TOML_PAYLOAD_KEY: SCRIPT_MACHINE,
                    SCRIPTS_PAYLOAD_KEY: {"scripts/run.py": good},
                },
                usd=0.01,
            ),
        ]
    )

    def fake_build(
        cfg: object, root: Path, profile: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            prompts.append(request.prompt)
            return next(responses)

        return run

    monkeypatch.setattr(cli, "_build_machine_agent_runner", fake_build)
    code = main(["machine", "create", "Run a script"])
    assert code == 0
    assert len(prompts) == 2
    assert "undefined_name" not in prompts[0]
    assert "`scripts/run.py`:" in prompts[1]
    assert "print(undefined_name)" in prompts[1]


# --- script bundle: the agent emits helper scripts alongside the .asm.toml ---

SCRIPT_MACHINE = """\
machine = "scripted"
version = 1
initial = "go"

[budget]
max_usd = 1.0
max_transitions = 10

[states.go]
kind = "tool"
command = ["python3", "scripts/run.py"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ran"

[states.fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""
SCRIPT_BODY = "import json\nprint(json.dumps({}))"


def test_extract_scripts_keeps_safe_entries_only() -> None:
    got = extract_scripts(
        {
            SCRIPTS_PAYLOAD_KEY: {
                "scripts/run.py": "x = 1",
                "./scripts/lib/util.py": "y = 2",  # normalized
                "scripts/../etc/passwd": "escape",  # dropped (..)
                "/abs/scripts/run.py": "abs",  # dropped (absolute)
                "notes.txt": "outside scripts/",  # dropped (not under scripts/)
                "scripts/x.py": 7,  # dropped (non-str content)
            }
        }
    )
    assert got == {"scripts/run.py": "x = 1", "scripts/lib/util.py": "y = 2"}


@pytest.mark.parametrize("payload", [None, {}, {SCRIPTS_PAYLOAD_KEY: "not-a-map"}])
def test_extract_scripts_empty(payload: dict[str, object] | None) -> None:
    assert extract_scripts(payload) == {}  # type: ignore[arg-type]


def test_create_writes_script_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            AgentExecResult(
                reason="finish_run",
                payload={
                    TOML_PAYLOAD_KEY: SCRIPT_MACHINE,
                    SCRIPTS_PAYLOAD_KEY: {"scripts/run.py": SCRIPT_BODY},
                },
                usd=0.02,
            )
        ],
    )
    code = main(["machine", "create", "Run a script"])
    assert code == 0
    out = capsys.readouterr()
    assert (tmp_path / "scripted.asm.toml").exists()
    script = tmp_path / "scripts" / "run.py"
    assert script.exists()
    assert script.read_text(encoding="utf-8") == SCRIPT_BODY + "\n"
    assert "1 script(s)" in out.err


def test_create_rejects_missing_script_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A TOML that runs scripts/run.py but ships no scripts must fail bundle
    validation (the user's bug), then succeed once the agent supplies it."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            # attempt 1: references the script but omits it -> rejected.
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: SCRIPT_MACHINE}, usd=0.01
            ),
            # attempt 2: now ships it -> accepted.
            AgentExecResult(
                reason="finish_run",
                payload={
                    TOML_PAYLOAD_KEY: SCRIPT_MACHINE,
                    SCRIPTS_PAYLOAD_KEY: {"scripts/run.py": SCRIPT_BODY},
                },
                usd=0.02,
            ),
        ],
    )
    code = main(["machine", "create", "Run a script"])
    assert code == 0
    out = capsys.readouterr()
    assert (tmp_path / "scripts" / "run.py").exists()
    assert "attempt 2/3" in out.err


def test_create_rejects_lint_bad_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A structurally-valid machine whose script has a lint error must NOT be
    written — ruff/ty run in the create loop and the failure is a diagnostic."""
    from agent6.cli import scriptcheck

    if "ruff" not in scriptcheck.available_tools():
        pytest.skip("ruff not installed")
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    bad = "import json\nprint(undefined_name)\n"  # F821 undefined name
    _stub_runner(
        monkeypatch,
        [
            AgentExecResult(
                reason="finish_run",
                payload={
                    TOML_PAYLOAD_KEY: SCRIPT_MACHINE,
                    SCRIPTS_PAYLOAD_KEY: {"scripts/run.py": bad},
                },
                usd=0.01,
            )
        ],
    )
    code = main(["machine", "create", "Run a script", "--max-attempts", "1"])
    assert code == 1
    out = capsys.readouterr()
    assert "ruff" in out.err
    assert not (tmp_path / "scripted.asm.toml").exists()


def test_create_never_ships_script_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            AgentExecResult(
                reason="finish_run", payload={TOML_PAYLOAD_KEY: SCRIPT_MACHINE}, usd=0.01
            )
        ],
    )
    code = main(["machine", "create", "Run a script", "--max-attempts", "1"])
    assert code == 1
    out = capsys.readouterr()
    assert "not found in bundle" in out.err
    # the diagnostic steers the agent to the right payload field.
    assert SCRIPTS_PAYLOAD_KEY in out.err
    # no half-written bundle left behind.
    assert not (tmp_path / "scripted.asm.toml").exists()
    assert not (tmp_path / "scripts").exists()
