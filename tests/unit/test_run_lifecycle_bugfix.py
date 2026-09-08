# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run/resume lifecycle regressions: a parked start's task survives a failed
first start, a leg discards a stop it never honored, a resumed plan makes no
commit notes, and an ask out of budget exits by the shared code map."""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from typing import Any

import pytest

import agent6.app._session as session_mod
import agent6.app.resume as resume_mod
import agent6.app.run as run_mod
from agent6.budget import BudgetExceeded
from agent6.paths import state_dir
from agent6.providers import ProviderResponse
from agent6.ui.cli import cli_main


def _use(prov: _Scripted) -> Any:
    """A build_role_provider stand-in that always yields *prov*."""

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    return _factory


def _none_isolation(*_a: Any, **_k: Any) -> str:
    return "none"


_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "no"

[models.worker]
provider = "p"
model = "m"

[models.planner]
provider = "p"
model = "m"

[providers.p]
api_format = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
trigger = "off"
"""


class _Scripted:
    """One canned ProviderResponse per call; the last repeats. A per-call hook
    fires before the response, for a side effect (write a marker, raise)."""

    def __init__(
        self,
        script: list[tuple[str, tuple[dict[str, Any], ...]]],
        *,
        before_call: Any = None,
    ) -> None:
        self.script = script
        self.calls = 0
        self._before = before_call

    def call(self, **_kw: Any) -> ProviderResponse:
        if self._before is not None:
            self._before(self.calls)
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        text, tools = self.script[i]
        return ProviderResponse(
            text=text,
            tool_uses=tools,
            stop_reason="tool_use" if tools else "end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [*([{"type": "text", "text": text}] if text else []), *tools]},
        )


def _edit(path: str, content: str) -> tuple[str, tuple[dict[str, Any], ...]]:
    return (
        "",
        (
            {
                "type": "tool_use",
                "id": "e1",
                "name": "apply_edit",
                "input": {
                    "path": path,
                    "old_string": "",
                    "new_string": content,
                    "kind": "overwrite",
                },
            },
        ),
    )


def _finish(summary: str = "done") -> tuple[str, tuple[dict[str, Any], ...]]:
    tool = {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": summary}}
    return ("", (tool,))


def _plan(md: str) -> tuple[str, tuple[dict[str, Any], ...]]:
    return (
        "",
        (
            {
                "type": "tool_use",
                "id": "p1",
                "name": "finish_planning",
                "input": {"summary": "a plan", "plan_markdown": md},
            },
        ),
    )


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(run_mod, "select_isolation", _none_isolation)
    monkeypatch.setattr(resume_mod, "select_isolation", _none_isolation)
    return repo


def test_a_failed_first_start_keeps_the_parked_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_task's manifest rewrite once un-parked the run before the leg ran, so
    a crash during the first start left it with no parked_task and no snapshot:
    the operator's saved words were unreachable. The park now survives until the
    leg has actually started."""
    repo = _setup(tmp_path, monkeypatch)
    sd = state_dir(repo) / "sessions" / "runs" / "pin-PARK01"
    sd.mkdir(parents=True)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "pin-PARK01",
                "mode": "run",
                "user_task": "the operator's queued words",
                "parked_task": "the operator's queued words",
                "parked_reason": "checkout busy",
                "base_branch": "main",
            }
        ),
        encoding="utf-8",
    )

    def _interrupt(*_a: object, **_k: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(session_mod, "build_role_provider", _interrupt)
    assert cli_main(["resume", "pin-PARK01"]) == 130
    capsys.readouterr()
    assert json.loads((sd / "manifest.json").read_text())["parked_task"] == (
        "the operator's queued words"
    )
    # The run still reads "parked", not "crashed": nothing ran.
    assert cli_main(["sessions", "show", "pin-PARK01"]) == 0
    assert "parked" in capsys.readouterr().out

    prov = _Scripted([_edit("z.txt", "1\n"), _finish("done at last")])
    monkeypatch.setattr(session_mod, "build_role_provider", _use(prov))
    assert cli_main(["resume", "pin-PARK01"]) == 0
    assert (repo / "z.txt").exists()
    assert not json.loads((sd / "manifest.json").read_text())["parked_task"]


def test_a_parked_resume_with_no_provider_key_refuses_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Starting a parked run bypassed run_task's own provider-key preflight
    (that check lives in the CLI wrapper, which the parked branch skips), so a
    missing key crashed with a traceback instead of the `agent6 connect`
    refusal a fresh run gives."""
    repo = _setup(tmp_path, monkeypatch)
    # An Anthropic route with no resolvable key: the case that refuses (an
    # OpenAI-compat endpoint legitimately runs keyless, so it would not).
    (tmp_path / "cfg" / "agent6" / "config.toml").write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "no"\n'
        '[models.worker]\nprovider = "anth"\nmodel = "claude-sonnet-4-5"\n'
        '[providers.anth]\napi_format = "anthropic"\napi_key_env = "ANTH_KEY_UNSET"\n'
        '[review]\ntrigger = "off"\n',
        encoding="utf-8",
    )
    sd = state_dir(repo) / "sessions" / "runs" / "pin-PARK02"
    sd.mkdir(parents=True)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "pin-PARK02",
                "mode": "run",
                "user_task": "queued",
                "parked_task": "queued",
                "parked_reason": "checkout busy",
                "base_branch": "main",
            }
        ),
        encoding="utf-8",
    )
    assert cli_main(["resume", "pin-PARK02"]) == 2
    err = capsys.readouterr().err
    assert "agent6 connect" in err
    assert "crashed" not in err and "traceback" not in err


def test_a_leg_discards_a_stop_it_never_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stop that lands after the last boundary poll (here, during the finish
    turn) is never honored by the ending leg; its teardown discards it, so it
    cannot leak into and abort the next leg."""
    from agent6.sessions.ipc import request_stop, stop_request_pending

    repo = _setup(tmp_path, monkeypatch)
    sd = state_dir(repo) / "sessions" / "runs" / "pin-STOP01"

    def _stop_on_finish(call_index: int) -> None:
        if call_index == 1:  # the second call is the finish turn
            request_stop(sd)

    prov = _Scripted([_edit("a.txt", "1\n"), _finish("done")], before_call=_stop_on_finish)
    monkeypatch.setattr(session_mod, "build_role_provider", _use(prov))
    assert cli_main(["run", "--session-id", "pin-STOP01", "task"]) == 0
    capsys.readouterr()
    assert not stop_request_pending(sd), "a stop the leg never honored must not survive it"


def test_a_resumed_plan_leg_makes_no_commit_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan leg commits nothing (its chain ref is None), so the run-only
    between-legs notes ("left out of this run's commits", "the tree holds
    changes no commit has") are false there, and untracked-at-start must not be
    written into a plan dir nothing reads."""
    from agent6.git_ops import chain_ref_for, chain_tip

    repo = _setup(tmp_path, monkeypatch)
    prov = _Scripted([_plan("# Plan: p\n\n1. one\n")])
    monkeypatch.setattr(session_mod, "build_role_provider", _use(prov))
    assert cli_main(["plan", "--session-id", "p-PLAN01", "figure it out"]) == 0
    capsys.readouterr()

    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # tracked, modified between legs
    (repo / "scratch.md").write_text("notes\n", encoding="utf-8")  # untracked

    prov2 = _Scripted([_plan("# Plan: p\n\n1. one\n2. two\n")])
    monkeypatch.setattr(session_mod, "build_role_provider", _use(prov2))
    assert cli_main(["resume", "p-PLAN01", "--steer", "add a step"]) == 0
    err = capsys.readouterr().err
    assert "commit" not in err
    sd = state_dir(repo) / "sessions" / "plans" / "p-PLAN01"
    assert not (sd / "untracked-at-start").exists()
    assert chain_tip(repo, chain_ref_for("p-PLAN01")) is None


def test_an_ask_out_of_budget_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ask leg mapped its own exit code (`0 if completed else 1`), so a
    budget-exhausted ask exited 1 where `run`/`resume` and the docs promise 3.
    It now returns through the one code map every mode shares."""
    _setup(tmp_path, monkeypatch)

    def _raise(_i: int) -> None:
        raise BudgetExceeded("USD budget exhausted")

    prov = _Scripted([("an answer", ())], before_call=_raise)
    monkeypatch.setattr(session_mod, "build_role_provider", _use(prov))
    assert cli_main(["ask", "query", "why is the sky blue?"]) == 3
    capsys.readouterr()

    prov2 = _Scripted([("the sky is blue", ())])
    monkeypatch.setattr(session_mod, "build_role_provider", _use(prov2))
    assert cli_main(["ask", "query", "why is the sky blue?"]) == 0
