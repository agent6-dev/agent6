# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run/resume lifecycle regressions.

A parked start's task survives a failed first start, an execution discards a stop it never honored,
a resumed plan makes no commit notes, and an ask out of budget exits by the shared code map.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import budget, paths
from agent6.app import _session, providers
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
run_commands = "deny"

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
    """One canned ProviderResponse per call; the last repeats.

    A per-call hook fires before the response, for a side effect (write a marker, raise).
    """

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


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(_session, "select_isolation", _none_isolation)
    monkeypatch.setattr(_session, "select_isolation", _none_isolation)
    return repo


def test_a_failed_first_start_keeps_the_parked_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed first start keeps the parked task.

    The park survives until the execution has started; a manifest rewrite that un-parks the run
    first leaves a crash during the first start with no parked_task and no snapshot, so the
    operator's saved words are unreachable.
    """
    repo = _setup(tmp_path, monkeypatch)
    sd = paths.state_dir(repo) / "sessions" / "runs" / "pin-PARK01"
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

    monkeypatch.setattr(providers, "build_role_provider", _interrupt)
    assert cli_main(["resume", "pin-PARK01"]) == 130
    capsys.readouterr()
    assert json.loads((sd / "manifest.json").read_text())["parked_task"] == (
        "the operator's queued words"
    )
    # The run still reads "parked", not "crashed": nothing ran.
    assert cli_main(["sessions", "show", "pin-PARK01"]) == 0
    assert "parked" in capsys.readouterr().out

    prov = _Scripted([_edit("z.txt", "1\n"), _finish("done at last")])
    monkeypatch.setattr(providers, "build_role_provider", _use(prov))
    assert cli_main(["resume", "pin-PARK01"]) == 0
    assert (repo / "z.txt").exists()
    assert not json.loads((sd / "manifest.json").read_text())["parked_task"]


def test_a_parked_resume_with_no_provider_key_refuses_cleanly(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parked resume with no provider key refuses cleanly.

    Starting a parked run goes through run_task's own provider-key preflight; a check that lives
    only in the CLI wrapper, which the parked branch skips, lets a missing key crash with a
    traceback instead of the `agent6 connect` refusal a fresh run gives.
    """
    repo = _setup(tmp_path, monkeypatch)
    # An Anthropic route with no resolvable key: the case that refuses (an
    # OpenAI-compat endpoint legitimately runs keyless, so it would not).
    (tmp_path / "cfg" / "agent6" / "config.toml").write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "deny"\n'
        '[models.worker]\nprovider = "anth"\nmodel = "claude-sonnet-4-5"\n'
        '[providers.anth]\napi_format = "anthropic"\napi_key_env = "ANTH_KEY_UNSET"\n'
        '[review]\ntrigger = "off"\n',
        encoding="utf-8",
    )
    sd = paths.state_dir(repo) / "sessions" / "runs" / "pin-PARK02"
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


def test_a_execution_discards_a_stop_it_never_honored(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An execution discards a stop it never honoured.

    A stop that lands after the last boundary poll (here, during the finish turn) is never honoured
    by the ending execution; its teardown discards it, so it cannot leak into and abort the next.
    """
    from agent6.sessions import ipc

    repo = _setup(tmp_path, monkeypatch)
    sd = paths.state_dir(repo) / "sessions" / "runs" / "pin-STOP01"

    def _stop_on_finish(call_index: int) -> None:
        if call_index == 1:  # the second call is the finish turn
            ipc.request_stop(sd)

    prov = _Scripted([_edit("a.txt", "1\n"), _finish("done")], before_call=_stop_on_finish)
    monkeypatch.setattr(providers, "build_role_provider", _use(prov))
    assert cli_main(["run", "--session-id", "pin-STOP01", "task"]) == 0
    capsys.readouterr()
    assert not ipc.stop_request_pending(sd), (
        "a stop the execution never honored must not survive it"
    )


def test_a_resumed_plan_execution_makes_no_commit_notes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resumed plan execution makes no commit notes.

    A plan execution commits nothing (its chain ref is None), so the run-only between-executions
    notes ("left out of this run's commits", "the tree holds changes no commit has") are false
    there, and untracked-at-start is not written into a plan dir nothing reads.
    """
    from agent6.vcs import git_read

    repo = _setup(tmp_path, monkeypatch)
    prov = _Scripted([_plan("# Plan: p\n\n1. one\n")])
    monkeypatch.setattr(providers, "build_role_provider", _use(prov))
    assert cli_main(["plan", "--session-id", "p-PLAN01", "figure it out"]) == 0
    capsys.readouterr()

    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # tracked, modified between executions
    (repo / "scratch.md").write_text("notes\n", encoding="utf-8")  # untracked

    prov2 = _Scripted([_plan("# Plan: p\n\n1. one\n2. two\n")])
    monkeypatch.setattr(providers, "build_role_provider", _use(prov2))
    assert cli_main(["resume", "p-PLAN01", "--steer", "add a step"]) == 0
    err = capsys.readouterr().err
    assert "commit" not in err
    sd = paths.state_dir(repo) / "sessions" / "plans" / "p-PLAN01"
    assert not (sd / "untracked-at-start").exists()
    assert git_read.chain_tip(repo, git_read.chain_ref_for("p-PLAN01")) is None


def test_an_ask_out_of_budget_exits_three(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ask out of budget exits 3.

    The ask execution returns through the one code map every mode shares; its own `0 if completed
    else 1` exits 1 where `run`, `resume` and the docs promise 3.
    """
    _setup(tmp_path, monkeypatch)

    def _raise(_i: int) -> None:
        raise budget.BudgetExceededError("USD budget exhausted")

    prov = _Scripted([("an answer", ())], before_call=_raise)
    monkeypatch.setattr(providers, "build_role_provider", _use(prov))
    assert cli_main(["ask", "query", "why is the sky blue?"]) == 3
    capsys.readouterr()

    prov2 = _Scripted([("the sky is blue", ())])
    monkeypatch.setattr(providers, "build_role_provider", _use(prov2))
    assert cli_main(["ask", "query", "why is the sky blue?"]) == 0


def test_a_plan_budget_stop_uses_one_label_and_prints_its_reason_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan cap stop repeated its reason and named its window two ways."""
    import time

    from agent6 import budget as agent6_budget
    from agent6 import secret_store
    from agent6.models import cache

    _setup(tmp_path, monkeypatch)
    (tmp_path / "cfg" / "agent6" / "config.toml").write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "deny"\n'
        '[models.worker]\nprovider = "chatgpt"\nmodel = "gpt-6-astra"\n'
        '[providers.chatgpt]\napi_format = "chatgpt"\n'
        '[review]\ntrigger = "off"\n'
        "[harness]\nverify_infer = false\n",
        encoding="utf-8",
    )
    secret_store.save_oauth_tokens(
        "chatgpt", secret_store.OAuthTokens("test-access", "test-refresh", time.time() + 3600)
    )

    def no_models(*_a: object, **_k: object) -> list[str]:
        return []

    monkeypatch.setattr(cache, "list_models", no_models)

    def build(*_a: object, budget: agent6_budget.BudgetTracker, **_k: object) -> _Scripted:
        budget.note_route("gpt-6-astra", "chatgpt")
        budget.record_plan_preflight(
            "gpt-6-astra",
            agent6_budget.PlanUsage.single(48, 10080, time.time() + 148 * 3600),
        )

        def before_call(_i: int) -> None:
            budget.check()
            budget.record(
                model="gpt-6-astra",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                plan_usage=agent6_budget.PlanUsage.single(49, 10080, time.time() + 148 * 3600),
            )

        return _Scripted(
            [
                (
                    "",
                    (
                        {
                            "type": "tool_use",
                            "id": "read1",
                            "name": "read_file",
                            "input": {"path": "a.py"},
                        },
                    ),
                )
            ],
            before_call=before_call,
        )

    monkeypatch.setattr(providers, "build_role_provider", build)
    assert (
        cli_main(["run", "--auto-approve", "--max-plan-points", "1", "Add sixteen functions"]) == 3
    )
    out = capsys.readouterr().out
    assert "failed · budget exhausted" in out
    assert out.count("this run consumed ~1.0 percentage points") == 1
    assert "account at 49% on its 7-day window" in out
    assert "plan usage (chatgpt): 49% of the 7-day window" in out
    assert out.count("plan budget exhausted") == 1
    assert "BUDGET EXCEEDED" not in out
    assert "primary window" not in out


def _resumed_config(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> Any:
    """Resume with the execution stubbed; return the config it was handed."""
    from agent6.app import _execution

    seen: list[Any] = []

    def _capture(cfg: Any, *_a: object, **_k: object) -> _execution.ExecutionEnd:
        seen.append(cfg)
        return _execution.ExecutionEnd(0)

    monkeypatch.setattr(_execution, "run_execution", _capture)
    assert cli_main(argv) == 0
    return seen[0]


def test_a_bare_resume_replays_the_caps_the_run_was_started_with(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resume with no budget flag runs under the `--max-*` values the run started with.

    The flags lived only in the first invocation's memory, so a bare resume fell back to the
    config caps: `--max-plan-points 2` resumed unbounded.
    """
    from agent6.sessions import manifest as sessions_manifest

    repo = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(providers, "build_role_provider", _use(_Scripted([_finish()])))
    argv = ["run", "--session-id", "cap-KEEP01", "--max-plan-points", "2", "--max-usd", "1.5", "t"]
    assert cli_main(argv) == 0
    sd = paths.state_dir(repo) / "sessions" / "runs" / "cap-KEEP01"
    recorded = sessions_manifest.read_manifest(sd).budget
    assert (recorded.max_plan_points, recorded.max_usd, recorded.max_tokens_fallback) == (
        2.0,
        1.5,
        None,
    )
    capsys.readouterr()

    cfg = _resumed_config(monkeypatch, ["resume", "cap-KEEP01", "--steer", "more"])

    assert (cfg.budget.max_plan_points, cfg.budget.max_usd) == (2.0, 1.5)
    assert "--max-usd 1.5 --max-plan-points 2.0" in capsys.readouterr().err


def test_resume_applies_the_manifest_plan_points_cap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest's max_plan_points survives resume and stops calls at the cap."""
    from agent6.app import _setup as app_setup

    repo = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(providers, "build_role_provider", _use(_Scripted([_finish()])))
    argv = ["run", "--session-id", "cap-POINTS01", "--max-plan-points", "2", "t"]
    assert cli_main(argv) == 0
    sd = paths.state_dir(repo) / "sessions" / "runs" / "cap-POINTS01"
    recorded = json.loads((sd / "manifest.json").read_text(encoding="utf-8"))
    assert recorded["budget"] == {
        "max_usd": None,
        "max_tokens_fallback": None,
        "max_plan_points": 2.0,
    }

    cfg = _resumed_config(monkeypatch, ["resume", "cap-POINTS01", "--steer", "more"])
    tracker = app_setup.budget_tracker(cfg)
    tracker.record_plan_preflight("m", budget.PlanUsage.single(40, 300, 0))
    tracker.check()
    tracker.record(
        model="m",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=budget.PlanUsage.single(42, 300, 0),
    )
    with pytest.raises(budget.BudgetExceededError, match="max_plan_points 2"):
        tracker.check()


def test_a_resume_budget_flag_outranks_the_recorded_cap_and_is_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--max-*` on a resume replaces that one recorded cap, for this and later resumes."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(providers, "build_role_provider", _use(_Scripted([_finish()])))
    argv = ["run", "--session-id", "cap-OVER01", "--max-plan-points", "2", "--max-usd", "1.5", "t"]
    assert cli_main(argv) == 0

    flagged = ["resume", "cap-OVER01", "--steer", "more", "--max-plan-points", "5"]
    cfg = _resumed_config(monkeypatch, flagged)
    assert (cfg.budget.max_plan_points, cfg.budget.max_usd) == (5.0, 1.5)
    cfg = _resumed_config(monkeypatch, ["resume", "cap-OVER01", "--steer", "more"])
    assert (cfg.budget.max_plan_points, cfg.budget.max_usd) == (5.0, 1.5)


def test_a_cap_set_in_config_is_read_from_config_on_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run started with no budget flag records none, so its resume follows the config."""
    from agent6.sessions import manifest as sessions_manifest

    repo = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(providers, "build_role_provider", _use(_Scripted([_finish()])))
    assert cli_main(["run", "--session-id", "cap-CONF01", "t"]) == 0
    sd = paths.state_dir(repo) / "sessions" / "runs" / "cap-CONF01"
    assert sessions_manifest.read_manifest(sd).budget == sessions_manifest.BudgetStamp()
    config = tmp_path / "cfg" / "agent6" / "config.toml"
    config.write_text(config.read_text() + "\n[budget]\nmax_plan_points = 7\n", encoding="utf-8")

    cfg = _resumed_config(monkeypatch, ["resume", "cap-CONF01", "--steer", "more"])

    assert cfg.budget.max_plan_points == 7.0
