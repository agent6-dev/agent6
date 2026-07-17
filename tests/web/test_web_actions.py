# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the web write side's argv building and spawn wiring (no HTTP)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.ui.cli.parser import (
    _inject_default_verb,  # pyright: ignore[reportPrivateUsage]
    build_parser,
)
from agent6.ui.web import actions

TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


# --- body-derived strings ride behind `--`, never parsed as flags -------------


def test_spawn_new_work_argv_ends_options_before_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path | None, str]:
        captured.append(list(argv))
        return None, "not started"

    monkeypatch.setattr(actions, "spawn_and_locate", _fake_locate)
    actions.spawn_new_work(tmp_path, "run", "--allow-root pwn", profile="quick")
    assert captured[-1][1:] == ["run", "--profile", "quick", "--", "--allow-root pwn"]


# --- the `/parallel` new-work directive: fan out lanes ------------------------


def _capture_locate(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    captured: list[list[str]] = []

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path | None, str]:
        captured.append(list(argv))
        return None, "not started"

    monkeypatch.setattr(actions, "spawn_and_locate", _fake_locate)
    return captured


def test_spawn_new_work_parallel_lane_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_locate(monkeypatch)
    actions.spawn_new_work(tmp_path, "run", "/parallel 2 add a greeting")
    assert captured[-1][1:] == ["run", "--parallel", "2", "--", "add a greeting"]


def test_spawn_new_work_parallel_model_list_with_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_locate(monkeypatch)
    actions.spawn_new_work(tmp_path, "run", "/parallel gpt-5,opus refactor", profile="quick")
    assert captured[-1][1:] == [
        "run", "--profile", "quick", "--parallel", "gpt-5,opus", "--", "refactor",
    ]  # fmt: skip


def test_spawn_new_work_malformed_parallel_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_locate(monkeypatch)
    run_id, err = actions.spawn_new_work(tmp_path, "run", "/parallel")
    assert run_id is None
    assert "/parallel" in err  # the directive error surfaces, no spawn happens
    assert captured == []


class _Eff:
    def __init__(self, cfg: object) -> None:
        self.config = cfg


def _provider_cfg() -> object:
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": "moonshotai/kimi-k2.6"}},
        }
    )


def test_spawn_new_work_parallel_refuses_unknown_model_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cache exists to validate against -> a typo'd model is the composer's normal
    # error path, nothing spawned.
    import json

    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "o.json").write_text(
        json.dumps({"models": ["moonshotai/kimi-k2.6"]}), encoding="utf-8"
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))

    def _eff(_cwd: object) -> _Eff:
        return _Eff(_provider_cfg())

    monkeypatch.setattr(actions, "load_effective", _eff)
    captured = _capture_locate(monkeypatch)
    run_id, err = actions.spawn_new_work(tmp_path, "run", "/parallel moonshotai/kimi-k2.7 fix it")
    assert run_id is None
    assert "unknown model 'moonshotai/kimi-k2.7'" in err
    assert "closest: moonshotai/kimi-k2.6" in err
    assert captured == []  # nothing spawned


def test_spawn_new_work_parallel_unknown_model_no_cache_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No cache to validate against -> never block; the detached lane's own
    # preflight warns. The spawn happens.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))

    def _eff(_cwd: object) -> _Eff:
        return _Eff(_provider_cfg())

    monkeypatch.setattr(actions, "load_effective", _eff)
    captured = _capture_locate(monkeypatch)
    actions.spawn_new_work(tmp_path, "run", "/parallel made-up/model fix it")
    assert captured[-1][1:] == ["run", "--parallel", "made-up/model", "--", "fix it"]


def test_spawn_new_work_parallel_only_for_run_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # plan/ask cannot fan out: the text is a literal task (rides behind `--`).
    captured = _capture_locate(monkeypatch)
    actions.spawn_new_work(tmp_path, "plan", "/parallel 2 add a greeting")
    assert captured[-1][1:] == ["plan", "--", "/parallel 2 add a greeting"]


def test_spawn_new_work_parallel_omitted_spec_is_one_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No spec -> one isolated lane: --parallel 1 (a clone lane, not an in-place run).
    captured = _capture_locate(monkeypatch)
    actions.spawn_new_work(tmp_path, "run", "/parallel refactor the parser")
    assert captured[-1][1:] == ["run", "--parallel", "1", "--", "refactor the parser"]


def test_spawn_new_work_parallel_multi_segment_spawns_one_fanout_per_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_locate(monkeypatch)
    actions.spawn_new_work(tmp_path, "run", "/parallel 2 task A /parallel gpt-5,opus task B")
    assert [c[1:] for c in captured] == [
        ["run", "--parallel", "2", "--", "task A"],
        ["run", "--parallel", "gpt-5,opus", "--", "task B"],
    ]


def test_spawn_new_work_multi_segment_malformed_spawns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # all-or-nothing: a later empty segment refuses the whole message, no spawn.
    captured = _capture_locate(monkeypatch)
    run_id, err = actions.spawn_new_work(tmp_path, "run", "/parallel 2 good task /parallel")
    assert run_id is None and "/parallel" in err
    assert captured == []


def test_spawn_machine_create_argv_ends_options_before_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path | None, str]:
        captured.append(list(argv))
        return None, "not started"

    monkeypatch.setattr(actions, "spawn_and_locate", _fake_locate)
    actions.spawn_machine_create(tmp_path, "-dashy task")
    assert captured[-1][1:] == ["machine", "create", "--", "-dashy task"]


def test_merge_and_config_argv_end_options_before_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _fake_capture(argv: list[str], cwd: Path, **_k: object) -> tuple[bool, str]:
        captured.append(list(argv))
        return True, "ok"

    monkeypatch.setattr(actions, "run_cli_capture", _fake_capture)
    actions.merge_run(tmp_path, "-rid", "squash")
    assert captured[-1][1:] == ["runs", "merge", "--strategy", "squash", "--", "-rid"]
    actions.set_config(tmp_path, "sandbox.protect_git", "-1", repo=True)
    assert captured[-1][1:] == ["config", "set", "--repo", "--", "sandbox.protect_git", "-1"]


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--profile", "quick", "--", "-dashy task"],
        ["run", "--parallel", "2", "--", "-dashy task"],
        ["run", "--profile", "quick", "--parallel", "gpt-5,opus", "--", "-dashy task"],
        ["plan", "--", "-dashy task"],
        ["ask", "--", "-dashy question"],
        ["machine", "create", "--", "-dashy task"],
        ["runs", "merge", "--strategy", "squash", "--", "-rid"],
        ["config", "set", "--repo", "--", "sandbox.protect_git", "-1"],
    ],
)
def test_cli_parser_accepts_double_dash_before_positionals(argv: list[str]) -> None:
    # The argv shapes the web actions build must parse: `--` ends options and the
    # dashy value lands in the positional.
    ns = build_parser().parse_args(_inject_default_verb(argv))
    positional = ns.task if hasattr(ns, "task") else getattr(ns, "run_id", None) or ns.value
    assert str(positional).startswith("-")


# --- machine run: refusals surface instead of a false "started" ---------------


def test_spawn_machine_run_propagates_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mf = tmp_path / "tiny.asm.toml"
    mf.write_text(TINY, encoding="utf-8")

    def _refuse(*_a: object, **_k: object) -> str:
        return "agent6 machine exited (1):\nlock held"

    monkeypatch.setattr(actions, "spawn_and_confirm", _refuse)
    ok, msg = actions.spawn_machine_run(tmp_path, str(mf))
    assert ok is False
    assert "lock held" in msg


def test_spawn_machine_run_started_signal_is_child_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # started(pid) fires only when the instance worker.pid holds the CHILD's own
    # pid: a live worker.pid from an already-running machine (lock held) must
    # not read as "this spawn started".
    from collections.abc import Callable

    from agent6.runs.ipc import write_worker_pid
    from agent6.ui.web import model

    mf = tmp_path / "tiny.asm.toml"
    mf.write_text(TINY, encoding="utf-8")
    captured_argv: list[list[str]] = []
    started_fns: list[Callable[[int], bool]] = []

    def _fake_confirm(
        argv: list[str],
        cwd: Path,
        *,
        started: Callable[[int], bool],
        timeout_s: float = 25.0,
    ) -> str:
        captured_argv.append(list(argv))
        started_fns.append(started)
        return ""

    monkeypatch.setattr(actions, "spawn_and_confirm", _fake_confirm)
    ok, msg = actions.spawn_machine_run(tmp_path, str(mf))
    assert ok is True and msg == "started"
    assert captured_argv[-1][1:] == ["machine", "run", str(mf)]
    started = started_fns[-1]
    instance = model.machines_root(tmp_path) / "tiny"
    instance.mkdir(parents=True)
    assert started(4242) is False  # no worker.pid yet
    write_worker_pid(instance, 4242)
    assert started(4242) is True  # the child owns the instance
    assert started(4243) is False  # someone else's pid (a prior runner)


# --- ended machines take no input ---------------------------------------------


def _ended_machine(cwd: Path, name: str) -> Path:
    """An instance whose journal records a MachineEnd, with one state-log dir."""
    inst = resolved_state_dir(cwd) / "machines" / name
    (inst / "states" / "0000-route").mkdir(parents=True)
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "states" / "0000-route" / "logs.jsonl").write_text("", encoding="utf-8")
    (inst / "journal.jsonl").write_text(
        '{"type":"machine.begin","ts":"2026-07-12T00:00:00+00:00","machine":"tiny","version":1}\n'
        '{"type":"machine.end","ts":"2026-07-12T00:00:01+00:00","status":"ok",'
        '"reason":"routed","state":"done","transitions":1}\n',
        encoding="utf-8",
    )
    return inst


def test_machine_poke_refuses_ended_machine(tmp_path: Path) -> None:
    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_poke(tmp_path, "tiny", message="wake up")
    assert not ok
    assert "ended" in msg
    assert not (inst / "signal").exists()  # nothing pretends to be delivered


def test_machine_steer_refuses_ended_machine(tmp_path: Path) -> None:
    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_steer(tmp_path, "tiny", "do more")
    assert not ok
    assert "ended" in msg
    assert not list((inst / "states" / "0000-route").glob("*.answer"))
