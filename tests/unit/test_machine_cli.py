# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for Phase 4 machine ergonomics: status, poke, run --exit-on-wait."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.machine import MachineJournal
from agent6.runs.ipc import clear_worker_pid
from agent6.ui.cli import main


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)


WAITER_DELAYED = """
machine = "waiter_delayed"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def _write_machine(tmp_path: Path) -> Path:
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER_DELAYED, encoding="utf-8")
    return f


def test_run_exit_on_wait_yields_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    out = capsys.readouterr().out
    assert "WAITING" in out
    # The wait was armed and persisted.
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    pending = MachineJournal(root).read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"


def test_run_prints_a_notify_on_the_foreground_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A notify was journal-only: the foreground run is its own watcher, so the
    # message must land on the terminal too (attach and the web already showed
    # it). The operator [machine.notify].on_event hook is unset here.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "waiter.asm.toml"
    f.write_text(
        WAITER_DELAYED.replace(
            'kind = "wait"',
            'kind = "wait"\nnotify = { message = "parked, awaiting a poke", level = "warn" }',
        ),
        encoding="utf-8",
    )
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    err = capsys.readouterr().err
    assert "[agent6] notify [warn] 'poll': parked, awaiting a poke" in err


def test_status_reports_waiting_state_and_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    # `machine run --exit-on-wait` exits the process, so the worker pid is dead;
    # in-process it is this live pytest, so clear it to model the parked reality.
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    clear_worker_pid(root)
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "waiter_delayed" in out
    # A parked instance reads "waiting" (the word run --exit-on-wait/web use), not
    # the engine's raw "incomplete".
    assert "status: waiting" in out
    assert "next wake:" in out
    assert "spend: $0.0000" in out


def test_status_hints_poke_for_a_live_foreground_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A foreground `machine run` blocked in a wait writes no pending-wait record
    # (that is --exit-on-wait's parked form); status must still say the machine
    # is waiting and how to poke it, not a bare "running" (attach already knew).
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    # Model the foreground shape: live worker (this pytest pid), no pending record.
    MachineJournal(root).clear_pending_wait()
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "status: running" in out
    assert "waiting in 'poll': agent6 machine poke waiter_delayed" in out


def test_status_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "status", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err


def test_uncommitted_refusal_logs_a_git_error_instead_of_silently_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The dirty-file gate fails OPEN on a GitError (it is review-discipline, not
    # security) but must never do so SILENTLY: a broken-git env stays visible.
    import agent6.app.machine.run as machine_run
    from agent6.git_ops import GitError

    _git_init(tmp_path)
    f = tmp_path / "m.asm.toml"
    f.write_text('machine="m"\nversion=1\ninitial="s"\n[states.s]\nkind="terminal"\n')

    def _boom(*_a: object, **_k: object) -> bool:
        raise GitError("git index is corrupt")

    monkeypatch.setattr(machine_run, "paths_dirty", _boom)
    assert machine_run.uncommitted_refusal(f, tmp_path) is None  # fail-open preserved
    err = capsys.readouterr().err
    assert "could not check" in err and "git index is corrupt" in err


def test_status_asm_file_path_hints_the_instance_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `machine run` takes a FILE (waiter.asm.toml); status/replay/poke/watch take
    # the instance ID (waiter_delayed). Passing the file where the id belongs must
    # suggest the id, not dead-end.
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)  # machine = "waiter_delayed"
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    clear_worker_pid(resolved_state_dir(tmp_path) / "machines" / "waiter_delayed")
    code = main(["machine", "status", "waiter.asm.toml"])
    assert code == 1
    err = capsys.readouterr().err
    assert "no machine instance" in err
    assert "waiter_delayed" in err  # the did-you-mean names the real instance id


# A no-I/O machine that reaches a terminal immediately (branch -> terminal), so
# `agent6 attach` on it takes the finished path (overview + end) without blocking
# in the follow loop and without needing a model or the jail.
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


def test_watch_finished_instance_shows_overview_and_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()  # drop run output
    # A finished instance has a journaled MachineEnd, so the unified `agent6 attach`
    # (which routes a machine name to the machine follower) prints the overview +
    # the final state and returns instead of entering the (blocking) follow loop.
    code = main(["attach", "tiny"])
    assert code == 0
    out = capsys.readouterr().out
    assert "machine: tiny" in out
    assert "> done" in out  # current state marked
    assert ". route" in out  # a visited state marked
    assert "OK: ended in 'done'" in out


def test_replay_pluralizes_the_transition_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine replay` counts "1 transition" (singular), matching `machine run`."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    assert main(["machine", "replay", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "after 1 transition (" in out and "1 transitions" not in out


def test_run_refuses_uncommitted_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # docs §7.1/§9: `machine run` only accepts a committed machine. An untracked
    # .asm.toml is refused before any execution.
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    code = main(["machine", "run", str(f)])
    assert code == 1
    err = capsys.readouterr().err
    assert "uncommitted" in err and "committed machine" in err
    # Refused before touching the state dir: no instance journal was created.
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    assert not (root / "journal.jsonl").exists()


def test_uncommitted_refusal_tracks_git_state(tmp_path: Path) -> None:
    from agent6.app.machine.run import uncommitted_refusal

    # Outside a git repo the gate never fires (nothing to commit against).
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert uncommitted_refusal(f, tmp_path) is None
    _git_init(tmp_path)
    assert uncommitted_refusal(f, tmp_path) is not None  # untracked
    subprocess.run(["git", "-C", str(tmp_path), "add", "tiny.asm.toml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert uncommitted_refusal(f, tmp_path) is None  # committed clean
    f.write_text(TINY + "\n", encoding="utf-8")
    assert uncommitted_refusal(f, tmp_path) is not None  # modified again


def test_run_refuses_rerun_of_ended_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An ended instance can only be replayed, never advanced. A rerun must refuse
    # BEFORE stamping worker.pid, so a dead machine never reads "running".
    from agent6.machine import drive, load_machine
    from agent6.runs.ipc import read_worker_pid, write_worker_pid

    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0  # runs to a terminal
    capsys.readouterr()
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    # Stand in for the previous worker having exited: a pid that is never alive.
    sentinel = 10**9
    write_worker_pid(root, sentinel)
    code = main(["machine", "run", str(f)])
    assert code == 1
    assert "already ended" in capsys.readouterr().err
    # worker.pid was NOT re-stamped with the (live) rerun process pid.
    assert read_worker_pid(root) == sentinel
    # The journal still reads terminal, unchanged.
    result = drive(load_machine(root / "machine.asm.toml"), MachineJournal(root), None, live=False)
    assert result.status == "ok"


def test_poke_drops_signal_for_waiting_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "waiter_delayed"])
    assert code == 0
    assert "poked" in capsys.readouterr().out
    # The signal is now pending for the next take_signal().
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() == (True, None)


def test_poke_carries_data_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", '{"cmd": "go"}']) == 0
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() == (True, {"cmd": "go"})
    # --message wraps a plain string.
    assert main(["machine", "poke", "waiter_delayed", "--message", "hello"]) == 0
    assert MachineJournal(root).take_signal() == (True, "hello")


def test_poke_rejects_invalid_json_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", "{not json}"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_poke_refuses_ended_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A terminal machine consumes no signals; poking it would sit unread, so the
    # CLI refuses instead of claiming "it will wake on its next signal check".
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "tiny"])
    assert code == 1
    assert "already ended" in capsys.readouterr().err
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    assert not (root / "signal").exists()  # no signal was dropped


def test_poke_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "poke", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err


AGENT_MACHINE_HARD = """
machine = "hard-usd"
version = 1
initial = "judge"

[budget]
max_usd = 1.0
max_transitions = 10

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


def _hard_usd_cfg(model: str):
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": model}},
        }
    )


def _load_spec(tmp_path: Path, body: str):
    from agent6.machine import load_machine

    f = tmp_path / "m.asm.toml"
    f.write_text(body, encoding="utf-8")
    return load_machine(f)


def test_hard_usd_preflight_refuses_unpriced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [budget].max_usd is a hard promise; an unpriced inherited worker model
    # cannot honor it, so machine run must refuse up front.
    from agent6.app.machine import hard_usd_preflight_error

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    spec = _load_spec(tmp_path, AGENT_MACHINE_HARD)
    err = hard_usd_preflight_error(spec, _hard_usd_cfg("nobody/unpriced"))
    assert err is not None
    assert "max_usd" in err and "nobody/unpriced" in err and "judge" in err


def test_hard_usd_preflight_passes_best_effort_and_priced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json as _json

    from agent6.app.machine import hard_usd_preflight_error

    # best_effort_usd_limit never refuses, priced or not.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    soft = AGENT_MACHINE_HARD.replace("max_usd = 1.0", "best_effort_usd_limit = 1.0")
    spec = _load_spec(tmp_path, soft)
    assert hard_usd_preflight_error(spec, _hard_usd_cfg("nobody/unpriced")) is None

    # max_usd passes once the model has price data.
    cache = tmp_path / "cache"
    (cache / "models").mkdir(parents=True)
    (cache / "models" / "p.json").write_text(
        _json.dumps({"models": ["priced/m"], "pricing": {"priced/m": [1.0, 2.0]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))
    spec_hard = _load_spec(tmp_path, AGENT_MACHINE_HARD)
    assert hard_usd_preflight_error(spec_hard, _hard_usd_cfg("priced/m")) is None


def test_hard_usd_preflight_checks_per_state_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A soft machine budget with a hard per-state max_usd still requires price
    # data for that state's model.
    from agent6.app.machine import hard_usd_preflight_error

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    body = AGENT_MACHINE_HARD.replace("max_usd = 1.0", "best_effort_usd_limit = 1.0").replace(
        'kind = "agent"', 'kind = "agent"\nmax_usd = 0.5\nmodel = "pinned/unpriced"', 1
    )
    spec = _load_spec(tmp_path, body)
    err = hard_usd_preflight_error(spec, _hard_usd_cfg("priced/m"))
    assert err is not None and "pinned/unpriced" in err
