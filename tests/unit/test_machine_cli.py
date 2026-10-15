# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for Phase 4 machine ergonomics: status, poke, run --exit-on-wait."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable

import pytest

from agent6 import errors, paths
from agent6.app import _setup, machine_agent
from agent6.machine import MachineJournal, write_stop_request
from agent6.sandbox import detect
from agent6.sessions import ipc
from agent6.ui.cli import main


def _git_init(path: pathlib.Path) -> None:
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


def _write_machine(tmp_path: pathlib.Path) -> pathlib.Path:
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER_DELAYED, encoding="utf-8")
    return f


def test_run_exit_on_wait_yields_waiting(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    out = capsys.readouterr().out
    assert "WAITING" in out
    # The wait was armed and persisted.
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    pending = MachineJournal(root).read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"


def test_a_foreground_wait_says_where_it_parked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A foreground wait says so on the terminal, so a parked `machine run` never looks hung."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "waiter.asm.toml"
    ticks_at_once = WAITER_DELAYED.replace(
        'secs = { type = "int", value = 3600 }', 'secs = { type = "int", value = 1 }'
    )
    f.write_text(ticks_at_once, encoding="utf-8")

    assert main(["machine", "run", str(f)]) == 0

    captured = capsys.readouterr()
    assert "waiting in 'poll': wakes at " in captured.err
    assert "agent6 machine poke waiter_delayed" in captured.err
    assert "OK: waiter_delayed ended in 'done'" in captured.out


def test_a_polling_run_prints_the_poke_hint_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Polling prints each wake time but the same poke command only once per invocation."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "poller.asm.toml"
    f.write_text(
        'machine = "poller"\nversion = 1\ninitial = "poll"\n'
        "[budget]\nmax_transitions = 2\n"
        '[states.poll]\nkind = "wait"\nevery_secs = "1"\n'
        'on = { tick = "poll", signal = "done" }\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "poked"\n',
        encoding="utf-8",
    )
    assert main(["machine", "run", str(f)]) == 1
    captured = capsys.readouterr()
    assert captured.err.count("wakes at") == 2
    assert captured.err.count("agent6 machine poke poller") == 1


def test_the_waiting_line_names_its_state_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The WAITING ending does not repeat its state inside the engine's reason."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    out = capsys.readouterr().out
    assert "WAITING: waiter_delayed paused in 'poll'" in out
    assert out.count("'poll'") == 1
    assert "until " in out


def test_run_prints_a_notify_on_the_foreground_terminal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The foreground run is its own watcher, so a notify must land on the terminal too.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    # `--exit-on-wait` exits the process; in-process the pid is this pytest, so clear it.
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    ipc.clear_worker_pid(root)
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "waiter_delayed" in out
    # A parked instance reads "waiting", not the engine's raw "incomplete".
    assert "status: waiting" in out
    assert "spend: $0.0000" in out


def test_status_of_a_parked_instance_with_no_worker_names_the_resume_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no worker alive, status names `machine run` and never says a poke wakes it now."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert not ipc.worker_is_alive(root)

    assert main(["machine", "status", "waiter_delayed"]) == 0

    out = capsys.readouterr().out
    assert "wakes it now" not in out
    assert "parked in 'poll' with no worker running (due at " in out
    assert "resume with `agent6 machine run waiter_delayed`" in out
    assert "a poke wakes it on that run: agent6 machine poke waiter_delayed" in out
    # The command as printed resumes the instance: `machine run` takes the name.
    assert main(["machine", "run", "waiter_delayed", "--exit-on-wait"]) == 0
    assert "WAITING: waiter_delayed paused in 'poll'" in capsys.readouterr().out


def test_status_after_an_operator_stop_names_the_resume_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run stopped inside its wait leaves no worker: status says so, with the resume command."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"

    def stop_once_parked() -> None:
        deadline = time.monotonic() + 30
        while not MachineJournal(root).wait_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        write_stop_request(root)

    stopper = threading.Thread(target=stop_once_parked)
    stopper.start()
    try:
        assert main(["machine", "run", str(f)]) == 0
    finally:
        stopper.join()
    stopped = capsys.readouterr().out
    assert "STOPPED: waiter_delayed parked in 'poll'" in stopped
    assert "resume with `agent6 machine run waiter_delayed`." in stopped

    assert main(["machine", "status", "waiter_delayed"]) == 0

    out = capsys.readouterr().out
    assert "wakes it now" not in out
    assert "parked in 'poll' with no worker running (due at " in out
    assert "resume with `agent6 machine run waiter_delayed`" in out


def test_attach_to_a_parked_instance_with_no_worker_names_the_resume_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no worker alive, the watch's exit line names `machine run`, not a poke that resumes."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output

    assert main(["attach", "waiter_delayed"]) == 0

    out = capsys.readouterr().out
    assert "poke to resume" not in out
    assert (
        "WAITING in 'poll' with no worker running: resume with"
        " `agent6 machine run waiter_delayed`" in out
    )


def test_status_hints_poke_for_a_live_foreground_wait(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A foreground `machine run` persists its wait record before sleeping; the poke line shows."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).read_pending_wait() is not None
    # The run cleared its own pid on exit; re-stamp a live worker (this pytest).
    ipc.write_worker_pid(root, os.getpid())
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "status: waiting" in out
    assert "waiting in 'poll': wakes at " in out
    assert "a poke wakes it now: agent6 machine poke waiter_delayed" in out


def test_status_names_a_poke_only_for_an_armed_wait(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The poke line comes from the wait record, never from the fold's guess."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    MachineJournal(root).clear_pending_wait()
    ipc.write_worker_pid(root, os.getpid())

    assert main(["machine", "status", "waiter_delayed"]) == 0

    out = capsys.readouterr().out
    assert "machine poke" not in out
    assert "waiting in" not in out


def test_status_shows_a_pending_poke_until_it_is_acked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A poke the machine has not acted on shows its payload until the wake's step is acked."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    ipc.clear_worker_pid(root)
    assert main(["machine", "poke", "waiter_delayed", "--message", "go"]) == 0
    capsys.readouterr()
    assert main(["machine", "status", "waiter_delayed"]) == 0
    assert "poke pending: 'go'" in capsys.readouterr().out
    journal = MachineJournal(root)
    assert journal.take_signal() == (True, "go")  # claimed, the step not yet durable
    assert main(["machine", "status", "waiter_delayed"]) == 0
    assert "poke pending: 'go'" in capsys.readouterr().out
    journal.ack_signal()
    assert main(["machine", "status", "waiter_delayed"]) == 0
    assert "poke pending" not in capsys.readouterr().out


def test_status_of_an_alive_but_parked_instance_reads_waiting(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """machine_word_for_dir checks `parked` before `alive`: alive-but-parked reads "waiting"."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    ipc.write_worker_pid(root, os.getpid())  # a LIVE worker alongside the persisted wait

    assert main(["machine", "status", "waiter_delayed"]) == 0
    out = capsys.readouterr().out
    assert "status: waiting" in out  # parked wins over alive
    assert "status: running" not in out


def test_status_tolerates_a_corrupt_pending_wait(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt wait.json does not abort the readout: status reads parked and notes the file."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    ipc.clear_worker_pid(root)
    MachineJournal(root).wait_path.write_text("{ not valid json", encoding="utf-8")

    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0  # not aborted
    out = capsys.readouterr().out
    assert "pending wait: unreadable" in out  # truthful, not silently swallowed
    assert "state:" in out  # the rest of the readout still prints


CRASHER = """
machine = "crasher"
version = 1
initial = "one"

[budget]
max_transitions = 100

[states.one]
kind = "tool"
command = ["true"]
timeout_secs = 60
on = { ok = "two", nonzero = "bad", timeout = "bad" }

[states.two]
kind = "tool"
command = ["true"]
timeout_secs = 60
on = { ok = "done", nonzero = "bad", timeout = "bad" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"

[states.bad]
kind = "terminal"
status = "failed"
reason = "bad"
"""


def test_status_reports_stopped_for_a_crashed_instance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A worker dead mid-state is "stopped" on every surface, through machine_status_word.
    from agent6.machine import journal as machine_journal

    monkeypatch.chdir(tmp_path)
    root = paths.state_dir(tmp_path) / "machines" / "crasher"
    root.mkdir(parents=True)
    (root / "machine.asm.toml").write_text(CRASHER, encoding="utf-8")
    journal = MachineJournal(root)
    journal.ensure_dirs()
    journal.begin(machine="crasher", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="one",
            label="ok",
            goto="two",
            fact=machine_journal.ToolFact(exit_code=0, stdout="", timed_out=False),
        )
    )
    # No worker.pid file -> not alive; no pending wait -> not parked.
    assert main(["machine", "status", "crasher"]) == 0
    out = capsys.readouterr().out
    assert "status: stopped" in out


def test_status_missing_instance_errors(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "status", "nope"])
    assert code == 2
    assert "no machine instance" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["status", "replay", "poke", "stop"])
@pytest.mark.parametrize("existing", [False, True])
def test_a_missing_instance_names_available_instances_and_the_run_command(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    existing: bool,
) -> None:
    """A missing id names this repo and the list/start commands, never its internal state path."""
    monkeypatch.chdir(tmp_path)
    if existing:
        f = _write_machine(tmp_path)
        assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
        capsys.readouterr()
    assert main(["machine", verb, "nosuch"]) == 2
    err = capsys.readouterr().err
    assert "no machine instance 'nosuch'" in err
    assert str(tmp_path) in err
    assert str(paths.state_dir(tmp_path)) not in err
    if existing:
        assert "Existing instances: 'waiter_delayed'" in err
    else:
        assert "No instances exist yet" in err
    assert "`agent6 machine` lists them" in err
    assert "`agent6 machine run <file>` starts one" in err


def test_uncommitted_refusal_logs_a_git_error_instead_of_silently_failing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The dirty-file gate fails open on a GitError but never silently.
    from agent6 import git_ops
    from agent6.app.machine import run as machine_run

    _git_init(tmp_path)
    f = tmp_path / "m.asm.toml"
    f.write_text('machine="m"\nversion=1\ninitial="s"\n[states.s]\nkind="terminal"\n')

    def _boom(*_a: object, **_k: object) -> bool:
        raise errors.GitError("git index is corrupt")

    monkeypatch.setattr(git_ops, "paths_dirty", _boom)
    assert machine_run.uncommitted_refusal(f, tmp_path) is None  # fail-open preserved
    err = capsys.readouterr().err
    assert "could not check" in err and "git index is corrupt" in err


def test_status_asm_file_path_hints_the_instance_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `machine run` takes a file; the other verbs take the instance id and suggest it for a file.
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)  # machine = "waiter_delayed"
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    ipc.clear_worker_pid(paths.state_dir(tmp_path) / "machines" / "waiter_delayed")
    code = main(["machine", "status", "waiter.asm.toml"])
    assert code == 2
    err = capsys.readouterr().err
    assert "no machine instance" in err
    assert "waiter_delayed" in err  # the did-you-mean names the real instance id


def test_status_on_an_invalid_machine_file_names_it_as_a_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unparsable .asm.toml where an id belongs is still a file, and the hint says so."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.asm.toml").write_text('machine = "bad"\n', encoding="utf-8")
    assert main(["machine", "status", "bad.asm.toml"]) == 2
    err = capsys.readouterr().err
    assert "no machine instance" in err
    assert "machine file" in err, err
    assert "agent6 machine check bad.asm.toml" in err, err  # `machine run` would refuse it too


# A no-I/O machine that reaches a terminal at once, so `attach` takes the finished path.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()  # drop run output
    # A finished instance has a MachineEnd, so `attach` prints the overview and returns.
    code = main(["attach", "tiny"])
    assert code == 0
    out = capsys.readouterr().out
    assert "machine: tiny" in out
    assert "▸ done" in out  # current state marked (machine_state_mark)
    assert "· route" in out  # a visited state marked
    assert "OK: ended in 'done'" in out


def _stalled_instance(tmp_path: pathlib.Path, *, parked: bool) -> None:
    """Return an instance whose journal has begun but not ended, with a dead worker pid.

    An armed pending wait when *parked*.
    """
    from agent6.machine import journal as machine_journal

    inst = paths.state_dir(tmp_path) / "machines" / "tiny"
    inst.mkdir(parents=True)
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "journal.jsonl").write_text(
        '{"type":"machine.begin","ts":"2026-07-12T00:00:00+00:00","machine":"tiny","version":1}\n',
        encoding="utf-8",
    )
    (inst / "worker.pid").write_text("999999", encoding="utf-8")
    if parked:
        machine_journal.MachineJournal(inst).write_pending_wait(
            machine_journal.PendingWait(state="route", wake_epoch=None)
        )


def _watch_in_thread(timeout_s: float) -> tuple[list[int], bool]:
    import threading

    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "tiny"])), daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result, t.is_alive()


def test_watch_exits_on_a_parked_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The watch exits on a parked (--exit-on-wait) machine, as its docstring promises."""
    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=True)
    result, still_running = _watch_in_thread(5.0)
    assert not still_running, "watch loop failed to terminate on a parked machine"
    assert result == [0]
    out = capsys.readouterr().out
    assert "WAITING" in out and "machine poke tiny" in out


def test_watch_follows_a_live_machine_in_a_wait(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live worker blocked in a foreground wait is followed; attach exits once it is gone."""
    import threading

    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=True)
    inst = paths.state_dir(tmp_path) / "machines" / "tiny"
    ipc.write_worker_pid(inst, os.getpid())
    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "tiny"])), daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert t.is_alive(), f"attach left a live machine: exit {result}"
    ipc.clear_worker_pid(inst)  # the worker goes: the parked instance has nothing to follow
    t.join(timeout=5.0)
    assert result == [0]
    assert "WAITING" in capsys.readouterr().out


def test_watch_exits_on_a_crashed_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crashed worker reads crashed in watch, never "watching..." forever."""
    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=False)
    result, still_running = _watch_in_thread(5.0)
    assert not still_running, "watch loop failed to terminate on a crashed machine"
    assert result == [1]
    assert "STOPPED" in capsys.readouterr().err


def test_watch_exits_on_a_stopped_machine_without_a_pid_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worker that cleared its pid while unwinding has no possible writer."""
    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=False)
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    (root / "worker.pid").unlink()

    result, still_running = _watch_in_thread(5.0)

    assert not still_running, "watch followed a stopped machine with no worker"
    assert result == [1]
    err = capsys.readouterr().err
    # An operator's `machine stop` leaves the same dir as a crash: the line asserts no cause.
    assert "STOPPED in 'route': no worker is running" in err
    assert "resume with `agent6 machine run tiny`" in err
    assert "exited" not in err


def test_replay_pluralizes_the_transition_count(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `machine run` only accepts a committed machine; an untracked file is refused first.
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    code = main(["machine", "run", str(f)])
    assert code == 2  # a refusal, like every REFUSING
    err = capsys.readouterr().err
    assert "uncommitted" in err and "committed machine" in err
    # Refused before touching the state dir: no instance journal was created.
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    assert not (root / "journal.jsonl").exists()


def test_uncommitted_refusal_tracks_git_state(tmp_path: pathlib.Path) -> None:
    from agent6.app.machine import run as machine_run

    # Outside a git repo the gate never fires (nothing to commit against).
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert machine_run.uncommitted_refusal(f, tmp_path) is None
    _git_init(tmp_path)
    assert machine_run.uncommitted_refusal(f, tmp_path) is not None  # untracked
    subprocess.run(["git", "-C", str(tmp_path), "add", "tiny.asm.toml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert machine_run.uncommitted_refusal(f, tmp_path) is None  # committed clean
    f.write_text(TINY + "\n", encoding="utf-8")
    assert machine_run.uncommitted_refusal(f, tmp_path) is not None  # modified again


def test_uncommitted_refusal_checks_the_machine_symlink_not_its_target(
    tmp_path: pathlib.Path,
) -> None:
    """Retargeting a tracked machine symlink is an uncommitted bundle edit."""
    from agent6.app.machine import run as machine_run

    _git_init(tmp_path)
    first = tmp_path / "first.asm.toml"
    second = tmp_path / "second.asm.toml"
    first.write_text(TINY, encoding="utf-8")
    second.write_text(TINY, encoding="utf-8")
    machine = tmp_path / "machine.asm.toml"
    machine.symlink_to(first.name)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)

    machine.unlink()
    machine.symlink_to(second.name)

    assert machine_run.uncommitted_refusal(machine, tmp_path) is not None


def test_uncommitted_refusal_follows_a_symlinked_repo_path(tmp_path: pathlib.Path) -> None:
    """A repo reached through a symlinked prefix keeps its committed-bundle gate."""
    from agent6.app.machine import run as machine_run

    real = tmp_path / "real"
    real.mkdir()
    _git_init(real)
    machine = real / "machine.asm.toml"
    machine.write_text(TINY, encoding="utf-8")
    subprocess.run(["git", "-C", str(real), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(real), "commit", "-q", "-m", "add"], check=True)
    machine.write_text(TINY + "\n# edited\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real)

    assert machine_run.uncommitted_refusal(link / "machine.asm.toml", link) is not None


def test_uncommitted_refusal_covers_the_scripts_bundle(tmp_path: pathlib.Path) -> None:
    """A dirty bundle refuses `machine run`; `machine test` stays the ungated iteration loop.

    A tool executes `scripts/` as trusted logic exactly like the .asm.toml.
    """
    from agent6.app.machine import run as machine_run

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "do.py").write_text("print('hi')\n", encoding="utf-8")
    # No git repo: no gate (nothing to commit against).
    assert machine_run.uncommitted_refusal(f, tmp_path) is None
    _git_init(tmp_path)
    assert machine_run.uncommitted_refusal(f, tmp_path) is not None  # untracked bundle
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert machine_run.uncommitted_refusal(f, tmp_path) is None  # committed clean
    (scripts / "do.py").write_text("print('changed')\n", encoding="utf-8")
    refusal = machine_run.uncommitted_refusal(f, tmp_path)
    assert refusal is not None and "scripts" in refusal  # modified script refuses


def test_first_run_records_the_bundle_and_drift_refuses_continuation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live instance runs the bundle it recorded; an edited script refuses continuation."""
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = _write_machine(tmp_path)  # waiter: parks WAITING under --exit-on-wait
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "do.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)

    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    assert "WAITING" in capsys.readouterr().out
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    recorded = root / "scripts" / "do.py"
    assert recorded.read_text(encoding="utf-8") == "print('hi')\n"

    (scripts / "do.py").write_text("print('changed')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "edit"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 2  # a refusal
    err = capsys.readouterr().err
    assert "scripts/do.py" in err and "archive the instance" in err

    (scripts / "do.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "restore"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    assert "WAITING" in capsys.readouterr().out


def test_continuation_refuses_an_edited_machine_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An edited machine file with the same name and version refuses continuation."""
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = _write_machine(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    f.write_text(WAITER_DELAYED.replace('reason = "ticked"', 'reason = "changed"'), "utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "edit"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 2  # a refusal
    err = capsys.readouterr().err
    assert "differs from the recorded" in err and "archive the instance" in err


def test_run_refuses_rerun_of_ended_instance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An ended instance can only be replayed; a rerun refuses before stamping worker.pid.
    from agent6.machine import drive, load_machine

    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0  # runs to a terminal
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    # Stand in for the previous worker having exited: a pid that is never alive.
    sentinel = 10**9
    ipc.write_worker_pid(root, sentinel)
    code = main(["machine", "run", str(f)])
    assert code == 2  # a refusal
    err = capsys.readouterr().err
    assert "already ended" in err
    assert str(root) in err  # the archive remedy names the instance dir
    # worker.pid was NOT re-stamped with the (live) rerun process pid.
    assert ipc.read_worker_pid(root) == sentinel
    # The journal still reads terminal, unchanged.
    result = drive(load_machine(root / "machine.asm.toml"), MachineJournal(root), None, live=False)
    assert result.status == "ok"


def test_the_fresh_start_remedy_is_a_command_outside_machines(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The printed archive command preserves the instance without leaving a listing row."""
    from agent6.viewmodel import machine_state

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path.parent / f"{tmp_path.name} state"))
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    state_dir = paths.state_dir(tmp_path)
    root = state_dir / "machines" / "tiny"
    # Moving inside machines/ still produces an instance row, even with an invalid id.
    ghost = root.with_name("tiny.archived-1")
    root.rename(ghost)
    assert machine_state.machine_instance_dirs(state_dir) == [ghost]
    ghost.rename(root)

    assert main(["machine", "run", str(f)]) == 2
    err = capsys.readouterr().err
    match = re.search(r"`(mv -- [^`]+)`", err)
    assert match is not None, err
    argv = shlex.split(match[1])
    source, target = map(pathlib.Path, argv[-2:])
    assert source == root
    assert target.parent == root.parent.parent
    assert target.name.startswith("tiny.archived-")
    subprocess.run(argv, check=True)
    assert not root.exists()
    assert (target / "machine.asm.toml").is_file()
    assert machine_state.machine_instance_dirs(state_dir) == []
    assert main(["machine", "run", str(f)]) == 0


def test_poke_drops_signal_for_waiting_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "waiter_delayed"])
    assert code == 0
    assert "poked" in capsys.readouterr().out
    # The signal is now pending for the next take_signal().
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() == (True, None)


def test_poke_carries_data_payload(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", '{"cmd": "go"}']) == 0
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    j = MachineJournal(root)
    assert j.take_signal() == (True, {"cmd": "go"})
    j.ack_signal()
    # --message wraps a plain string.
    assert main(["machine", "poke", "waiter_delayed", "--message", "hello"]) == 0
    assert j.take_signal() == (True, "hello")


def test_poke_rejects_invalid_json_data(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", "{not json}"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_poke_refuses_ended_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A terminal machine consumes no signals, so poking it is refused.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "tiny"])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("REFUSING:") and "already ended" in err
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    assert not (root / "signal").exists()  # no signal was dropped


@pytest.mark.parametrize("alive", [False, True])
def test_poke_refuses_an_instance_without_an_open_wait(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    alive: bool,
) -> None:
    """A poke only wakes the wait that is open now, never a future wait."""
    monkeypatch.chdir(tmp_path)
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    root.mkdir(parents=True)
    (root / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    MachineJournal(root).begin(machine="tiny", version=1)
    if alive:
        ipc.write_worker_pid(root, os.getpid())

    assert main(["machine", "poke", "tiny"]) == 2

    assert "machine 'tiny' has no open wait to poke" in capsys.readouterr().err
    assert not (root / "signal").exists()


def test_poke_missing_instance_errors(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "poke", "nope"])
    assert code == 2
    assert "no machine instance" in capsys.readouterr().err


def test_poke_rejects_an_absolute_path_as_an_instance_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ID cannot redirect the signal write outside the machine state root."""
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "not-a-machine"
    victim.mkdir()

    assert main(["machine", "poke", str(victim)]) == 2

    assert "no machine instance" in capsys.readouterr().err
    assert not (victim / "signal").exists()


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


AGENT_RUN_MACHINE = AGENT_MACHINE_HARD.replace(
    'machine = "hard-usd"', 'machine = "run-warn"'
).replace('kind = "agent"', 'kind = "agent"\nmode = "run"', 1)


WAIT_THEN_RUN = (
    WAITER_DELAYED.replace('machine = "waiter_delayed"', 'machine = "wait-then-run"')
    .replace('on = { tick = "done", signal = "woken" }', 'on = { tick = "work", signal = "work" }')
    .replace(
        "[states.done]",
        """[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.work]
kind = "agent"
mode = "run"
prompt = "fix it"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]""",
    )
    .replace('[states.woken]\nkind = "terminal"\nstatus = "ok"\nreason = "signalled"\n', "")
)


def test_machine_start_and_resume_share_one_begin_and_setup_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume uses the journal's commit after HEAD moves and carries setup in the IPC overlay."""
    from agent6 import git_ops, kinds
    from agent6.config import Config
    from agent6.machine import journal as machine_journal
    from agent6.sessions import layout

    config_home = tmp_path.parent / f"{tmp_path.name}-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    (config_home / "agent6").mkdir(parents=True)
    (config_home / "agent6" / "config.toml").write_text(
        """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
[models.worker]
provider = "anthropic"
model = "x"
""",
        encoding="utf-8",
    )
    _git_init(tmp_path)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "wait.asm.toml"
    source.write_text(WAIT_THEN_RUN, encoding="utf-8")
    agents = tmp_path / "AGENTS.md"
    agents.write_text("## Setup command\n\n```sh\ntrue\n```\n")
    subprocess.run(["git", "add", "wait.asm.toml", "AGENTS.md"], check=True)
    subprocess.run(["git", "commit", "-qm", "machine base"], check=True)
    base = git_ops.status(tmp_path).head_sha
    carried: list[tuple[str, ...]] = []
    real_overlay = machine_agent.state_overlay

    def capture(cfg: Config, isolation: kinds.IsolationLevel) -> machine_agent.StateOverlay:
        overlay = real_overlay(cfg, isolation)
        carried.append(tuple(overlay.overlay["parallel"]["setup_command"]))
        return overlay

    monkeypatch.setattr(machine_agent, "state_overlay", capture)
    argv = ["machine", "run", str(source), "--exit-on-wait", "--auto-approve"]
    assert main(argv) == 0
    root = layout.machines_root(paths.state_dir(tmp_path)) / "wait-then-run"
    journal = MachineJournal(root)
    first = journal.read()[0]
    assert isinstance(first, machine_journal.MachineBegin)
    assert first.base_commit == base
    agents.write_text("## Setup command\n\n```sh\nfalse\n```\n")
    subprocess.run(["git", "commit", "-qam", "move checkout"], check=True)
    assert git_ops.status(tmp_path).head_sha != base
    assert main(argv) == 0
    assert carried == [("true",), ("true",)]
    begins = [event for event in journal.read() if isinstance(event, machine_journal.MachineBegin)]
    assert begins == [first]


def test_run_says_where_a_machines_work_landed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine with run states commits to `agent6/machine-<id>` and never touches the checkout."""
    cfg_home = tmp_path.parent / (tmp_path.name + "-cfg")  # outside the workspace
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    (cfg_home / "agent6").mkdir(parents=True, exist_ok=True)
    (cfg_home / "agent6" / "config.toml").write_text(
        "\n".join(
            (
                "[agent6]",
                "config_version = 1",
                "[providers.anthropic]",
                'api_format = "anthropic"',
                'api_key_env = "ANTHROPIC_API_KEY"',
                "[models.worker]",
                'provider = "anthropic"',
                'model = "x"',
            )
        ),
        encoding="utf-8",
    )
    _git_init(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    # A prior instance's work sits on the branch, above what HEAD holds.
    for argv in (
        ["checkout", "-q", "-b", "agent6/machine-wait-then-run"],
        ["commit", "-q", "--allow-empty", "-m", "a run state's work"],
        ["checkout", "-q", "-"],
    ):
        subprocess.run(["git", "-C", str(tmp_path), *argv], check=True)
    f = tmp_path / "wtr.asm.toml"
    f.write_text(WAIT_THEN_RUN, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "wtr.asm.toml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "machine"], check=True)

    assert main(["machine", "run", str(f), "--exit-on-wait", "--auto-approve"]) == 0

    out = capsys.readouterr().out
    assert "WAITING" in out
    assert "changes are on agent6/machine-wait-then-run" in out
    assert "git merge agent6/machine-wait-then-run" in out


def test_a_fully_pinned_agent_state_needs_no_default_worker_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state whose `config` table pins provider and model runs without a worker default."""
    from agent6.machine import AgentExecResult

    monkeypatch.chdir(tmp_path)
    config_home = tmp_path.parent / f"{tmp_path.name}-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    machine = tmp_path / "pinned.asm.toml"
    machine.write_text(
        AGENT_MACHINE_HARD + '\n[states.judge.config.models.worker]\nprovider = "p"\nmodel = "m"\n',
        encoding="utf-8",
    )
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[providers.p]\napi_format = "openai"\nbase_url = "http://127.0.0.1:9"\n',
        encoding="utf-8",
    )

    def build(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        def run(*_args: object, **_kwargs: object) -> AgentExecResult:
            return AgentExecResult(reason="finish_session", payload={"ok": True})

        return run

    monkeypatch.setattr(machine_agent, "build_machine_agent_runner", build)
    assert main(["--config", str(cfg), "machine", "run", str(machine)]) == 0


def _stub_agent_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.machine import AgentExecResult, AgentRequest

    def build(
        *_args: object, **_kwargs: object
    ) -> Callable[[AgentRequest, pathlib.Path | None], AgentExecResult]:
        def run(_request: AgentRequest, _log: pathlib.Path | None) -> AgentExecResult:
            return AgentExecResult(
                reason="finish_session",
                payload={"ok": True},
                usd=0.02,
                input_tokens=123,
                output_tokens=45,
            )

        return run

    monkeypatch.setattr(machine_agent, "build_machine_agent_runner", build)


@pytest.mark.parametrize(
    ("pin", "role", "percent", "signed_in", "plan_note"),
    [
        (True, "", 10, True, False),
        (True, "", 0, True, False),
        (True, "", 10, False, False),
        (False, "", 10, True, True),
        (True, "reviewer", 10, True, True),
        (True, "planner", 10, True, True),
    ],
)
def test_machine_preflight_judges_the_states_routes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pin: bool,
    role: str,
    percent: int,
    signed_in: bool,
    plan_note: bool,
) -> None:
    """A pinned worker leaves out an unused operator route, but inherited roles stay judged."""
    from agent6 import secret_store
    from agent6.models import cache

    monkeypatch.chdir(tmp_path)
    _stub_agent_result(monkeypatch)
    monkeypatch.setattr(secret_store, "load_secrets", dict)

    def stored(*_args: object, **_kwargs: object) -> object:
        return object()

    def listed(*_args: object, **_kwargs: object) -> list[str]:
        return []

    if signed_in:
        monkeypatch.setattr(secret_store, "load_oauth_tokens", stored)
    monkeypatch.setattr(cache, "list_models", listed)
    cards = paths.cache_dir() / "models"
    cards.mkdir(parents=True, exist_ok=True)
    (cards / "paid.json").write_text(
        json.dumps({"models": ["priced-model"], "pricing": {"priced-model": [1, 2]}}),
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text(
        '[sandbox]\nisolation = "none"\n'
        '[providers.plan]\napi_format = "chatgpt"\n'
        '[providers.paid]\napi_format = "openai"\nbase_url = "http://127.0.0.1:9"\n'
        '[models.worker]\nprovider = "plan"\nmodel = "plan-model"\n'
        f"[budget]\nmax_percent = {percent}\n"
        + (f'[models.{role}]\nprovider = "plan"\nmodel = "plan-model"\n' if role else ""),
        encoding="utf-8",
    )
    machine = tmp_path / "pinned.asm.toml"
    machine.write_text(
        AGENT_MACHINE_HARD
        + (
            '\n[states.judge.config.models.worker]\nprovider = "paid"\nmodel = "priced-model"\n'
            if pin
            else ""
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config), "machine", "run", str(machine)]) == 0
    err = capsys.readouterr().err
    assert ("'plan-model' draws on a subscription plan" in err) is plan_note


@pytest.mark.parametrize("ending", ["ok", "waiting", "stopped"])
def test_machine_run_end_line_carries_spend_and_tokens(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ending: str,
) -> None:
    """Run reports journaled dollars and tokens whether it ends, parks or stops."""
    monkeypatch.chdir(tmp_path)
    _stub_agent_result(monkeypatch)
    config = tmp_path / "config.toml"
    config.write_text(
        '[sandbox]\nisolation = "none"\n'
        '[providers.p]\napi_format = "openai"\nbase_url = "http://127.0.0.1:9"\n'
        '[models.worker]\nprovider = "p"\nmodel = "m"\n',
        encoding="utf-8",
    )
    machine = tmp_path / "spend.asm.toml"
    text = AGENT_MACHINE_HARD
    if ending != "ok":
        text = text.replace('= "done"', '= "park"').replace(
            "[states.done]",
            '[states.park]\nkind = "wait"\nevery_secs = "3600"\n'
            'on = { tick = "done", signal = "done" }\n\n[states.done]',
        )
    machine.write_text(text, encoding="utf-8")
    root = paths.state_dir(tmp_path) / "machines" / "hard-usd"

    def stop_once_parked() -> None:
        deadline = time.monotonic() + 30
        while not MachineJournal(root).wait_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        write_stop_request(root)

    stopper = threading.Thread(target=stop_once_parked) if ending == "stopped" else None
    if stopper is not None:
        stopper.start()
    flags = ["--exit-on-wait"] if ending == "waiting" else []
    try:
        assert main(["--config", str(config), "machine", "run", str(machine), *flags]) == 0
    finally:
        if stopper is not None:
            stopper.join()
    out = capsys.readouterr().out
    assert f"{ending.upper()}: hard-usd" in out
    assert "spent $0.02 (in=123 tok, out=45 tok)" in out


def test_run_refuses_a_legacy_journal_without_a_starting_commit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pre-record instance must restart rather than infer setup from a different commit."""
    from agent6.sessions import layout

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    source = tmp_path / "legacy.asm.toml"
    source.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    subprocess.run(["git", "add", "legacy.asm.toml"], check=True)
    subprocess.run(["git", "commit", "-qm", "legacy machine"], check=True)
    root = layout.machines_root(paths.state_dir(tmp_path)) / "run-warn"
    journal = MachineJournal(root)
    journal.ensure_dirs()
    journal.journal_path.write_text(
        '{"type":"machine.begin","ts":"t","machine":"run-warn","version":1}\n',
        encoding="utf-8",
    )
    assert main(["machine", "run", str(source)]) == 2
    output = capsys.readouterr().err
    assert (
        "journal has no starting commit.\nTo start fresh, archive the instance directory"
    ) in output
    assert "mv --" in output


def test_run_refuses_clone_setup_under_ask_at_preflight(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine needing setup refuses ask once, before creating a clone or calling a model."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    machine_file = tmp_path / "setup.asm.toml"
    machine_file.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("## Setup command\n\n```sh\ntrue\n```\n")
    subprocess.run(["git", "add", "setup.asm.toml", "AGENTS.md"], check=True)
    subprocess.run(["git", "commit", "-qm", "machine"], check=True)
    assert main(["machine", "run", str(machine_file)]) == 2
    output = capsys.readouterr().err
    assert output.count("REFUSING:") == 1
    assert "needs workspace setup" in output
    assert "sandbox.run_commands='ask'" in output
    assert "--auto-approve" in output
    assert "sandbox.run_commands='yes'" in output


def test_run_warns_on_mode_run_states_under_ask_policy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unattended 'ask' machine auto-denies run_command; the note names both remedies up front.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    code = main(["machine", "run", str(f)])
    err = capsys.readouterr().err
    assert code == 2  # no worker configured: refused right after the note
    assert "auto-denies" in err and "--auto-approve" in err


def test_run_warns_naming_the_config_file_the_ask_came_from(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under `--config FILE` the remedy is that file; `config set --repo` would not reach it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    config = tmp_path / "ask.toml"
    config.write_text('[sandbox]\nrun_commands = "ask"\n', encoding="utf-8")
    assert main(["--config", str(config), "machine", "run", str(f)]) == 2
    err = capsys.readouterr().err
    assert f'set sandbox.run_commands = "yes" in {config}' in err
    assert "config set --repo" not in err


def test_run_auto_approve_suppresses_the_warning_and_leaves_the_env_alone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The grant travels in each state's overlay; the supervisor's own process is not approved."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("AGENT6_AUTO_APPROVE", raising=False)
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    code = main(["machine", "run", str(f), "--auto-approve"])
    err = capsys.readouterr().err
    assert code == 2  # still refused on the missing worker; that is fine
    assert "auto-denies" not in err  # the grant removed the dead-end
    assert "AGENT6_AUTO_APPROVE" not in os.environ


def test_a_fresh_instance_over_a_stale_chain_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine chain ref that outlives its instance dir refuses the run, naming both remedies."""
    import subprocess

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.setenv("AGENT6_AUTO_APPROVE", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    f = repo / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "runwarn.asm.toml"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    # The old instance's work: a commit past HEAD that nothing merged.
    work = subprocess.run(
        ["git", "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "work"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/agent6/machine-run-warn/head", work], cwd=repo, check=True
    )
    cfg = tmp_path / "agent6.toml"
    cfg.write_text(
        """[providers.p]
api_format = "openai"
base_url = "http://127.0.0.1:9"
api_key_env = "AGENT6_TEST_KEY"

[models.worker]
provider = "p"
model = "m"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_TEST_KEY", "x")
    code = main(["--config", str(cfg), "machine", "run", str(f)])
    err = capsys.readouterr().err
    assert code == 2
    assert "chain branch 'agent6/machine-run-warn' exists" in err
    assert "git branch -D agent6/machine-run-warn" in err
    # A preflight refusal is not a running worker; the stamped pid must be cleared.

    root = paths.state_dir(repo) / "machines" / "run-warn"
    assert ipc.read_worker_pid(root) is None


PARKED_RUN_MACHINE = """
machine = "run-warn"
version = 1
initial = "park"

[budget]
max_usd = 1.0
max_transitions = 10

[vars.operator]
secs = { type = "int", value = 3600 }

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.park]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "judge", signal = "judge" }

[states.judge]
kind = "agent"
mode = "run"
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


def test_a_fresh_instance_over_a_merged_chain_starts_from_head(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chain whose tip HEAD already holds is spent: the instance drops it and starts from HEAD."""
    import subprocess

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.setenv("AGENT6_AUTO_APPROVE", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    f = repo / "runwarn.asm.toml"
    f.write_text(PARKED_RUN_MACHINE, encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "runwarn.asm.toml"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed"],
        ["git", "update-ref", "refs/agent6/machine-run-warn/head", "HEAD"],
        ["git", "branch", "agent6/machine-run-warn", "HEAD"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    cfg = tmp_path / "agent6.toml"
    cfg.write_text(
        """[providers.p]
api_format = "openai"
base_url = "http://127.0.0.1:9"
api_key_env = "AGENT6_TEST_KEY"

[models.worker]
provider = "p"
model = "m"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_TEST_KEY", "x")
    code = main(["--config", str(cfg), "machine", "run", str(f), "--exit-on-wait"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "REFUSING" not in captured.err
    assert "work on 'agent6/machine-run-warn' is in HEAD" in captured.err
    assert "WAITING" in captured.out
    # This instance committed nothing and HEAD holds the branch: no merge line.
    assert "changes are on" not in captured.out
    gone = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/agent6/machine-run-warn/head"],
        cwd=repo,
        check=False,
    )
    assert gone.returncode != 0  # the spent chain ref is dropped


def test_run_flags_never_land_in_the_supervisors_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flags travel in the resolved configs; the process spawns nothing under them."""
    keys = ("AGENT6_NO_COMMANDS", "AGENT6_DANGEROUSLY_DISABLE_SANDBOX")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "nocmd.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    main(["machine", "run", str(f), "--no-commands", "--dangerously-disable-sandbox"])
    assert not any(key in os.environ for key in keys)


def test_apply_operator_env_grants_upgrades_ask_never_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent6.app import machine_agent
    from agent6.config import Config

    monkeypatch.setenv("AGENT6_AUTO_APPROVE", "1")
    ask = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    assert machine_agent._apply_operator_env_grants(ask).sandbox.run_commands == "yes"
    no = Config.model_validate({"sandbox": {"run_commands": "no"}})
    assert (
        machine_agent._apply_operator_env_grants(no).sandbox.run_commands == "no"
    )  # never resurrected
    monkeypatch.delenv("AGENT6_AUTO_APPROVE")
    assert (
        machine_agent._apply_operator_env_grants(ask).sandbox.run_commands == "ask"
    )  # no grant, no change


TOOL_PROBE_MACHINE = """
machine = "probe-check"
version = 1
initial = "lint"

[budget]
max_usd = 1.0
max_transitions = 10

[states.lint]
kind = "tool"
command = ["definitely-not-a-binary-xyz", "check"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "clean"

[states.fail]
kind = "terminal"
status = "failed"
reason = "lint"
"""


@pytest.mark.parametrize("verb", ["check", "test"])
def test_offline_validation_reads_the_explicit_config_layer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """The top-level `--config` layer applies to every machine command, check and test included."""
    monkeypatch.chdir(tmp_path)
    machine = tmp_path / "tiny.asm.toml"
    machine.write_text(TINY, encoding="utf-8")
    config = tmp_path / "broken.toml"
    config.write_text('[harness]\nnonsense_key = "x"\n', encoding="utf-8")

    assert main(["--config", str(config), "machine", verb, str(machine)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "nonsense_key" in err


def test_check_validates_the_config_overlay_run_will_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine check` and `test` merge the file's [config] table as `machine run` does."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "bad.asm.toml"
    f.write_text(
        TINY + '\n[config.harness]\nnonsense_key = "x"\n',
        encoding="utf-8",
    )
    assert main(["machine", "check", str(f)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "nonsense_key" in err
    assert main(["machine", "test", str(f)]) == 1


def test_check_warns_on_binaries_unreachable_in_the_jail(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The probe covers tool-state command[0] and literal subprocess argv in bundle scripts.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "probe.asm.toml"
    f.write_text(TOOL_PROBE_MACHINE, encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "helper.py").write_text(
        '"""Bundle helper."""\n\nimport subprocess\n\n\n'
        "def main() -> int:\n"
        '    subprocess.run(["also-missing-tool-abc", "x"], check=False)\n'
        '    subprocess.run(["python3", "-c", "pass"], check=False)\n'
        "    return 0\n",
        encoding="utf-8",
    )
    code = main(["machine", "check", str(f)])
    out = capsys.readouterr()
    assert code == 0  # advisory: the operator may install the tool later
    assert "OK:" in out.out
    assert out.err.startswith("[agent6] WARNING:")
    assert "`definitely-not-a-binary-xyz` ([states.lint] command)" in out.err
    assert "`also-missing-tool-abc` (scripts/helper.py)" in out.err
    assert "python3" not in out.err  # reachable binaries stay quiet


def test_machine_stop_marks_a_running_worker_and_notes_a_dead_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine stop` writes the durable marker only for a live worker.

    A parked or dead instance gets the note and exit 0, never a marker that ambushes the next run.
    """
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    assert main(["machine", "stop", "tiny"]) == 0  # ended: nothing to stop
    err = capsys.readouterr().err
    assert err.startswith("[agent6] ") and "already ended" in err and "nothing to stop" in err
    assert not (root / "stop").exists()

    w = _write_machine(tmp_path)  # waiter: parks WAITING, journal not ended
    assert main(["machine", "run", str(w), "--exit-on-wait"]) == 0
    capsys.readouterr()
    wroot = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert main(["machine", "stop", "waiter_delayed"]) == 0  # parked, worker dead
    err = capsys.readouterr().err
    assert err.startswith("[agent6] ") and "not running" in err
    assert not (wroot / "stop").exists()

    def _alive(_root: pathlib.Path) -> bool:
        return True

    monkeypatch.setattr(ipc, "worker_is_alive", _alive)  # the verb gate's owner
    assert main(["machine", "stop", "waiter_delayed"]) == 0
    assert "stop requested" in capsys.readouterr().out
    assert (wroot / "stop").is_file()


def test_run_start_clears_a_stale_stop_marker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A leftover stop marker does not park the next invocation at its first boundary."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    root.mkdir(parents=True)
    (root / "stop").touch()
    assert main(["machine", "run", str(f)]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out and "STOPPED" not in out
    assert not (root / "stop").exists()


def test_hub_spawn_away_mode_reaches_the_instance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hub-spawned machine records "wait" on its instance dir at run start.

    Every agent state's bridges then park prompts for the front-end, whenever its viewer registers.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    assert ipc.away_mode(root) == "wait"


def test_attach_degrades_a_corrupt_journal_like_status(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine status` and the watch print a clean ERROR for a corrupt journal, no traceback."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    (root / "journal.jsonl").write_text('{"type": "machine.begin"\n', encoding="utf-8")
    code = main(["attach", "tiny"])
    assert code == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err


def test_run_refuses_an_explicit_protect_git_the_host_cannot_enforce(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine run` makes the same protect_git check `run` and `ask` make (`config_refusal`)."""
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True, exist_ok=True)
    (cfg_home / "agent6" / "config.toml").write_text(
        "[sandbox]\nprotect_git = true\n", encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "repo"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    class _Env:
        detected_isolation = "hardened"

    monkeypatch.setattr(_setup, "detect_env", _Env)

    def _hardened(_req: str, _env: object) -> str:
        return "hardened"

    monkeypatch.setattr(detect, "resolve_isolation", _hardened)
    f = workspace / "probe.asm.toml"
    f.write_text(TOOL_PROBE_MACHINE, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "protect_git" in err


def test_run_refuses_a_state_dir_inside_the_workspace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine run` refuses a state base inside the workspace, as `run` does."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "inside-state"))
    monkeypatch.chdir(tmp_path)

    class _Env:
        detected_isolation = "strict"

    monkeypatch.setattr(_setup, "detect_env", _Env)

    def _strict(_req: str, _env: object) -> str:
        return "strict"

    monkeypatch.setattr(detect, "resolve_isolation", _strict)
    f = tmp_path / "probe.asm.toml"
    f.write_text(TOOL_PROBE_MACHINE, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "private directory" in err


def test_list_joins_instances_with_their_files_and_names_the_rest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 machine` (`machine list`) is the CLI's machines page.

    Each instance's status and current state joined with the authored file, then the files no
    instance has run, and an unparsable file kept by path alone.
    """
    monkeypatch.chdir(tmp_path)
    assert main(["machine"]) == 0
    assert "no machines yet" in capsys.readouterr().out
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    (tmp_path / "waiter.asm.toml").write_text(WAITER_DELAYED, encoding="utf-8")
    (tmp_path / "broken.asm.toml").write_text("not toml {{", encoding="utf-8")
    (tmp_path / "broken2.asm.toml").write_text("also not toml {{", encoding="utf-8")
    assert main(["machine", "run", str(tmp_path / "tiny.asm.toml")]) == 0
    capsys.readouterr()
    assert main(["machine"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["updated", "status", "state", "machine", "spec", "file"]
    tiny = next(line for line in lines if "tiny.asm.toml" in line)
    assert tiny.split()[2:] == ["ok", "done", "tiny", "valid", "tiny.asm.toml"]  # [2:]: the when
    waiter = next(line for line in lines if "waiter.asm.toml" in line)
    assert waiter.split() == ["-", "-", "-", "waiter_delayed", "valid", "waiter.asm.toml"]
    broken = next(line for line in lines if "broken.asm.toml" in line)
    assert broken.split() == ["-", "-", "-", "-", "invalid", "broken.asm.toml"]
    # Two unparsable files (both named "-") keep two rows.
    assert any("broken2.asm.toml" in line for line in lines)


def test_list_names_a_corrupt_machine_journal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The listing keeps the shared unreadable word and says what is corrupt."""
    monkeypatch.chdir(tmp_path)
    root = paths.state_dir(tmp_path) / "machines" / "tiny"
    root.mkdir(parents=True)
    (root / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (root / "journal.jsonl").write_text("{not json\n", encoding="utf-8")

    assert main(["machine", "list"]) == 0

    row = next(line for line in capsys.readouterr().out.splitlines() if "tiny" in line)
    assert "unreadable · corrupt journal line 1" in row


def test_status_and_list_name_a_parked_approval(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worker holding an unanswered approval reads "waiting" and status names the state."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    MachineJournal(root).clear_pending_wait()
    ipc.write_worker_pid(root, os.getpid())
    execution = root / "states" / "0001-attempt"
    execution.mkdir(parents=True)
    (execution / "logs.jsonl").write_text(
        json.dumps({"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: x"})
        + "\n",
        encoding="utf-8",
    )
    assert main(["machine", "status", "waiter_delayed"]) == 0
    out = capsys.readouterr().out
    assert "status: waiting" in out
    assert "an approval open in 0001-attempt" in out
    assert main(["machine", "list"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "waiter_delayed" in line)
    assert "waiting" in row and "0001-attempt" in row


def test_machine_stop_refuses_an_instance_whose_journal_it_cannot_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt journal is not "nothing to stop": the verb refuses with exit 2 and no marker."""
    import os

    monkeypatch.chdir(tmp_path)
    w = _write_machine(tmp_path)
    assert main(["machine", "run", str(w), "--exit-on-wait"]) == 0
    capsys.readouterr()
    root = paths.state_dir(tmp_path) / "machines" / "waiter_delayed"
    journal = root / "journal.jsonl"
    # Corrupt at the head with a valid tail: a tail-only guard let it through to the fold.
    journal.write_text("{not json\n" + journal.read_text(encoding="utf-8"), encoding="utf-8")
    for alive in (False, True):
        if alive:
            ipc.write_worker_pid(root, os.getpid())
        assert main(["machine", "stop", "waiter_delayed"]) == 2
        err = capsys.readouterr().err
        assert err.startswith("REFUSING: machine 'waiter_delayed':") and "journal" in err
        assert not (root / "stop").exists()
    # Poke reads the whole journal too: a tail-only read wrote a signal into a dead instance.
    assert main(["machine", "poke", "waiter_delayed"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("REFUSING: machine 'waiter_delayed':") and "journal" in err
    assert not any(p.name.startswith("signal") for p in root.iterdir())


def test_a_machine_verb_takes_the_name_the_listing_prints(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine list` prints a name and a file; check, test, graph and run take either."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "machines").mkdir()
    (tmp_path / "machines" / "nightly.asm.toml").write_text(WAITER_DELAYED, encoding="utf-8")
    assert main(["machine", "check", "waiter_delayed"]) == 0
    assert "machines/nightly.asm.toml" in capsys.readouterr().out
    assert main(["machine", "graph", "waiter_delayed"]) == 0
    # An existing path wins over a name.
    (tmp_path / "waiter_delayed").write_text("not a machine", encoding="utf-8")
    assert main(["machine", "check", "waiter_delayed"]) == 1


def test_a_name_two_files_declare_is_refused_with_both(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("a.asm.toml", "b.asm.toml"):
        (tmp_path / name).write_text(WAITER_DELAYED, encoding="utf-8")
    with pytest.raises(errors.OperatorError, match=r"a\.asm\.toml, b\.asm\.toml\); name the file"):
        main(["machine", "check", "waiter_delayed"])


FAILING_TOOL = """
machine = "failing_tool"
version = 1
initial = "check"

[budget]
max_transitions = 5

[states.check]
kind = "tool"
command = ["true"]
timeout_secs = 5
on = { ok = "done", nonzero = "broken", timeout = "broken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "checked"

[states.broken]
kind = "terminal"
status = "failed"
reason = "the check failed"
"""


def test_a_failed_tool_step_says_why_in_the_run_and_in_status(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `nonzero` edge reads as progress: the exit code and last output line ride along."""
    from agent6 import kinds
    from agent6.app import reporter as app_reporter
    from agent6.app.machine import run
    from agent6.sandbox import jail

    def failing(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return kinds.CommandResult(
            argv=policy.argv,
            returncode=3,
            stdout="",
            stderr="warming up\npython3: can't open file 'scripts/x.py'\n",
            duration_s=0.0,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jail, "run_in_jail", failing)
    mfile = tmp_path / "failing.asm.toml"
    mfile.write_text(FAILING_TOOL, encoding="utf-8")
    said: list[str] = []
    assert run.run_machine(mfile, app_reporter.Reporter(out=said.append, err=said.append)) == 1
    assert "[agent6] state 'check': exit 3: python3: can't open file 'scripts/x.py'" in said
    assert main(["machine", "status", "failing_tool"]) == 0
    assert (
        "[0] check --nonzero--> broken -- exit 3: python3: can't open file 'scripts/x.py'"
        in capsys.readouterr().out
    )
