# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the deterministic engine: run, capture, branch, replay, recovery."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest

from agent6 import kinds
from agent6.app.machine import run
from agent6.config import Config
from agent6.machine import _semantics
from agent6.machine import engine as machine_engine
from agent6.machine import journal as machine_journal

# A minimal tool/branch/terminal machine: scan -> (branch on items) -> record -> stop.
COUNTER = """
machine = "counter"
version = 1
initial = "scan"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
items = { type = "list[str]", default = [] }

[schemas.scan_result]
items = "list[str]"

[states.scan]
kind = "tool"
command = ["scan"]
output_schema = "scan_result"
capture = { set = { items = "{{ result.items }}" } }
timeout_secs = 5
on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

[states.check]
kind = "branch"
when = [
  { if = "len(items) == 0", goto = "stop_ok" },
  { else = true, goto = "record" },
]

[states.record]
kind = "tool"
command = ["record", "{{ items }}"]
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "tool failed"
"""

# A waiting machine.
WAITER = """
machine = "waiter"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 1 }

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

# A mutable wait interval can become invalid at runtime; the engine fails cleanly.
WAITER_DYNAMIC_ZERO = """
machine = "waiter_dynamic_zero"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
secs = { type = "int", default = 0 }

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

# A waiting machine with a non-zero poll interval (for --exit-on-wait).
WAITER_DELAYED = """
machine = "waiter_delayed"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 60 }

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

# A wait with no timer: park until a poke, then a tool consumes the poke payload.
FOREVER = """
machine = "forever"
version = 1
initial = "park"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
last = { type = "json", default = {} }

[states.park]
kind = "wait"
on = { signal = "record" }

[states.record]
kind = "tool"
command = ["record"]
capture = { stdout_json = "last" }
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "fail"
"""


# A terminal's `notify` template that references an absent optional field raises at render.
NOTIFY_FAIL = """
machine = "notify_fail"
version = 1
initial = "route"

[budget]
max_transitions = 10

[schemas.r]
note = { type = "str", optional = true }

[vars.agent]
out = { type = "r", default = {} }

[states.route]
kind = "branch"
when = [ { else = true, goto = "done" } ]

[states.done]
kind = "terminal"
notify = "bye {{ out.note }}"
status = "ok"
reason = "finished"
"""


# A machine with a `notify` message on a tool state, plus a terminal.
NOTIFIER = """
machine = "notifier"
version = 1
initial = "work"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
who = { type = "str", value = "ops" }

[states.work]
kind = "tool"
notify = { message = "hi {{ who }}", level = "warn" }
command = ["noop"]
timeout_secs = 5
on = { ok = "done", nonzero = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "finished"
"""


# An unbounded loop, guarded only by max_transitions.
SPINNER = """
machine = "spinner"
version = 1
initial = "spin"

[budget]
max_usd = 1.0
max_transitions = 3

[states.spin]
kind = "tool"
command = ["noop"]
timeout_secs = 5
on = { ok = "spin", nonzero = "spin", timeout = "spin" }
"""


# An agent state: review -> (branch on verdict.approved) -> stop_ok / stop_fail.
REVIEWER = """
machine = "reviewer"
version = 1
initial = "review"

[budget]
max_usd = 1.0
max_transitions = 100

[schemas.verdict]
approved = "bool"
note = "str"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.review]
kind = "agent"
model = "claude-sonnet-4-5"
prompt = "Review the change."
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_secs = 600
on = { ok = "route", failed = "stop_fail", budget_exhausted = "halt", timeout = "expired" }

[states.route]
kind = "branch"
when = [
  { if = "verdict.approved", goto = "stop_ok" },
  { else = true, goto = "stop_fail" },
]

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "approved"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "rejected"

[states.halt]
kind = "terminal"
status = "failed"
reason = "budget"

[states.expired]
kind = "terminal"
status = "failed"
reason = "timeout"
"""


# An agent state capturing a scalar field via `set` into a declared var.
SCORER = """
machine = "scorer"
version = 1
initial = "score"

[budget]
max_usd = 1.0
max_transitions = 100

[schemas.score_result]
points = "int"

[vars.agent]
total = { type = "int", default = 0 }

[states.score]
kind = "agent"
model = "m"
prompt = "Score it."
output_schema = "score_result"
capture = { set = { total = "{{ result.points }}" } }
timeout_secs = 60
on = { ok = "stop_ok", failed = "stop_fail", budget_exhausted = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "fail"
"""


@dataclasses.dataclass
class FakeWorld:
    """A deterministic :class:`World`: programmed tool results and wakes."""

    tool_results: dict[str, machine_engine.ToolExecResult]
    wakes: list[machine_engine.WaitWake] = dataclasses.field(default_factory=list)
    clock: float = 1000.0
    calls: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
    net_calls: list[tuple[tuple[str, ...], kinds.NetworkMode]] = dataclasses.field(
        default_factory=list
    )
    agent_results: list[machine_engine.AgentExecResult] = dataclasses.field(default_factory=list)
    agent_calls: list[machine_engine.AgentRequest] = dataclasses.field(default_factory=list)
    sleep_deadlines: list[float | None] = dataclasses.field(default_factory=list)
    materialized: list[Any] = dataclasses.field(default_factory=list)
    notifications: list[tuple[str, str, str, str]] = dataclasses.field(default_factory=list)

    def run_tool(
        self,
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        network: kinds.NetworkMode = "none",
        pass_env: tuple[str, ...] = (),
    ) -> machine_engine.ToolExecResult:
        self.calls.append(argv)
        self.net_calls.append((argv, network))
        return self.tool_results[argv[0]]

    def run_agent(self, request: machine_engine.AgentRequest) -> machine_engine.AgentExecResult:
        self.agent_calls.append(request)
        return self.agent_results.pop(0)

    def now(self) -> float:
        return self.clock

    def sleep_until(self, wake_epoch: float | None) -> machine_engine.WaitWake:
        self.sleep_deadlines.append(wake_epoch)
        return self.wakes.pop(0) if self.wakes else machine_engine.WaitWake("tick")

    def materialize_poke(self, payload: Any) -> None:
        self.materialized.append(payload)

    def notify(self, kind: str, state: str, message: str, level: str) -> None:
        self.notifications.append((kind, state, message, level))


def _ok(stdout: str = "") -> machine_engine.ToolExecResult:
    return machine_engine.ToolExecResult(exit_code=0, stdout=stdout, timed_out=False)


def _load(tmp_path: pathlib.Path, text: str) -> tuple[machine_journal.MachineJournal, pathlib.Path]:
    f = tmp_path / "m.asm.toml"
    f.write_text(text, encoding="utf-8")
    return machine_journal.MachineJournal(tmp_path / "inst"), f


def test_full_run_reaches_ok_terminal(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world: machine_engine.World = FakeWorld({"scan": _ok('{"items": ["a", "b"]}'), "record": _ok()})
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "done", "stop_ok", 3)
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["items"] == ["a", "b"]


def test_branch_routes_to_ok_when_empty(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": []}')})
    result = machine_engine.drive(spec, journal, world, live=True)
    # scan -> check -> stop_ok (record never runs)
    assert result == machine_engine.MachineResult("ok", "done", "stop_ok", 2)
    assert world.calls == [("scan",)]


def test_tool_stdout_violating_output_schema_halts(tmp_path: pathlib.Path) -> None:
    """A tool output that fails its declared schema halts the machine with a clean failed result."""
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": "notalist"}')})
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "output_schema" in result.reason
    # The capture never ran, so no snapshot carries a corrupted `items`.
    snap = journal.latest_snapshot()
    assert snap is None or snap.blackboard.get("items") == []


def test_tool_nonzero_routes_to_failure(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {"scan": machine_engine.ToolExecResult(exit_code=1, stdout="", timed_out=False)}
    )
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"


def test_tool_stderr_is_journaled_on_the_fact(tmp_path: pathlib.Path) -> None:
    # A failing tool's stderr flows into the journal; routing still keys off exit_code.
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {
            "scan": machine_engine.ToolExecResult(
                exit_code=2, stdout="", timed_out=False, stderr="boom: bad flag\n"
            )
        }
    )
    assert machine_engine.drive(spec, journal, world, live=True).state == "stop_fail"
    tool_steps = [
        e
        for e in journal.read()
        if isinstance(e, machine_journal.StepEvent) and isinstance(e.fact, machine_journal.ToolFact)
    ]
    assert tool_steps
    fact = tool_steps[0].fact
    assert isinstance(fact, machine_journal.ToolFact) and fact.stderr == "boom: bad flag\n"


def test_tool_timeout_routes_to_failure(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {"scan": machine_engine.ToolExecResult(exit_code=0, stdout="", timed_out=True)}
    )
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"


def test_empty_tool_stdout_is_not_json_for_an_opaque_capture(tmp_path: pathlib.Path) -> None:
    """`stdout_json` means parse one JSON value; an empty stdout is malformed, not a null."""
    journal, f = _load(tmp_path, FOREVER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"record": _ok("")}, wakes=[machine_engine.WaitWake("signal")])

    result = machine_engine.drive(spec, journal, world, live=True)

    assert result.status == "failed"
    assert "not valid JSON" in result.reason
    snapshot = journal.latest_snapshot()
    assert snapshot is not None and snapshot.blackboard["last"] == {}


def test_tool_bad_stdout_fails_clean_without_poisoning_journal(tmp_path: pathlib.Path) -> None:
    # Non-JSON stdout halts FAILED cleanly and never journals the poison fact.
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok("not json at all")})
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "not valid JSON" in result.reason
    # No StepEvent was written: only MachineBegin + MachineEnd.
    events = journal.read()
    assert not any(isinstance(e, machine_journal.StepEvent) for e in events)
    assert isinstance(events[-1], machine_journal.MachineEnd)
    # Replay over the same journal returns the failure, it does not raise.
    replayed = machine_engine.drive(spec, journal, None, live=False)
    assert replayed.status == "failed"


def test_recovery_rejects_a_tool_route_that_disagrees_with_its_fact(tmp_path: pathlib.Path) -> None:
    """A successful tool fact determines `ok`; a journal cannot relabel it during crash recovery."""
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="nonzero",
            goto="stop_fail",
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        )
    )
    journal.append(
        machine_journal.MachineEnd(
            ts="t", status="failed", reason="tool failed", state="stop_fail", transitions=1
        )
    )

    with pytest.raises(machine_engine.EngineError, match="fact implies label 'ok'"):
        machine_engine.drive(spec, journal, None, live=False)


def test_recovery_rejects_a_branch_choice_that_the_blackboard_did_not_take(
    tmp_path: pathlib.Path,
) -> None:
    """A branch is pure, so replay recomputes its winning clause and trusts no journaled index."""
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        )
    )
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=1,
            state="check",
            label="else",
            goto="record",
            fact=machine_journal.BranchFact(clause_index=1),
        )
    )

    with pytest.raises(machine_engine.EngineError, match=r"branch fact selects clause 1.*clause 0"):
        machine_engine.drive(spec, journal, None, live=False)


def test_recovery_rejects_a_journal_without_its_begin_event(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        )
    )

    with pytest.raises(machine_engine.EngineError, match=r"must start with a machine\.begin"):
        machine_engine.drive(spec, journal, None, live=False)


def test_recovery_rejects_a_goto_the_state_never_declared(tmp_path: pathlib.Path) -> None:
    # A fabricated destination fails loudly at its own event.
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="ghost",  # not an edge scan declares
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        )
    )
    with pytest.raises(machine_engine.EngineError, match="not an edge"):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)


def test_recovery_rejects_a_state_the_edited_file_dropped(tmp_path: pathlib.Path) -> None:
    # A journal against a machine file edited to drop the destination fails loudly, not KeyError.
    journal, _f = _load(tmp_path, COUNTER)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",  # a real edge when recorded
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"items": ["a"]}', timed_out=False),
        )
    )
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=1,
            state="check",
            label="[1]",
            goto="record",
            fact=machine_journal.BranchFact(clause_index=1),
        ),
    )
    edited = COUNTER.replace(
        """[states.record]
kind = "tool"
command = ["record", "{{ items }}"]
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

""",
        "",
    ).replace('{ else = true, goto = "record" }', '{ else = true, goto = "stop_ok" }')
    edited_dir = tmp_path / "edited"
    edited_dir.mkdir()
    spec = _semantics.load_machine(_load(edited_dir, edited)[1])
    with pytest.raises(machine_engine.EngineError, match=r"no longer declares|not an edge"):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)


def test_recovery_rejects_machine_id_mismatch_after_the_journal_ended(
    tmp_path: pathlib.Path,
) -> None:
    """A terminal journal still belongs to its recorded machine, not a spec reusing the dir."""
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    machine_engine.drive(spec, journal, FakeWorld({"scan": _ok('{"items": []}')}), live=True)
    other_path = tmp_path / "other.asm.toml"
    other_path.write_text(
        COUNTER.replace('machine = "counter"', 'machine = "other"'), encoding="utf-8"
    )
    other = _semantics.load_machine(other_path)

    with pytest.raises(machine_engine.EngineError, match="started by machine 'counter'"):
        machine_engine.drive(other, journal, None, live=True)


def test_recovery_rejects_machine_id_mismatch(tmp_path: pathlib.Path) -> None:
    # A different machine reusing the same instance id is caught up front.
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="someone_else", version=1)
    with pytest.raises(machine_engine.EngineError, match="someone_else"):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)


def test_record_splices_captured_list(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": ["x", "y", "z"]}'), "record": _ok()})
    machine_engine.drive(spec, journal, world, live=True)
    assert world.calls == [("scan",), ("record", "x", "y", "z")]


def test_replay_reproduces_path_without_world(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": ["a"]}'), "record": _ok()})
    live = machine_engine.drive(spec, journal, world, live=True)
    replayed = machine_engine.drive(spec, journal, None, live=False)
    assert replayed == live


def test_replay_of_incomplete_journal(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"items": ["a"]}', timed_out=False),
        )
    )
    result = machine_engine.drive(spec, journal, None, live=False)
    assert result.status == "incomplete"
    assert result.state == "check"
    assert result.transitions == 1


def test_crash_recovery_continues_without_redoing_step(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    # A crash after `scan`: recovery rebuilds `items` from the fact and continues from `check`.
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=machine_journal.ToolFact(
                exit_code=0, stdout='{"items": ["a", "b"]}', timed_out=False
            ),
        )
    )
    world = FakeWorld({"record": _ok()})  # no "scan" — proving it is not re-run
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "done", "stop_ok", 3)
    assert world.calls == [("record", "a", "b")]


def test_resume_finished_machine_is_idempotent(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": []}')})
    first = machine_engine.drive(spec, journal, world, live=True)
    # A second run with no world at all returns the recorded terminal result.
    again = machine_engine.drive(spec, journal, None, live=True)
    assert again == first


def test_max_transitions_halts_loop(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, SPINNER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"noop": _ok()})
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "max_transitions" in result.reason
    assert result.transitions == 3


def test_wait_tick_path(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, wakes=[machine_engine.WaitWake("tick")])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "ticked", "done", 1)
    events = journal.read()
    step = next(e for e in events if isinstance(e, machine_journal.StepEvent))
    assert step.label == "tick"


def test_wait_zero_dynamic_interval_fails_cleanly(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER_DYNAMIC_ZERO)
    spec = _semantics.load_machine(f)
    result = machine_engine.drive(spec, journal, FakeWorld({}), live=True)
    assert result.status == "failed"
    assert "`every_secs` must be >= 1" in result.reason
    assert not any(isinstance(event, machine_journal.StepEvent) for event in journal.read())


def test_run_tool_uses_the_injected_jail_runner(tmp_path: pathlib.Path) -> None:
    """The tool step executes through LiveWorld.jail_runner, the seam a run-machine overrides."""
    seen: list[kinds.JailPolicy] = []

    def _runner(policy: kinds.JailPolicy) -> kinds.CommandResult:
        seen.append(policy)
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="{}", stderr="", duration_s=0.0
        )

    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=machine_journal.MachineJournal(tmp_path / "i"),
        tool_policy=lambda argv, timeout_s, network, pass_env: kinds.JailPolicy(
            cwd=tmp_path, argv=argv
        ),
        jail_runner=_runner,
    )
    res = world.run_tool(("echo", "hi"), 5.0)
    assert res.exit_code == 0 and not res.timed_out
    assert [p.argv for p in seen] == [("echo", "hi")]


def test_exit_on_wait_zero_dynamic_interval_fails_cleanly(tmp_path: pathlib.Path) -> None:
    # The --exit-on-wait path halts FAILED with a journaled MachineEnd too.
    journal, f = _load(tmp_path, WAITER_DYNAMIC_ZERO)
    spec = _semantics.load_machine(f)
    result = machine_engine.drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "failed"
    assert "`every_secs` must be >= 1" in result.reason
    assert isinstance(journal.read()[-1], machine_journal.MachineEnd)


def test_wait_signal_path(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, wakes=[machine_engine.WaitWake("signal")])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "signalled", "woken", 1)


def test_a_foreground_wait_persists_its_deadline_before_sleeping(tmp_path: pathlib.Path) -> None:
    """The foreground wait persists the same durable PendingWait as --exit-on-wait does.

    A restart resumes the pre-existing pending for the state instead of recomputing from now().
    """

    class _ClockedWorld:
        def __init__(self, journal: machine_journal.MachineJournal) -> None:
            self._journal = journal
            self.slept: list[float | None] = []
            self.persisted: list[float | None] = []

        def run_tool(
            self, argv: Any, timeout_s: Any, *, network: Any = "none", pass_env: Any = ()
        ) -> Any:
            raise AssertionError("no tool states")

        def run_agent(self, request: Any, events_log: Any = None) -> Any:
            raise AssertionError("no agent states")

        def now(self) -> float:
            return 1000.0

        def sleep_until(self, wake_epoch: float | None) -> machine_engine.WaitWake:
            pending = self._journal.read_pending_wait()
            self.persisted.append(None if pending is None else pending.wake_epoch)
            self.slept.append(wake_epoch)
            return machine_engine.WaitWake("tick")

        def materialize_poke(self, payload: Any) -> None:
            pass

        def notify(self, kind: str, state: str, message: str, level: str) -> None:
            pass

    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = _semantics.load_machine(f)
    world = _ClockedWorld(journal)
    assert machine_engine.drive(spec, journal, world, live=True).status == "ok"
    # The deadline was durable before the sleep, and the sleep received it.
    assert world.persisted == [1060.0]
    assert world.slept == [1060.0]
    assert journal.read_pending_wait() is None  # consumed wake leaves no stale park

    # Restart mid-sleep: the persisted instant is resumed, never recomputed.
    other = tmp_path / "other"
    journal2 = machine_journal.MachineJournal(other / "inst")
    journal2.write_pending_wait(machine_journal.PendingWait(state="poll", wake_epoch=1013.5))
    world2 = _ClockedWorld(journal2)
    assert machine_engine.drive(spec, journal2, world2, live=True).status == "ok"
    assert world2.slept == [1013.5]


def test_a_signal_wake_acks_its_poke_after_the_step(tmp_path: pathlib.Path) -> None:
    """A poke's claim file outlives take_signal and is dropped once the step is durable."""
    journal, f = _load(tmp_path, WAITER)
    spec = _semantics.load_machine(f)
    journal.poke("p")
    assert journal.take_signal() == (True, "p")  # the claim is made, unacked
    world = FakeWorld({}, wakes=[machine_engine.WaitWake("signal", "p")])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "signalled", "woken", 1)
    assert not journal.signal_path.with_suffix(".consuming").exists()
    assert journal.take_signal() == (False, None)


def test_a_wake_record_outlives_the_step_that_consumes_it(tmp_path: pathlib.Path) -> None:
    """The wait record is dropped only once the transition it produced is in the journal."""
    journal, f = _load(tmp_path, WAITER)
    spec = _semantics.load_machine(f)
    armed: list[machine_journal.PendingWait | None] = []
    appended = journal.append

    def _die_on_the_step(event: object) -> None:
        if isinstance(event, machine_journal.StepEvent):
            armed.append(journal.read_pending_wait())
            raise KeyboardInterrupt("died between the wake and its StepEvent")
        appended(event)  # pyright: ignore[reportArgumentType]

    journal.append = _die_on_the_step  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        machine_engine.drive(
            spec, journal, FakeWorld({}, wakes=[machine_engine.WaitWake("tick")]), live=True
        )
    journal.append = appended  # type: ignore[method-assign]

    # The armed record survives for the restart, so the wait resumes on the same instant.
    assert armed and armed[0] is not None
    assert journal.read_pending_wait() == armed[0]


def test_notify_journals_event_and_fires_hook(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, NOTIFIER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"noop": _ok()})
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "ok"
    # The `notify` message rendered and journaled on entry to `work`.
    note = next(e for e in journal.read() if isinstance(e, machine_journal.MachineNotify))
    assert note.state == "work"
    assert note.message == "hi ops"
    assert note.level == "warn"
    # The operator hook fired for the notify AND the terminal end.
    assert ("notify", "work", "hi ops", "warn") in world.notifications
    assert ("end", "done", "finished", "ok") in world.notifications


def test_terminal_notify_render_failure_keeps_status_but_is_never_silent(
    tmp_path: pathlib.Path,
) -> None:
    """`notify` is presentation only: a render failure is journaled, never a status flip."""
    journal, f = _load(tmp_path, NOTIFY_FAIL)
    spec = _semantics.load_machine(f)
    world = FakeWorld({})
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "ok"
    assert result.reason == "finished"
    note = next(e for e in journal.read() if isinstance(e, machine_journal.MachineNotify))
    assert note.level == "error" and "notify failed" in note.message
    assert any(
        kind == "notify" and level == "error" and "notify failed" in message
        for kind, _state, message, level in world.notifications
    )
    events_ok = journal.read()
    assert isinstance(events_ok[-1], machine_journal.MachineEnd)
    assert events_ok[-1].status == "ok"


def test_live_world_materializes_poke_atomically(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data"
    world = machine_engine.LiveWorld(
        cwd=tmp_path, journal=machine_journal.MachineJournal(tmp_path / "i"), data_dir=data_dir
    )
    world.materialize_poke({"cmd": "go", "n": 2})
    poke = data_dir / "poke.json"
    assert json.loads(poke.read_text(encoding="utf-8")) == {"cmd": "go", "n": 2}
    # No leftover temp file (atomic temp+rename), so a reader never sees a torn file.
    assert not (data_dir / "poke.json.tmp").exists()
    # No data dir -> a silent no-op, never a crash.
    machine_engine.LiveWorld(
        cwd=tmp_path, journal=machine_journal.MachineJournal(tmp_path / "i")
    ).materialize_poke("x")


def test_data_dir_env_matches_jail_mount(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The data dir lives outside cwd; the jail mounts it at its real path in every isolation.
    from agent6.machine import engine

    data_dir = tmp_path / "state" / "machines" / "m" / "data"
    captured: dict[str, kinds.JailPolicy] = {}

    def fake_run_in_jail(policy: kinds.JailPolicy) -> kinds.CommandResult:
        captured["policy"] = policy
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="{}", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(engine, "run_in_jail", fake_run_in_jail)

    levels: tuple[kinds.IsolationLevel, ...] = ("strict", "hardened")
    for isolation in levels:
        world = machine_engine.LiveWorld(
            cwd=tmp_path,
            journal=machine_journal.MachineJournal(tmp_path / "i"),
            data_dir=data_dir,
            tool_policy=run.machine_tool_policy_factory(
                Config(), tmp_path, isolation, protect_paths=(), data_dir=data_dir
            ),
        )
        world.run_tool(("python3", "x.py"), 5.0)
        policy = captured["policy"]
        assert dict(policy.env)["AGENT6_MACHINE_DATA_DIR"] == str(data_dir)
        assert data_dir in policy.extra_rw_paths


def test_tool_jails_carry_the_operator_hide_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine's tool jails are jailed commands like any other; hide_paths reaches them."""
    from agent6.machine import engine

    captured: dict[str, kinds.JailPolicy] = {}

    def fake_run_in_jail(policy: kinds.JailPolicy) -> kinds.CommandResult:
        captured["policy"] = policy
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(engine, "run_in_jail", fake_run_in_jail)
    hidden = tmp_path / "cred.txt"
    cfg = Config.model_validate({"sandbox": {"hide_paths": [str(hidden)]}})
    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=machine_journal.MachineJournal(tmp_path / "i"),
        tool_policy=run.machine_tool_policy_factory(
            cfg, tmp_path, "strict", protect_paths=(), data_dir=None
        ),
    )
    world.run_tool(("python3", "x.py"), 5.0)
    assert hidden in captured["policy"].hide_paths


def test_live_world_run_tool_maps_rc124_to_timed_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LiveWorld.run_tool derives timed_out from rc 124, so on.timeout is reachable."""
    from agent6.machine import engine

    def fake_run_in_jail(policy: kinds.JailPolicy) -> kinds.CommandResult:
        # The launcher SIGKILLed the child at the deadline and reported rc=124.
        return kinds.CommandResult(
            argv=policy.argv, returncode=124, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(engine, "run_in_jail", fake_run_in_jail)
    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=machine_journal.MachineJournal(tmp_path / "i"),
        tool_policy=run.machine_tool_policy_factory(
            Config(), tmp_path, "strict", protect_paths=(), data_dir=None
        ),
    )
    res = world.run_tool(("sleep", "99"), 1.0)
    assert res.timed_out is True
    assert res.exit_code == 124

    # A normal nonzero exit stays not-timed-out.
    def plain_nonzero(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return kinds.CommandResult(
            argv=policy.argv, returncode=2, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(engine, "run_in_jail", plain_nonzero)
    res2 = world.run_tool(("false",), 1.0)
    assert res2.timed_out is False
    assert res2.exit_code == 2


def test_notify_does_not_change_replay_path(tmp_path: pathlib.Path) -> None:
    # machine.notify is presentation only: replay ignores it and reproduces.
    journal, f = _load(tmp_path, NOTIFIER)
    spec = _semantics.load_machine(f)
    live = machine_engine.drive(spec, journal, FakeWorld({"noop": _ok()}), live=True)
    replayed = machine_engine.drive(spec, journal, None, live=False)
    assert replayed == live


def test_wait_forever_blocks_on_signal_only(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, FOREVER)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {"record": _ok('{"got": true}')},
        wakes=[machine_engine.WaitWake("signal", {"cmd": "go", "n": 2})],
    )
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "done", "stop_ok", 2)
    # A no-timer wait passes None as the deadline (park until a signal).
    assert world.sleep_deadlines == [None]
    # The poke payload was materialized for the next tool and journaled.
    assert world.materialized == [{"cmd": "go", "n": 2}]

    wait_step = next(
        e
        for e in journal.read()
        if isinstance(e, machine_journal.StepEvent) and isinstance(e.fact, machine_journal.WaitFact)
    )
    assert isinstance(wait_step.fact, machine_journal.WaitFact)
    assert wait_step.fact.wake_epoch is None
    assert wait_step.fact.payload == {"cmd": "go", "n": 2}


def test_wait_forever_payload_reproduces_on_replay(tmp_path: pathlib.Path) -> None:
    # The journaled poke payload is a fact: replay rebuilds the identical path.
    journal, f = _load(tmp_path, FOREVER)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {"record": _ok('{"got": true}')}, wakes=[machine_engine.WaitWake("signal", {"cmd": "go"})]
    )
    live = machine_engine.drive(spec, journal, world, live=True)
    replayed = machine_engine.drive(spec, journal, None, live=False)
    assert replayed == live


def test_exit_on_wait_forever_parks_until_signal(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, FOREVER)
    spec = _semantics.load_machine(f)
    # No timer: --exit-on-wait persists a signal-only pending wait (no instant).
    result = machine_engine.drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "waiting"
    assert "signal poke" in result.reason
    pending = journal.read_pending_wait()
    assert pending is not None
    assert pending.wake_epoch is None
    # A poke with a payload fires it on the next scheduler invocation.
    journal.poke({"cmd": "go"})
    result = machine_engine.drive(
        spec,
        journal,
        FakeWorld({"record": _ok('{"got": true}')}),
        live=True,
        exit_on_wait=True,
    )
    assert result == machine_engine.MachineResult("ok", "done", "stop_ok", 2)
    wait_step = next(e for e in journal.read() if isinstance(e, machine_journal.StepEvent))

    assert isinstance(wait_step.fact, machine_journal.WaitFact)
    assert wait_step.fact.payload == {"cmd": "go"}


def test_exit_on_wait_arms_and_yields_waiting(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, clock=1000.0)
    result = machine_engine.drive(spec, journal, world, live=True, exit_on_wait=True)
    assert result.status == "waiting"
    assert result.state == "poll"
    assert result.transitions == 0
    # The wake instant was persisted once; no step was appended yet.
    pending = journal.read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"
    assert pending.wake_epoch == 1060.0
    assert not any(isinstance(e, machine_journal.StepEvent) for e in journal.read())
    # Both surfaces read one owner, so the banner and `machine status` agree on the wake time.
    assert pending.wake_at == "1970-01-01T00:17:40+00:00"
    assert pending.wake_at in result.reason


def test_exit_on_wait_fires_tick_when_due(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = _semantics.load_machine(f)
    # First invocation arms the wait (wake at 1060).
    machine_engine.drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    # A later scheduler tick re-invokes once the instant has passed.
    result = machine_engine.drive(
        spec, journal, FakeWorld({}, clock=1060.0), live=True, exit_on_wait=True
    )
    assert result == machine_engine.MachineResult("ok", "ticked", "done", 1)
    step = next(e for e in journal.read() if isinstance(e, machine_journal.StepEvent))
    assert step.label == "tick"
    # The persisted wait was cleared once it fired.
    assert journal.read_pending_wait() is None


def test_exit_on_wait_fires_signal_before_due(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = _semantics.load_machine(f)
    machine_engine.drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    # Operator poke arrives before the wake instant.
    journal.poke()
    result = machine_engine.drive(
        spec, journal, FakeWorld({}, clock=1005.0), live=True, exit_on_wait=True
    )
    assert result == machine_engine.MachineResult("ok", "signalled", "woken", 1)
    assert journal.read_pending_wait() is None


def test_blocking_wait_clears_persisted_wait(tmp_path: pathlib.Path) -> None:
    # A blocking run that consumes the wake clears the stale wait.json with it.
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = _semantics.load_machine(f)
    machine_engine.drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    assert journal.read_pending_wait() is not None
    result = machine_engine.drive(
        spec, journal, FakeWorld({}, wakes=[machine_engine.WaitWake("tick")]), live=True
    )
    assert result == machine_engine.MachineResult("ok", "ticked", "done", 1)
    assert journal.read_pending_wait() is None


def test_exit_on_wait_wake_epoch_computed_once(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = _semantics.load_machine(f)
    machine_engine.drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    # A second not-ready invocation must NOT re-arm (would be 1030+60 = 1090).
    result = machine_engine.drive(
        spec, journal, FakeWorld({}, clock=1030.0), live=True, exit_on_wait=True
    )
    assert result.status == "waiting"
    pending = journal.read_pending_wait()
    assert pending is not None
    assert pending.wake_epoch == 1060.0


def test_journal_begins_once(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": []}')})
    machine_engine.drive(spec, journal, world, live=True)
    events = journal.read()
    assert sum(isinstance(e, machine_journal.MachineBegin) for e in events) == 1
    assert sum(isinstance(e, machine_journal.MachineEnd) for e in events) == 1


# Agent state.


def _agent(reason: str, payload: dict[str, Any] | None) -> machine_engine.AgentExecResult:
    return machine_engine.AgentExecResult(reason=reason, payload=payload)


def test_agent_ok_captures_payload_and_routes(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    payload = {"approved": True, "note": "lgtm"}
    world = FakeWorld({}, agent_results=[_agent("finish_session", payload)])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "approved", "stop_ok", 2)
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["verdict"] == payload
    # The rendered prompt was passed through to the runner.
    assert world.agent_calls[0].model == "claude-sonnet-4-5"
    assert world.agent_calls[0].prompt == "Review the change."


def test_agent_per_state_knobs_threaded_to_request(tmp_path: pathlib.Path) -> None:
    body = REVIEWER.replace(
        'prompt = "Review the change."',
        'prompt = "Review the change."\n'
        'provider = "anthropic"\n'
        'effort = "high"\n'
        "temperature = 0.3\n"
        "max_usd = 2.5\n"
        "max_tokens_fallback = 90000",
    )
    journal, f = _load(tmp_path, body)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("finish_session", {"approved": True})])
    machine_engine.drive(spec, journal, world, live=True)
    req = world.agent_calls[0]
    assert req.provider == "anthropic"
    assert req.effort == "high"
    assert req.temperature == 0.3
    # min(2.5 state cap, 1.0 machine budget): a state never gets more than the machine has.
    assert req.max_usd == 1.0
    assert req.max_tokens_fallback == 90000


def test_agent_ok_but_rejected_routes_fail(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    payload = {"approved": False, "note": "needs work"}
    world = FakeWorld({}, agent_results=[_agent("finish_session", payload)])
    result = machine_engine.drive(spec, journal, world, live=True)
    # Valid payload (label ok) captured, then branch routes to stop_fail.
    assert result.status == "failed"
    assert result.state == "stop_fail"
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["verdict"] == payload


def test_agent_invalid_payload_routes_failed_no_capture(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    # Missing the required `note` field -> schema validation fails -> "failed".
    world = FakeWorld({}, agent_results=[_agent("finish_session", {"approved": True})])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"
    snap = journal.latest_snapshot()
    assert snap is not None
    # `verdict` keeps its declared default; nothing was captured.
    assert snap.blackboard["verdict"] == {}


def test_agent_finish_session_without_payload_routes_failed(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("finish_session", None)])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"


def test_agent_budget_exhausted_label(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("budget_exhausted", None)])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "halt"
    assert result.reason == "budget"


def test_agent_timeout_label(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("timeout", None)])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "expired"
    assert result.reason == "timeout"


_SPENDER = """
machine = "spender"
version = 1
initial = "work"

[budget]
max_usd = 0.05
max_transitions = 100

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.work]
kind = "agent"
model = "m"
prompt = "do"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "work", failed = "stop_fail", budget_exhausted = "stop_fail", timeout = "stop_fail" }

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "fail"
"""


def test_machine_stops_when_cumulative_max_usd_exceeded(tmp_path: pathlib.Path) -> None:
    # Each step costs $0.10 against a $0.05 budget, so the budget guard must stop the loop.
    journal, f = _load(tmp_path, _SPENDER)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {},
        agent_results=[
            machine_engine.AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.10)
        ],
    )
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "max_usd" in result.reason
    assert len(world.agent_calls) == 1  # one step ran, then the budget guard fired


def test_the_agent_request_cap_is_clamped_to_the_remaining_machine_budget(
    tmp_path: pathlib.Path,
) -> None:
    """A child agent's cap is min(state cap, remaining machine budget)."""
    body = _SPENDER.replace('prompt = "do"', 'prompt = "do"\nmax_usd = 0.10')
    journal, f = _load(tmp_path, body)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {},
        agent_results=[
            machine_engine.AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.04),
            machine_engine.AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.04),
        ],
    )
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "failed" and "max_usd" in result.reason
    caps = [req.max_usd for req in world.agent_calls]
    assert caps[0] == 0.05  # min(0.10 state cap, 0.05 machine budget)
    assert caps[1] == pytest.approx(0.01)  # the machine's remaining cent, not 0.10


def test_agent_spend_threaded_into_fact(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    result = machine_engine.AgentExecResult(
        reason="finish_session",
        payload={"approved": True, "note": "ok"},
        usd=0.25,
        input_tokens=2000,
        output_tokens=300,
    )
    world = FakeWorld({}, agent_results=[result])
    machine_engine.drive(spec, journal, world, live=True)
    step = next(
        e
        for e in journal.read()
        if isinstance(e, machine_journal.StepEvent)
        and isinstance(e.fact, machine_journal.AgentFact)
    )
    assert isinstance(step.fact, machine_journal.AgentFact)
    assert step.fact.usd == 0.25
    assert step.fact.input_tokens == 2000
    assert step.fact.output_tokens == 300


def test_agent_set_capture_extracts_scalar_field(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, SCORER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("finish_session", {"points": 7})])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "done", "stop_ok", 1)
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["total"] == 7


def test_agent_replay_reproduces_path_without_world(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    payload = {"approved": True, "note": "ok"}
    world = FakeWorld({}, agent_results=[_agent("finish_session", payload)])
    live = machine_engine.drive(spec, journal, world, live=True)
    replayed = machine_engine.drive(spec, journal, None, live=False)
    assert replayed == live


def test_agent_recovery_revalidates_the_journaled_payload(tmp_path: pathlib.Path) -> None:
    """The machine's record schema rejects a fabricated `ok` payload before it reaches the board."""
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="reviewer", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="review",
            label="ok",
            goto="route",
            fact=machine_journal.AgentFact(
                outcome="ok",
                reason="finish_session",
                payload={"approved": True},
            ),
        )
    )

    with pytest.raises(machine_engine.EngineError, match="missing required field 'note'"):
        machine_engine.drive(spec, journal, None, live=False)


def test_agent_crash_recovery_does_not_rerun(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    # A crash after the agent ran: recovery rebuilds `verdict` without calling the runner again.
    journal.ensure_dirs()
    journal.begin(machine="reviewer", version=1)
    journal.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="review",
            label="ok",
            goto="route",
            fact=machine_journal.AgentFact(
                outcome="ok",
                reason="finish_session",
                payload={"approved": True, "note": "lgtm"},
            ),
        )
    )
    world = FakeWorld({})  # no programmed agent results — proving it is not re-run
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result == machine_engine.MachineResult("ok", "approved", "stop_ok", 2)
    assert world.agent_calls == []


def test_agent_without_runner_raises(tmp_path: pathlib.Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = _semantics.load_machine(f)
    journal.ensure_dirs()
    world = machine_engine.LiveWorld(cwd=tmp_path, journal=journal, agent_runner=None)
    with pytest.raises(machine_engine.EngineError, match="no agent runner"):
        machine_engine.drive(spec, journal, world, live=True)


def test_per_state_agent_log_path_and_prune(tmp_path: pathlib.Path) -> None:
    """Each agent state gets its own logs.jsonl, pruned to the most recent state_log_keep."""
    journal = machine_journal.MachineJournal(tmp_path / "inst")
    states_root = tmp_path / "inst" / "states"
    captured: list[pathlib.Path | None] = []

    def fake_runner(
        req: machine_engine.AgentRequest, events_log: pathlib.Path | None
    ) -> machine_engine.AgentExecResult:
        captured.append(events_log)
        if events_log is not None:  # the real subprocess creates the log; simulate it
            events_log.parent.mkdir(parents=True, exist_ok=True)
            events_log.write_text("{}\n", encoding="utf-8")
        return machine_engine.AgentExecResult(reason="finish_session", payload=None)

    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=journal,
        agent_runner=fake_runner,
        state_log_root=states_root,
        state_log_keep=3,
    )
    for seq in range(5):
        world.run_agent(
            machine_engine.AgentRequest(prompt="x", timeout_s=1.0, state_name="s", step_seq=seq)
        )

    # Each call got its own seq-named log path.
    assert captured[0] == states_root / "0000-s" / "logs.jsonl"
    assert captured[4] == states_root / "0004-s" / "logs.jsonl"
    # Pruned to keep=3: only the three most recent state dirs survive on disk.
    assert sorted(p.name for p in states_root.iterdir()) == ["0002-s", "0003-s", "0004-s"]


def test_state_log_keep_zero_disables_pruning(tmp_path: pathlib.Path) -> None:
    """`[machine].state_log_keep = 0` keeps every per-state log dir, like `snapshot_keep`."""
    journal = machine_journal.MachineJournal(tmp_path / "inst")
    states_root = tmp_path / "inst" / "states"

    def fake_runner(
        req: machine_engine.AgentRequest, events_log: pathlib.Path | None
    ) -> machine_engine.AgentExecResult:
        if events_log is not None:
            events_log.parent.mkdir(parents=True, exist_ok=True)
            events_log.write_text("{}\n", encoding="utf-8")
        return machine_engine.AgentExecResult(reason="finish_session", payload=None)

    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=journal,
        agent_runner=fake_runner,
        state_log_root=states_root,
        state_log_keep=0,
    )
    for seq in range(5):
        world.run_agent(
            machine_engine.AgentRequest(prompt="x", timeout_s=1.0, state_name="s", step_seq=seq)
        )
    assert len(list(states_root.iterdir())) == 5


def test_per_state_log_disabled_without_root(tmp_path: pathlib.Path) -> None:
    """Without a state_log_root there is no per-state log."""
    seen: list[pathlib.Path | None] = []

    def fake_runner(
        req: machine_engine.AgentRequest, events_log: pathlib.Path | None
    ) -> machine_engine.AgentExecResult:
        seen.append(events_log)
        return machine_engine.AgentExecResult(reason="finish_session", payload=None)

    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=machine_journal.MachineJournal(tmp_path / "i"),
        agent_runner=fake_runner,
    )
    world.run_agent(
        machine_engine.AgentRequest(prompt="x", timeout_s=1.0, state_name="s", step_seq=0)
    )
    assert seen == [None]


def test_best_effort_usd_limit_no_longer_validates(tmp_path: pathlib.Path) -> None:
    # The old soft field must fail the grammar loudly, never load as an ignored knob.
    from agent6.machine import MachineError

    body = _SPENDER.replace("max_usd = 0.05", "best_effort_usd_limit = 0.05")
    f = tmp_path / "m.asm.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(MachineError, match="best_effort_usd_limit"):
        _semantics.load_machine(f)


def test_agent_state_max_tokens_fallback_flows_to_request(tmp_path: pathlib.Path) -> None:
    body = _SPENDER.replace('kind = "agent"', 'kind = "agent"\nmax_tokens_fallback = 5000', 1)
    journal, f = _load(tmp_path, body)
    spec = _semantics.load_machine(f)
    world = FakeWorld(
        {},
        agent_results=[
            machine_engine.AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.10)
        ],
    )
    machine_engine.drive(spec, journal, world, live=True)
    assert world.agent_calls[0].max_tokens_fallback == 5000


NOTIFY_WAIT = """
machine = "notifywait"
version = 1
initial = "park"

[budget]
max_usd = 1.0
max_transitions = 100

[states.park]
kind = "wait"
notify = { message = "machine parked, poke me", level = "info" }
on = { signal = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "finished"
"""


def test_parked_wait_notify_fires_once_across_scheduler_ticks(tmp_path: pathlib.Path) -> None:
    # Re-driving a parked machine must not re-fire the wait's notify: it belongs to state entry.

    journal, f = _load(tmp_path, NOTIFY_WAIT)
    spec = _semantics.load_machine(f)
    hook_notifies = 0
    for _ in range(3):
        world = FakeWorld({})
        result = machine_engine.drive(spec, journal, world, live=True, exit_on_wait=True)
        assert result.status == "waiting"
        hook_notifies += sum(1 for n in world.notifications if n[0] == "notify")
    assert hook_notifies == 1
    assert sum(1 for e in journal.read() if isinstance(e, machine_journal.MachineNotify)) == 1
    # The firing tick (poke consumed) adds no duplicate either.
    journal.poke(None)
    result = machine_engine.drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "ok"
    assert sum(1 for e in journal.read() if isinstance(e, machine_journal.MachineNotify)) == 1


def test_poke_atomic_write_leaves_no_temp_and_keeps_payload(tmp_path: pathlib.Path) -> None:
    # poke() publishes atomically; a plain write let take_signal consume a partial file.
    journal, _ = _load(tmp_path, FOREVER)
    journal.poke({"cmd": "deploy", "target": "prod"})
    assert not any(p.name.endswith(".tmp") for p in journal.root.iterdir())
    present, payload = journal.take_signal()
    assert present is True
    assert payload == {"cmd": "deploy", "target": "prod"}


def test_machine_is_parked_reflects_pending_wait(tmp_path: pathlib.Path) -> None:
    from agent6.viewmodel import fold_machine, probe_instance

    journal, f = _load(tmp_path, FOREVER)
    spec = _semantics.load_machine(f)
    assert probe_instance(journal.root, fold_machine(spec, journal.read())).parked is False
    result = machine_engine.drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "waiting"
    # The engine's own record (state and seq) reads as the armed wait.
    assert probe_instance(journal.root, fold_machine(spec, journal.read())).parked is True


def test_live_world_run_tool_uses_the_shared_jail_tool_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tool-state jail resolves operator tools exactly as run_command's jail does.
    from agent6.machine import engine as engine_mod

    captured: dict[str, kinds.JailPolicy] = {}

    def fake_run_in_jail(policy: kinds.JailPolicy) -> kinds.CommandResult:
        captured["policy"] = policy
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="{}", stderr="", duration_s=0.0
        )

    def fake_tool_paths() -> tuple[str, tuple[pathlib.Path, ...]]:
        return "/usr/bin:/bin:/fake/bin", (pathlib.Path("/fake/real-tools"),)

    monkeypatch.setattr(engine_mod, "run_in_jail", fake_run_in_jail)
    monkeypatch.setattr("agent6.tools.policy.operator_tool_paths", fake_tool_paths)
    monkeypatch.setenv("PATH", "/host/only/path")  # must NOT leak into the jail
    world = machine_engine.LiveWorld(
        cwd=tmp_path,
        journal=machine_journal.MachineJournal(tmp_path),
        tool_policy=run.machine_tool_policy_factory(
            Config(), tmp_path, "strict", protect_paths=(), data_dir=None
        ),
    )
    world.run_tool(("sometool", "arg"), timeout_s=5.0)

    policy = captured["policy"]
    env = dict(policy.env)
    assert env["PATH"] == "/usr/bin:/bin:/fake/bin"  # the jail-correct PATH
    assert policy.tool_paths == (pathlib.Path("/fake/real-tools"),)  # with its mounts
    assert "/host/only/path" not in env.values()
    assert env["UV_NO_SYNC"] == "1"  # same offline-jail rule as run_command


def test_a_captured_lone_surrogate_never_reaches_the_blackboard(tmp_path: pathlib.Path) -> None:
    """A lone surrogate is scrubbed on the blackboard, so the next request payload serializes."""
    import pydantic

    from agent6.machine import spec as machine_spec

    class _Payload(pydantic.BaseModel):  # the agent request's shape, minimally
        task: str

    _journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    state = spec.states["scan"]
    assert isinstance(state, machine_spec.ToolState)
    blackboard: dict[str, Any] = {}
    # The journal sanitizes on write, so only the IN-MEMORY value matters here.
    machine_engine._apply_capture(spec, state, '{"items": ["emoji tail \\ud83d"]}', blackboard)

    captured = blackboard["items"][0]
    assert "emoji tail" in captured  # kept, not dropped
    _Payload(task=captured).model_dump_json()  # the sink that raised


def test_replay_rejects_a_diverged_state_seq_or_fact_kind(tmp_path: pathlib.Path) -> None:
    """The fold verifies each step's state, seq and fact kind against the replayed position."""
    base = [
        dict(ts="t", seq=0, state="scan", label="ok", goto="check"),
    ]
    counter = iter(range(10))

    def journal_with(**overrides: object) -> tuple[Any, Any]:
        case_dir = tmp_path / f"case{next(counter)}"
        case_dir.mkdir()
        journal, f = _load(case_dir, COUNTER)
        journal.ensure_dirs()
        journal.begin(machine="counter", version=1)
        fields: dict[str, Any] = {
            **base[0],
            "fact": machine_journal.ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        }
        fields.update(overrides)
        journal.append(machine_journal.StepEvent(**fields))
        return journal, _semantics.load_machine(f)

    journal, spec = journal_with(state="record")  # not the replayed position
    with pytest.raises(machine_engine.EngineError, match="diverges"):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)

    journal, spec = journal_with(seq=41)  # noncontiguous
    with pytest.raises(machine_engine.EngineError, match="not contiguous"):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)

    journal, spec = journal_with(
        fact=machine_journal.BranchFact(clause_index=0)
    )  # wrong kind for a tool state
    with pytest.raises(machine_engine.EngineError, match="cannot produce"):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)


def test_stop_request_parks_at_the_transition_boundary(tmp_path: pathlib.Path) -> None:
    """A stop marker written mid-state parks the machine at the next boundary.

    The finished state's fact is journaled, no MachineEnd is written, the marker is consumed, and a
    later drive continues.
    """
    journal, f = _load(tmp_path, COUNTER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": ["a"]}'), "record": _ok()})
    real_run_tool = world.run_tool

    def stop_during_first_tool(
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        network: kinds.NetworkMode = "none",
        pass_env: Any = (),
    ) -> machine_engine.ToolExecResult:
        machine_journal.write_stop_request(journal.root)
        return real_run_tool(argv, timeout_s, network=network)

    world.run_tool = stop_during_first_tool  # type: ignore[method-assign]
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "stopped"
    assert not machine_journal.stop_requested(journal.root)  # consumed at the park
    events = journal.read()
    assert not isinstance(events[-1], machine_journal.MachineEnd)  # resumable, no end journaled
    done = machine_engine.drive(
        spec, journal, FakeWorld({"scan": _ok('{"items": ["a"]}'), "record": _ok()}), live=True
    )
    assert done.status == "ok"


def test_stop_interrupts_a_foreground_wait_and_keeps_it_armed(tmp_path: pathlib.Path) -> None:
    """A stop that lands mid-sleep parks without consuming the wait; a later run resumes it."""
    journal, f = _load(tmp_path, WAITER)
    spec = _semantics.load_machine(f)
    world = FakeWorld({}, wakes=[machine_engine.WaitWake("stop")])
    result = machine_engine.drive(spec, journal, world, live=True)
    assert result.status == "stopped"
    pending = journal.read_pending_wait()
    assert pending is not None and pending.state == "poll"


def test_live_world_sleep_wakes_on_a_stop_request(tmp_path: pathlib.Path) -> None:
    """The stop marker interrupts a real sleep."""
    import time as _time

    journal = machine_journal.MachineJournal(tmp_path / "inst")
    journal.ensure_dirs()
    world = machine_engine.LiveWorld(cwd=tmp_path, journal=journal, poll_interval_s=0.01)
    machine_journal.write_stop_request(journal.root)
    assert world.sleep_until(_time.time() + 3600).woke_by == "stop"


# Two waits in a row: the second must not inherit the first's wake instant.
WAITER_CHAIN = """
machine = "waiter_chain"
version = 1
initial = "first"

[budget]
max_usd = 1.0
max_transitions = 100

[states.first]
kind = "wait"
every_secs = "60"
on = { tick = "second", signal = "second" }

[states.second]
kind = "wait"
every_secs = "3600"
on = { tick = "done", signal = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"
"""


def test_a_wait_never_inherits_the_previous_waits_record(tmp_path: pathlib.Path) -> None:
    """The driver clears the wait record only after the StepEvent is durable.

    Arming keys on the state name: `second` computes its own instant instead of firing on `first`'s.
    """
    journal, f = _load(tmp_path, WAITER_CHAIN)
    spec = _semantics.load_machine(f)
    appended = journal.append

    def _die_after_the_step(event: object) -> None:
        appended(event)  # pyright: ignore[reportArgumentType]
        if isinstance(event, machine_journal.StepEvent):
            raise KeyboardInterrupt("died between the wake's StepEvent and the clear")

    journal.append = _die_after_the_step  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        machine_engine.drive(spec, journal, FakeWorld({}), live=True)
    journal.append = appended  # type: ignore[method-assign]
    assert journal.read_pending_wait() == machine_journal.PendingWait(
        state="first", wake_epoch=1060.0
    )

    world = FakeWorld({})
    assert machine_engine.drive(spec, journal, world, live=True).status == "ok"
    assert world.sleep_deadlines == [4600.0]


LOOP_WAIT = """
machine = "loop_wait"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
hop = { type = "json", default = 0 }

[states.poll]
kind = "wait"
every_secs = "60"
on = { tick = "route", signal = "route" }

[states.route]
kind = "branch"
when = [
  { if = "hop == 0", goto = "bump" },
  { else = true, goto = "done" },
]

[states.bump]
kind = "tool"
command = ["bump"]
capture = { stdout_json = "hop" }
timeout_secs = 5
on = { ok = "poll", nonzero = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "looped"
"""


def test_a_revisited_wait_state_arms_its_own_fresh_instant(tmp_path: pathlib.Path) -> None:
    """A state reached twice arms its own instant on the second visit, never a stale record."""
    journal, f = _load(tmp_path, LOOP_WAIT)
    spec = _semantics.load_machine(f)
    appended = journal.append

    def _die_after_the_step(event: object) -> None:
        appended(event)  # pyright: ignore[reportArgumentType]
        if isinstance(event, machine_journal.StepEvent) and event.seq == 0:
            raise KeyboardInterrupt("died between the wake's StepEvent and the clear")

    journal.append = _die_after_the_step  # type: ignore[method-assign]
    first = FakeWorld({"bump": _ok("1")})
    first.clock = 1000.0
    with pytest.raises(KeyboardInterrupt):
        machine_engine.drive(spec, journal, first, live=True)
    journal.append = appended  # type: ignore[method-assign]
    assert journal.read_pending_wait() == machine_journal.PendingWait(
        state="poll", wake_epoch=1060.0
    )

    second = FakeWorld({"bump": _ok("1")})
    second.clock = 5000.0
    assert machine_engine.drive(spec, journal, second, live=True).status == "ok"
    # The second `poll` arms 5060.0 off the new clock, never the first visit's 1060.0.
    assert second.sleep_deadlines == [5060.0]


def test_a_corrupt_wait_record_refuses_before_the_notify_re_fires(tmp_path: pathlib.Path) -> None:
    """A corrupt wait record refuses the parked entry instead of paging on every tick."""
    journal, f = _load(tmp_path, NOTIFY_WAIT)
    spec = _semantics.load_machine(f)
    assert (
        machine_engine.drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True).status
        == "waiting"
    )
    journal.wait_path.write_text("{not json", encoding="utf-8")
    for _ in range(2):
        world = FakeWorld({})
        with pytest.raises(machine_journal.JournalError):
            machine_engine.drive(spec, journal, world, live=True, exit_on_wait=True)
        assert world.notifications == []
    assert sum(1 for e in journal.read() if isinstance(e, machine_journal.MachineNotify)) == 1
