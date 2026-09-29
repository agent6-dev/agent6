# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the append-only journal, snapshots, and lock (agent6.machine.journal)."""

from __future__ import annotations

import pathlib

import pytest

from agent6.machine import journal as machine_journal

_DATA = pathlib.Path(__file__).parent / "data"


def _journal(tmp_path: pathlib.Path) -> machine_journal.MachineJournal:
    j = machine_journal.MachineJournal(tmp_path / "m")
    j.ensure_dirs()
    return j


def _golden_events() -> list[object]:
    """One event of every journal family (all four Facts) with fixed timestamps."""
    return [
        machine_journal.MachineBegin(
            ts="2026-07-16T00:00:00.000000+00:00", machine="demo", version=1
        ),
        machine_journal.StepEvent(
            ts="2026-07-16T00:00:01.000000+00:00",
            seq=0,
            state="scan",
            label="ok",
            goto="branch",
            fact=machine_journal.ToolFact(
                exit_code=0, stdout='{"note": "ok"}', timed_out=False, stderr="warn: slow\n"
            ),
        ),
        machine_journal.StepEvent(
            ts="2026-07-16T00:00:02.000000+00:00",
            seq=1,
            state="branch",
            label="else",
            goto="poll",
            fact=machine_journal.BranchFact(clause_index=2),
        ),
        machine_journal.StepEvent(
            ts="2026-07-16T00:00:03.000000+00:00",
            seq=2,
            state="poll",
            label="signal",
            goto="review",
            fact=machine_journal.WaitFact(
                wake_epoch=None, woke_by="signal", payload={"from": "operator"}
            ),
        ),
        machine_journal.StepEvent(
            ts="2026-07-16T00:00:04.000000+00:00",
            seq=3,
            state="review",
            label="ok",
            goto="stop_ok",
            fact=machine_journal.AgentFact(
                outcome="ok",
                reason="finish_session",
                payload={"approved": True},
                usd=0.25,
                input_tokens=1000,
                output_tokens=200,
            ),
        ),
        machine_journal.MachineNotify(
            ts="2026-07-16T00:00:05.000000+00:00",
            state="review",
            message="all checks passed",
            level="info",
        ),
        machine_journal.MachineEnd(
            ts="2026-07-16T00:00:06.000000+00:00",
            status="ok",
            reason="approved",
            state="stop_ok",
            transitions=4,
        ),
    ]


def test_journal_line_format_matches_golden(tmp_path: pathlib.Path) -> None:
    # Byte pin of the line format: a drift silently breaks replay of every older journal.
    j = _journal(tmp_path)
    for event in _golden_events():
        j.append(event)  # type: ignore[arg-type]
    written = j.journal_path.read_text(encoding="utf-8")
    assert written == (_DATA / "golden_journal.jsonl").read_text(encoding="utf-8")


def test_replay_of_golden_journal_bytes_reproduces_state(tmp_path: pathlib.Path) -> None:
    # Those exact bytes replay back to the same typed events.
    j = _journal(tmp_path)
    j.journal_path.write_bytes((_DATA / "golden_journal.jsonl").read_bytes())
    events = j.read()
    assert [type(e) for e in events] == [
        machine_journal.MachineBegin,
        machine_journal.StepEvent,
        machine_journal.StepEvent,
        machine_journal.StepEvent,
        machine_journal.StepEvent,
        machine_journal.MachineNotify,
        machine_journal.MachineEnd,
    ]
    assert [type(e.fact) for e in events if isinstance(e, machine_journal.StepEvent)] == [
        machine_journal.ToolFact,
        machine_journal.BranchFact,
        machine_journal.WaitFact,
        machine_journal.AgentFact,
    ]
    tool = events[1]
    assert isinstance(tool, machine_journal.StepEvent) and isinstance(
        tool.fact, machine_journal.ToolFact
    )
    assert tool.fact.stderr == "warn: slow\n"  # stderr round-trips through the wire
    wait = events[3]
    assert isinstance(wait, machine_journal.StepEvent) and isinstance(
        wait.fact, machine_journal.WaitFact
    )
    assert wait.fact.wake_epoch is None and wait.fact.payload == {"from": "operator"}
    agent = events[4]
    assert isinstance(agent, machine_journal.StepEvent) and isinstance(
        agent.fact, machine_journal.AgentFact
    )
    assert agent.fact.usd == 0.25 and agent.fact.output_tokens == 200
    end = events[6]
    assert isinstance(end, machine_journal.MachineEnd) and end.transitions == 4


def test_old_tool_fact_without_stderr_still_parses(tmp_path: pathlib.Path) -> None:
    # A line without `stderr` still replays with ""; extra="forbid" rejects unknown keys only.
    j = _journal(tmp_path)
    old_line = (
        '{"type":"step","ts":"t","seq":0,"state":"scan","label":"ok","goto":"done",'
        '"fact":{"kind":"tool","exit_code":1,"stdout":"","timed_out":false}}\n'
    )
    j.journal_path.write_text(old_line, encoding="utf-8")
    events = j.read()
    assert len(events) == 1
    step = events[0]
    assert isinstance(step, machine_journal.StepEvent) and isinstance(
        step.fact, machine_journal.ToolFact
    )
    assert step.fact.stderr == ""


def test_read_survives_unicode_line_separators(tmp_path: pathlib.Path) -> None:
    # U+2028/U+2029/U+0085 inside JSON strings are not line breaks; splitlines would shred the line.
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    poison = "a\u2028b\u2029c\u0085d"  # line/para/next-line separators
    j.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="done",
            fact=machine_journal.ToolFact(
                exit_code=0, stdout=f'{{"note": "{poison}"}}', timed_out=False
            ),
        )
    )
    events = j.read()
    assert len(events) == 2  # begin + one step, not fragmented
    assert isinstance(events[1], machine_journal.StepEvent)
    assert isinstance(events[1].fact, machine_journal.ToolFact)
    assert poison in events[1].fact.stdout


def test_read_tolerates_and_append_heals_torn_final_line(tmp_path: pathlib.Path) -> None:
    # A final line with no newline is dropped by `read` and healed by the next `append`.
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    with j.journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "machine.end", "ts": "t", "status": "ok"')  # torn, no newline
    events = j.read()
    assert len(events) == 1  # just the begin; the torn tail is ignored
    assert isinstance(events[0], machine_journal.MachineBegin)
    j.append(
        machine_journal.MachineEnd(ts="t", status="failed", reason="r", state="s", transitions=1)
    )
    healed = j.read()
    assert len(healed) == 2
    assert isinstance(healed[-1], machine_journal.MachineEnd)


def test_read_tolerates_torn_final_utf8_sequence(tmp_path: pathlib.Path) -> None:
    # A split multibyte character on the final line is dropped before decoding, then healed.
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    torn = b'{"type":"machine.end","ts":"t","status":"failed","reason":"caf' + "é".encode()[:1]
    with j.journal_path.open("ab") as fh:
        fh.write(torn)
    events = j.read()
    assert len(events) == 1
    assert isinstance(events[0], machine_journal.MachineBegin)
    j.append(
        machine_journal.MachineEnd(ts="t", status="failed", reason="r", state="s", transitions=1)
    )
    healed = j.read()
    assert len(healed) == 2
    assert isinstance(healed[-1], machine_journal.MachineEnd)


def test_latest_snapshot_falls_back_past_corrupt_newest(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.write_snapshot(machine_journal.Snapshot(seq=1, state="a", blackboard={"n": 1}))
    j.write_snapshot(machine_journal.Snapshot(seq=2, state="b", blackboard={"n": 2}))
    # Invalid UTF-8 is corruption too; the older retained snapshot still restores the readout.
    (j.snapshots_dir / "2.json").write_bytes(b"\xff\xfe")
    snap = j.latest_snapshot()
    assert snap is not None
    assert snap.seq == 1
    # All corrupt -> None (the journal is authoritative), never an exception.
    (j.snapshots_dir / "1.json").write_text("nope", encoding="utf-8")
    assert j.latest_snapshot() is None


def test_append_and_read_roundtrip(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    j.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=machine_journal.ToolFact(exit_code=0, stdout='{"x": 1}', timed_out=False),
        )
    )
    j.append(
        machine_journal.StepEvent(
            ts="t",
            seq=1,
            state="check",
            label="else",
            goto="poll",
            fact=machine_journal.BranchFact(clause_index=1),
        )
    )
    j.append(
        machine_journal.StepEvent(
            ts="t",
            seq=2,
            state="poll",
            label="tick",
            goto="done",
            fact=machine_journal.WaitFact(wake_epoch=12.0, woke_by="tick"),
        )
    )
    j.append(
        machine_journal.MachineEnd(ts="t", status="ok", reason="done", state="done", transitions=3)
    )

    events = j.read()
    assert isinstance(events[0], machine_journal.MachineBegin)
    assert isinstance(events[1], machine_journal.StepEvent)
    assert isinstance(events[1].fact, machine_journal.ToolFact)
    assert isinstance(events[2].fact, machine_journal.BranchFact)
    assert isinstance(events[3].fact, machine_journal.WaitFact)
    assert isinstance(events[4], machine_journal.MachineEnd)
    assert events[4].transitions == 3


def test_read_missing_journal_is_empty(tmp_path: pathlib.Path) -> None:
    assert machine_journal.MachineJournal(tmp_path / "nope").read() == []


def test_agent_fact_roundtrip(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="reviewer", version=1)
    j.append(
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
    j.append(
        machine_journal.MachineEnd(
            ts="t", status="ok", reason="approved", state="stop_ok", transitions=2
        )
    )
    events = j.read()
    step = events[1]
    assert isinstance(step, machine_journal.StepEvent)
    assert isinstance(step.fact, machine_journal.AgentFact)
    assert step.fact.outcome == "ok"
    assert step.fact.reason == "finish_session"
    assert step.fact.payload == {"approved": True, "note": "lgtm"}


def test_agent_fact_spend_roundtrip(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="reviewer", version=1)
    j.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="review",
            label="ok",
            goto="route",
            fact=machine_journal.AgentFact(
                outcome="ok",
                reason="finish_session",
                payload=None,
                usd=0.1234,
                input_tokens=1500,
                output_tokens=420,
            ),
        )
    )
    events = j.read()
    step = events[1]
    assert isinstance(step, machine_journal.StepEvent)
    assert isinstance(step.fact, machine_journal.AgentFact)
    assert step.fact.usd == 0.1234
    assert step.fact.input_tokens == 1500
    assert step.fact.output_tokens == 420


def test_agent_fact_spend_defaults_to_zero(tmp_path: pathlib.Path) -> None:
    fact = machine_journal.AgentFact(outcome="ok", reason="finish_session", payload=None)
    assert fact.usd == 0.0
    assert fact.input_tokens == 0
    assert fact.output_tokens == 0


def test_corrupt_journal_line_raises(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    j.journal_path.write_text('{"type": "step", "bogus": true}\n', encoding="utf-8")
    with pytest.raises(machine_journal.JournalError):
        j.read()


def test_journal_error_is_a_machine_error() -> None:
    # Surfaces that degrade on a broken machine file degrade the same way on a broken journal.
    from agent6.machine import spec

    exc = machine_journal.JournalError("corrupt journal line 3")
    assert isinstance(exc, spec.MachineError)
    assert exc.problems == ["corrupt journal line 3"]
    assert str(exc) == "corrupt journal line 3"


def test_snapshot_write_and_latest(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.write_snapshot(machine_journal.Snapshot(seq=0, state="a", blackboard={"n": 1}))
    j.write_snapshot(machine_journal.Snapshot(seq=1, state="b", blackboard={"n": 2}))
    latest = j.latest_snapshot()
    assert latest is not None
    assert latest.seq == 1
    assert latest.state == "b"
    assert latest.blackboard == {"n": 2}


def test_latest_snapshot_none_when_empty(tmp_path: pathlib.Path) -> None:
    assert _journal(tmp_path).latest_snapshot() is None


def test_snapshot_pruning_keeps_configured_tail(tmp_path: pathlib.Path) -> None:
    # Only latest_snapshot is read; old snapshots are pruned to [machine] snapshot_keep.
    j = _journal(tmp_path)
    for seq in range(20):
        j.write_snapshot(machine_journal.Snapshot(seq=seq, state="s", blackboard={"n": seq}))
    kept = sorted(int(p.stem) for p in j.snapshots_dir.glob("*.json"))
    assert kept == [15, 16, 17, 18, 19]
    latest = j.latest_snapshot()
    assert latest is not None and latest.seq == 19


def test_snapshot_keep_zero_disables_pruning(tmp_path: pathlib.Path) -> None:
    j = machine_journal.MachineJournal(tmp_path / "inst", snapshot_keep=0)
    j.ensure_dirs()
    for seq in range(10):
        j.write_snapshot(machine_journal.Snapshot(seq=seq, state="s", blackboard={}))
    assert len(list(j.snapshots_dir.glob("*.json"))) == 10


def test_take_signal_consumes_file(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    assert j.take_signal() == (False, None)
    j.signal_path.write_text("", encoding="utf-8")  # a hand-touched empty poke
    assert j.take_signal() == (True, None)
    j.ack_signal()
    assert j.take_signal() == (False, None)


def test_poke_writes_signal_consumed_by_take_signal(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    assert j.take_signal() == (False, None)
    j.poke()
    assert j.take_signal() == (True, None)
    j.ack_signal()
    assert j.take_signal() == (False, None)


def test_a_poke_survives_until_its_step_is_acked(tmp_path: pathlib.Path) -> None:
    """The poke's claim outlives take_signal and is re-delivered until the step is acked."""
    j = _journal(tmp_path)
    j.poke({"cmd": "reload"})
    assert j.take_signal() == (True, {"cmd": "reload"})
    # Death before the step append: a fresh journal (restart) re-delivers.
    j2 = _journal(tmp_path)
    assert j2.take_signal() == (True, {"cmd": "reload"})
    j2.ack_signal()
    assert j2.take_signal() == (False, None)


def test_take_signal_preserves_poke_landing_mid_consume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A poke racing take_signal lands at signal_path while the renamed-away copy is consumed.
    j = _journal(tmp_path)
    j.poke("first")
    real_read_text = pathlib.Path.read_text

    def racing_read_text(self: pathlib.Path, *args: object, **kwargs: object) -> str:
        if self.name.startswith("signal"):
            j.poke("second")  # a poke lands mid-consume
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pathlib.Path, "read_text", racing_read_text)
    assert j.take_signal() == (True, "first")
    monkeypatch.undo()
    j.ack_signal()
    assert j.take_signal() == (True, "second")  # the mid-consume poke survived


def test_take_signal_recovers_stranded_consuming_file(tmp_path: pathlib.Path) -> None:
    """An unacked claim is re-delivered before any fresh signal, its payload intact."""
    j = _journal(tmp_path)
    j.signal_path.with_suffix(".consuming").write_text('"stranded"', encoding="utf-8")
    j.poke("fresh")
    assert j.take_signal() == (True, "stranded")
    j.ack_signal()
    assert j.take_signal() == (True, "fresh")
    j.ack_signal()
    assert j.take_signal() == (False, None)


def test_poke_carries_payload(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    j.poke({"cmd": "reload", "n": 3})
    assert j.take_signal() == (True, {"cmd": "reload", "n": 3})
    j.ack_signal()
    # A --message-style string payload round-trips too.
    j.poke("hello")
    assert j.take_signal() == (True, "hello")


def test_pending_wait_roundtrip_and_clear(tmp_path: pathlib.Path) -> None:
    j = _journal(tmp_path)
    assert j.read_pending_wait() is None
    j.write_pending_wait(machine_journal.PendingWait(state="poll", wake_epoch=1234.5))
    pending = j.read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"
    assert pending.wake_epoch == 1234.5
    j.clear_pending_wait()
    assert j.read_pending_wait() is None
    # clearing again is a no-op (missing_ok)
    j.clear_pending_wait()


def test_machine_lock_refuses_second_holder(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "m"
    with machine_journal.machine_lock(root):  # noqa: SIM117 - inner lock must be acquired while outer is held
        with pytest.raises(machine_journal.JournalError):
            with machine_journal.machine_lock(root):
                pass


def test_source_roundtrip(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "m"
    machine_journal.write_source(root, "machine = 'x'\n")
    assert machine_journal.read_source(root) == "machine = 'x'\n"


def test_read_source_missing_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(machine_journal.JournalError):
        machine_journal.read_source(tmp_path / "absent")


def test_read_source_reports_invalid_utf8_as_a_journal_error(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "m"
    root.mkdir()
    (root / "machine.asm.toml").write_bytes(b"\xff\xfe")

    with pytest.raises(machine_journal.JournalError, match="cannot read persisted machine source"):
        machine_journal.read_source(root)


def test_append_and_snapshot_survive_a_lone_surrogate(tmp_path: pathlib.Path) -> None:
    r"""A lone surrogate from a \udXXX escape is encoded lossily, never a snapshot crash.

    A tool's captured stdout and a `machine poke --data` payload both feed json.loads into the
    journal.
    """
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    poison = "emoji tail \ud83d"  # a split surrogate pair, as json.loads yields it
    j.append(
        machine_journal.StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="done",
            fact=machine_journal.ToolFact(exit_code=0, stdout=poison, timed_out=False),
        )
    )
    events = j.read()
    assert len(events) == 2  # written and re-readable, not a crash
    assert isinstance(events[1], machine_journal.StepEvent)
    assert isinstance(events[1].fact, machine_journal.ToolFact)
    assert "emoji tail" in events[1].fact.stdout
    # The snapshot writer takes the same value on every subsequent step.
    j.write_snapshot(machine_journal.Snapshot(seq=0, state="scan", blackboard={"note": poison}))
    assert j.latest_snapshot() is not None


def test_healing_a_torn_tail_never_empties_the_journal(tmp_path: pathlib.Path) -> None:
    """The torn-tail heal truncates in place, so a reader never sees an empty journal."""
    import json
    import os

    inst = tmp_path / "inst"
    inst.mkdir()
    journal = inst / "journal.jsonl"
    committed = (
        json.dumps({"type": "machine.begin", "ts": "t", "machine": "m", "version": 1}) + "\n"
    ) * 3
    journal.write_text(committed + '{"type": "step", "ts": "t"', encoding="utf-8")
    opened = os.open(journal, os.O_RDONLY)  # a reader holding the file open
    try:
        machine_journal.MachineJournal(inst)._heal_torn_tail()  # pyright: ignore[reportPrivateUsage]
    finally:
        os.close(opened)

    assert journal.read_text(encoding="utf-8") == committed
    assert len(machine_journal.MachineJournal(inst).read()) == 3
    # Structural: a whole-file rewrite let a kill or a lockless reader see an empty journal.
    import inspect

    body = inspect.getsource(machine_journal.MachineJournal._heal_torn_tail)  # pyright: ignore[reportPrivateUsage]
    assert "write_bytes" not in body
    assert "os.truncate" in body


def test_end_event_reads_a_record_longer_than_its_tail_window(tmp_path: pathlib.Path) -> None:
    """The tail window skips a partial first line and falls back to a full read when none fits."""
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    long_reason = "x" * (machine_journal._TAIL_WINDOW + 1024)
    j.append(
        machine_journal.MachineEnd(
            ts="t", status="ok", reason=long_reason, state="s", transitions=1
        )
    )
    end = j.end_event()
    assert end is not None and end.reason == long_reason

    j.journal_path.write_bytes(j.journal_path.read_bytes() + b'{"type": "step_event", "st')
    torn = j.end_event()
    assert torn is not None and torn.reason == long_reason
