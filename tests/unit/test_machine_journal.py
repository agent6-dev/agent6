# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the append-only journal, snapshots, and lock (agent6.machine.journal)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.machine.journal import (
    AgentFact,
    BranchFact,
    JournalError,
    MachineBegin,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    PendingWait,
    Snapshot,
    StepEvent,
    ToolFact,
    WaitFact,
    machine_lock,
    read_source,
    write_source,
)

_DATA = Path(__file__).parent / "data"


def _journal(tmp_path: Path) -> MachineJournal:
    j = MachineJournal(tmp_path / "m")
    j.ensure_dirs()
    return j


def _golden_events() -> list[object]:
    """One event of every journal family (all four Facts), fixed timestamps: the
    sequence the golden line format is pinned against."""
    return [
        MachineBegin(ts="2026-07-16T00:00:00.000000+00:00", machine="demo", version=1),
        StepEvent(
            ts="2026-07-16T00:00:01.000000+00:00",
            seq=0,
            state="scan",
            label="ok",
            goto="branch",
            fact=ToolFact(
                exit_code=0, stdout='{"note": "ok"}', timed_out=False, stderr="warn: slow\n"
            ),
        ),
        StepEvent(
            ts="2026-07-16T00:00:02.000000+00:00",
            seq=1,
            state="branch",
            label="else",
            goto="poll",
            fact=BranchFact(clause_index=2),
        ),
        StepEvent(
            ts="2026-07-16T00:00:03.000000+00:00",
            seq=2,
            state="poll",
            label="signal",
            goto="review",
            fact=WaitFact(wake_epoch=None, woke_by="signal", payload={"from": "operator"}),
        ),
        StepEvent(
            ts="2026-07-16T00:00:04.000000+00:00",
            seq=3,
            state="review",
            label="ok",
            goto="stop_ok",
            fact=AgentFact(
                outcome="ok",
                reason="finish_run",
                payload={"approved": True},
                usd=0.25,
                input_tokens=1000,
                output_tokens=200,
            ),
        ),
        MachineNotify(
            ts="2026-07-16T00:00:05.000000+00:00",
            state="review",
            message="all checks passed",
            level="info",
        ),
        MachineEnd(
            ts="2026-07-16T00:00:06.000000+00:00",
            status="ok",
            reason="approved",
            state="stop_ok",
            transitions=4,
        ),
    ]


def test_journal_line_format_matches_golden(tmp_path: Path) -> None:
    # Byte pin of the replay-critical line format: appending the fixed event
    # sequence must reproduce the golden journal exactly (one line per event,
    # compact JSON, discriminator keys first). A drift here silently breaks
    # replay of every journal an older instance wrote.
    j = _journal(tmp_path)
    for event in _golden_events():
        j.append(event)  # type: ignore[arg-type]
    written = j.journal_path.read_text(encoding="utf-8")
    assert written == (_DATA / "golden_journal.jsonl").read_text(encoding="utf-8")


def test_replay_of_golden_journal_bytes_reproduces_state(tmp_path: Path) -> None:
    # The other half of the contract: those exact bytes replay back to the same
    # typed events and values, so an on-disk journal keeps reducing identically.
    j = _journal(tmp_path)
    j.journal_path.write_bytes((_DATA / "golden_journal.jsonl").read_bytes())
    events = j.read()
    assert [type(e) for e in events] == [
        MachineBegin,
        StepEvent,
        StepEvent,
        StepEvent,
        StepEvent,
        MachineNotify,
        MachineEnd,
    ]
    assert [type(e.fact) for e in events if isinstance(e, StepEvent)] == [
        ToolFact,
        BranchFact,
        WaitFact,
        AgentFact,
    ]
    tool = events[1]
    assert isinstance(tool, StepEvent) and isinstance(tool.fact, ToolFact)
    assert tool.fact.stderr == "warn: slow\n"  # stderr round-trips through the wire
    wait = events[3]
    assert isinstance(wait, StepEvent) and isinstance(wait.fact, WaitFact)
    assert wait.fact.wake_epoch is None and wait.fact.payload == {"from": "operator"}
    agent = events[4]
    assert isinstance(agent, StepEvent) and isinstance(agent.fact, AgentFact)
    assert agent.fact.usd == 0.25 and agent.fact.output_tokens == 200
    end = events[6]
    assert isinstance(end, MachineEnd) and end.transitions == 4


def test_old_tool_fact_without_stderr_still_parses(tmp_path: Path) -> None:
    # stderr is additive: a journal line written before the field existed (no
    # `stderr` key) must still replay, with stderr defaulting to "". extra="forbid"
    # rejects UNKNOWN keys, never a missing defaulted one.
    j = _journal(tmp_path)
    old_line = (
        '{"type":"step","ts":"t","seq":0,"state":"scan","label":"ok","goto":"done",'
        '"fact":{"kind":"tool","exit_code":1,"stdout":"","timed_out":false}}\n'
    )
    j.journal_path.write_text(old_line, encoding="utf-8")
    events = j.read()
    assert len(events) == 1
    step = events[0]
    assert isinstance(step, StepEvent) and isinstance(step.fact, ToolFact)
    assert step.fact.stderr == ""


def test_read_survives_unicode_line_separators(tmp_path: Path) -> None:
    # U+2028/U+2029/U+0085 are written literally inside JSON strings; `read` must
    # not treat them as line breaks (splitlines does), or one captured value with
    # one would shred a journal line and brick the instance.
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    poison = "a\u2028b\u2029c\u0085d"  # line/para/next-line separators
    j.append(
        StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="done",
            fact=ToolFact(exit_code=0, stdout=f'{{"note": "{poison}"}}', timed_out=False),
        )
    )
    events = j.read()
    assert len(events) == 2  # begin + one step, not fragmented
    assert isinstance(events[1], StepEvent)
    assert isinstance(events[1].fact, ToolFact)
    assert poison in events[1].fact.stdout


def test_read_tolerates_and_append_heals_torn_final_line(tmp_path: Path) -> None:
    # A crash mid-append leaves a final line with no trailing newline. `read`
    # drops it instead of bricking, and the next `append` heals the file so the
    # new event lands on its own line (not concatenated onto the fragment).
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    with j.journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "machine.end", "ts": "t", "status": "ok"')  # torn, no newline
    events = j.read()
    assert len(events) == 1  # just the begin; the torn tail is ignored
    assert isinstance(events[0], MachineBegin)
    j.append(MachineEnd(ts="t", status="failed", reason="r", state="s", transitions=1))
    healed = j.read()
    assert len(healed) == 2
    assert isinstance(healed[-1], MachineEnd)


def test_read_tolerates_torn_final_utf8_sequence(tmp_path: Path) -> None:
    # A crash can split a multibyte UTF-8 character on the final line. `read`
    # must drop that byte tail before decoding, then append must heal it.
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    torn = b'{"type":"machine.end","ts":"t","status":"failed","reason":"caf' + "é".encode()[:1]
    with j.journal_path.open("ab") as fh:
        fh.write(torn)
    events = j.read()
    assert len(events) == 1
    assert isinstance(events[0], MachineBegin)
    j.append(MachineEnd(ts="t", status="failed", reason="r", state="s", transitions=1))
    healed = j.read()
    assert len(healed) == 2
    assert isinstance(healed[-1], MachineEnd)


def test_latest_snapshot_falls_back_past_corrupt_newest(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.write_snapshot(Snapshot(seq=1, state="a", blackboard={"n": 1}))
    j.write_snapshot(Snapshot(seq=2, state="b", blackboard={"n": 2}))
    # Corrupt the newest snapshot; latest_snapshot must fall back to seq=1.
    (j.snapshots_dir / "2.json").write_text("{ not valid", encoding="utf-8")
    snap = j.latest_snapshot()
    assert snap is not None
    assert snap.seq == 1
    # All corrupt -> None (the journal is authoritative), never an exception.
    (j.snapshots_dir / "1.json").write_text("nope", encoding="utf-8")
    assert j.latest_snapshot() is None


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    j.append(
        StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=ToolFact(exit_code=0, stdout='{"x": 1}', timed_out=False),
        )
    )
    j.append(
        StepEvent(
            ts="t", seq=1, state="check", label="else", goto="poll", fact=BranchFact(clause_index=1)
        )
    )
    j.append(
        StepEvent(
            ts="t",
            seq=2,
            state="poll",
            label="tick",
            goto="done",
            fact=WaitFact(wake_epoch=12.0, woke_by="tick"),
        )
    )
    j.append(MachineEnd(ts="t", status="ok", reason="done", state="done", transitions=3))

    events = j.read()
    assert isinstance(events[0], MachineBegin)
    assert isinstance(events[1], StepEvent)
    assert isinstance(events[1].fact, ToolFact)
    assert isinstance(events[2].fact, BranchFact)
    assert isinstance(events[3].fact, WaitFact)
    assert isinstance(events[4], MachineEnd)
    assert events[4].transitions == 3


def test_read_missing_journal_is_empty(tmp_path: Path) -> None:
    assert MachineJournal(tmp_path / "nope").read() == []


def test_agent_fact_roundtrip(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="reviewer", version=1)
    j.append(
        StepEvent(
            ts="t",
            seq=0,
            state="review",
            label="ok",
            goto="route",
            fact=AgentFact(
                outcome="ok",
                reason="finish_run",
                payload={"approved": True, "note": "lgtm"},
            ),
        )
    )
    j.append(MachineEnd(ts="t", status="ok", reason="approved", state="stop_ok", transitions=2))
    events = j.read()
    step = events[1]
    assert isinstance(step, StepEvent)
    assert isinstance(step.fact, AgentFact)
    assert step.fact.outcome == "ok"
    assert step.fact.reason == "finish_run"
    assert step.fact.payload == {"approved": True, "note": "lgtm"}


def test_agent_fact_spend_roundtrip(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="reviewer", version=1)
    j.append(
        StepEvent(
            ts="t",
            seq=0,
            state="review",
            label="ok",
            goto="route",
            fact=AgentFact(
                outcome="ok",
                reason="finish_run",
                payload=None,
                usd=0.1234,
                input_tokens=1500,
                output_tokens=420,
            ),
        )
    )
    events = j.read()
    step = events[1]
    assert isinstance(step, StepEvent)
    assert isinstance(step.fact, AgentFact)
    assert step.fact.usd == 0.1234
    assert step.fact.input_tokens == 1500
    assert step.fact.output_tokens == 420


def test_agent_fact_spend_defaults_to_zero(tmp_path: Path) -> None:
    fact = AgentFact(outcome="ok", reason="finish_run", payload=None)
    assert fact.usd == 0.0
    assert fact.input_tokens == 0
    assert fact.output_tokens == 0


def test_corrupt_journal_line_raises(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.begin(machine="demo", version=1)
    j.journal_path.write_text('{"type": "step", "bogus": true}\n', encoding="utf-8")
    with pytest.raises(JournalError):
        j.read()


def test_journal_error_is_a_machine_error() -> None:
    # Surfaces that degrade on a broken machine file (`except MachineError`,
    # reading exc.problems) must degrade the same way on a broken journal.
    from agent6.machine.model import MachineError

    exc = JournalError("corrupt journal line 3")
    assert isinstance(exc, MachineError)
    assert exc.problems == ["corrupt journal line 3"]
    assert str(exc) == "corrupt journal line 3"


def test_snapshot_write_and_latest(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.write_snapshot(Snapshot(seq=0, state="a", blackboard={"n": 1}))
    j.write_snapshot(Snapshot(seq=1, state="b", blackboard={"n": 2}))
    latest = j.latest_snapshot()
    assert latest is not None
    assert latest.seq == 1
    assert latest.state == "b"
    assert latest.blackboard == {"n": 2}


def test_latest_snapshot_none_when_empty(tmp_path: Path) -> None:
    assert _journal(tmp_path).latest_snapshot() is None


def test_snapshot_pruning_keeps_configured_tail(tmp_path: Path) -> None:
    # Only latest_snapshot is ever read; old snapshots get pruned per the
    # [machine] snapshot_keep config (default 5) so a long-running machine
    # does not accumulate one file per transition.
    j = _journal(tmp_path)
    for seq in range(20):
        j.write_snapshot(Snapshot(seq=seq, state="s", blackboard={"n": seq}))
    kept = sorted(int(p.stem) for p in j.snapshots_dir.glob("*.json"))
    assert kept == [15, 16, 17, 18, 19]
    latest = j.latest_snapshot()
    assert latest is not None and latest.seq == 19


def test_snapshot_keep_zero_disables_pruning(tmp_path: Path) -> None:
    j = MachineJournal(tmp_path / "inst", snapshot_keep=0)
    j.ensure_dirs()
    for seq in range(10):
        j.write_snapshot(Snapshot(seq=seq, state="s", blackboard={}))
    assert len(list(j.snapshots_dir.glob("*.json"))) == 10


def test_take_signal_consumes_file(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    assert j.take_signal() == (False, None)
    j.signal_path.write_text("", encoding="utf-8")  # a hand-touched empty poke
    assert j.take_signal() == (True, None)
    assert j.take_signal() == (False, None)


def test_poke_writes_signal_consumed_by_take_signal(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    assert j.take_signal() == (False, None)
    j.poke()
    assert j.take_signal() == (True, None)
    assert j.take_signal() == (False, None)


def test_take_signal_preserves_poke_landing_mid_consume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A poke that renames a fresh signal into place between take_signal's read
    # and its unlink must survive for the next check (the old read-then-unlink
    # destroyed it). The claim-by-rename makes the window structural: the racing
    # poke lands at signal_path while we consume the renamed-away copy.
    j = _journal(tmp_path)
    j.poke("first")
    real_read_text = Path.read_text

    def racing_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name.startswith("signal"):
            j.poke("second")  # a poke lands mid-consume
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", racing_read_text)
    assert j.take_signal() == (True, "first")
    monkeypatch.undo()
    assert j.take_signal() == (True, "second")  # the mid-consume poke survived


def test_poke_carries_payload(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.poke({"cmd": "reload", "n": 3})
    assert j.take_signal() == (True, {"cmd": "reload", "n": 3})
    # A --message-style string payload round-trips too.
    j.poke("hello")
    assert j.take_signal() == (True, "hello")


def test_pending_wait_roundtrip_and_clear(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    assert j.read_pending_wait() is None
    j.write_pending_wait(PendingWait(state="poll", wake_epoch=1234.5))
    pending = j.read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"
    assert pending.wake_epoch == 1234.5
    j.clear_pending_wait()
    assert j.read_pending_wait() is None
    # clearing again is a no-op (missing_ok)
    j.clear_pending_wait()


def test_read_pending_wait_corrupt_raises(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    j.wait_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(JournalError):
        j.read_pending_wait()


def test_machine_lock_refuses_second_holder(tmp_path: Path) -> None:
    root = tmp_path / "m"
    with machine_lock(root):  # noqa: SIM117 - inner lock must be acquired while outer is held
        with pytest.raises(JournalError):
            with machine_lock(root):
                pass


def test_source_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "m"
    write_source(root, "machine = 'x'\n")
    assert read_source(root) == "machine = 'x'\n"


def test_read_source_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(JournalError):
        read_source(tmp_path / "absent")
