# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure machine-journal fold in agent6.viewmodel.machine_state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent6.machine import load_machine
from agent6.machine.journal import (
    BranchFact,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    PendingWait,
    StepEvent,
)
from agent6.viewmodel.machine_state import (
    _NOTIFY_KEEP,  # pyright: ignore[reportPrivateUsage]
    NotificationView,
    fold_machine,
    machine_state_as_dict,
    machine_status_word,
    machine_word_for_dir,
    newest_state_log,
    notification_key,
    probe_instance,
)

# A branch -> terminal machine: two states, no I/O, valid to load.
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


def _spec(tmp_path: Path):
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    return load_machine(f)


def test_fold_empty_starts_at_initial(tmp_path: Path) -> None:
    # No journal yet: the machine is at its initial state, nothing visited.
    ms = fold_machine(_spec(tmp_path), [])
    assert (ms.machine, ms.version, ms.initial, ms.current) == ("tiny", 1, "route", "route")
    assert ms.transitions == ()
    assert ms.ended is None
    assert [s.name for s in ms.states] == ["route", "done"]  # spec order preserved
    route = next(s for s in ms.states if s.name == "route")
    assert route.is_current and not route.is_visited and route.kind == "branch"
    assert all(not s.is_visited for s in ms.states)


def test_fold_tracks_position_transitions_and_end(tmp_path: Path) -> None:
    events = [
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    ms = fold_machine(_spec(tmp_path), events)
    by = {s.name: s for s in ms.states}
    # current = goto of the last transition; both endpoints are visited.
    assert ms.current == "done"
    assert by["done"].is_current and not by["route"].is_current
    assert by["route"].is_visited and by["done"].is_visited
    path = [(t.seq, t.state, t.label, t.goto) for t in ms.transitions]
    assert path == [(0, "route", "else", "done")]
    assert ms.ended is not None
    assert (ms.ended.status, ms.ended.reason, ms.ended.state, ms.ended.transitions) == (
        "ok",
        "routed",
        "done",
        1,
    )


def test_machine_status_word_distinguishes_waiting_from_running(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    ended = fold_machine(
        spec, [MachineEnd(ts="t", status="failed", reason="boom", state="done", transitions=1)]
    )
    # A terminal instance reports its end status regardless of liveness probes.
    assert machine_status_word(ended, parked=True, alive=True) == "failed"

    live = fold_machine(spec, [])  # not ended
    # Parked (an armed --exit-on-wait wait) reads waiting, even if a stale pid
    # probe were to lie alive; running only when live and not parked; a dead pid
    # that is neither parked nor ended is stopped.
    assert machine_status_word(live, parked=True, alive=False) == "waiting"
    assert machine_status_word(live, parked=True, alive=True) == "waiting"
    assert machine_status_word(live, parked=False, alive=True) == "running"
    assert machine_status_word(live, parked=False, alive=False) == "stopped"

    # A live worker blocked in a foreground `wait` state is "waiting", not
    # "running" (the default `machine run` persists no PendingWait).
    wf = tmp_path / "w.asm.toml"
    wf.write_text(
        'machine = "w"\nversion = 1\ninitial = "poll"\n[budget]\nmax_transitions = 10\n'
        '[states.poll]\nkind = "wait"\nevery_secs = "3600"\n'
        'on = { tick = "done", signal = "done" }\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "d"\n',
        encoding="utf-8",
    )
    waiting = fold_machine(load_machine(wf), [])
    assert machine_status_word(waiting, parked=False, alive=True) == "waiting"


def test_machine_word_for_dir_pairs_the_dir_probes(tmp_path: Path) -> None:
    """The dir-level owner: the pure word fed the armed-wait and worker-pid
    probes, so surfaces cannot pair them differently."""
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    assert machine_word_for_dir(live, d) == "stopped"  # no wait, no worker
    MachineJournal(d).write_pending_wait(PendingWait(state="route", wake_epoch=None))
    assert machine_word_for_dir(live, d) == "waiting"  # armed wait, no worker
    MachineJournal(d).clear_pending_wait()
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert machine_word_for_dir(live, d) == "running"
    ended = fold_machine(
        spec, [MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1)]
    )
    assert machine_word_for_dir(ended, d) == "ok"  # the end outranks the probes


def test_machine_state_as_dict_stamps_the_dir_word(tmp_path: Path) -> None:
    # With a dir in hand the wire form carries the dir-aware word; a genuinely
    # dir-less stream keeps the bare fold (no fabricated liveness claim).
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    MachineJournal(d).write_pending_wait(PendingWait(state="route", wake_epoch=None))
    assert machine_state_as_dict(live, d)["status"] == "waiting"
    assert "status" not in machine_state_as_dict(live)


def test_the_wire_form_carries_every_verb_refusal(tmp_path: Path) -> None:
    """A front-end paints all four verbs from the same readiness facts."""
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    refusals = machine_state_as_dict(live, d)["refusals"]

    assert refusals == {
        "stop": "",
        "poke": "machine 'inst' has no open wait to poke",
        "steer": "machine 'inst' has no open agent state to steer",
        "answer": "machine 'inst' has no open prompt to answer",
    }
    MachineJournal(d).write_pending_wait(PendingWait(state="route", wake_epoch=None))
    parked = machine_state_as_dict(live, d)
    assert parked["status"] == "waiting"
    assert parked["refusals"]["poke"] == ""
    assert "no open prompt" in parked["refusals"]["answer"]
    assert "reads no steer" in parked["refusals"]["steer"]


def test_a_live_machine_without_an_open_agent_wait_refuses_by_name(tmp_path: Path) -> None:
    """A live worker alone does not make a past or unborn agent state a reader."""
    from agent6.viewmodel.machine_state import machine_verb_refusal

    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    MachineJournal(d).begin(machine="tiny", version=1)
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    assert machine_verb_refusal(d, "tiny", "steer") == (
        "machine 'tiny' has no open agent state to steer"
    )
    assert machine_verb_refusal(d, "tiny", "answer") == (
        "machine 'tiny' has no open prompt to answer"
    )


def test_the_wire_form_reads_the_journal_once(tmp_path: Path) -> None:
    """The refusals ride on the fold the caller already has. Asking
    `machine_verb_refusals` for them re-read the journal and re-folded the
    machine, so every SSE frame did the work twice (50 ms -> 9 ms per frame on
    a 5,000-event journal)."""
    import agent6.viewmodel.machine_state as mod

    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    reads = 0
    real_read = MachineJournal.read

    def counting_read(self: MachineJournal) -> list[object]:
        nonlocal reads
        reads += 1
        return real_read(self)

    mod.MachineJournal.read = counting_read  # type: ignore[method-assign]
    try:
        d_out = machine_state_as_dict(live, d)
    finally:
        mod.MachineJournal.read = real_read  # type: ignore[method-assign]

    assert d_out["refusals"]["stop"] == ""
    assert reads == 0, f"the wire form re-read the journal {reads} time(s)"


def test_a_corrupt_wait_file_counts_as_parked(tmp_path: Path) -> None:
    # Better to render "waiting" than to guess "stopped"/close the stream over
    # an unreadable wait record; the one rule every surface shares.
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    (d / "wait.json").write_text("{ not json", encoding="utf-8")
    assert probe_instance(d, live).parked is True
    assert machine_word_for_dir(live, d) == "waiting"
    # The verbs name the corruption on the wire as the CLI does: the poke
    # button read enabled while the POST behind it refused.
    refusals = machine_state_as_dict(live, d)["refusals"]
    assert all("corrupt pending wait" in text for text in refusals.values()), refusals


def test_newest_state_log_picks_highest_seq(tmp_path: Path) -> None:
    states = tmp_path / "states"
    for name in ("0000-greet", "0002-review", "0001-greet"):
        (states / name).mkdir(parents=True)
        (states / name / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    # A dir without a log yet (the agent hasn't written) must be ignored.
    (states / "0009-pending").mkdir()
    assert newest_state_log(tmp_path) == states / "0002-review" / "logs.jsonl"
    assert newest_state_log(tmp_path / "absent") is None


def test_fold_collects_notifications(tmp_path: Path) -> None:
    events = [
        MachineNotify(ts="t1", state="route", message="starting", level="info"),
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineNotify(ts="t2", state="done", message="all done", level="warn"),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    ms = fold_machine(_spec(tmp_path), events)
    assert [(n.state, n.message, n.level) for n in ms.notifications] == [
        ("route", "starting", "info"),
        ("done", "all done", "warn"),
    ]


def test_notifications_are_a_capped_sliding_window(tmp_path: Path) -> None:
    # notifications is capped to the recent tail: a front-end must dedup by
    # notification_key, NOT by a count index (which would miss every one past
    # the cap once the window slides).
    events = [
        MachineNotify(ts=f"t{i}", state="route", message=f"n{i}", level="info")
        for i in range(_NOTIFY_KEEP + 5)
    ]
    ms = fold_machine(_spec(tmp_path), events)
    assert len(ms.notifications) == _NOTIFY_KEEP
    assert ms.notifications[-1].message == f"n{_NOTIFY_KEEP + 4}"  # newest kept
    assert ms.notifications[0].message == "n5"  # oldest dropped


def test_notification_key_is_stable_identity() -> None:
    n = NotificationView(ts="t1", state="poll", message="hi", level="warn")
    assert notification_key(n) == ("t1", "poll", "hi")


def test_machine_state_as_dict_is_json_serializable(tmp_path: Path) -> None:
    import json

    from agent6.viewmodel.machine_state import machine_state_as_dict

    events = [
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    d = machine_state_as_dict(fold_machine(_spec(tmp_path), events))
    assert d["machine"] == "tiny" and d["current"] == "done"
    assert d["states"][0]["name"] == "route"  # tuple -> list, dataclass -> dict
    assert d["ended"]["status"] == "ok"
    json.dumps(d)  # the wire form must serialize


def test_machine_verb_refusal_is_one_reading_per_state_and_verb(tmp_path: Path) -> None:
    """The one gate every surface's stop/poke/steer/answer runs: an unknown
    machine is named as unknown; an ended one takes nothing; a stopped one
    takes only a poke when a wait is armed; a live machine takes stop, poke only
    with an open wait, steer only with an agent state, and answer only with a prompt."""
    from agent6.viewmodel.machine_state import machine_verb_refusal

    verbs = ("stop", "poke", "steer", "answer")
    missing = tmp_path / "ghost"
    assert all(machine_verb_refusal(missing, "ghost", v) == "no machine 'ghost'" for v in verbs)

    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = MachineJournal(d)
    journal.begin(machine="tiny", version=1)
    # Stopped: no worker, no wait. No verb goes through.
    assert "no open wait" in machine_verb_refusal(d, "tiny", "poke")
    for verb in ("stop", "steer", "answer"):
        assert "is not running" in machine_verb_refusal(d, "tiny", verb), verb
    # Live in an armed wait: stop and poke reach it; answer has no prompt and
    # steer has no agent loop.
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    journal.write_pending_wait(PendingWait(state="route", wake_epoch=None))
    assert machine_verb_refusal(d, "tiny", "stop") == ""
    assert machine_verb_refusal(d, "tiny", "poke") == ""
    assert "no open prompt" in machine_verb_refusal(d, "tiny", "answer")
    assert "reads no steer" in machine_verb_refusal(d, "tiny", "steer")
    journal.clear_pending_wait()
    # An open agent loop takes steer, and its open question takes answer.
    log = d / "states" / "0000-route" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        '{"type":"session.start","mode":"run","user_task":"t"}\n'
        '{"type":"question.prompt","id":"q1","questions":[{"question":"Which?"}]}\n',
        encoding="utf-8",
    )
    assert machine_verb_refusal(d, "tiny", "poke")
    assert all(machine_verb_refusal(d, "tiny", v) == "" for v in ("stop", "steer", "answer"))
    # Ended: nothing does, and the end is named.
    journal.append(MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1))
    for verb in verbs:
        msg = machine_verb_refusal(d, "tiny", verb)
        assert "already ended in 'done' (ok: routed)" in msg, (verb, msg)


def test_an_open_prompt_in_the_newest_state_blocks_the_machine(tmp_path: Path) -> None:
    """The newest state log's unanswered approval names the state the machine
    waits on; an answered one does not, and a live blocked worker is "waiting"."""
    from agent6.viewmodel.machine_state import machine_status_word, newest_agent_leg

    states = tmp_path / "states"
    (states / "0001-attempt").mkdir(parents=True)
    log = states / "0001-attempt" / "logs.jsonl"
    prompt = {"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: pytest"}
    log.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    assert newest_agent_leg(tmp_path).blocked_in == "0001-attempt"
    answer = {"type": "approval.answer", "id": "a1", "approved": True}
    log.write_text(json.dumps(prompt) + "\n" + json.dumps(answer) + "\n", encoding="utf-8")
    assert newest_agent_leg(tmp_path).blocked_in == ""
    ms = fold_machine(_spec(tmp_path), [])
    assert machine_status_word(ms, parked=False, alive=True, blocked=True) == "waiting"
    assert machine_status_word(ms, parked=False, alive=True) == "running"


def test_a_blocked_summary_names_an_answer_whichever_prompt_waits(tmp_path: Path) -> None:
    """A machine held on an unanswered `ask_user` question read "waiting on
    an approval": the summary's reason named one prompt kind for both."""
    from agent6.viewmodel.machine_state import summarize_machine_dir

    (tmp_path / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    log = tmp_path / "states" / "0001-attempt" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    prompt = {"type": "question.prompt", "id": "q1", "questions": [{"question": "Which?"}]}
    log.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    assert summarize_machine_dir(tmp_path).reason == "waiting on an answer in 0001-attempt"
    # A stopped machine's prompt has no reader (its leg restarts on resume).
    (tmp_path / "worker.pid").unlink()
    assert summarize_machine_dir(tmp_path).reason == ""


@pytest.mark.parametrize(
    "stale", [PendingWait(state="elsewhere"), PendingWait(state="route", seq=1)]
)
def test_a_wait_record_of_another_occurrence_is_not_an_open_wait(
    tmp_path: Path, stale: PendingWait
) -> None:
    """A record a death left behind an earlier visit (another state, or this
    state at another transition) read as an open wait: the machine executing a
    tool state showed "waiting" and took a poke its next wait would consume as
    its wake. The engine's own test (state and seq) is the one reading."""
    from agent6.viewmodel.machine_state import machine_verb_refusal

    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = MachineJournal(d)
    journal.begin(machine="tiny", version=1)
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    journal.write_pending_wait(stale)
    assert "no open wait" in machine_verb_refusal(d, "tiny", "poke")
    assert machine_state_as_dict(live, d)["status"] == "running"

    journal.write_pending_wait(PendingWait(state="route", seq=0))
    assert machine_verb_refusal(d, "tiny", "poke") == ""
    assert machine_state_as_dict(live, d)["status"] == "waiting"


def test_verb_refusals_fold_no_state_log_unless_a_live_leg_could_read(tmp_path: Path) -> None:
    """The refusals folded the newest state log for every instance asked, an
    ended or stopped one included (a TAB over the instance dirs, the machine
    screen's poll), though only a live, unended machine has a leg to read a
    steer or an answer."""
    import agent6.viewmodel.machine_state as mod
    from agent6.viewmodel.machine_state import machine_verb_refusals

    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = MachineJournal(d)
    journal.begin(machine="tiny", version=1)
    log = d / "states" / "0000-route" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"type":"session.start","mode":"run","user_task":"t"}\n', encoding="utf-8")
    folds = 0
    real_tail = mod.tail_events

    def counting_tail(*args: Any, **kwargs: Any) -> Any:
        nonlocal folds
        folds += 1
        return real_tail(*args, **kwargs)

    mod.tail_events = counting_tail  # type: ignore[assignment]
    try:
        machine_verb_refusals(d, "tiny")  # stopped: no worker
        assert folds == 0
        (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
        machine_verb_refusals(d, "tiny")  # live
        assert folds == 1
        journal.append(
            MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1)
        )
        machine_verb_refusals(d, "tiny")  # ended
        assert folds == 1
    finally:
        mod.tail_events = real_tail


def test_an_unreadable_summary_keeps_its_reason_to_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spec with several problems put every line into the listing's reason
    cell, and the row ran over the table."""
    import agent6.viewmodel.machine_state as mod
    from agent6.machine import MachineError
    from agent6.viewmodel.machine_state import summarize_machine_dir

    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")

    def broken(_path: Path) -> Any:
        raise MachineError(["state 'a' is unreachable", "state 'b' is unreachable"])

    monkeypatch.setattr(mod, "load_machine", broken)
    summary = summarize_machine_dir(d)
    assert summary.status == "unreadable"
    assert summary.reason == "state 'a' is unreachable"


def test_the_wire_form_carries_the_status_level(tmp_path: Path) -> None:
    """The hub row stamps a level beside its status word; the machine page's
    own header had none, so a failed machine read plain on its page."""
    from agent6.viewmodel.format import status_level

    spec = _spec(tmp_path)
    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    live = fold_machine(spec, [])
    running = machine_state_as_dict(live, d)
    assert running["level"] == status_level(running["status"])
    failed = fold_machine(
        spec, [MachineEnd(ts="t", status="failed", reason="budget", state="done", transitions=1)]
    )
    ended = machine_state_as_dict(failed, d)
    assert (ended["status"], ended["level"]) == ("failed", status_level("failed"))


def test_the_newest_leg_fold_reads_only_what_the_log_gained(tmp_path: Path) -> None:
    """A poll loop folded the newest state log from scratch on every tick, once
    for the refusals and once for the prompts or the reasoning; the held fold
    reads the appended bytes only, follows the machine into a newer agent
    state, and starts over when a log was rewritten."""
    import agent6.viewmodel.machine_state as mod
    from agent6.viewmodel.machine_state import AgentLeg, NewestLegFold
    from agent6.viewmodel.tail import tail_events

    d = tmp_path / "inst"
    log = d / "states" / "0000-route" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"type":"session.start","mode":"run","user_task":"t"}\n', encoding="utf-8")
    fold = NewestLegFold()
    assert fold.refresh(d) == log
    assert fold.leg() == AgentLeg(open=True, blocked_in="")

    def no_full_read(*_a: object, **_k: object) -> Any:
        raise AssertionError("the whole log was read again")

    mod.tail_events = no_full_read  # type: ignore[assignment]
    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write('{"type":"question.prompt","id":"q1","questions":[{"question":"?"}]}\n')
        fold.refresh(d)
        assert fold.leg() == AgentLeg(open=True, blocked_in="0000-route")
        # A newer agent state: the fold moves to its log.
        newer = d / "states" / "0001-work" / "logs.jsonl"
        newer.parent.mkdir(parents=True)
        newer.write_text(
            '{"type":"session.start","mode":"run","user_task":"t"}\n', encoding="utf-8"
        )
        assert fold.refresh(d) == newer
        assert fold.leg() == AgentLeg(open=True, blocked_in="")
        # A rewritten (shorter) log: the fold starts over rather than folding
        # the new bytes onto the old state.
        newer.write_text('{"type":"session.end","reason":"finish_session"}\n', encoding="utf-8")
        fold.refresh(d)
        assert fold.leg() == AgentLeg(open=False, blocked_in="")
    finally:
        mod.tail_events = tail_events
