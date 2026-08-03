# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine status folds the in-flight state's live spend, not just booked steps."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.app.machine import machine_spend
from agent6.machine import AgentFact, StepEvent


def _agent_step(seq: int, usd: float) -> StepEvent:
    return StepEvent(
        ts="t",
        seq=seq,
        state=f"s{seq}",
        label="ok",
        goto="next",
        fact=AgentFact(
            outcome="ok",
            reason="finish_run",
            payload=None,
            usd=usd,
            input_tokens=100,
            output_tokens=50,
        ),
    )


def _state_log(root: Path, seq: int, name: str, usd: float) -> None:
    d = root / "states" / f"{seq:04d}-{name}"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text(
        json.dumps(
            {"type": "budget.update", "usd_total": usd, "input_total": 70, "output_total": 30}
        )
        + "\n",
        encoding="utf-8",
    )


def test_spend_folds_the_running_state_when_alive(tmp_path: Path) -> None:
    # One completed (booked) step at seq 0, plus a running state at seq 1 whose
    # StepEvent is not written yet -- its live spend must be added.
    events = [_agent_step(0, 0.10)]
    _state_log(tmp_path, 1, "hunt", 0.059)  # in-flight, unbooked
    spend, inflight = machine_spend(events, tmp_path, alive=True)
    assert abs(spend.usd - 0.159) < 1e-9  # 0.10 booked + 0.059 live
    assert inflight == "hunt"
    assert spend.input_tokens == 170 and spend.output_tokens == 80


def test_spend_ignores_the_state_log_when_not_alive(tmp_path: Path) -> None:
    # A dead/parked machine: do not fold a stale in-flight log (only booked steps).
    events = [_agent_step(0, 0.10)]
    _state_log(tmp_path, 1, "hunt", 0.059)
    spend, inflight = machine_spend(events, tmp_path, alive=False)
    assert abs(spend.usd - 0.10) < 1e-9 and inflight == ""


def test_spend_does_not_double_count_a_booked_state(tmp_path: Path) -> None:
    # The newest state log's seq matches a booked StepEvent (state completed):
    # its cost is already in the AgentFact, so it must NOT be added again.
    events = [_agent_step(0, 0.10)]
    _state_log(tmp_path, 0, "s0", 0.10)  # same seq as the booked step
    spend, inflight = machine_spend(events, tmp_path, alive=True)
    assert abs(spend.usd - 0.10) < 1e-9 and inflight == ""


def test_read_budget_totals_offset_scopes_to_one_call(tmp_path: Path) -> None:
    """machine create shares ONE draft log across attempts; a retry that died
    before its first budget.update must salvage $0, not the prior attempt's
    cumulative totals (which double-booked spend and lied on the draft
    dashboard). from_offset scopes the read to events after the caller's
    spawn point."""
    import json

    from agent6.app.machine._spend import Spend, read_budget_totals

    log = tmp_path / "logs.jsonl"
    log.write_text(
        json.dumps(
            {"type": "budget.update", "usd_total": 0.90, "input_total": 9, "output_total": 3}
        )
        + "\n",
        encoding="utf-8",
    )
    offset = log.stat().st_size
    # Attempt 2 died before any budget.update: nothing after the offset.
    assert read_budget_totals(log, from_offset=offset) == Spend()
    # Without the offset the prior attempt's totals still read (machine states
    # pass 0 on their fresh per-state logs).
    assert read_budget_totals(log).usd == 0.90
    # Attempt 2 then emits its own update: only ITS totals salvage.
    with log.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"type": "budget.update", "usd_total": 0.05, "input_total": 2, "output_total": 1}
            )
            + "\n"
        )
    assert read_budget_totals(log, from_offset=offset) == Spend(0.05, 2, 1)


def test_unpriced_spend_reads_as_a_partial_lower_bound(tmp_path: Path) -> None:
    """An unpriced model's spend is a LOWER BOUND: the run surface marks it
    '~', and machine status must agree instead of rendering '$0.0000' as if
    exact -- the machine ledger burning real money against a $0 figure."""
    import json

    from agent6.app.machine._spend import Spend, read_budget_totals
    from agent6.viewmodel.format import format_cost

    log = tmp_path / "logs.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "budget.update",
                "usd_total": 0.0,
                "usd_partial": True,
                "input_total": 900,
                "output_total": 50,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spend = read_budget_totals(log)
    assert spend.partial is True
    assert format_cost(spend.usd, partial=spend.partial).startswith("~$")
    # The flag survives folding with priced (non-partial) slices.
    assert (spend + Spend(1.0)).partial is True


def test_spend_of_a_state_whose_capture_failed_is_still_booked(tmp_path: Path) -> None:
    """A capture that cannot be reduced halts BEFORE the StepEvent is journaled
    -- deliberately, since a fact whose capture fails would re-crash every later
    replay -- which also discarded the agent's real usd and tokens. `machine run`
    then reported spent $0.0000 for a state that had burned money. The end event
    carries the unbooked slice so the ledger still sees it."""
    from agent6.machine.journal import MachineEnd

    root = tmp_path / "inst"
    root.mkdir()
    events = [
        MachineEnd(
            ts="2026-07-27T00:00:00+00:00",
            status="failed",
            reason="state 'judge': record has no field 'note'",
            state="judge",
            transitions=0,
            usd=1.23,
            usd_partial=True,
            input_tokens=40_000,
            output_tokens=8_000,
        )
    ]
    spend, _inflight = machine_spend(events, root, alive=False)
    assert spend.usd == 1.23, "the dollars actually spent were dropped"
    assert spend.input_tokens == 40_000
    assert spend.output_tokens == 8_000
    assert spend.partial is True  # an unpriced slice keeps its lower-bound flag


def test_an_unpriced_unbooked_slice_keeps_its_tokens_and_its_marker(tmp_path: Path) -> None:
    """An UNPRICED slice reports usd 0.0 with usd_partial True. Guarding the end
    event's fold on `event.usd` being truthy skipped it entirely, so the tokens
    and the sticky lower-bound flag went with it: the ledger claimed an EXACT
    $0.0000 (in=0 tok) for a state that burned 48k tokens."""
    from agent6.machine.journal import MachineEnd

    root = tmp_path / "inst"
    root.mkdir()
    end = MachineEnd(
        ts="2026-07-28T00:00:00+00:00",
        status="failed",
        reason="state 'judge': record has no field 'note'",
        state="judge",
        transitions=0,
        usd=0.0,
        usd_partial=True,
        input_tokens=40_000,
        output_tokens=8_000,
    )
    spend, _inflight = machine_spend([end], root, alive=False)
    assert spend.input_tokens == 40_000
    assert spend.output_tokens == 8_000
    assert spend.partial is True, "the '~' lower-bound marker was dropped"
