# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine status folds the in-flight state's live spend, not just booked steps."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.machine import AgentFact, StepEvent
from agent6.ui.cli.machine_cmds import _machine_spend  # pyright: ignore[reportPrivateUsage]


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
    usd, tin, tout, inflight = _machine_spend(events, tmp_path, alive=True)
    assert abs(usd - 0.159) < 1e-9  # 0.10 booked + 0.059 live
    assert inflight == "hunt"
    assert tin == 170 and tout == 80


def test_spend_ignores_the_state_log_when_not_alive(tmp_path: Path) -> None:
    # A dead/parked machine: do not fold a stale in-flight log (only booked steps).
    events = [_agent_step(0, 0.10)]
    _state_log(tmp_path, 1, "hunt", 0.059)
    usd, _, _, inflight = _machine_spend(events, tmp_path, alive=False)
    assert abs(usd - 0.10) < 1e-9 and inflight == ""


def test_spend_does_not_double_count_a_booked_state(tmp_path: Path) -> None:
    # The newest state log's seq matches a booked StepEvent (state completed):
    # its cost is already in the AgentFact, so it must NOT be added again.
    events = [_agent_step(0, 0.10)]
    _state_log(tmp_path, 0, "s0", 0.10)  # same seq as the booked step
    usd, _, _, inflight = _machine_spend(events, tmp_path, alive=True)
    assert abs(usd - 0.10) < 1e-9 and inflight == ""
