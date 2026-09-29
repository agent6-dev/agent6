# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Book the spend of an agent state whose `result.json` never landed.

The harness writes a `budget.update` event per turn with cumulative totals; the last one in a
state's log is that state's running total.
"""

from __future__ import annotations

import datetime
import pathlib

from agent6.machine import AttemptSpend, MachineJournal, StepEvent
from agent6.viewmodel import machine_state


def book_crashed_attempt(journal: MachineJournal, root: pathlib.Path) -> None:
    """Journal an `AttemptSpend` for an orphaned state log and retire the log dir.

    A supervisor death mid-state leaves provider spend recorded only in the per-state log;
    the resuming supervisor books it before the drive re-runs the state. The dir is renamed
    `crashed-<timestamp>-<original>`, out of every seq matcher, so the slice folds once and
    the re-run starts a fresh log. No orphan log, a booked seq or an empty total books nothing.

    Args:
        journal: The machine's journal.
        root: The machine instance's directory.
    """
    newest = machine_state.newest_state_log(root)
    if newest is None:
        return
    seq = machine_state.state_dir_seq(newest.parent.name)
    if seq is None:
        return
    if any(isinstance(e, StepEvent) and e.seq == seq for e in journal.read()):
        return
    state = newest.parent.name.split("-", 1)[-1]
    # Rename first: the seq does not advance across a crash, so a seq-derived name collides.
    # A crash between the rename and the append loses one booking, never duplicates one.
    ts = datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds")
    retired = newest.parent.with_name(f"crashed-{ts.replace(':', '')}-{newest.parent.name}")
    newest.parent.rename(retired)
    spend = machine_state.read_budget_totals(retired / newest.name)
    if spend.usd or spend.input_tokens or spend.output_tokens:
        journal.append(
            AttemptSpend(
                ts=ts,
                seq=seq,
                state=state,
                usd=spend.usd,
                usd_partial=spend.partial,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )
        )
