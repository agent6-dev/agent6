# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A corrupt `wait.json` refuses, and says how to get moving again."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.machine.journal import JournalError, MachineJournal


def test_a_corrupt_pending_wait_names_the_file_and_the_fix(tmp_path: Path) -> None:
    """The engine must not guess a wake instant from a corrupt record -- an early
    or skipped wait is worse than a refusal. But every other refusal in agent6
    names its remedy, and this one left the operator to infer that deleting the
    file re-arms the wait on the next run.
    """
    journal = MachineJournal(tmp_path)
    journal.wait_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(JournalError) as exc:
        journal.read_pending_wait()

    message = str(exc.value)
    assert str(journal.wait_path) in message, "the operator cannot act without the path"
    assert "delete" in message, "a refusal with no remedy leaves the machine stuck"
