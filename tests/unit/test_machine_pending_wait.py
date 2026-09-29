# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A corrupt `wait.json` refuses, and says how to get moving again."""

from __future__ import annotations

import pathlib

import pytest

from agent6.machine import journal as machine_journal


@pytest.mark.parametrize("contents", [b"{not json", b"\xff\xfe"])
def test_a_corrupt_pending_wait_names_the_file_and_the_fix(
    tmp_path: pathlib.Path, contents: bytes
) -> None:
    """A corrupt wait record is refused with its remedy: deleting the file re-arms the wait."""
    journal = machine_journal.MachineJournal(tmp_path)
    journal.wait_path.write_bytes(contents)

    with pytest.raises(machine_journal.JournalError) as exc:
        journal.read_pending_wait()

    message = str(exc.value)
    assert str(journal.wait_path) in message, "the operator cannot act without the path"
    assert "delete" in message, "a refusal with no remedy leaves the machine stuck"
