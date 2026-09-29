# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI finds a draft where `machine create` writes one.

Creating a machine from the hub spawns `agent6 machine create` and locates the
new draft by diffing the draft directory before and after. That is a two-sided
contract with nothing checking it: move the write and the locator sees no new
directory, so the TUI reports "could not start machine create" over a draft
that started fine.
"""

from __future__ import annotations

import pathlib

from agent6 import kinds
from agent6.app.machine import create as create_mod
from agent6.sessions import layout
from agent6.ui.tui import machines  # pyright: ignore[reportPrivateUsage]


def test_the_locator_reads_the_directory_the_writer_writes(tmp_path: pathlib.Path) -> None:
    written = layout.bucket_dir(tmp_path, kinds.session_bucket("machine")) / "eager-forge-AAAAAA"
    written.mkdir(parents=True)

    assert machines._list_drafts(tmp_path) == [written]


def test_the_locator_lists_the_directory_new_draft_dir_derives(tmp_path: pathlib.Path) -> None:
    drafted = create_mod.new_draft_dir(tmp_path)
    drafted.mkdir(parents=True)

    assert machines._list_drafts(tmp_path) == [drafted]


def test_an_empty_state_dir_lists_nothing(tmp_path: pathlib.Path) -> None:
    assert machines._list_drafts(tmp_path) == []
