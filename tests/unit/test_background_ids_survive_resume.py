# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Background ids continue across a resume instead of restarting.

A resumed run reuses its session dir, and `_open_log` refuses an id whose log directory exists.
Every command that outlives `command_checkin_s` is handed back as a background shell.
"""

from __future__ import annotations

import pathlib

from agent6.tools import background


def _execution(root: pathlib.Path) -> background.BackgroundShells:
    """A fresh roster over the same session dir, as a resume builds."""
    return background.BackgroundShells(root)


def test_a_resumed_execution_does_not_reuse_an_id(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "shells"
    first = _execution(root)
    first._open_log("bg1")  # pyright: ignore[reportPrivateUsage]

    resumed = _execution(root)
    assert resumed._seq == 1, "the new execution must continue the numbering"  # pyright: ignore[reportPrivateUsage]
    # The next id it hands out is free, so opening its log succeeds.
    resumed._open_log("bg2")  # pyright: ignore[reportPrivateUsage]


def test_the_scan_covers_a_execution_that_died_between_its_two_dirs(tmp_path: pathlib.Path) -> None:
    """The scan covers an execution that died between its two dirs.

    `start` creates <root>/bg<N> and `_open_log` creates <root>/logs/bg<N>; either alone counts.
    """
    root = tmp_path / "shells"
    (root / "logs").mkdir(parents=True)
    (root / "bg7").mkdir()  # shell dir only
    assert _execution(root)._seq == 7  # pyright: ignore[reportPrivateUsage]

    other = tmp_path / "other"
    (other / "logs" / "bg4").mkdir(parents=True)  # log dir only
    assert _execution(other)._seq == 4  # pyright: ignore[reportPrivateUsage]


def test_a_fresh_run_still_starts_at_one(tmp_path: pathlib.Path) -> None:
    """The negative control: nothing recorded means nothing to continue from."""
    assert _execution(tmp_path / "shells")._seq == 0  # pyright: ignore[reportPrivateUsage]


def test_unrelated_names_are_not_mistaken_for_ids(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "shells"
    (root / "logs").mkdir(parents=True)
    for junk in ("bgus", "bg", "background", "bg1x"):
        (root / junk).mkdir()
    assert _execution(root)._seq == 0  # pyright: ignore[reportPrivateUsage]
