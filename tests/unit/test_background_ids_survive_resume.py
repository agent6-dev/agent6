# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Background ids continue across a resume instead of restarting.

A resumed run reuses its session dir, so numbering from `bg1` again handed the
next command an id whose log directory already existed. `_open_log` refuses
that -- two commands sharing one log would mix their output -- with a message
blaming a command for planting the directory, so a resumed run's first
background command failed on a collision it had caused itself.

Every long command reaches this since the check-in landed: one that outlives
`command_checkin_s` is handed back as a background shell.
"""

from __future__ import annotations

from pathlib import Path

from agent6.tools.background import BackgroundShells


def _leg(root: Path) -> BackgroundShells:
    """A fresh roster over the same session dir, as a resume builds."""
    return BackgroundShells(root)


def test_a_resumed_leg_does_not_reuse_an_id(tmp_path: Path) -> None:
    root = tmp_path / "shells"
    first = _leg(root)
    first._open_log("bg1")  # pyright: ignore[reportPrivateUsage]

    resumed = _leg(root)
    assert resumed._seq == 1, "the new leg must continue the numbering"  # pyright: ignore[reportPrivateUsage]
    # The next id it hands out is free, so opening its log succeeds.
    resumed._open_log("bg2")  # pyright: ignore[reportPrivateUsage]


def test_the_scan_covers_a_leg_that_died_between_its_two_dirs(tmp_path: Path) -> None:
    """`start` creates <root>/bg<N> and `_open_log` creates <root>/logs/bg<N>;
    a leg killed between them leaves only one, and either must still count."""
    root = tmp_path / "shells"
    (root / "logs").mkdir(parents=True)
    (root / "bg7").mkdir()  # shell dir only
    assert _leg(root)._seq == 7  # pyright: ignore[reportPrivateUsage]

    other = tmp_path / "other"
    (other / "logs" / "bg4").mkdir(parents=True)  # log dir only
    assert _leg(other)._seq == 4  # pyright: ignore[reportPrivateUsage]


def test_a_fresh_run_still_starts_at_one(tmp_path: Path) -> None:
    """The negative control: nothing recorded means nothing to continue from."""
    assert _leg(tmp_path / "shells")._seq == 0  # pyright: ignore[reportPrivateUsage]


def test_unrelated_names_are_not_mistaken_for_ids(tmp_path: Path) -> None:
    root = tmp_path / "shells"
    (root / "logs").mkdir(parents=True)
    for junk in ("bgus", "bg", "background", "bg1x"):
        (root / junk).mkdir()
    assert _leg(root)._seq == 0  # pyright: ignore[reportPrivateUsage]
