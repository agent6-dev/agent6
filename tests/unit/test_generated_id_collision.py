# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A generated session id never lands on a directory that already exists.

The "already exists, use resume" guard is reached only for an EXPLICIT
--session-id. A generated one that collided wrote a fresh manifest and
loop_state into a live session's dir, beside its graph, checkpoints and
transcripts -- the mixed state that guard exists to refuse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.run import _unused_session_id  # pyright: ignore[reportPrivateUsage]


def test_a_generated_id_skips_a_taken_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.app import run as run_mod

    minted = iter(["taken-one-AAAAAA", "free-two-BBBBBB"])
    monkeypatch.setattr(run_mod, "new_friendly_id", lambda: next(minted))
    (tmp_path / "runs" / "taken-one-AAAAAA").mkdir(parents=True)

    assert _unused_session_id(tmp_path, "runs") == "free-two-BBBBBB"


def test_a_free_id_is_taken_as_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.app import run as run_mod

    monkeypatch.setattr(run_mod, "new_friendly_id", lambda: "free-one-AAAAAA")
    assert _unused_session_id(tmp_path, "runs") == "free-one-AAAAAA"


def test_it_gives_up_loudly_rather_than_reusing_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every candidate taken means something is wrong with minting, not that
    the run should write into someone else's directory."""
    from agent6.app import run as run_mod

    monkeypatch.setattr(run_mod, "new_friendly_id", lambda: "taken-one-AAAAAA")
    (tmp_path / "runs" / "taken-one-AAAAAA").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="could not mint"):
        _unused_session_id(tmp_path, "runs")


def test_the_bucket_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same id in a DIFFERENT bucket is not a collision: a plan and a run
    are separate directories."""
    from agent6.app import run as run_mod

    monkeypatch.setattr(run_mod, "new_friendly_id", lambda: "same-name-AAAAAA")
    (tmp_path / "runs" / "same-name-AAAAAA").mkdir(parents=True)

    assert _unused_session_id(tmp_path, "plans") == "same-name-AAAAAA"
