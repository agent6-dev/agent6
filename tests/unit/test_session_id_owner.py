# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`unused_session_id` is the only minter that names a session directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.sessions.id import friendly_token, unused_session_id


def test_the_owner_skips_a_taken_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.sessions import id as id_mod

    minted = iter(["taken-one-AAAAAA", "free-two-BBBBBB"])
    monkeypatch.setattr(id_mod, "friendly_token", lambda: next(minted))
    (tmp_path / "sessions" / "machines" / "taken-one-AAAAAA").mkdir(parents=True)

    assert unused_session_id(tmp_path, "machines") == "free-two-BBBBBB"


def test_the_owner_gives_up_rather_than_reusing_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.sessions import id as id_mod

    monkeypatch.setattr(id_mod, "friendly_token", lambda: "taken-one-AAAAAA")
    (tmp_path / "sessions" / "runs" / "taken-one-AAAAAA").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="could not mint"):
        unused_session_id(tmp_path, "runs")


def test_a_free_id_is_taken_as_is(tmp_path: Path) -> None:
    assert unused_session_id(tmp_path, "runs") != ""
    assert friendly_token() != unused_session_id(tmp_path, "runs")


def test_the_bucket_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same id in a DIFFERENT bucket is not a collision: a plan and a run
    are separate directories."""
    from agent6.sessions import id as id_mod

    monkeypatch.setattr(id_mod, "friendly_token", lambda: "same-name-AAAAAA")
    (tmp_path / "sessions" / "runs" / "same-name-AAAAAA").mkdir(parents=True)

    assert unused_session_id(tmp_path, "plans") == "same-name-AAAAAA"
