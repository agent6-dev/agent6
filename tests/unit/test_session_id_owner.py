# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`unused_session_id` is the only minter that names a session directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.sessions.id import unused_session_id


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


def test_an_id_taken_in_another_bucket_is_not_minted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ids are one public namespace: the CLI resolver, the web lookup, and the
    agent6/<id> branch all address a session by bare id, so a plan minted with
    a run's id was ambiguous on every surface (the CLI refused it as ambiguous,
    the web silently picked one). The mint skips a candidate that exists in ANY
    bucket."""
    from agent6.sessions import id as id_mod

    minted = iter(["same-name-AAAAAA", "fresh-name-BBBBBB"])
    monkeypatch.setattr(id_mod, "friendly_token", lambda: next(minted))
    (tmp_path / "sessions" / "runs" / "same-name-AAAAAA").mkdir(parents=True)

    assert unused_session_id(tmp_path, "plans") == "fresh-name-BBBBBB"


def test_session_id_bucket_names_the_holder(tmp_path: Path) -> None:
    from agent6.sessions.id import session_id_bucket

    (tmp_path / "sessions" / "plans" / "demo").mkdir(parents=True)
    assert session_id_bucket(tmp_path, "demo") == "plans"
    assert session_id_bucket(tmp_path, "other") is None
