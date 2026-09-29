# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`unused_session_id` is the only minter that names a session directory."""

from __future__ import annotations

import pathlib

import pytest

from agent6.sessions import id


def test_the_owner_skips_a_taken_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.sessions import id as id_mod

    minted = iter(["taken-one-AAAAAA", "free-two-BBBBBB"])
    monkeypatch.setattr(id_mod, "friendly_token", lambda: next(minted))
    (tmp_path / "sessions" / "machines" / "taken-one-AAAAAA").mkdir(parents=True)

    assert id.unused_session_id(tmp_path, "machines") == "free-two-BBBBBB"


def test_the_owner_gives_up_rather_than_reusing_a_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.sessions import id as id_mod

    monkeypatch.setattr(id_mod, "friendly_token", lambda: "taken-one-AAAAAA")
    (tmp_path / "sessions" / "runs" / "taken-one-AAAAAA").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="could not mint"):
        id.unused_session_id(tmp_path, "runs")


def test_an_id_taken_in_another_bucket_is_not_minted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mint skips a candidate id that exists in any bucket.

    Ids are one public namespace (the CLI resolver, the web lookup, the agent6/<id> branch),
    so a plan minted with a run's id was ambiguous on every surface.
    """
    from agent6.sessions import id as id_mod

    minted = iter(["same-name-AAAAAA", "fresh-name-BBBBBB"])
    monkeypatch.setattr(id_mod, "friendly_token", lambda: next(minted))
    (tmp_path / "sessions" / "runs" / "same-name-AAAAAA").mkdir(parents=True)

    assert id.unused_session_id(tmp_path, "plans") == "fresh-name-BBBBBB"


def test_session_id_bucket_names_the_holder(tmp_path: pathlib.Path) -> None:
    (tmp_path / "sessions" / "plans" / "demo").mkdir(parents=True)
    assert id.session_id_bucket(tmp_path, "demo") == "plans"
    assert id.session_id_bucket(tmp_path, "other") is None
