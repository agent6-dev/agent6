# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A session directory is never named by a raw `new_friendly_id()`.

Ids carry 4 timestamp chars and 2 random ones, so two minted in the same
millisecond collide about once in 30 million. `unused_session_id` is what turns
that into "mint another"; a site that calls the generator directly and uses the
result as a directory name routes around it -- which is how `machine create`
came to `mkdir(exist_ok=True)` over a live draft.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent6.sessions.id import new_friendly_id, unused_session_id

_SRC = Path(__file__).resolve().parents[2] / "src" / "agent6"

# Where minting a bare id is correct, with the reason it is not a session dir.
_ALLOWED = {
    # The owner itself.
    "sessions/id.py",
    # ACP's own per-CONNECTION id. Never a directory: agent6's session id for
    # that connection is minted separately, through the owner.
    "ui/acp/session.py",
    # The fan-out GROUP id: names a workdir root and seeds each lane's
    # `<coordinator>-<group>-l<i>`, and every lane is spawned with an explicit
    # --session-id that the run lifecycle then guards.
    "ui/cli/parallel.py",
    # The forked CHILD id, which refuses outright when its dir already exists
    # rather than writing into it.
    "app/fork.py",
}


def test_no_new_site_names_a_session_dir_with_a_raw_id() -> None:
    callers = {
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if re.search(r"\bnew_friendly_id\(\)", path.read_text(encoding="utf-8"))
    }
    unreviewed = sorted(callers - _ALLOWED)
    assert not unreviewed, (
        "these mint an id without checking the bucket; use unused_session_id,"
        f" or add them to _ALLOWED with the reason they are not a session dir: {unreviewed}"
    )


def test_the_owner_skips_a_taken_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.sessions import id as id_mod

    minted = iter(["taken-one-AAAAAA", "free-two-BBBBBB"])
    monkeypatch.setattr(id_mod, "new_friendly_id", lambda: next(minted))
    (tmp_path / "sessions" / "machines" / "taken-one-AAAAAA").mkdir(parents=True)

    assert unused_session_id(tmp_path, "machines") == "free-two-BBBBBB"


def test_the_owner_gives_up_rather_than_reusing_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.sessions import id as id_mod

    monkeypatch.setattr(id_mod, "new_friendly_id", lambda: "taken-one-AAAAAA")
    (tmp_path / "sessions" / "runs" / "taken-one-AAAAAA").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="could not mint"):
        unused_session_id(tmp_path, "runs")


def test_a_free_id_is_taken_as_is(tmp_path: Path) -> None:
    assert unused_session_id(tmp_path, "runs") != ""
    assert new_friendly_id() != unused_session_id(tmp_path, "runs")


def test_the_bucket_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same id in a DIFFERENT bucket is not a collision: a plan and a run
    are separate directories."""
    from agent6.sessions import id as id_mod

    monkeypatch.setattr(id_mod, "new_friendly_id", lambda: "same-name-AAAAAA")
    (tmp_path / "sessions" / "runs" / "same-name-AAAAAA").mkdir(parents=True)

    assert unused_session_id(tmp_path, "plans") == "same-name-AAAAAA"
