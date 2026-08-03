# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A completer offers exactly what its argument accepts.

Offering less is a lie by omission: the operator tabs, sees no plan or ask, and
concludes the verb does not take one -- when it does. Offering more is worse,
since the suggestion is refused on Enter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.layout import bucket_dir
from agent6.ui.cli import completers


def _seed(tmp_path: Path) -> None:
    state = resolved_state_dir(tmp_path)
    for bucket, mode, sid in (
        ("runs", "run", "runny-one-AAAAAA"),
        ("plans", "plan", "planny-two-BBBBB"),
        ("asks", "ask", "asky-three-CCCCC"),
        ("machines", "machine", "drafty-four-DDDD"),
    ):
        session = bucket_dir(state, bucket) / sid
        session.mkdir(parents=True)
        (session / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": mode}) + "\n", encoding="utf-8"
        )
    (state / "machines" / "live-machine").mkdir(parents=True)


def test_every_session_id_is_offered_where_any_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sessions show|diff|transcript|...` resolve across every bucket."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_session_ids(""))  # pyright: ignore[reportPrivateUsage]
    assert offered == {
        "runny-one-AAAAAA",
        "planny-two-BBBBB",
        "asky-three-CCCCC",
        "drafty-four-DDDD",
    }


def test_resume_offers_only_what_it_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine draft is a session, but `resume` refuses it -- so suggesting it
    would be a suggestion the operator cannot act on."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_resumable_ids(""))  # pyright: ignore[reportPrivateUsage]
    assert offered == {"runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC"}


def test_attach_offers_every_session_and_every_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_watch_targets(""))  # pyright: ignore[reportPrivateUsage]
    assert "live-machine" in offered
    assert {"runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC"} <= offered
