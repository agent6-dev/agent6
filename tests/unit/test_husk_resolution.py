# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A husk session refuses at the resolver; only `sessions rm` still names one."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.sessions import id
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import _common


def _husk(repo: pathlib.Path, session_id: str = "husky-one-AAAAAA") -> pathlib.Path:
    """A session dir with neither manifest.json nor logs.jsonl crashed before it ever started."""
    d = sessions_layout.bucket_dir(paths.state_dir(repo), "runs") / session_id
    d.mkdir(parents=True)
    return d


def test_an_explicit_husk_id_refuses_with_the_remedy(tmp_path: pathlib.Path) -> None:
    """The resolver answers a husk once, for every surface, never as a session to resume."""
    _husk(tmp_path)
    with pytest.raises(id.SessionIdError, match="crashed before it ever started"):
        _common.resolve_session_layout(tmp_path, "husky-one-AAAAAA")
    with pytest.raises(id.SessionIdError, match="sessions rm"):
        _common.resolve_session_layout(tmp_path, "husky")  # by prefix too


def test_rm_still_resolves_a_husk(tmp_path: pathlib.Path) -> None:
    """Cleanup must keep working: rm is the surface that deletes exactly this."""
    d = _husk(tmp_path)
    layout = _common.resolve_session_layout(tmp_path, "husky-one-AAAAAA", allow_husk=True)
    assert layout.session_dir == d
    from_newest = _common.resolve_or_newest_layout(tmp_path, "husky-one-AAAAAA", allow_husk=True)
    assert from_newest is not None and from_newest.session_dir == d


def test_a_real_session_still_resolves(tmp_path: pathlib.Path) -> None:
    d = sessions_layout.bucket_dir(paths.state_dir(tmp_path), "runs") / "realy-two-BBBBBB"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text('{"type":"session.start","mode":"run"}\n', encoding="utf-8")
    assert _common.resolve_session_layout(tmp_path, "realy-two-BBBBBB").session_dir == d
