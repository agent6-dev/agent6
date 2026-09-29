# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An id that resolves nothing is not "no run".

`sessions show`, `attach` and `resume` reach every bucket, so the miss names the id, not a kind.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6.sessions import id, layout
from agent6.ui.cli import _common  # pyright: ignore[reportPrivateUsage]


def test_the_cross_bucket_resolver_says_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(id.SessionIdError) as caught:
        _common.resolve_session_layout(tmp_path, "nope-nope-NOPE00")
    assert "no session matches" in str(caught.value), str(caught.value)


def test_the_bucket_scoped_resolver_says_session(tmp_path: pathlib.Path) -> None:
    bucket = layout.bucket_dir(tmp_path, "runs")
    bucket.mkdir(parents=True)
    with pytest.raises(id.SessionIdError) as caught:
        id.resolve_session(tmp_path, "nope-nope-NOPE00", buckets=("runs",))
    assert "no session matches" in str(caught.value), str(caught.value)
    assert str(tmp_path) in str(caught.value)  # the state dir searched
