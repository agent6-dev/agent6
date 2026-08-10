# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-process tools' path containment (`tools/_path_safety`)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.tools._path_safety import SafePath, contain, open_contained
from agent6.tools.dispatch import ToolError


def test_contain_refuses_an_uncontained_relative_path(tmp_path: Path) -> None:
    """Containment is the walk, and the walk cannot express `..`: every caller
    resolves first, so the invariant was a nine-caller convention. Held by the
    SafePath now, so a caller that forgets is refused instead of walking out."""
    (tmp_path / "root").mkdir()
    (tmp_path / "outside.txt").write_text("host\n", encoding="utf-8")
    with pytest.raises(ToolError, match=r"\.\."):
        contain(tmp_path / "root", "../outside.txt")


def test_contain_refuses_an_absolute_path(tmp_path: Path) -> None:
    """An absolute rel_path drops the base entirely (pathlib's join rule), so
    the fd would be on a host file no containment check ever saw."""
    (tmp_path / "root").mkdir()
    with pytest.raises(ToolError, match="Absolute"):
        contain(tmp_path / "root", "/etc/hostname")


@pytest.mark.parametrize("rel", ["/etc/hostname", "../outside.txt"])
def test_open_contained_re_checks_a_hand_built_safe_path(tmp_path: Path, rel: str) -> None:
    """The walk keeps its own `..`/absolute guard rather than trusting the
    SafePath: containment must hold even for one built directly, since the type
    is constructible without going through `contain` or a `Workspace`."""
    (tmp_path / "root").mkdir()
    forged = SafePath(base=tmp_path / "root", rel_path=Path(rel), abs_path=Path(rel))
    with pytest.raises(ToolError):
        open_contained(forged, os.O_RDONLY)


def test_open_contained_reads_a_contained_path(tmp_path: Path) -> None:
    (tmp_path / "root" / "sub").mkdir(parents=True)
    (tmp_path / "root" / "sub" / "f.txt").write_text("ok\n", encoding="utf-8")
    fd = open_contained(contain(tmp_path / "root", "sub/f.txt"), os.O_RDONLY)
    with os.fdopen(fd, encoding="utf-8") as handle:
        assert handle.read() == "ok\n"
