# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config show` renders TOML an operator can copy back out."""

from __future__ import annotations

from pathlib import Path

from agent6.config.layer import load_effective
from agent6.viewmodel.config_view import render_show


def test_a_top_level_scalar_is_not_dressed_as_a_table(tmp_path: Path) -> None:
    """`preset` is a bare top-level key, not a `[preset]` table.

    The renderer grouped every leaf by its first dotted segment, so a key with
    no dot became its own one-row "section" under a `[preset]` header. Copying
    that into a config file writes invalid TOML, and `config fill` -- the other
    half of the same feature -- already emits top-level scalars correctly.
    """
    out = render_show(load_effective(tmp_path, preset="quick"))

    assert "[preset]" not in out, "a scalar rendered as a TOML table header"
    assert "preset" in out, "the setting itself must still be shown"
    # It belongs above the tables, exactly where TOML requires it.
    assert out.index("preset") < out.index("["), "a top-level scalar must precede every section"
