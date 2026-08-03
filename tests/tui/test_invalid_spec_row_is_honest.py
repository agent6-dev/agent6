# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A file that does not parse has no machine name, so the row does not invent one.

`path.stem` on `lint-and-test.asm.toml` renders `lint-and-test.asm`: half a
filename, and neither the machine's declared name nor the file's. The name is
simply not known until the spec loads, and the `file` column already says which
file it was.
"""

from __future__ import annotations

from pathlib import Path

from agent6.ui.tui.machines import _machine_row  # pyright: ignore[reportPrivateUsage]

_VALID = """
machine = "lint-and-test"
version = 1
initial = "stop_ok"

[budget]
max_usd = 1.0
max_transitions = 10

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "nothing to do"
"""


def test_an_unparsable_file_claims_no_name(tmp_path: Path) -> None:
    path = tmp_path / "lint-and-test.asm.toml"
    path.write_text("this is not toml {{{", encoding="utf-8")

    name, states, spec = _machine_row(path)
    assert spec == "invalid"
    assert states == "-"
    assert name == "-", f"invented a name: {name!r}"


def test_a_valid_file_shows_its_declared_name(tmp_path: Path) -> None:
    """The declared name, not the filename: the two can differ."""
    path = tmp_path / "some-other-filename.asm.toml"
    path.write_text(_VALID, encoding="utf-8")

    name, _states, spec = _machine_row(path)
    assert name == "lint-and-test"
    assert spec != "invalid"
