# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A file that does not parse has no machine name, so the row does not invent one.

`path.stem` on `lint-and-test.asm.toml` is half a filename; the `file` column already says
which file it was. The TUI machines page and `machine list` share the row.
"""

from __future__ import annotations

import pathlib

from agent6.app.machine import summarize_machine_file

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


def test_an_unparsable_file_claims_no_name(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "lint-and-test.asm.toml"
    path.write_text("this is not toml {{{", encoding="utf-8")

    row = summarize_machine_file(path)
    assert row.spec == "invalid"
    assert row.states == "-"
    assert row.name == "-", f"invented a name: {row.name!r}"


def test_a_valid_file_shows_its_declared_name(tmp_path: pathlib.Path) -> None:
    """The declared name, not the filename: the two can differ."""
    path = tmp_path / "some-other-filename.asm.toml"
    path.write_text(_VALID, encoding="utf-8")

    row = summarize_machine_file(path)
    assert row.name == "lint-and-test"
    assert row.spec != "invalid"


def test_row_validity_covers_the_scripts_bundle(tmp_path: pathlib.Path) -> None:
    """A machine whose `scripts/` reference is missing is flagged, as `machine check` refuses it."""
    f = tmp_path / "runner.asm.toml"
    f.write_text(
        """\
machine = "runner"
version = 1
initial = "go"

[budget]
max_transitions = 5

[states.go]
kind = "tool"
command = ["bash", "scripts/missing.sh"]
timeout_secs = 60
on = { ok = "done", nonzero = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "r"
""",
        encoding="utf-8",
    )
    row = summarize_machine_file(f)
    assert row.name == "runner"
    assert row.spec != "valid" and "issue" in row.spec
