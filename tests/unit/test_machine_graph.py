# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.machine.graph — mermaid + dot rendering."""

from __future__ import annotations

from pathlib import Path

from agent6.machine._semantics import load_machine
from agent6.machine.graph import render_dot, render_mermaid
from tests.unit.test_machine_model import VALID_MACHINE


def _spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    path = tmp_path / "m.asm.toml"
    path.write_text(VALID_MACHINE, encoding="utf-8")
    return load_machine(path)


def test_mermaid_has_entry_and_terminal(tmp_path: Path) -> None:
    out = render_mermaid(_spec(tmp_path))
    assert out.startswith("stateDiagram-v2\n")
    assert "[*] --> poll" in out
    assert "halt --> [*]" in out
    assert "scan --> have_items: ok" in out


def test_dot_has_start_point_and_terminal_shape(tmp_path: Path) -> None:
    out = render_dot(_spec(tmp_path))
    assert out.startswith('digraph "item-classifier" {')
    assert "__start__ [shape=point];" in out
    assert '"halt" [shape=doublecircle];' in out
    assert '__start__ -> "poll";' in out
    assert '"scan" -> "have_items" [label="ok"];' in out
