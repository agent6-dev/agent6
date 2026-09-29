# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shipped bench content stays loadable.

No bench/machines example or bench config literal carries a field the schema has deleted; configs
load under extra="forbid", so one stale name refuses the whole file.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6.machine import MachineError, load_machine

_REPO = pathlib.Path(__file__).resolve().parents[2]
_BENCH = _REPO / "bench"

_EXPECTED_INVALID: set[str] = set()


def _example_tomls() -> list[pathlib.Path]:
    tomls = sorted((_BENCH / "machines").glob("*/*.asm.toml"))
    assert tomls, f"no machine examples under {_BENCH / 'machines'}"
    return tomls


@pytest.mark.parametrize("toml", _example_tomls(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shipped_machine_examples_validate(toml: pathlib.Path) -> None:
    if toml.name in _EXPECTED_INVALID:
        with pytest.raises(MachineError):
            load_machine(toml)
    else:
        load_machine(toml)


# Fields the budget redesign and the git.allow_* removal deleted; recorded outputs are exempt.
_DELETED_FIELDS = (
    "best_effort_usd_limit",
    "max_input_tokens",
    "max_output_tokens",
    "allow_push",
    "allow_force",
    "allow_history_rewrite",
)
_HISTORICAL = {"swebench/FINDINGS.md"}
_HISTORICAL_PREFIXES = ("machines/_create-bench/",)
_SCANNED_SUFFIXES = {".sh", ".toml", ".md", ".py"}


def test_bench_carries_no_deleted_config_fields() -> None:
    offenders: list[str] = []
    for path in sorted(_BENCH.rglob("*")):
        rel = path.relative_to(_BENCH).as_posix()
        if not path.is_file() or path.suffix not in _SCANNED_SUFFIXES or rel in _HISTORICAL:
            continue
        if rel.startswith(_HISTORICAL_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8")
        for field in _DELETED_FIELDS:
            if field in text:
                offenders.append(f"{rel}: {field}")
    assert not offenders, "deleted config fields still referenced under bench/:\n" + "\n".join(
        offenders
    )
