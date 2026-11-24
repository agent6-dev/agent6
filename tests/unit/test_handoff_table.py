"""The handoff table keeps failed runs and missing readings in their arm."""

from __future__ import annotations

import json
import pathlib

import pytest


def test_table_shows_acceptance_coverage_and_spread(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from bench.handoff import table

    base = dict(
        split="scripted",
        model="scripted/worker",
        effort="medium",
        revision="tree",
        seed="seed",
        arm="text",
        issues=[],
        all_accepted=True,
        stuck_notices=0,
        turns=10,
        input_tokens=60,
        output_tokens=20,
        cache_read_tokens=40,
        cache_write_tokens=None,
        started_twice=0,
        never_started=0,
        under_two_s=1.5,
        off_arm_calls=0,
    )
    other = base | dict(
        issues=["unreported input_tokens"],
        all_accepted=False,
        turns=4,
        input_tokens=None,
        started_twice=1,
        never_started=2,
        under_two_s=3.5,
    )
    absent = dict.fromkeys(base) | dict(
        arm="id",
        issues=["no parent stop"],
        split="scripted",
        model="scripted/worker",
        effort="medium",
        revision="tree",
        seed="seed",
    )
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (base, other, absent)))
    table.print_set(path)
    text = capsys.readouterr().out
    assert "text | 2 | 1/2 | 1 | 0 [0,0] | 7 [4,10] | 60 [60,60] (1/2)" in text
    assert "0.5 [0,1] | 1 [0,2] | 2.5 [1.5,3.5]" in text
    assert "id | 1 | ? | 1 | ? | ?" in text
