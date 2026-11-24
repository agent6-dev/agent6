"""The table keeps replicate spread and missing evidence visible."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


def test_table_shows_spread_including_capped_runs_and_missing_metrics(
    tmp_path: pathlib.Path,
) -> None:
    base = dict(
        revision="tree",
        model="local/m",
        effort="medium",
        max_plan_points=2,
        isolation="full",
        arm="short",
        split="eval",
        task="bugs",
        stop_reason="finish_session",
        patch_calls=2,
        patch_failures=1,
        edit_calls=3,
        edit_failures=0,
        passed=True,
        turns=3,
        input_tokens=20,
        output_tokens=2,
        error=None,
    )
    rows = [
        base,
        base
        | dict(
            stop_reason="budget_exhausted",
            passed=False,
            patch_failures=0,
            turns=7,
            input_tokens=None,
        ),
        {k: base[k] for k in ("revision", "model", "effort", "max_plan_points", "isolation")}
        | dict(arm="control", task="textkit", split="dev", error="no journal", passed=None),
    ]
    data = tmp_path / "results.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = subprocess.run(
        [sys.executable, "-m", "bench.tools.table", data],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "eval | bugs | short | 2 | 1/2 | 1 | 0 | 1/4; 0.5 [0,1] | 0/6; 0 [0,0] | "
        "5 [3,7] | 20 [20,20] (1/2)" in result.stdout
    )
    assert "dev | textkit | control | 1 | 0/0 | 0 | 1 | ? | ? | ? | ? | ?" in result.stdout
