"""The benchmark accepts one bounded report from the registered scorer."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

import pytest
from bench.tools import grading, journals


def _report() -> dict[str, Any]:
    return dict(
        task="textkit",
        score=0.0,
        cases_passed=0,
        cases_total=30,
        components_passed=0,
        components_total=5,
        components={
            k: {"passed": 0, "total": n}
            for k, n in zip(
                (
                    "normalize_whitespace",
                    "word_count",
                    "most_common_words",
                    "wrap_text",
                    "to_snake_case",
                ),
                (5, 5, 7, 6, 7),
                strict=True,
            )
        },
    )


def _grade(where: pathlib.Path, output: bytes) -> dict[str, Any]:
    workspace = where / "grading"
    workspace.mkdir()
    scorer = where / "textkit/grade.py"
    scorer.parent.mkdir()
    scorer.write_bytes(output)
    # A fixed local command exercises the runner's pipes without executing the submission.
    binary = where / "fixture"
    binary.write_text(
        f"#!{sys.executable} -I\nimport sys\nsys.stderr.write('fixture diagnostic\\n')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
    )
    binary.chmod(0o700)
    return grading.grade(binary, where / "run.toml", workspace, os.environ.copy(), scorer)


def test_grader_stderr_is_private(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert _grade(tmp_path, json.dumps(_report()).encode())["cases_total"] == 30
    assert "fixture diagnostic" not in capfd.readouterr().err
    assert (tmp_path / "grader.stderr").read_bytes() == b"fixture diagnostic\n"


@pytest.mark.parametrize(
    "output",
    [
        json.dumps(_report()).encode() + b'\n{"cases_passed":1,"cases_total":1}\n',
        b"[]",
        b"not JSON",
        b"\xff",
        json.dumps(_report() | {"cases_total": 1, "cases_passed": 1}).encode(),
        json.dumps(_report() | {"task": "bugs"}).encode(),
        json.dumps(_report() | {"score": 1.0}).encode(),
        json.dumps(_report() | {"components": {}}).encode(),
        b" " * 65537 + json.dumps(_report()).encode(),
    ],
    ids=(
        "two-reports",
        "array",
        "text",
        "encoding",
        "wrong-total",
        "wrong-task",
        "wrong-score",
        "components",
        "oversize",
    ),
)
def test_invalid_grader_output_is_a_measurement_error(
    tmp_path: pathlib.Path, output: bytes
) -> None:
    with pytest.raises(journals.MeasurementError):
        _grade(tmp_path, output)


def test_official_import_error_is_a_scored_failure(tmp_path: pathlib.Path) -> None:
    result = _report() | {"components": {}, "cases_total": 1, "import_error": "missing module"}
    assert _grade(tmp_path, json.dumps(result).encode()) == {
        "passed": False,
        "cases_total": 1,
        "cases_passed": 0,
    }
