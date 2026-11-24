"""The benchmark accepts one bounded report from the registered scorer."""

from __future__ import annotations

import hashlib
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


def _grade(
    where: pathlib.Path,
    script: bytes,
    fixture: str = (
        "sys.stderr.write('fixture diagnostic\\n')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
    ),
) -> dict[str, Any]:
    repo = where / "repo"
    repo.mkdir()
    scorer = where / "textkit/grade.py"
    scorer.parent.mkdir()
    scorer.write_bytes(script)
    # A fixed local command exercises the runner's pipes without executing the submission.
    binary = where / "fixture"
    binary.write_text(f"#!{sys.executable} -I\nimport sys\n{fixture}")
    binary.chmod(0o700)
    return grading.grade(binary, where / "run.toml", repo, os.environ.copy(), scorer)


def test_grader_gets_only_pipe_stdio_and_unchanged_script_bytes(tmp_path: pathlib.Path) -> None:
    """A script larger than a pipe arrives intact without an open scorer file."""
    script = b"# pipe fixture\r\n" * 16384
    fixture = (
        "import hashlib, os, stat\n"
        "sys.stderr.write('.' * 48000 + '\\n')\n"
        "print(*(stat.S_ISFIFO(os.fstat(fd).st_mode) for fd in (0, 1, 2)), file=sys.stderr)\n"
        "print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest(), file=sys.stderr)\n"
        f"print({json.dumps(_report())!r})\n"
    )
    assert _grade(tmp_path, script, fixture)["cases_total"] == 30
    assert (tmp_path / "grader.stderr").read_text().splitlines() == [
        "." * 48000,
        "True True True",
        hashlib.sha256(script).hexdigest(),
    ]


def test_grader_stderr_is_private(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert _grade(tmp_path, json.dumps(_report()).encode())["cases_total"] == 30
    assert "fixture diagnostic" not in capfd.readouterr().err
    assert (tmp_path / "grader.stderr").read_bytes() == b"fixture diagnostic\n"


def test_grader_refusal_before_reading_stdin_keeps_its_diagnostic(tmp_path: pathlib.Path) -> None:
    fixture = "print('fixture unavailable', file=sys.stderr)\nsys.exit(2)\n"
    with pytest.raises(journals.MeasurementError, match=r"grader failed; see grader\.stderr"):
        _grade(tmp_path, b"# fixture\n" * 32768, fixture)
    assert (tmp_path / "grader.stderr").read_text() == "fixture unavailable\n"


def test_grader_stderr_limit_applies_during_input(tmp_path: pathlib.Path) -> None:
    with pytest.raises(journals.MeasurementError, match="grader stderr exceeded 64 KiB"):
        _grade(tmp_path, b"# fixture\n" * 32768, "sys.stderr.buffer.write(b'.' * 65537)\n")
    assert (tmp_path / "grader.stderr").stat().st_size <= 65536


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
