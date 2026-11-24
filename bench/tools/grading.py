"""Run an unchanged coreagent scorer and read its bounded report."""

from __future__ import annotations

import json
import os
import pathlib
import selectors
import subprocess
import time
from typing import Any

from bench.tools import journals

OUTPUT_LIMIT = 64 * 1024
COMPONENTS = {
    "bugs": {
        "clamp": 8,
        "mean": 7,
        "median": 8,
        "gcd": 8,
        "is_prime": 8,
        "roman": 8,
        "running_max": 8,
    },
    "textkit": {
        "normalize_whitespace": 5,
        "word_count": 5,
        "most_common_words": 7,
        "wrap_text": 6,
        "to_snake_case": 7,
    },
}


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse two values for a field in a score report."""
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise journals.MeasurementError("duplicate grader field")
    return fields


def score(task: str, output: bytes) -> dict[str, Any]:
    """Accept one complete report with the registered task's totals."""
    try:
        report = json.loads(output.decode("utf-8"), object_pairs_hook=unique_fields)
    except (ValueError, RecursionError) as exc:
        raise journals.MeasurementError("expected one JSON grader report") from exc
    invalid = journals.MeasurementError(f"invalid {task} grader report")
    counts = {"cases_passed", "cases_total", "components_passed", "components_total"}
    if not isinstance(report, dict) or any(type(report.get(k)) is not int for k in counts):
        raise invalid
    keys = counts | {"score"}
    passed, total = report["cases_passed"], report["cases_total"]
    completed, components = report["components_passed"], report["components_total"]
    if task == "eventflow":
        if total != 25 or components != 8:
            raise invalid
    else:
        expected = COMPONENTS[task]
        keys |= {"task", "components"}
        parts = report.get("components")
        if report.get("task") != task or not isinstance(parts, dict) or components != len(expected):
            raise invalid
        if "import_error" in report:
            keys.add("import_error")
            if not isinstance(report["import_error"], str) or (passed, total, completed, parts) != (
                0,
                1,
                0,
                {},
            ):
                raise invalid
        else:
            if parts.keys() != expected.keys() or total != sum(expected.values()):
                raise invalid
            results: list[int] = []
            for name, n in expected.items():
                part = parts[name]
                if not isinstance(part, dict) or part.keys() != {"passed", "total"}:
                    raise invalid
                p = part["passed"]
                if (
                    type(p) is not int
                    or not 0 <= p <= n
                    or type(part["total"]) is not int
                    or part["total"] != n
                ):
                    raise invalid
                results.append(p)
            if passed != sum(results) or completed != sum(
                p == n for p, n in zip(results, expected.values(), strict=True)
            ):
                raise invalid
    if (
        report.keys() != keys
        or not 0 <= passed <= total
        or not 0 <= completed <= components
        or (passed == total) != (completed == components)
        or type(report["score"]) not in (int, float)
        or report["score"] != round(passed / total, 4)
    ):
        raise invalid
    return {"passed": passed == total, "cases_passed": passed, "cases_total": total}


def grade(
    binary: pathlib.Path,
    config: pathlib.Path,
    workspace: pathlib.Path,
    env: dict[str, str],
    scorer: pathlib.Path,
) -> dict[str, Any]:
    """Pipe the archived grader into a fresh jail. Neither output stream reaches the terminal."""
    report = bytearray()
    deadline = time.monotonic() + 120
    with (
        scorer.open("rb") as script,
        (workspace.parent / "grader.stderr").open("xb") as log,
        subprocess.Popen(
            [binary, "--config", config, "exec", "--new", "python3", "-I", "-", "repo"],
            cwd=workspace,
            env=env,
            stdin=script,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as proc,
        selectors.DefaultSelector() as pipes,
    ):
        assert proc.stdout is not None and proc.stderr is not None
        pipes.register(proc.stdout, selectors.EVENT_READ)
        pipes.register(proc.stderr, selectors.EVENT_READ)
        try:
            while pipes.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not (ready := pipes.select(remaining)):
                    raise journals.MeasurementError("grader timed out")
                for key, _ in ready:
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        pipes.unregister(key.fd)
                    elif key.fileobj is proc.stdout:
                        if len(report) + len(chunk) > OUTPUT_LIMIT:
                            raise journals.MeasurementError("grader stdout exceeded 64 KiB")
                        report.extend(chunk)
                    elif log.tell() + len(chunk) <= OUTPUT_LIMIT:
                        log.write(chunk)
                    else:
                        raise journals.MeasurementError("grader stderr exceeded 64 KiB")
            if proc.wait(timeout=max(0.0, deadline - time.monotonic())) != 0:
                raise journals.MeasurementError("grader failed; see grader.stderr")
        except subprocess.TimeoutExpired as exc:
            raise journals.MeasurementError("grader timed out") from exc
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return score(scorer.parent.name, bytes(report))
