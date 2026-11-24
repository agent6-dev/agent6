# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Checks for the project's type policy."""

from __future__ import annotations

import pathlib
import subprocess
import sys


def test_an_untyped_dependency_needs_a_local_exception(tmp_path: pathlib.Path) -> None:
    """An exception for the schema tests hid every untyped third-party import."""
    source = tmp_path / "consumer.py"
    source.write_text("import jsonschema\n", encoding="utf-8")
    config = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"

    result = subprocess.run(
        [sys.executable, "-m", "pyrefly", "check", "--config", config, source],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result
    assert "[untyped-import]" in result.stdout
