# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail crate is held to the same standard as the Python, by the same gate.

`agent6-jail` IS the security boundary, and CI only ever built it -- so its
formatting drifted and its lints went unread. Rather than adding two more
commands an operator has to remember, the checks run inside the suite everyone
already runs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_CRATE = Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail"

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None, reason="no rust toolchain (the wheel build needs one)"
)


def _cargo(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", *args],
        cwd=_CRATE,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_crate_is_rustfmt_clean() -> None:
    done = _cargo("fmt", "--check")
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_crate_has_no_clippy_warnings() -> None:
    """`-D warnings`: a lint on the boundary binary is not advisory."""
    done = _cargo("clippy", "--release", "--", "-D", "warnings")
    assert done.returncode == 0, done.stdout + done.stderr
