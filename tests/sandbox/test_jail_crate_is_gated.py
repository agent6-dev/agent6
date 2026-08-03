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


def test_the_binary_the_suite_runs_is_not_older_than_the_sources() -> None:
    """Every jail-invariant test outside the smoke file goes through
    `run_in_jail`, which loads the BUNDLED binary -- and only the smoke file's
    fixture checked freshness, against the `target/` build it prefers.

    So editing main.rs and running the suite exercised the PREVIOUS boundary:
    observed on a mount fix that passed by hand and failed under pytest against
    the stale bundle. Green must mean green for the code in the tree.
    """
    from agent6.sandbox.jail import locate_jail_binary

    binary = locate_jail_binary()
    if binary is None:
        pytest.skip("no jail binary bundled or on PATH")
    sources = [*_CRATE.glob("src/*.rs"), _CRATE / "Cargo.toml"]
    newest = max((p.stat().st_mtime for p in sources if p.is_file()), default=0.0)
    assert newest <= binary.stat().st_mtime, (
        f"{binary} predates the jail sources: rebuild (`uv sync`, or `cargo build --release`"
        " and point AGENT6_JAIL_BIN at it) before trusting these tests"
    )
