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


def test_the_crate_tests_pass() -> None:
    """The crate's #[cfg(test)] suite (mountinfo filtering, stream capping)
    runs nowhere else: the gate checked format and lints but never executed
    the boundary binary's own tests."""
    done = _cargo("test")
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("target", ["x86_64-unknown-linux-musl", "aarch64-unknown-linux-musl"])
def test_the_crate_compiles_for_every_target_the_release_builds(target: str) -> None:
    """The wheels bundle a static musl binary per arch, and only the HOST target
    was ever checked here -- so `libc::SYS_chmod`, which arm64 does not have,
    landed in the seccomp filter and broke the arm64 wheel build outright. It
    compiled everywhere the suite looked.

    `clippy` rather than `build`: it runs the whole front end (this is where an
    arch-missing constant fails) without needing a cross-linker. Skipped when
    the target is not installed, since a contributor need not carry both.
    """
    try:
        installed = subprocess.run(
            ["rustup", "target", "list", "--installed"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        pytest.skip("rustup not on PATH")
    if target not in installed.stdout:
        pytest.skip(f"{target} not installed (`rustup target add {target}`)")
    done = _cargo("clippy", "--release", "--locked", "--target", target, "--", "-D", "warnings")
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
