# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail crate is held to the same standard as the Python, by the same gate.

`agent6-jail` is the security boundary; its formatting, lints and tests run inside the
suite everyone already runs rather than through commands an operator has to remember.
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
    """The crate's #[cfg(test)] suite (mountinfo filtering, stream capping) runs in the gate.

    It runs nowhere else; format and lints alone never execute the boundary binary's own tests.
    """
    done = _cargo("test")
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("target", ["x86_64-unknown-linux-musl", "aarch64-unknown-linux-musl"])
def test_the_crate_compiles_for_every_target_the_release_builds(target: str) -> None:
    """The crate compiles for every wheel target, not only the host.

    The wheels bundle a static musl binary per arch, and a constant one arch lacks
    (`libc::SYS_chmod` on arm64) in the seccomp filter breaks that wheel's build. `clippy`
    rather than `build`: it runs the whole front end, where an arch-missing constant fails,
    without a cross-linker. Skipped when the target is not installed.
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
    """The bundled binary every jail-invariant test loads is as fresh as the crate source.

    Every test outside the smoke file goes through `run_in_jail`, which loads the bundled
    binary; a stale bundle makes the suite exercise the previous boundary, so green must mean
    green for the code in the tree.
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
