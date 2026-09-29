# SPDX-License-Identifier: Apache-2.0
"""A launcher that hangs past the timeout is killed with its process group and reads as rc 124.

A fake launcher script backgrounds a process that inherits the stdout pipe and blocks, so
communicate() never sees EOF.
"""

from __future__ import annotations

import stat
import time
from pathlib import Path

import pytest

from agent6.kinds import JailPolicy
from agent6.sandbox import jail


def _write_fake_launcher(tmp_path: Path) -> Path:
    # A child holds stdout open and blocks forever; the marker file shows whether the group died.
    marker = tmp_path / "grandchild_alive"
    script = tmp_path / "fake-jail.sh"
    script.write_text(
        "#!/bin/sh\n"
        # The grandchild keeps fd 1 open and refreshes a marker, so its survival is detectable.
        f"( while true; do echo alive > '{marker}'; sleep 0.2; done ) &\n"
        # parent launcher blocks forever -> communicate() must time out.
        "sleep 600\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return script


def test_jail_timeout_returns_124_and_kills_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_launcher(tmp_path)
    monkeypatch.setattr(jail, "locate_jail_binary", lambda: fake)

    def _policy_spec(policy: JailPolicy) -> dict[str, object]:
        return {}

    monkeypatch.setattr(jail, "_policy_spec", _policy_spec)

    policy = JailPolicy(
        cwd=tmp_path,
        argv=("/bin/true",),
        isolation="strict",
        network="none",
        # launcher sleeps 600s; timeout is timeout_s + 5.0, so keep it tiny.
        timeout_s=0.5,
    )

    start = time.monotonic()
    result = jail.run_in_jail(policy)
    elapsed = time.monotonic() - start

    # Must return the documented timeout contract, not raise.
    assert result.returncode == 124
    assert result.argv == ("/bin/true",)
    # Bounded: must not block for the full 600s sleep.
    assert elapsed < 30.0

    # The grandchild was reaped by the group kill: the marker stops being refreshed.
    marker = tmp_path / "grandchild_alive"
    time.sleep(1.0)
    if marker.exists():
        first = marker.stat().st_mtime
        time.sleep(1.0)
        second = marker.stat().st_mtime
        assert first == second, "grandchild still alive after group kill"
