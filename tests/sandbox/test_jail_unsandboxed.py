# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one-shot `none` path: a plain subprocess that still keeps the policy's resource caps."""

from __future__ import annotations

import pathlib
import sys
import time

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_jobs


@pytest.mark.parametrize("cap_mib", [64, 0])
def test_the_one_shot_none_path_applies_the_memory_cap(
    tmp_path: pathlib.Path, cap_mib: int
) -> None:
    """`memory_limit_mib` bounds a one-shot command under `none` as the launcher bounds the run's.

    The plain subprocess ran with no rlimit, so a machine tool state or a workspace setup
    command under `none` allocated past the cap the `none` warning says still applies.
    """
    result = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "bytearray(400 << 20)"),
            isolation="none",
            network="host",
            memory_limit_mib=cap_mib,
            timeout_s=60.0,
        )
    )
    if cap_mib:
        assert result.returncode != 0 and "MemoryError" in result.stderr
    else:
        assert result.returncode == 0


@pytest.mark.parametrize("cap_mib", [64, 0])
def test_the_detached_none_path_applies_the_memory_cap(
    tmp_path: pathlib.Path, cap_mib: int
) -> None:
    """`memory_limit_mib` bounds a background command under `none` as it bounds a one-shot one.

    The detached `Popen` set no rlimit, so `run_command(background=true)` under `none` ran past
    the cap when no jail session served the run.
    """
    job = jail_jobs.start_in_jail(
        kinds.JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "bytearray(400 << 20)"),
            isolation="none",
            network="host",
            memory_limit_mib=cap_mib,
            timeout_s=60.0,
        ),
        outcome_dir=tmp_path / "outcome",
    )
    deadline = time.monotonic() + 60
    while (status := job.status()).running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not status.running
    assert (status.returncode != 0) is bool(cap_mib)
