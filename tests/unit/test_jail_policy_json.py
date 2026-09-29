# SPDX-License-Identifier: Apache-2.0
"""Contract test for the policy JSON the Python side sends the launcher.

The launcher serde-defaults a missing `memory_limit_mb`, so a Python side that stops sending it
would not fail there; the field is always present, the 0 opt-out included.
"""

from __future__ import annotations

from pathlib import Path

from agent6.kinds import JailPolicy
from agent6.sandbox.jail import _policy_spec  # pyright: ignore[reportPrivateUsage]


def _fields(policy: JailPolicy) -> dict[str, object]:
    return _policy_spec(policy)


def test_policy_json_carries_the_uncapped_default_memory_limit(tmp_path: Path) -> None:
    """0 is off, matching [sandbox].memory_limit_mb, so both sides agree on the default."""
    fields = _fields(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",)))
    assert fields["memory_limit_mb"] == 0


def test_policy_json_carries_explicit_and_zero_memory_limit(tmp_path: Path) -> None:
    assert (
        _fields(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), memory_limit_mb=512))[
            "memory_limit_mb"
        ]
        == 512
    )
    assert (
        _fields(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), memory_limit_mb=0))[
            "memory_limit_mb"
        ]
        == 0
    )
