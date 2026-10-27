# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve the external binaries used by the `collector` tests.

AGENT6_OTELCOL is a test-only path to otelcol-contrib, checked before PATH.
Promtool is resolved only from PATH.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import pytest


def require_collector(binary: str = "otelcol-contrib") -> str:
    """Return the executable path or skip with the missing binary's name."""
    if binary == "otelcol-contrib":
        override = os.environ.get("AGENT6_OTELCOL", "")
        if override and (found := shutil.which(override)):
            return str(pathlib.Path(found).resolve())
    if found := shutil.which(binary):
        return str(pathlib.Path(found).resolve())
    if binary == "otelcol-contrib":
        pytest.skip("otelcol-contrib is not on PATH or at AGENT6_OTELCOL")
    pytest.skip(f"{binary} is not on PATH")
