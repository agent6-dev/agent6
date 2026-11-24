# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web children list prints a child's facts as the CLI and the TUI print them."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agent6 import budget
from tests.web import client_assets


def test_a_childs_spend_prints_as_every_surface_prints_dollars() -> None:
    """A spend of $0.049765199999999996 printed with every digit of its float."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise the children client")
    cases = [(0.049765199999999996, False), (0.0004321, True), (0, False), (0, True), (12.5, True)]
    scenario = (
        f"for (const [usd, partial] of {json.dumps(cases)}) console.log(childFacts({{"
        "verdict_order: [], launched: true, spend_usd: usd, usd_partial: partial}));"
    )
    result = subprocess.run(
        [node, "-"],
        input=client_assets.script("children") + scenario,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.split() == [budget.format_usd(usd, partial=p) for usd, p in cases]
