# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Define the presentation seam `ui/cli` injects into machine run and create."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

from agent6 import kinds
from agent6.app import reporter as app_reporter
from agent6.app.machine import _preflight
from agent6.config import Config
from agent6.machine import ToolState

# Resolves a tool-network refusal at a TTY: the fixed (cfg, isolation), or an exit code.
ResolveNetworkFix = Callable[
    [
        pathlib.Path,
        _preflight.NetworkRefusal,
        Config,
        kinds.IsolationLevel,
        list[ToolState],
        pathlib.Path,
        dict[str, Any],
    ],
    "int | tuple[Config, kinds.IsolationLevel]",
]


@dataclasses.dataclass(frozen=True, slots=True)
class MachineFrontend:
    """Hold the callables machine run and create drive, injected by `ui/cli`.

    The thin sibling of `SessionFrontend`: machines are headless-first, so a `Reporter` for
    status output and one interactive callback suffice. `create_machine` uses only `reporter`.

    Attributes:
        reporter: The two output channels.
        resolve_network_fix: Explains a tool-network refusal and offers the config fix.
    """

    reporter: app_reporter.Reporter
    resolve_network_fix: ResolveNetworkFix
