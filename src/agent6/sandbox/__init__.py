# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Host and kernel detection, the Landlock ABI probe, and the jail launcher.

The bottom layer: every layer may import it, and it imports only root-level modules.
Main modules: `jail`, `jail_session`, `detect`, `container`, `landlock`.
"""

from __future__ import annotations
