# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one pydantic ConfigDict every config model shares."""

from __future__ import annotations

from pydantic import ConfigDict

MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
