# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""State machines: declarative, replayable mini-agents.

A shared package: it imports `sandbox` and no layer above it.
The engine drives a machine deterministically over an append-only journal.
The journal gives crash recovery and offline replay.
An `agent` state runs an agent6 loop through an injected runner.
Main modules: `schema`, `semantics`, `engine`, `execution`, `journal`, `live_world`, `graph`.
"""

from __future__ import annotations
