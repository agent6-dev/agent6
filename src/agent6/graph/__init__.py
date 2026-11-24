# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The task graph: one session's persistent plan.

A shared package: any layer may import it, and it imports no layer.
The worker, the planner and the operator's steering mutate it through one `GraphCurator`.
The curator owns the graph files in the session dir: `graph.jsonl`, `graph/*.md`, `cursor.json`.
The `worker.lock` flock keeps one writer, and the per-mutation flock also serialises fork's copy.
The session dir lives under the state base, which the jail masks under `full`.
Main modules: `curator`, `models`, `storage`, `order`, `replay`.
"""

from __future__ import annotations
