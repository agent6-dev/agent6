# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The task graph: one session's persistent plan.

The curator owns the graph files in the session dir (graph.jsonl, graph/*.md, cursor.json);
the worker, the planner and the operator's steering all mutate through the one in-process
`GraphCurator`. The `worker.lock` flock keeps one writer, and the per-mutation flock also
serialises fork's copy. The session dir lives under the state base, which the jail masks
under `full`.
"""

from __future__ import annotations
