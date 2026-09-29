# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The task graph: one session's persistent plan.

The curator owns the graph files in the session dir (graph.jsonl, graph/*.md, cursor.json);
the worker, the planner and the operator's steering all mutate through the one in-process
`GraphCurator`. The session dir lives outside the workspace, so a jailed command never
reaches it; the `worker.lock` flock keeps one writer, and the curator's own per-mutation
flock guards against a concurrent operator-CLI read or write of the same files.
"""

from __future__ import annotations
