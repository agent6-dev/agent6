# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Task-graph package.

The graph is the persistent representation of a single `agent6 run`. It lives in
the run dir (under the per-repo state dir). The curator owns the task GRAPH files
(graph.jsonl, graph/*.md, cursor.json); mutations from worker / planner / critic /
alignment-guard agents go through the curator over a Unix-domain socket. The main
agent process writes the resume snapshot (loop_state.json), the event log
(logs.jsonl), and transcripts in-process. The run dir stays safe from jailed
commands because it lives OUT of the workspace: jailed commands run on the repo
cwd and the state dir is outside it, so safety comes from the out-of-tree
location, not from curator-exclusivity or a single-writer invariant.

Submodules:
  - `models`:   pydantic models for task nodes and curator intents.
  - `ulid`:     tiny self-contained ULID generator (no new dep).
  - `storage`:  on-disk format, markdown+YAML-frontmatter per node, jsonl
                journal, dot generation, atomic writes via fcntl flock +
                tmp-then-rename.
  - `curator`:  in-process `GraphCurator` (the same logic the subprocess uses;
                exposed directly for fast unit tests that don't want a socket).
  - `ipc`:      length-prefixed JSON request/response protocol and pydantic
                envelopes shared by client and server.
  - `server`:   UDS server loop that hosts a `GraphCurator` per run id.
  - `client`:   blocking client used by the workflow process.
"""
