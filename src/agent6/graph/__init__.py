# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Task-graph package.

The graph is the persistent representation of a single `agent6 run`. It lives at
`.agent6/runs/<run-id>/` and is owned exclusively by the curator subprocess. All
mutations from worker / planner / critic / alignment-guard agents go through the
curator over a Unix-domain socket; the worker pool's landlock policy denies
writes to `.agent6/` so this is a kernel-enforced invariant, not a convention.

Submodules:
  - `models`:   pydantic models for nodes, intents, snapshots, resume diffs.
  - `ulid`:     tiny self-contained ULID generator (no new dep).
  - `storage`:  on-disk format — markdown+YAML-frontmatter per node, jsonl
                journal, dot generation, atomic writes via fcntl flock +
                tmp-then-rename.
  - `curator`:  in-process `GraphCurator` (the same logic the subprocess uses;
                exposed directly for fast unit tests that don't want a socket).
  - `ipc`:      length-prefixed JSON request/response protocol and pydantic
                envelopes shared by client and server.
  - `server`:   UDS server loop that hosts a `GraphCurator` per run id.
  - `client`:   blocking client used by the workflow process.
"""
