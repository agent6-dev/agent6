# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the agent loop over one execution of a session.

Harness is the loop plus every primitive it calls in a turn.
`Harness` in `loop` drives one execution: provider calls, tool dispatch, guards, gates, snapshots.
What spans executions belongs to `app`, which composes executions into lifecycles.
Harness imports `tools`, `sandbox` and the shared packages, never `app` or `ui`.
The package re-exports nothing: import the submodules directly.
Main modules: `loop`, `provider_call`, `guards`, `finish_gates`, `parallel`, `run_chain`.
"""

from __future__ import annotations
