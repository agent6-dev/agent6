# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the agent loop over one execution of a session.

Harness is the loop plus every primitive it calls in a turn.
`Harness` in `loop` drives one execution: provider calls, tool dispatch, guards, gates, snapshots.
What spans executions belongs to `app`, which composes executions into lifecycles.
Harness imports `tools`, `sandbox` and the shared packages, never `app` or `ui`.
`system_prompt_for` and `model_exchange_for` give `agent6 prompt show` the run's first exchange.
Main modules: `loop`, `provider_call`, `guards`, `finish_gates`, `parallel`, `run_chain`.
"""

from __future__ import annotations

from agent6.harness.prompt import preview as prompt_preview
from agent6.harness.prompt import repo_context

ModelExchange = prompt_preview.ModelExchange
agents_md_text = repo_context.agents_md_text
model_exchange_for = prompt_preview.model_exchange_for
system_prompt_for = prompt_preview.system_prompt_for

__all__ = [
    "ModelExchange",
    "agents_md_text",
    "model_exchange_for",
    "system_prompt_for",
]
