# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the agent loop over a session.

`Harness` (`loop.py`) drives one execution: the provider calls, the tool dispatch, the guards
and gates, and the snapshot a resume re-enters; sibling modules hold its helpers.
`system_prompt_for` and `model_exchange_for` expose the run's first exchange to
`agent6 prompt show`.
"""

from __future__ import annotations

from agent6.harness import prompt_preview, repo_context

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
