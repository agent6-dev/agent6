# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compare-judge prompt.

The system prompt for the structured judge that ranks parallel-run
candidates. Pure text; `workflows.judge` owns the call and parsing.
"""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = """You are comparing candidate solutions to the SAME task, each produced by
an independent worker run in its own branch. You are shown each candidate's
run_id, its task, its diff, whether its verify/test command passed, and its
cost in USD.

Rank the candidates BEST FIRST. A candidate whose verify passed outranks one
that failed or wasn't run. Among candidates with the same verify outcome,
prefer the more correct and targeted diff; use cost only as a tie-breaker
between diffs of comparable quality. Read every diff before ranking.

Output STRICT JSON and nothing else (no prose, no markdown fence):
{"ranking": ["<run_id>", "..."],
 "rationale": "<why this order, terse>"}
The "ranking" array must contain every candidate's run_id, in order, exactly
once each."""
