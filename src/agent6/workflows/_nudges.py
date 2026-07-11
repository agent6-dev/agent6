# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run harness interjections: when the loop speaks and what it says.

Each nudge/gate is a threshold (when it fires) plus a directive (the text
injected as a user-role harness message). The loop owns detection and
injection; this module owns the tuning values and the words.
"""

from __future__ import annotations

# Plan-mode wrap-up: nudge once the budget fraction drops below the threshold,
# or after this many iterations without having finished (or even started) a
# plan at all. A plan rarely needs more than a handful of reads.
PLAN_BUDGET_NUDGE_BELOW = 0.35
PLAN_NUDGE_AFTER_ITERS = 12

# Task finish-gate: when the worker has broken the run into subtasks, don't let
# it finish (or silently stop) while subtasks are still open -- re-prompt with
# the open list instead. A weak model on a long task tends to quit early with
# work pending (observed live: silent_finish at iter 7 with 7 subtasks open).
# Capped so a worker that genuinely can't close a task (and won't mark it
# obsolete/skipped) can't bounce the loop forever; after the cap the finish is
# honoured. Only SUBTASKS gate -- the always-pending auto-root would deadlock.
TASK_FINISH_PATIENCE = 3

# Opt-in hard finish gate (`require_verify_to_finish`): how many times to bounce a
# finish_run over a red/stale verify before honouring it anyway (as an honest
# all_passed=False "finished"). Bounded so a task that genuinely can't pass can't
# pin the loop to the iteration cap.
VERIFY_FINISH_PATIENCE = 3
VERIFY_FINISH_GATE = (
    "[harness] finish_run refused: the verify command is not green"
    " (require_verify_to_finish is on). Run run_verify_command, fix what it"
    " reports, and only call finish_run once it passes. If this task genuinely"
    " cannot pass verify, say so and stop rather than finishing."
)

# verify-settled completion (run mode). A non-metric run has no positive "done"
# signal, clean exit depends on the worker volunteering finish_run, and a weak
# worker keeps re-running read-only commands after success (Kimi K2.6 observed:
# 128 iters when done at ~45). Once verify has passed, count iterations that
# make no progress (no new commit + no edit): nudge to finish at the first
# threshold, hard-stop at the second. NOT "green verify = instant stop", verify
# fires per-edit and is often lenient, so green-but-still-editing must continue.
# Thresholds are deliberately generous: the failure mode is only a little wasted
# budget on an already-done run, whereas a too-tight window could cut off a
# worker still reading toward its next edit in a big multi-file change.
VERIFY_SETTLED_NUDGE_AFTER = 3
VERIFY_SETTLED_STOP_AFTER = 6

VERIFY_SETTLED_NUDGE = (
    "[harness settled] Your recent changes are committed and your last turns"
    " made no new changes (no commit, no edit). If the task is complete, call"
    " finish_run now with a short summary. If not, make a concrete edit toward"
    " what remains — do not keep re-running read-only commands."
)

# A non-metric `run` injects a one-shot wrap-up directive when the budget gets
# low. Observed live (Kimi K2.6): the worker solves the task, never re-runs
# verify, never calls finish_run, and burns the remaining budget on read-only
# commands; the verify-settled detector cannot engage without a green verify.
RUN_BUDGET_NUDGE_BELOW = 0.25

RUN_BUDGET_NUDGE = (
    "[harness budget] You are running low on budget. Run `run_verify_command`"
    " NOW. If it passes, call finish_run immediately with a short summary. If"
    " it fails, fix ONLY the smallest blocking issue, re-verify, and finish."
    " Do not run any other commands."
)

# Gateless variant (no verify command this run): there is nothing to verify, so
# steer straight to finish_run.
RUN_BUDGET_NUDGE_GATELESS = (
    "[harness budget] You are running low on budget. Call finish_run NOW with a"
    " short summary of what you changed. Do not run any other commands."
)

PLAN_BUDGET_NUDGE = (
    "[harness budget] You are running low on token budget and have NOT yet"
    " called finish_planning. Stop reading and reasoning now and call"
    " finish_planning immediately with the best plan you have — a concise, even"
    " rough, plan that is actually delivered is far more useful than an"
    " exhaustive one you never emit. Do not call any other tool first."
)


QUESTION_NUDGE = (
    "[harness] You ended your turn by asking a question, but you did not call a"
    " tool, so nobody can answer it and the run is about to end. If you need the"
    " operator's input, call `ask_user` with the question (a front-end delivers"
    " it and returns the answer). If you have enough to proceed, do the work. If"
    " you are truly done, call `finish_run`. Do not just write the question as"
    " text again."
)


# Cross-run memory write nudges. Measured (bench/longhorizon FINDINGS #2):
# 46 legs across 2 models produced ZERO unprompted add_memory calls, so the
# <memories> header alone never causes writes. Surface the tool at the two
# moments a durable discovery is actually in hand: the first red-to-green
# verify flip (advisory, free) and the first finish_run after such a recovery
# (deferred once, the backstop). Each fires at most once per run, only in run
# mode with a memory store wired, and only while the worker has recorded
# nothing; a run whose verify never failed is never nudged.
MEMORY_FLIP_NUDGE = (
    "[harness memory] Verify just went green after failing. If the fix rested"
    " on a durable non-obvious fact about this repository (a generated file, a"
    " hidden coupling, a convention), record that fact now with"
    " add_memory(scope, body) so future runs skip the rediscovery. If the"
    " failure was ordinary, carry on."
)

MEMORY_FINISH_NUDGE = (
    "[harness memory] finish_run deferred once: verify failed earlier in this"
    " run before going green, and nothing was recorded for future runs. If the"
    " root cause was a durable non-obvious fact about this repository (a"
    " generated file, a hidden coupling, a convention), record it with"
    " add_memory(scope, body), then call finish_run again. If nothing"
    " qualifies, call finish_run again immediately."
)


def ends_with_question(text: str) -> bool:
    """Best-effort: the model's prose ends by asking the operator something. The
    last non-empty line ending in '?' catches the common 'Should I proceed?' /
    'Which option do you want?' close that a model writes instead of calling
    ask_user."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].endswith("?")
