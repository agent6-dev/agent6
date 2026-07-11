# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run harness interjections: when the loop speaks and what it says.

Each nudge/gate is a threshold (when it fires) plus a directive (the text
injected as a user-role harness message). The loop owns detection and
injection; this module owns the tuning values and the words.
"""

from __future__ import annotations

import hashlib
import re

# No-progress spiral guard (run mode). Observed on mistral-small (2026-07-11):
# nine consecutive verify failures with the IDENTICAL normalized error while
# the worker kept editing the same file, spending a third of the run's budget
# repeating one failure. The guard fires only on that pathological pattern
# (N consecutive fails, one signature), so a healthy run never pays for it:
# a green verify or a DIFFERENT failure (real progress through the error
# list) resets the streak. Signatures ignore line numbers, addresses, and
# durations so cosmetic drift between otherwise-identical failures does not
# defeat the detector.
NO_PROGRESS_NUDGE_AFTER = 4
NO_PROGRESS_ESCALATE_AFTER = 7
# Third stage: measured (guard2 waves, n=14) -- the detector fired on exactly
# the doomed runs and never on a healthy one, but nudged runs still burned to
# the iteration cap at score 0. Ten consecutive identical failures (both
# nudges delivered and unheeded) is past any observed recovery; stop the run
# honestly instead of burning the remaining budget on a proven non-strategy.
NO_PROGRESS_STOP_AFTER = 10

# Tool-error spiral guard (run mode). Distinct from the verify streak: this
# counts consecutive tool calls that raise the SAME error (name + error text
# with digits stripped, so a runaway that varies its args but trips the same
# "arguments not valid JSON" / "pattern too long" error still accumulates).
# Observed on SWE-bench: kimi re-issuing malformed grep calls until the run
# timed out. Any successful tool call, or a different error, resets it.
TOOL_ERROR_NUDGE_AFTER = 3
TOOL_ERROR_ESCALATE_AFTER = 5
TOOL_ERROR_STOP_AFTER = 8

TOOL_ERROR_NUDGE = (
    "[harness tool-error] The same tool call has failed several times in a"
    " row with the SAME error. The call itself is wrong (malformed arguments,"
    " a bad path, or the wrong tool), not the code. Stop repeating it: re-read"
    " the tool's requirements, fix the call shape, or use a different tool."
)
TOOL_ERROR_ESCALATION = (
    "[harness tool-error] The identical tool error persists. Do not send this"
    " call again. Switch approach entirely: a smaller/simpler call, a"
    " different tool, or proceed with what you already have."
)


# A verify command that exited nonzero almost instantly with one of these
# signatures did not RUN the tests -- the runner itself is absent/broken.
# Treating that as a normal red misleads the model into "fixing" passing code
# or finishing on an unchecked patch (observed on SWE-bench sympy testbeds:
# `python -m pytest` with pytest absent, exit 1 in 0.0s, across three models).
_VERIFY_DEAD_SIGNATURES = (
    "no module named pytest",
    "no module named _pytest",
    "no module named nose",
    "command not found",
    "no such file or directory",
    "can't open file",
    "is not recognized as an internal or external command",
    "modulenotfounderror",
    "importerror while loading conftest",
)

VERIFY_BROKEN_NUDGE = (
    "[harness verify-broken] The verify command exited immediately WITHOUT"
    " running any tests -- its output indicates the test runner itself is"
    " missing or misconfigured, not that tests failed. Do NOT treat this as a"
    " real test failure and do NOT change working code to satisfy it. Find the"
    " project's actual test command yourself (check setup.cfg / tox.ini /"
    " pyproject for the test config, a bin/test or runtests script, or the"
    " right module) and run it with run_command, then continue."
)


def verify_did_not_run(stdout_tail: str, stderr_tail: str, duration_s: float) -> bool:
    """True when a FAILED verify almost certainly did not execute any tests
    (the runner is absent), so the loop can flag it instead of passing the
    blind failure to the model. Requires a fast exit to avoid flagging a real
    suite that happens to import-error deep in a long run."""
    if duration_s > 3.0:
        return False
    blob = f"{stdout_tail}\n{stderr_tail}".lower()
    return any(sig in blob for sig in _VERIFY_DEAD_SIGNATURES)


def tool_error_signature(name: str, error_text: str) -> str:
    """Stable signature of a tool error, insensitive to varying numbers so a
    runaway that changes its args but trips the same error still matches."""
    return f"{name}:{re.sub(r'[0-9]+', '#', error_text)[:200]}"


NO_PROGRESS_NUDGE = (
    "[harness no-progress] The verify command has failed several times in a"
    " row with the SAME error; your edits are not changing the outcome. Stop"
    " editing. Re-read the failure output above and the code path it"
    " exercises, state the root cause in one sentence, then make ONE fix"
    " aimed at that cause, not at the symptom."
)

NO_PROGRESS_ESCALATION = (
    "[harness no-progress] The identical verify failure persists after"
    " further edits. Step back: re-read the relevant part of the spec and"
    " the failing test. If earlier edits may have made things worse, restore"
    " a file's last committed state (read it with `git show HEAD:<path>`,"
    " then apply_edit it back) and make one minimal fix for the stated root"
    " cause."
)

_SIG_NOISE = re.compile(r"line \d+|0x[0-9a-fA-F]+|\d+\.\d+s\b|:\d+:|/tmp/\S+|\bin \d+(\.\d+)?s\b")


def verify_failure_signature(stdout_tail: str, stderr_tail: str) -> str:
    """Stable hash of a verify failure, insensitive to cosmetic drift."""
    tail = f"{stdout_tail}\n{stderr_tail}".strip()[-800:]
    digest = hashlib.md5(
        _SIG_NOISE.sub("#", tail).encode("utf-8", "replace"), usedforsecurity=False
    )
    return digest.hexdigest()


# Opt-in spec-recheck finish gate ([workflow].spec_recheck_on_finish).
# Measured motivation (bench/coreagent eventflow, 2026-07-11): when the
# committed suite covers only a subset of the spec, models finish on the
# first green verify with requirements unmet; injecting a re-check directive
# via a skill raised scores on every model tested (haiku 0.907->0.960 with
# the variance collapsing to zero). This gate is the same mechanism as a
# one-turn native bounce: the FIRST finish_run over a green verify is
# revoked once with the directive below. Off by default until the A/B
# quantifies the cost on tasks whose suite IS the full spec.
SPEC_RECHECK_NUDGE = (
    "[harness spec-check] Verify is green, but the test suite may cover only"
    " part of the requirements. Before finishing: re-read the task and its"
    " spec, check EACH stated requirement against your implementation, and"
    " fix anything unmet. Then call finish_run again."
)

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


# Silent finish before any work (run mode). Observed on SWE-bench with
# kimi-k2.7: the model answered the problem statement in PROSE at iteration
# 2 (a chat-tuned habit), no edit or verify had happened, and the loop
# accepted it as an implicit finish -- the whole run ended patchless with
# the budget unspent. An EARLY prose turn (first iterations) on an untouched
# tree is a stall, not a finish; steer back to the tools a bounded number of
# times. Later prose finishes stay honored: a run that read its fill and
# answers in prose is the legitimate implicit-finish path.
SILENT_NO_WORK_PATIENCE = 2
SILENT_NO_WORK_NUDGE = (
    "[harness] You replied with prose, but no tool call, and you have not"
    " changed anything yet. Text alone cannot finish this task. Use your"
    " tools: read_file/grep/outline to explore, apply_edit or apply_patch to"
    " change code, run_verify_command to check. If you are truly blocked,"
    " call finish_run and say why."
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
# "State the rule, not the instance": measured on orchard leg 3 (FINDINGS #2
# day 3) — a store that spelled the house convention in words transferred to
# a new computation; a store carrying only the formula it was first seen in
# did not.
MEMORY_FLIP_NUDGE = (
    "[harness memory] Verify just went green after failing. If the fix rested"
    " on a durable non-obvious fact about this repository (a generated file, a"
    " hidden coupling, a convention), record that fact now with"
    " add_memory(scope, body) so future runs skip the rediscovery. State the"
    " general rule in words, not just the instance you fixed. If the failure"
    " was ordinary, carry on."
)

MEMORY_FINISH_NUDGE = (
    "[harness memory] finish_run deferred once: verify failed earlier in this"
    " run before going green, and nothing was recorded for future runs. If the"
    " root cause was a durable non-obvious fact about this repository (a"
    " generated file, a hidden coupling, a convention), record it with"
    " add_memory(scope, body), stating the general rule in words rather than"
    " one instance, then call finish_run again. If nothing qualifies, call"
    " finish_run again immediately."
)


def ends_with_question(text: str) -> bool:
    """Best-effort: the model's prose ends by asking the operator something. The
    last non-empty line ending in '?' catches the common 'Should I proceed?' /
    'Which option do you want?' close that a model writes instead of calling
    ask_user."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].endswith("?")
