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

# No-progress spiral guard (run mode): N consecutive verify failures sharing
# ONE normalized signature. A green verify or a DIFFERENT failure (progress
# through the error list) resets the streak, so a healthy run never pays for
# it. Signatures ignore line numbers, addresses, and durations, else cosmetic
# drift between identical failures defeats the detector.
# Thresholds + evidence: bench/coreagent/FINDINGS.md.
NO_PROGRESS_NUDGE_AFTER = 4
NO_PROGRESS_ESCALATE_AFTER = 7
# Third stage: both nudges delivered and unheeded, so stop honestly rather
# than spend the rest of the budget on a proven non-strategy.
NO_PROGRESS_STOP_AFTER = 10

# Tool-error spiral guard (run mode). Distinct from the verify streak: this
# counts consecutive tool calls that raise the SAME error (name + error text
# with digits stripped, so a runaway that varies its args but trips the same
# "arguments not valid JSON" / "pattern too long" error still accumulates).
# Any successful tool call, or a different error, resets it.
TOOL_ERROR_NUDGE_AFTER = 3
TOOL_ERROR_ESCALATE_AFTER = 5
TOOL_ERROR_STOP_AFTER = 8

TOOL_ERROR_NUDGE = (
    "[harness tool-error] The same call failed repeatedly with the same"
    " error: the call is wrong, not the code. Fix the call shape or use a"
    " different tool."
)
TOOL_ERROR_ESCALATION = (
    "[harness tool-error] The identical error persists. Do not send this"
    " call again: switch tool or approach, or proceed with what you have."
)

# A streak of ToolDenied refusals (approval policy, the git guard): the call
# was REFUSED, not malformed, so the generic "fix the call shape" text would
# be false and invite pointless reshuffling of the same command.
TOOL_DENIED_NUDGE = (
    "[harness tool-error] Refused by policy, not failure; repeating or"
    " reshaping it cannot change the outcome. Follow the refusal's guidance,"
    " use tools that need no approval, or finish and report."
)


# A verify command that exited nonzero almost instantly with one of these
# signatures did not RUN the tests -- the runner itself is absent/broken.
# Treating that as a normal red misleads the model into "fixing" passing code
# or finishing on an unchecked patch.
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

BASELINE_RED_NOTICE = (
    "[harness] That verify ran on an unmodified tree: the gate was already"
    " failing before your changes. Those failures are not yours; do the task"
    " and note them in your summary."
)

VERIFY_BROKEN_NUDGE = (
    "[harness verify-broken] Verify exited at once without running tests:"
    " the runner is missing or misconfigured, not a real failure. Do not"
    " change working code for it. Find the project's real test command"
    " (setup.cfg, tox.ini, pyproject, bin/test) and run it via run_command."
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
    "[harness no-progress] Verify failed repeatedly with the same error;"
    " your edits are not changing the outcome. Stop editing, state the root"
    " cause in one sentence, then make one fix aimed at it."
)

NO_PROGRESS_ESCALATION = (
    "[harness no-progress] The identical failure persists. Re-read the"
    " failing test; if earlier edits made things worse, restore a file"
    " (`git show HEAD:<path>`, apply_edit it back) and make one minimal fix."
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
# When the committed suite covers only a subset of the spec, models finish on
# the first green verify with requirements unmet; a re-check directive raised
# scores on every model tested (measured: bench/coreagent eventflow). Same
# mechanism as a
# one-turn native bounce: the FIRST finish_session over a green verify is
# revoked once with the directive below. Off by default until the A/B
# quantifies the cost on tasks whose suite IS the full spec.
SPEC_RECHECK_NUDGE = (
    "[harness spec-check] Verify is green but may cover only part of the"
    " requirements. Re-read the task, check each stated requirement, fix"
    " anything unmet, then call finish_session again."
)

# Plan-mode wrap-up: nudge once the budget fraction drops below the threshold,
# or after this many iterations without having finished (or even started) a
# plan at all. A plan rarely needs more than a handful of reads.
PLAN_BUDGET_NUDGE_BELOW = 0.35
PLAN_NUDGE_AFTER_ITERS = 12

# Task finish-gate: when the worker has broken the run into subtasks, don't let
# it finish (or silently stop) while subtasks are still open -- re-prompt with
# the open list instead. A weak model on a long task tends to quit early with
# work pending.
# Capped so a worker that genuinely can't close a task (and won't mark it
# obsolete/skipped) can't bounce the loop forever; after the cap the finish is
# honoured. Only SUBTASKS gate -- the always-pending auto-root would deadlock.
TASK_FINISH_PATIENCE = 3

# Opt-in hard finish gate (`require_verify_to_finish`): how many times to bounce a
# finish_session over a red/stale verify before honouring it anyway (as an honest
# all_passed=False "finished"). Bounded so a task that genuinely can't pass can't
# pin the loop to the iteration cap.
VERIFY_FINISH_PATIENCE = 3
VERIFY_FINISH_GATE = (
    "[harness] finish_session refused: verify is not green"
    " (require_verify_to_finish). Fix what it reports and finish once it"
    " passes; if the task genuinely cannot pass, say so and stop."
)

# verify-settled completion (run mode). A non-metric run has no positive "done"
# signal, clean exit depends on the worker volunteering finish_session, and a weak
# worker keeps re-running read-only commands after success. Once verify has
# passed, count iterations that
# make no progress (no new commit + no edit): nudge to finish at the first
# threshold, hard-stop at the second. NOT "green verify = instant stop", verify
# fires per-edit and is often lenient, so green-but-still-editing must continue.
# Thresholds are deliberately generous: the failure mode is only a little wasted
# budget on an already-done run, whereas a too-tight window could cut off a
# worker still reading toward its next edit in a big multi-file change.
VERIFY_SETTLED_NUDGE_AFTER = 3
VERIFY_SETTLED_STOP_AFTER = 6

VERIFY_SETTLED_NUDGE = (
    "[harness settled] Changes are committed and recent turns changed"
    " nothing. If the task is complete, call finish_session now; if not,"
    " make a concrete edit, not more read-only commands."
)

# A non-metric `run` injects a one-shot wrap-up directive when the budget gets
# low: a worker that solves the task but never re-runs verify leaves the
# settled detector unable to engage (it needs a green verify) and burns the
# remainder on read-only commands.
RUN_BUDGET_NUDGE_BELOW = 0.25

RUN_BUDGET_NUDGE = (
    "[harness budget] Budget is low. Run run_verify_command now; if green,"
    " finish_session immediately; if red, fix only the smallest blocker,"
    " re-verify, finish. Nothing else."
)

# Gateless variant (no verify command this run): there is nothing to verify, so
# steer straight to finish_session.
RUN_BUDGET_NUDGE_GATELESS = (
    "[harness budget] Budget is low. Call finish_session now with a short summary. Nothing else."
)

# plan.md on disk is the plan; the planner's conversation only ever holds a
# copy. The operator answers open questions with `agent6 plan edit`, so the
# loop re-reads the file each turn and prepends this header when it differs
# from what the planner was last shown.
PLAN_ON_DISK_HEADER = (
    "[harness plan] plan.md on disk now reads as follows. The operator may have"
    " edited it (answers under `**A:**`, new constraints, deletions), and it"
    " supersedes any earlier version of the plan in this conversation. Carry"
    " these edits into the plan_markdown you pass to finish_planning, which"
    " overwrites the file."
)

PLAN_BUDGET_NUDGE = (
    "[harness budget] Budget is low and finish_planning has not been called."
    " Call it now with the best plan you have; a rough delivered plan beats"
    " an exhaustive one never emitted."
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
    "[harness] Prose with no tool call, and nothing changed yet; text alone"
    " cannot finish this task. Use the tools to do the work, or call"
    " finish_session and say why you are blocked."
)


QUESTION_NUDGE = (
    "[harness] You asked a question in prose; nobody sees it. Use ask_user"
    " for operator input, proceed if you can, or finish_session if done."
)


# Cross-run memory write nudges. Measured (bench/longhorizon FINDINGS #2):
# 46 legs across 2 models produced ZERO unprompted add_memory calls, so the
# <memories> header alone never causes writes. Surface the tool at the two
# moments a durable discovery is actually in hand: the first red-to-green
# verify flip (advisory, free) and the first finish_session after such a recovery
# (deferred once, the backstop). Each fires at most once per run, only in run
# mode with a memory store wired, and only while the worker has recorded
# nothing; a run whose verify never failed is never nudged.
# "State the rule, not the instance": measured on orchard leg 3 (FINDINGS #2
# day 3) — a store that spelled the house convention in words transferred to
# a new computation; a store carrying only the formula it was first seen in
# did not.
MEMORY_FLIP_NUDGE = (
    "[harness memory] Verify flipped green. If the fix rested on a durable"
    " non-obvious fact about this repo, add_memory the general rule so"
    " future runs skip the rediscovery; if ordinary, carry on."
)

MEMORY_FINISH_NUDGE = (
    "[harness memory] finish_session deferred once: verify recovered earlier"
    " and nothing was recorded. If the root cause was a durable non-obvious"
    " repo fact, add_memory the general rule; then call finish_session"
    " again either way."
)


def ends_with_question(text: str) -> bool:
    """Best-effort: the model's prose ends by asking the operator something. The
    last non-empty line ending in '?' catches the common 'Should I proceed?' /
    'Which option do you want?' close that a model writes instead of calling
    ask_user."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].endswith("?")


def standing_resume_nudge(reason: str, task_id: str, title: str) -> str:
    """The soft-end conversion for a run with a standing task: instead of
    ending, the loop re-enters the standing goal with this notice."""
    return (
        f"[harness] The run would have ended here ({reason}), but the standing"
        f" task ({task_id}: {title}) continues. Re-enter it now: pick the next"
        " piece of that goal, insert any new work you discover with add_task"
        " (ordinary tasks always run first), and write decisions down rather"
        " than asking questions. The run ends on its budget or an operator"
        " stop."
    )
