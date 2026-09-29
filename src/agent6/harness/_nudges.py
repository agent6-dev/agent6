# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the harness's mid-run notices: when the loop speaks and what it says.

Each nudge is a threshold and a directive, injected as a user-role harness message. The loop
detects and injects; this module holds the tuning values and the words.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

# No-progress guard (run mode): consecutive verify failures sharing one normalized signature.
# A green verify or a different failure resets the streak (evidence: bench/coreagent/FINDINGS.md).
NO_PROGRESS_NUDGE_AFTER = 4
NO_PROGRESS_ESCALATE_AFTER = 7
# Both nudges delivered and unheeded: the run stops.
NO_PROGRESS_STOP_AFTER = 10

# Tool-error guard (run mode): consecutive tool calls raising the same error, digits stripped.
# Any successful call, or a different error, resets it.
TOOL_ERROR_NUDGE_AFTER = 3
TOOL_ERROR_ESCALATE_AFTER = 5
TOOL_ERROR_STOP_AFTER = 8

TOOL_ERROR_NUDGE = (
    "[harness tool-error] The same call has failed three times with the same"
    " error; the error comes from the call's shape, not from the code."
)
TOOL_ERROR_ESCALATION = (
    "[harness tool-error] The identical error persists; the run ends at the eighth."
)

# A streak of ToolDeniedError refusals: the call was refused, not malformed.
TOOL_DENIED_NUDGE = (
    "[harness tool-error] Refused by policy, not a failure: the same call gets"
    " the same refusal, and the refusal names what applies. Tools that need no"
    " approval, and finish_session, are available."
)


# The repeat notice: the same (tool, args) call this many times in a row.
LOOP_GUARD_NOTICE_AFTER = 3


def loop_guard_words(tool: str, streak: int) -> str:
    """Return the notice after a streak of identical calls of one tool."""
    return (
        f"[loop-guard] You have called `{tool}` with"
        f" identical arguments {streak} times in a row."
        " Re-issuing the same call will not move the run forward. Change"
        " your approach: try different arguments, a different"
        " tool, commit to an edit, or call `finish_session` if"
        " you have already done what the task requires."
    )


def unreachable_tool_notice(binary: str) -> str:
    """Return the note for a binary the host has but the jail cannot execute."""
    return (
        f"NOTE: `{binary}` is installed on this machine but the sandbox"
        " cannot execute it: a reachability problem (a per-user or"
        " version-manager install the jail does not mount), not a problem"
        " with your code. Tell the operator to install it into a standard"
        " bin dir (~/.local/bin, /usr/local/bin) or grant its real"
        " directory via sandbox.extra_read_paths; if the tool exists"
        " inside the workspace, call it by that path. Do not keep probing"
        " for it."
    )


# The empty turn (no text, no tool_use); a starved reasoner gets its own nudge below.
WENT_QUIET_NUDGE = (
    "[harness] Your previous turn was empty (no text, no tool call); this"
    " message is the harness's. A tool call continues the run; finish_session"
    " ends it."
)


def reasoning_starved_nudge(output_tokens: int) -> str:
    """Return the nudge after a turn that spent its whole output cap on reasoning."""
    return (
        f"[harness] Your previous turn spent its whole output budget ({output_tokens}"
        " tokens) on reasoning, with no visible content and no tool call; this"
        " message is the harness's. A tool call continues the run; finish_session"
        " ends it."
    )


# A verify that failed at once with one of these ran no tests: the runner itself is absent.
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


def unrunnable_signature(argv: tuple[str, ...], rc: int | None, stdout: str, stderr: str) -> str:
    """Return why an adopted gate cannot run here, or "" for an ordinary red.

    Exit 127 (no such executable), or "No module named <mod>" naming the module the adopted
    `-m <mod>` runs. A failing suite exits 1 or 2 with test output and matches neither.

    Args:
        argv: The gate's argv.
        rc: Its exit code, or None.
        stdout: Its stdout.
        stderr: Its stderr.

    Returns:
        The reason, or "".
    """
    if rc == 127:
        return "exit 127, the command is not found"
    if "-m" in argv:
        mod = argv[argv.index("-m") + 1] if argv.index("-m") + 1 < len(argv) else ""
        blob = f"{stdout}\n{stderr}".lower()
        if mod and f"no module named {mod.lower()}" in blob:
            return f"no module named {mod}"
    return ""


VERIFY_UNADOPTED_NOTICE = (
    "[harness] The adopted verify command `{cmd}` cannot run here ({why});"
    " the run is gateless again: run_verify_command is not available and"
    " the harness commits each editing step."
)

BASELINE_RED_NOTICE = (
    "[harness] That verify ran on an unmodified tree: the gate was already"
    " failing before your changes; those failures predate the task."
)

VERIFY_BROKEN_NUDGE = (
    "[harness verify-broken] Verify exited at once without running tests:"
    " the runner is missing or misconfigured, not a test failure. The"
    " project's own test command (setup.cfg, tox.ini, pyproject, bin/test)"
    " runs via run_command."
)


def verify_did_not_run(stdout_tail: str, stderr_tail: str, duration_s: float) -> bool:
    """Return whether a failed verify ran no tests because the runner is absent.

    Requires a fast exit, so a real suite that import-errors deep in a long run is not flagged.

    Args:
        stdout_tail: The verify's last stdout bytes.
        stderr_tail: The verify's last stderr bytes.
        duration_s: How long the verify ran.

    Returns:
        Whether the failure carries a dead-runner signature within three seconds.
    """
    if duration_s > 3.0:
        return False
    blob = f"{stdout_tail}\n{stderr_tail}".lower()
    return any(sig in blob for sig in _VERIFY_DEAD_SIGNATURES)


def tool_error_signature(name: str, error_text: str) -> str:
    """Return a tool error's signature with digits masked, so varied numbers still match."""
    return f"{name}:{re.sub(r'[0-9]+', '#', error_text)[:200]}"


NO_PROGRESS_NUDGE = (
    "[harness no-progress] Verify has failed four times with the same error;"
    " the edits so far have not changed the outcome."
)

NO_PROGRESS_ESCALATION = (
    "[harness no-progress] The identical failure persists; the run ends at"
    " the tenth. Earlier file content is readable (`git show HEAD:<path>`)"
    " and restorable with apply_edit."
)

_SIG_NOISE = re.compile(r"line \d+|0x[0-9a-fA-F]+|\d+\.\d+s\b|:\d+:|/tmp/\S+|\bin \d+(\.\d+)?s\b")


def verify_failure_signature(stdout_tail: str, stderr_tail: str) -> str:
    """Return a verify failure's hash, insensitive to cosmetic drift."""
    tail = f"{stdout_tail}\n{stderr_tail}".strip()[-800:]
    digest = hashlib.md5(
        _SIG_NOISE.sub("#", tail).encode("utf-8", "replace"), usedforsecurity=False
    )
    return digest.hexdigest()


# Plan-mode wrap-up: nudge below this budget fraction, or after this many turns without a plan.
PLAN_BUDGET_NUDGE_BELOW = 0.35
PLAN_NUDGE_AFTER_ITERS = 12

# Task finish gate: a finish with open subtasks is re-prompted with the open list, this many times.
# Only subtasks gate; the always-pending root would deadlock.
TASK_FINISH_PATIENCE = 3

# Verify-settled completion (run mode): after a green verify, turns with no commit and no edit.
# Nudge at the first threshold, stop at the second; a green verify alone never stops a run.
# Generous on purpose: a tight window would cut off a worker reading toward its next edit.
VERIFY_SETTLED_NUDGE_AFTER = 3
VERIFY_SETTLED_STOP_AFTER = 6

VERIFY_SETTLED_NUDGE = (
    "[harness settled] Changes are committed and the last three turns changed"
    " nothing; finish_session ends the run, and at six unchanged turns the"
    " run ends on its own."
)

# Injected once when a run has spent `stagnation_notice_after_s` with no edit and no verify.
STAGNATION_NUDGE = (
    "[stagnation] {minutes} minutes in, no edit and no verify yet; the budget"
    " is finite, and finish_session ends the run with its summary."
)

# The gateless variant names no verify step: there is no gate to run.
STAGNATION_NUDGE_GATELESS = (
    "[stagnation] {minutes} minutes in and nothing edited yet; the budget is"
    " finite, and finish_session ends the run with its summary."
)

# A non-metric run gets one wrap-up directive when the budget runs low.
# A worker that never re-runs verify leaves the settled detector unable to engage.
RUN_BUDGET_NUDGE_BELOW = 0.25

RUN_BUDGET_NUDGE = (
    "[harness budget] Under a quarter of the budget remains; the loop halts"
    " when a cap is crossed. run_verify_command certifies the work and"
    " finish_session ends the run."
)

# Gateless variant: nothing to verify, so straight to finish_session.
RUN_BUDGET_NUDGE_GATELESS = (
    "[harness budget] Under a quarter of the budget remains; the loop halts"
    " when a cap is crossed. finish_session ends the run."
)

# plan.md on disk is the plan; the planner's conversation only holds a copy.
# The loop re-reads the file each turn and prepends this header when it differs from the last shown.
PLAN_ON_DISK_HEADER = (
    "[harness plan] plan.md on disk now reads as follows; it supersedes every"
    " earlier version in this conversation (operator edits: answers under"
    " `**A:**`, new constraints, deletions). The plan_markdown passed to"
    " finish_planning overwrites the file."
)

PLAN_BUDGET_NUDGE = (
    "[harness budget] finish_planning has not been called, and the pass is"
    " past its turn allowance or low on budget; the loop halts when a cap is"
    " crossed, and the plan exists only once finish_planning writes it."
)


# Silent finish before any work (run mode): an early prose turn on an untouched tree is a stall.
# A later prose finish stays honored as the implicit-finish path.
SILENT_NO_WORK_PATIENCE = 2
SILENT_NO_WORK_NUDGE = (
    "[harness] Prose with no tool call on an untouched tree is not a finish"
    " here; the tools do the work, and finish_session ends the run (its"
    " summary carries a blocker)."
)


QUESTION_NUDGE = (
    "[harness] A question in prose reaches nobody; ask_user reaches the"
    " operator, and finish_session ends the run."
)


# Memory write nudges fire at the first red-to-green verify flip and the first finish after it.
# Each fires at most once per run, in run mode with a memory store, while nothing is recorded.
# Measured (bench/longhorizon/FINDINGS.md): a stored rule transfers, a stored instance does not.
MEMORY_FLIP_NUDGE = (
    "[harness memory] Verify flipped green and nothing is recorded in the"
    " memory dir this run; it takes a durable non-obvious repo fact as a"
    " general rule (a new <name>.md plus its MEMORY.md line)."
)

MEMORY_FINISH_NUDGE = (
    "[harness memory] finish_session deferred once: verify recovered earlier"
    " and nothing is recorded. The memory dir takes a durable non-obvious"
    " repo fact as a general rule; the next finish_session call is honored"
    " either way."
)


# A line that asks: its last '?' is followed only by decoration and at most one short parenthesis.
_ASKS = re.compile(r"\?[\s)\"'`*_\]]*(?:\([^()]{0,24}\))?[\s)\"'`*_\]]*$")
# An option line under a question ("1. yes", "- keep", "a) first"); options after the ask keep it.
_OPTION_LINE = re.compile(r"^(?:[-*]|\d{1,2}[.)]|[a-z][.)])\s")


def _asks(line: str) -> bool:
    """Return whether the line ends in a question."""
    return _ASKS.search(line) is not None


def ending_question(text: str) -> str:
    """Return the question at the end of the model's prose, or "".

    Option lines may follow the question; the asking line is returned rather than the final
    option, so a later steer is recorded against the right ruling.

    Args:
        text: The model's prose.

    Returns:
        The asking line, or "".
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if _asks(lines[-1]):
        return lines[-1]
    for back in range(2, min(6, len(lines) + 1)):
        if _asks(lines[-back]) and all(_OPTION_LINE.match(after) for after in lines[-back + 1 :]):
            return lines[-back]
    return ""


def ends_with_question(text: str) -> bool:
    """Return whether the model's prose ends with a question and optional choices."""
    return bool(ending_question(text))


def standing_fruitless_nudge(reason: str, task_id: str, title: str, streak: int) -> str:
    """Return the re-entry notice for a round that landed nothing."""
    return (
        f"[harness] The run would have ended here ({reason}), and nothing has"
        f" landed since the last re-entry (fruitless round {streak}). The"
        f" standing task ({task_id}: {title}) continues: dig deeper or try a"
        " different approach -- a different angle, tool, or part of the repo;"
        " do not repeat the previous round. The run ends on its budget or an"
        " operator stop."
    )


def standing_resume_nudge(reason: str, task_id: str, title: str) -> str:
    """Return the notice that re-enters the standing goal in place of a soft end."""
    return (
        f"[harness] The run would have ended here ({reason}), but the standing"
        f" task ({task_id}: {title}) continues. Re-enter it now: pick the next"
        " piece of that goal, insert any new work you discover with add_task"
        " (ordinary tasks always run first), and write decisions down rather"
        " than asking questions. The run ends on its budget or an operator"
        " stop."
    )


def is_test_path(path: str) -> bool:
    """Return whether the path names a test file by the common Python conventions.

    Args:
        path: The path, repo-relative.

    Returns:
        True for a `test_*.py` or `*_test.py` basename, or any `tests` or `test` directory
        segment.
    """
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return (
        (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or any(p in ("tests", "test") for p in parts[:-1])
    )


# Paths the test-only flip notice lists before counting the rest.
TEST_ONLY_LIST_CAP = 12


def test_only_green_notice(paths: Sequence[str]) -> str:
    """Return the notice for a red-to-green flip whose changed files are all tests.

    The flip alone renders as an ordinary success, so the notice names the files.

    Args:
        paths: The files changed between the two verifies.

    Returns:
        The notice, listing at most `TEST_ONLY_LIST_CAP` paths.
    """
    shown = sorted(paths)[:TEST_ONLY_LIST_CAP]
    more = len(paths) - len(shown)
    listed = ", ".join(shown) + (f" (+{more} more)" if more else "")
    return (
        "[harness verify] The gate was red at the last verify and green at this one;"
        f" every file changed in between is a test file: {listed}."
    )
