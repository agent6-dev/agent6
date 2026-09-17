# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The harness-run verify gate (`[workflow].verify_when`): when the harness
runs the gate itself, and what the model is told about a run it did not
start.

`finish` certifies the tree a run ends on; `step` also judges every editing
turn; `never` leaves every gate run to the model's own `run_verify_command`
calls. A turn whose own verify call already judged the tree is never judged
twice. These are pure decisions over the turn's facts; the loop owns the
running and the bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent6.git_ops import GitError, tree_diff_paths
from agent6.git_ops import status as git_status
from agent6.tools.dispatch import ToolDenied, ToolDispatcher, ToolError
from agent6.tools.results import ExecResult
from agent6.verify_infer import infer_verify_command, read_agents_md
from agent6.workflows._chain import RunChain
from agent6.workflows._conversation import Notice
from agent6.workflows._nearest_tests import diff_changed_paths, is_bare_pytest, nearest_test_paths
from agent6.workflows._nudges import (
    BASELINE_RED_NOTICE,
    VERIFY_BROKEN_NUDGE,
    VERIFY_UNADOPTED_NOTICE,
    is_test_path,
    test_only_green_notice,
    unrunnable_signature,
    verify_did_not_run,
    verify_failure_signature,
)
from agent6.workflows._session_state import Verification
from agent6.workflows._verify_verdict import VerifyVerdict

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState, TurnState

VerifyWhen = Literal["finish", "step", "never"]
HarnessVerifyWhy = Literal["finish", "step"]

# Characters of gate output the model sees, from the end.
VERIFY_TAIL_CHARS = 2000


def harness_verify_due(
    *,
    when: VerifyWhen,
    gate_present: bool,
    tree_judged: bool,
    changed_this_turn: bool,
    finishing: bool,
) -> HarnessVerifyWhy | None:
    """Why the harness runs the gate after this turn, or None: `step` after a
    turn that changed the tree, `finish` when a run ends over a tree no verify
    covers; never on top of a verdict the run already holds for this tree.

    `tree_judged` is that verdict, green OR red: the model's own
    run_verify_command this turn, or a standing verdict from an earlier turn
    with nothing edited since. A red tree nothing has touched needs no re-run
    (the finish reports the red it already knows); an edit since the verdict
    clears it and the gate runs again."""
    if when == "never" or not gate_present or tree_judged:
        return None
    if finishing:
        return "finish"
    if when == "step" and changed_this_turn:
        return "step"
    return None


def harness_verify_notice(result: ExecResult, why: HarnessVerifyWhy) -> str:
    """What the model sees of a gate run it did not start: the verdict and
    the output tail, labelled by what triggered it."""
    verdict = "passed" if result.returncode == 0 else f"exit {result.returncode}"
    tail = f"{result.stdout}\n{result.stderr}".strip()[-VERIFY_TAIL_CHARS:]
    head = f"[harness verify] {why}: verify_command {verdict} ({result.duration_s:.0f}s)."
    return f"{head}\n{tail}" if tail else head


def scoped_verify_notice(result: ExecResult, *, timeout_s: float, paths: tuple[str, ...]) -> str:
    """What the model sees of a scoped gate run: the full command overran its
    budget (this run or an earlier one; scoping stays armed), so the gate ran
    only the tests nearest the run's change, listed by path. A scoped green
    certifies less than a full pass and the notice says so. One notice for
    the harness gate and the follow-up to the model's own timed-out call."""
    verdict = "passed" if result.returncode == 0 else f"exit {result.returncode}"
    tail = f"{result.stdout}\n{result.stderr}".strip()[-VERIFY_TAIL_CHARS:]
    head = (
        f"[verify] verify_command overran its {timeout_s:.0f}s budget;"
        f" the gate ran scoped to the tests nearest the run's change ({', '.join(paths)}):"
        f" {verdict} ({result.duration_s:.0f}s). A scoped green is not a full-suite pass."
    )
    return f"{head}\n{tail}" if tail else head


def gate_withheld_notice(head: str, exc: Exception) -> str:
    """A gate run not approved (a human's no, or the unattended auto-deny):
    withheld for the rest of the run, whichever site asked."""
    return (
        f"{head}: not run: {exc}."
        " The gate is withheld for the rest of the run; the run ends unverified."
    )


def finish_red_notice(*, used: int, retries: int) -> str:
    """The finish came back: the gate did not certify the tree."""
    left = retries - used
    ending = (
        "the next red finish ends the run, reported as finished, not passed"
        if left == 0
        else (
            f"{left} more red finish{'es return' if left != 1 else ' returns'}"
            " before the run ends red"
        )
    )
    return (
        f"[harness] finish_session not honoured: the verify gate did not certify the tree"
        f" (return {used} of {retries}); {ending}."
    )


# A jailed command that hit its timeout, per sandbox.jail's contract.
EXIT_TIMEOUT = 124


@dataclass(slots=True)
class VerifyGate:
    """The run's verify gate: its command (configured, or adopted mid-run by
    a gateless run that commits; `()` = gateless), when the harness runs it
    (`[workflow].verify_when`), its retries and timeout, and the run facts
    it reads. One owner for whether a gate is present, whether the tree is
    green, what a verify result means for the verdict, and the harness's
    own gate runs (the scoped re-run after a timeout included)."""

    command: tuple[str, ...]
    when: Literal["finish", "step", "never"]
    retries: int
    timeout_s: float
    infer: bool
    mode: Literal["run", "plan", "ask", "agent"]
    chain: RunChain
    dispatcher: ToolDispatcher
    log: Callable[[str], None]
    emit: Callable[..., None]

    def judged_the_base_commit(self, verdict: VerifyVerdict, result: ExecResult) -> bool:
        """True when this verify judged the commit the RUN started from.

        "The model has not edited yet" is the wrong test: every reason an
        operator resumes -- a budget stop, an iteration cap, a provider error --
        commits the leg's work first, so leg two opens on a clean tree whose
        HEAD already carries leg one's breakage, and reading that as the base
        would tell the worker its own failures are inherited. `/parallel` does
        the same by merging lane commits into the workspace.

        So: HEAD must still BE the base commit, the tree must be clean, and the
        gate must have actually produced a verdict -- a runner that was absent
        (instant exit) or timed out (124) never judged anything, and recording
        either would excuse every real failure for the rest of the run.

        A run that has already made the gate GREEN is answerable for a later
        red: it demonstrably could pass.

        Fails CLOSED. Every other user of `RunChain.dirty` treats an
        unreadable git as "assume clean"; here that would be a false
        exoneration, so an unreadable git records nothing.
        """
        if (
            verdict.ever_passed
            or result.exec_failed
            or result.returncode == EXIT_TIMEOUT
            or verify_did_not_run(result.stdout, result.stderr, result.duration_s)
            or not self.chain.base_sha
        ):
            return False
        try:
            status = git_status(self.chain.root, exclude=self.chain.untracked_at_start)
        except (GitError, OSError):
            return False
        return status.is_clean and status.head_sha == self.chain.base_sha

    def note_result(self, state: LoopState, turn: TurnState, result: ExecResult) -> None:
        """Verify bookkeeping: pass/fail flags, the grounding tail, and the
        no-progress streak (consecutive fails sharing one signature)."""
        rc = result.returncode
        verdict = state.verify
        if rc == 0:
            turn.verify_just_passed = True
            if verdict.last_ok is False:
                turn.verify_flipped_green = True
                if paths := self.test_only_paths_since_red(verdict.red_tree):
                    turn.tool_results.append(Notice(test_only_green_notice(paths)))
                    self.log(f"  verify flipped green over test-only edits: {' '.join(paths)}")
                    self.emit(
                        "loop.test_only_green.notice", iteration=turn.iteration, paths=list(paths)
                    )
            # This verify validated the current tree; any earlier
            # edit is now covered.
            turn.edit_since_verify_pass = False
        else:
            turn.verify_just_failed = True
            if verdict.adopted and (
                why := unrunnable_signature(verdict.adopted, rc, result.stdout, result.stderr)
            ):
                # An ADOPTED gate that cannot run here: un-adopt (the run is
                # gateless again, the argv never re-adopted) and say so. A
                # configured gate stays a loud red.
                cmd = " ".join(verdict.adopted)
                verdict.unadoptable.add(verdict.adopted)
                verdict.adopted = ()
                # The gate produced no verdict and no longer exists: the turn
                # is not "verify failed" (an on_verify_fail panel and the
                # checkpoint logic key on it).
                turn.verify_just_failed = False
                self.command = ()
                self.dispatcher.drop_verify_command()
                self.log(f"LOOP: verify un-adopted ({why}): {cmd}")
                self.emit(
                    "loop.verify_inferred",
                    command=[],
                    source="unadopted",
                    adopted_at=turn.iteration,
                )
                turn.tool_results.append(Notice(VERIFY_UNADOPTED_NOTICE.format(cmd=cmd, why=why)))
                return
            # A verify that exited instantly without running any tests (runner
            # absent) is a broken verify, not a real failure: flag it once so
            # the model does not "fix" working code or finish unchecked.
            if not verdict.broken_warned and verify_did_not_run(
                result.stdout, result.stderr, result.duration_s
            ):
                verdict.broken_warned = True
                turn.tool_results.append(Notice(VERIFY_BROKEN_NUDGE))
                self.emit("loop.verify_broken.nudge", iteration=turn.iteration)
        if verdict.baseline_ok is None and self.judged_the_base_commit(verdict, result):
            # This verify judged the run's BASE commit, so it IS the
            # baseline: no second gate run is needed to learn the same answer.
            verdict.baseline_ok = rc == 0
            self.emit("loop.baseline", ok=rc == 0, iteration=turn.iteration)
            if rc != 0:
                turn.tool_results.append(Notice(BASELINE_RED_NOTICE))
        tail = f"{result.stdout}\n{result.stderr}"
        verdict.last_tail = tail.strip()[-2000:]
        if rc == 0:
            verdict.note_pass()
            state.no_progress.nudges_used = 0
            return
        verdict.note_fail(verify_failure_signature(result.stdout, result.stderr))
        verdict.red_tree = self.chain.tree_sha()
        if verdict.fail_streak == 1:
            # A NEW stuck point: the nudge allowance starts over with it.
            state.no_progress.nudges_used = 0

    def maybe_adopt(self, state: LoopState, turn: TurnState) -> None:
        """A gateless run that commits has just materialized project files the
        preflight inference never saw (an empty repo infers nothing, then the
        run creates a pyproject two minutes later and finishes ungated). Re-run
        the DETERMINISTIC inference tiers (an AGENTS.md fence, repo signals;
        never the LLM tier) at each gateless commit until one lands, then adopt
        it for the rest of the run: the loop's gates, the dispatcher's
        run_verify_command, and the resume snapshot all read the adopted
        command. The model is told, so the gate flip is never silent; first
        adoption wins (the config gaining a command ends the gateless branch).
        `verify_infer = false` pins gatelessness: no adoption either."""
        if not self.infer:
            return
        inferred = infer_verify_command(
            self.chain.root, read_agents_md(self.chain.root), llm_call=None
        )
        if inferred is None or inferred.argv in state.verify.unadoptable:
            return
        if not self.dispatcher.adopt_verify_command(inferred.argv):
            # An inferred runner the jail cannot execute: adopting it would
            # turn the honest settle into an unexecutable-verify abort. Stay
            # gateless; re-inferred (and re-declined) at the next commit.
            self.log(f"LOOP: verify inference declined; {inferred.argv[0]} not on the jail PATH")
            return
        self.command = inferred.argv
        state.verify.adopted = inferred.argv
        cmd = " ".join(inferred.argv)
        self.log(f"LOOP: verify adopted from {inferred.source}: {cmd}")
        self.emit(
            "loop.verify_inferred",
            command=list(inferred.argv),
            source=inferred.source,
            adopted_at=turn.iteration,
        )
        turn.tool_results.append(
            Notice(
                "[harness] The repo now has a recognizable project, so a verify"
                f" command was adopted and gates the rest of this run: `{cmd}`."
                " Run run_verify_command to check your work."
            )
        )

    def may_run(self, *, denied: bool) -> bool:
        """Whether anyone may run a verify command in this run: `run_commands =
        "no"` withholds it from the harness as from the model, and so does a
        denied gate (`denied`: an ask answered no, or the unattended
        auto-deny)."""
        return self.dispatcher.command_policy() != "no" and not denied

    def present(self, *, denied: bool) -> bool:
        """Whether a verify gate can judge this run's steps: a command is
        configured (or adopted) and someone may run it. The one answer behind
        the harness gate, the per-step commit, the nudges, the verdict and the
        prompt's commit rule, so none of them can disagree."""
        return bool(self.command) and self.may_run(denied=denied)

    def harness_verify(self, state: LoopState, turn: TurnState, *, ending: bool = False) -> None:
        """Run the gate the harness owes this turn (`[workflow].verify_when`):
        after an editing turn under `step`, and when the run is ending (a
        finish_session, or `ending`: an end the harness declares) over a tree
        no green run covers under `step` or `finish`. The model's own
        run_verify_command this turn already judged the tree, so nothing runs
        on top of it. `run_commands = "no"` withholds the gate from the
        harness as it does from the model, and a DENIED gate (ask: a human's
        no, or the unattended auto-deny) is withheld for the rest of the run
        the same way. An unexecutable operator command raises
        `OperatorCommandUnexecutable` for the loop to end the run on."""
        why = harness_verify_due(
            when=self.when,
            gate_present=self.mode == "run" and self.present(denied=state.verify.denied),
            # The verdict the run holds over the tree AS IT STANDS -- this
            # turn's own verify or a standing one nothing has edited since.
            # A red tree nothing touched is not re-judged (the finish reports
            # the red), and its one red is counted once, not twice.
            tree_judged=state.verify.judged_and_untouched,
            changed_this_turn=turn.edit_since_verify_pass,
            finishing=ending
            or (turn.finish_signal is not None and turn.finish_kind == "finish_session"),
        )
        if why is None:
            return
        self.log(f"LOOP: harness verify ({why}) at iter {turn.iteration}")
        self.emit("loop.verify_harness", why=why, iteration=turn.iteration)
        try:
            scope = self.scope_paths() if state.verify.scoped else ()
            result = self.dispatcher.run_verify(extra_argv=scope)
        except ToolDenied as exc:
            state.verify.denied = True
            turn.tool_results.append(Notice(gate_withheld_notice(f"[harness verify] {why}", exc)))
            return
        except ToolError as exc:
            turn.tool_results.append(Notice(f"[harness verify] {why}: not run: {exc}"))
            return
        if (
            result.returncode == EXIT_TIMEOUT
            and not state.verify.scoped
            and self.scoped_followup(state, turn) is not None
        ):
            return
        self.note_result(state, turn, result)
        notice = (
            scoped_verify_notice(result, timeout_s=self.timeout_s, paths=scope)
            if scope
            else harness_verify_notice(result, why)
        )
        turn.tool_results.append(Notice(notice))

    def scope_paths(self) -> tuple[str, ...]:
        """The scoped-gate selection: tests nearest the run's cumulative diff.
        Empty unless the gate is a pytest argv naming no paths (the one shape
        that takes appended test files as its selection), or when nothing
        near the change exists to run."""
        if not is_bare_pytest(tuple(self.command)):
            return ()
        return nearest_test_paths(self.chain.root, diff_changed_paths(self.chain.diff_since_base()))

    def scoped_followup(self, state: LoopState, turn: TurnState) -> ExecResult | None:
        """The scoped re-run after a full gate overran its budget, wherever
        that gate ran (the harness's own, or the model's run_verify_command):
        the same command over the tests nearest the run's diff, noted and
        noticed like any gate run. Arms `verdict.scoped`, so later harness
        gates skip the doomed full run. None when the gate is not pytest or
        nothing near the change exists to run."""
        scope = self.scope_paths()
        if not scope:
            return None
        state.verify.scoped = True
        self.log(f"LOOP: verify overran; gate scoped to {len(scope)} test files")
        self.emit("loop.verify_scoped", paths=list(scope), iteration=turn.iteration)
        try:
            result = self.dispatcher.run_verify(extra_argv=scope)
        except ToolDenied as exc:
            state.verify.denied = True
            turn.tool_results.append(Notice(gate_withheld_notice("[verify] scoped re-run", exc)))
            return None
        except ToolError as exc:
            turn.tool_results.append(Notice(f"[verify] scoped re-run not run: {exc}"))
            return None
        self.note_result(state, turn, result)
        turn.tool_results.append(
            Notice(scoped_verify_notice(result, timeout_s=self.timeout_s, paths=scope))
        )
        return result

    def test_only_paths_since_red(self, red_tree: str) -> tuple[str, ...]:
        """Paths whose content differs between *red_tree* (the tree at the
        last red verify) and the current tree, when every one is a test file;
        () when either tree is unknown, nothing changed, or a non-test file did.
        Asked of git, so a run_command edit counts like an apply_edit."""
        if not red_tree:
            return ()
        tree = self.chain.tree_sha()
        if not tree:
            return ()
        try:
            paths = tree_diff_paths(self.chain.root, red_tree, tree)
        except (GitError, OSError):
            return ()
        if paths and all(is_test_path(p) for p in paths):
            return tuple(sorted(paths))
        return ()

    def tree_green(self, verdict: VerifyVerdict) -> bool | None:
        """Is the current tree in a verified-green state? None when no verify
        command is configured (nothing to gate on); else True iff the last verify
        was green AND nothing has been edited since, so a gate nobody may run
        leaves the run unverified, as documented. Grounds both the honest
        finish signal and the opt-in hard finish gate, so 'passed' can never
        mean 'finished over a red or stale verify'."""
        if not self.command:
            return None
        return verdict.green_and_untouched

    def verification(self, verdict: VerifyVerdict) -> Verification:
        """The verify verdict for the SessionResult, grounded on what the gate
        last saw of the tree. Not-green splits on that observation: "failed"
        claims someone SAW a red gate, so a leg where no verify ran (or edits
        landed after the last green) is "unverified" instead -- both exit 4,
        but only one sends the operator chasing a red that never happened.

        Only a run is gated: plan and ask finish clean whatever the tree looks
        like (finish_planning and the ask answer both emit all_passed=True), and
        preflight still INFERS a verify command for a plan that never runs one,
        so grounding on the tree there would report failure against their own
        events."""
        if self.mode != "run":
            return "not_applicable"
        green = self.tree_green(verdict)
        if green is None:
            return "not_applicable"
        if green:
            return "passed"
        return "failed" if verdict.last_ok is False else "unverified"
