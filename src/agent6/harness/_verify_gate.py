# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the verify gate the harness owes a turn, per `[harness].verify_when`.

`finish` certifies the tree a run ends on; `step` also judges every editing turn; `never`
leaves every gate run to the model. Under `finish` and `step` a fresh run also runs the gate once
before its first turn, which records the baseline. A tree the run already holds a verdict for is
never judged twice. `VerifyGate` runs the gate and its scoped follow-up and keeps the verdict
bookkeeping.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from agent6 import errors, git_ops, terminal, verify_infer
from agent6.harness import (
    _chain,
    _conversation,
    _nearest_tests,
    _nudges,
    _snapshot,
    _verify_verdict,
)
from agent6.tools import dispatch, results
from agent6.tools import errors as tool_errors

if TYPE_CHECKING:
    from agent6.harness import _loop_state

VerifyWhen = Literal["finish", "step", "never"]
HarnessVerifyWhy = Literal["finish", "step", "start"]

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
    """Return why the harness runs the gate after this turn, or None.

    Args:
        when: `[harness].verify_when`.
        gate_present: Whether a gate can judge the run.
        tree_judged: Whether the run holds a verdict, green or red, for the tree as it stands.
        changed_this_turn: Whether the turn edited the tree.
        finishing: Whether the run is ending.

    Returns:
        `step` after an editing turn, `finish` when the run ends over an unjudged tree, else None.
    """
    if when == "never" or not gate_present or tree_judged:
        return None
    if finishing:
        return "finish"
    if when == "step" and changed_this_turn:
        return "step"
    return None


def harness_verify_notice(result: results.ExecResult, why: HarnessVerifyWhy) -> str:
    """Return the notice for a gate run the harness started: the verdict and output tail.

    Args:
        result: The gate's result.
        why: What triggered the run.

    Returns:
        The notice text.
    """
    verdict = "passed" if result.returncode == 0 else f"exit {result.returncode}"
    tail = f"{result.stdout}\n{result.stderr}".strip()[-VERIFY_TAIL_CHARS:]
    head = f"[harness verify] {why}: verify_command {verdict} ({result.duration_s:.0f}s)."
    return f"{head}\n{tail}" if tail else head


def scoped_verify_notice(
    result: results.ExecResult, *, timeout_s: float, paths: tuple[str, ...]
) -> str:
    """Return the notice for a scoped gate run, which names its paths and what a scoped green means.

    Args:
        result: The gate's result.
        timeout_s: The budget the full command overran.
        paths: The test paths the gate ran.

    Returns:
        The notice text.
    """
    verdict = "passed" if result.returncode == 0 else f"exit {result.returncode}"
    tail = f"{result.stdout}\n{result.stderr}".strip()[-VERIFY_TAIL_CHARS:]
    head = (
        f"[verify] verify_command overran its {timeout_s:.0f}s budget;"
        f" the gate ran scoped to the tests nearest the run's change ({', '.join(paths)}):"
        f" {verdict} ({result.duration_s:.0f}s). A scoped green is not a full-suite pass."
    )
    return f"{head}\n{tail}" if tail else head


def gate_withheld_notice(head: str, exc: Exception) -> str:
    """Return the notice for a gate run an approval denied: withheld for the rest of the run.

    Args:
        head: The notice's label.
        exc: The denial.

    Returns:
        The notice text.
    """
    return (
        f"{head}: not run: {exc}."
        " The gate is withheld for the rest of the run; the run ends unverified."
    )


def finish_red_notice(*, used: int, retries: int) -> str:
    """Return the notice for a finish the gate did not certify, with the returns left.

    Args:
        used: The red finishes so far.
        retries: The red finishes allowed.

    Returns:
        The notice text.
    """
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


def settled_red_notice(*, used: int, retries: int) -> str:
    """Return the notice for a settled stop the gate did not certify, with the returns left.

    Args:
        used: The red settled stops so far.
        retries: The red settled stops allowed.

    Returns:
        The notice text.
    """
    left = retries - used
    ending = (
        "the next settled stop ends the run, reported as finished, not passed"
        if left == 0
        else (
            f"{left} more settled stop{'s return' if left != 1 else ' returns'}"
            " before the run ends red"
        )
    )
    return (
        "[harness settled] The tree stopped changing, but the verify gate is red, so the run"
        f" goes on (return {used} of {retries}); {ending}."
    )


# A jailed command that hit its timeout, per sandbox.jail's contract.
EXIT_TIMEOUT = 124


@dataclasses.dataclass(slots=True)
class VerifyGate:
    """Own the run's verify gate: whether one is present, what a result means, and the harness runs.

    Attributes:
        configured: The configured command, () when none.
        when: `[harness].verify_when`.
        at_start: `[harness].verify_at_start`: a fresh run's gate runs once before its first turn.
        retries: The red finishes allowed.
        timeout_s: The gate's budget.
        infer: Whether a gateless run adopts an inferred command at a commit.
        hide: `[sandbox].hide_paths`, which the inference reads nothing from.
        mode: The loop mode; only a run is gated.
        chain: The run's chain.
        dispatcher: The run's dispatcher, which runs the gate.
        log: The run's text logger.
        emit: The run's event emitter.
    """

    configured: tuple[str, ...]
    when: Literal["finish", "step", "never"]
    at_start: bool
    retries: int
    timeout_s: float
    infer: bool
    hide: tuple[str, ...]
    mode: Literal["run", "plan", "ask", "agent"]
    chain: _chain.RunChain
    dispatcher: dispatch.ToolDispatcher
    log: Callable[[str], None]
    emit: Callable[..., None]

    def judged_the_base_commit(
        self, verdict: _verify_verdict.VerifyVerdict, result: results.ExecResult
    ) -> bool:
        """Return whether this verify judged the commit the run started from.

        HEAD must be the base commit on a clean tree (a resumed execution opens on its
        predecessor's commits), the gate must have produced a verdict (an absent or timed-out
        runner judged nothing), and a run that ever went green answers for a later red.
        Fails closed: an unreadable git records nothing.

        Args:
            verdict: The run's verify bookkeeping.
            result: The verify's result.

        Returns:
            True when the result is the base commit's own.
        """
        if (
            verdict.ever_passed
            or result.exec_failed
            or result.returncode == EXIT_TIMEOUT
            or _nudges.verify_did_not_run(result.stdout, result.stderr, result.duration_s)
            or not self.chain.base_sha
        ):
            return False
        try:
            status = git_ops.status(self.chain.root, exclude=self.chain.untracked_at_start)
        except (errors.GitError, OSError):
            return False
        return status.is_clean and status.head_sha == self.chain.base_sha

    def note_result(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, result: results.ExecResult
    ) -> None:
        """Record a verify result: the flags, the grounding tail, the no-progress streak.

        Args:
            state: The execution's state.
            turn: The turn the verify ran in.
            result: The verify's result.
        """
        rc = result.returncode
        verdict = state.verify
        if rc == 0:
            turn.verify_just_passed = True
            if verdict.last_ok is False:
                turn.verify_flipped_green = True
                if paths := self.test_only_paths_since_red(verdict.red_tree):
                    turn.tool_results.append(
                        _conversation.Notice(_nudges.test_only_green_notice(paths))
                    )
                    shown = " ".join(map(terminal.printable_name, paths))
                    self.log(f"  verify flipped green over test-only edits: {shown}")
                    self.emit("loop.test_only_green.notice", turn=turn.turn, paths=list(paths))
            turn.edit_since_verify_pass = False
        else:
            turn.verify_just_failed = True
            if verdict.adopted and (
                why := _nudges.unrunnable_signature(
                    verdict.adopted, rc, result.stdout, result.stderr
                )
            ):
                # An adopted gate that cannot run here is un-adopted; a configured one stays red.
                cmd = " ".join(verdict.adopted)
                verdict.unadoptable.add(verdict.adopted)
                verdict.adopted = ()
                turn.verify_just_failed = False  # no verdict was produced

                self.dispatcher.drop_verify_command()
                self.log(f"LOOP: verify un-adopted ({why}): {cmd}")
                self.emit(
                    "loop.verify_inferred",
                    command=[],
                    source="unadopted",
                    adopted_at=turn.turn,
                )
                turn.tool_results.append(
                    _conversation.Notice(_nudges.VERIFY_UNADOPTED_NOTICE.format(cmd=cmd, why=why))
                )
                return
            # An instant exit with no tests run is a broken gate, flagged once, not a failure.
            if not verdict.broken_warned and _nudges.verify_did_not_run(
                result.stdout, result.stderr, result.duration_s
            ):
                verdict.broken_warned = True
                turn.tool_results.append(_conversation.Notice(_nudges.VERIFY_BROKEN_NUDGE))
                self.emit("loop.verify_broken.nudge", turn=turn.turn)
        judged_base = self.judged_the_base_commit(verdict, result)
        if judged_base and verdict.baseline_ok is None:
            verdict.baseline_ok = rc == 0
            self.emit("loop.baseline", ok=rc == 0, turn=turn.turn)
            if rc != 0:
                turn.tool_results.append(_conversation.Notice(_nudges.BASELINE_RED_NOTICE))
        tail = f"{result.stdout}\n{result.stderr}"
        verdict.last_tail = tail.strip()[-2000:]
        self._record_verdict(state, result, judged_base=judged_base)

    def _record_verdict(
        self, state: _loop_state.LoopState, result: results.ExecResult, *, judged_base: bool
    ) -> None:
        """Record the verdict's flags and streak.

        Args:
            state: The execution's state.
            result: The verify's result.
            judged_base: Whether the verdict is on the untouched base. Such a verdict is the
                baseline's, whoever ran it. The settled guard waits for a pass of the run's
                own, and a red base verdict starts no streak.
        """
        verdict = state.verify
        ok = result.returncode == 0
        if judged_base:
            verdict.note_baseline(ok=ok)
        elif ok:
            verdict.note_pass()
        else:
            verdict.note_fail(_nudges.verify_failure_signature(result.stdout, result.stderr))
        if not ok:
            verdict.red_tree = self.chain.tree_sha()
        # A green, or a new stuck point, re-arms the nudges.
        if ok or verdict.fail_streak == 1:
            state.no_progress.rearm()

    def maybe_adopt(self, state: _loop_state.LoopState, turn: _loop_state.TurnState) -> None:
        """Adopt a verify command a gateless run's commit makes inferable.

        The deterministic inference tiers (an AGENTS.md fence, repo signals; never the model
        tier) run at each gateless commit until one lands; the model is told of the adoption.
        `verify_infer = false` pins gatelessness.

        Args:
            state: The execution's state.
            turn: The turn that committed.
        """
        if not self.infer:
            return
        inferred = verify_infer.infer_verify_command(
            self.chain.root,
            verify_infer.read_agents_md(self.chain.root, self.hide),
            llm_call=None,
            hide=self.hide,
        )
        if not inferred.argv:
            # A runner the jail cannot execute stays unadopted: the run settles, no abort.
            passed = verify_infer.passed_over_text(inferred)
            if passed and passed != state.verify.declined:
                state.verify.declined = passed
                self.log(f"LOOP: verify inference declined; {passed}")
            return
        if inferred.argv in state.verify.unadoptable:
            return
        if not self.dispatcher.adopt_verify_command(inferred.argv):
            return
        state.verify.adopted = inferred.argv
        cmd = " ".join(inferred.argv)
        self.log(f"LOOP: verify adopted from {inferred.label}: {cmd}")
        self.emit(
            "loop.verify_inferred",
            command=list(inferred.argv),
            source=inferred.source,
            adopted_at=turn.turn,
        )
        turn.tool_results.append(
            _conversation.Notice(
                "[harness] The repo now has a recognizable project, so a verify"
                f" command was adopted and gates the rest of this run: `{cmd}`."
                " Run run_verify_command to check your work."
            )
        )

    def may_run(self, *, withheld: bool) -> bool:
        """Return whether anyone may run a verify command in this run.

        `run_commands = "deny"` and a denied gate withhold it from the harness as from the model.

        Args:
            withheld: Whether an approval withheld the gate.

        Returns:
            True when the gate may run.
        """
        return self.dispatcher.command_policy() != "deny" and not withheld

    def command(self, verdict: _verify_verdict.VerifyVerdict) -> tuple[str, ...]:
        """Return the command in force: the configured one, else the adopted one, else ()."""
        return self.configured or verdict.adopted

    def present(self, verdict: _verify_verdict.VerifyVerdict) -> bool:
        """Return whether a gate can judge this run: a command is in force and someone may run it.

        Args:
            verdict: The run's verify bookkeeping.

        Returns:
            The one answer the harness gate, the per-step commit, the nudges and the prompt read.
        """
        return bool(self.command(verdict)) and self.may_run(withheld=verdict.withheld)

    def start_baseline(self, state: _loop_state.LoopState) -> list[_conversation.Notice]:
        """Run the gate on the unedited tree before a fresh run's first turn; record the baseline.

        The verdict is the baseline's, not a pass or a failure of the run's own: the settled
        guard and the memory advisory wait for those. A gate that produced no verdict records
        nothing. An unexecutable operator command raises `OperatorCommandUnexecutableError` for
        the loop to end on.

        Args:
            state: The execution's state.

        Returns:
            The first turn's notices: a red gate's verdict and the base-red notice, or the
            withheld notice after a denial; none after a green.
        """
        verdict = state.verify
        runs = self.at_start and self.mode == "run" and self.when != "never"
        if not (runs and self.present(verdict)):
            return []
        self.log("LOOP: harness verify (start)")
        self.emit("loop.verify_harness", why="start", turn=0)
        try:
            result = self.dispatcher.run_verify()
        except tool_errors.ToolDeniedError as exc:
            verdict.withheld = True
            return [_conversation.Notice(gate_withheld_notice("[harness verify] start", exc))]
        except tool_errors.ToolError:
            return []
        ok = result.returncode == 0
        no_verdict = (
            result.exec_failed
            or result.returncode == EXIT_TIMEOUT
            or _nudges.verify_did_not_run(result.stdout, result.stderr, result.duration_s)
        )
        if no_verdict:
            return []  # nothing to record or tell
        word = "green" if ok else f"red (exit {result.returncode})"
        judged = self.judged_the_base_commit(verdict, result)
        if judged:
            verdict.note_baseline(ok=ok)
            self.emit("loop.baseline", ok=ok, turn=0)
            self.log(f"LOOP: gate {word} before the run")
            verdict.last_tail = f"{result.stdout}\n{result.stderr}".strip()[-2000:]
            if not ok:
                verdict.red_tree = self.chain.tree_sha()
        else:
            # The tree is past the base commit (an included dirty tree, a setup command that
            # rewrote a tracked file): the verdict is no baseline, and the model still reads it.
            self.log(f"LOOP: gate {word} before the run, on a tree past the base: no baseline")
        if ok:
            return []
        notices = [_conversation.Notice(harness_verify_notice(result, "start"))]
        if judged:
            notices.append(_conversation.Notice(_nudges.BASELINE_RED_NOTICE))
        return notices

    def harness_verify(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, *, ending: bool = False
    ) -> None:
        """Run the gate the harness owes this turn, per `harness_verify_due`.

        A denied gate is withheld for the rest of the run. An unexecutable operator command
        raises `OperatorCommandUnexecutableError` out of the dispatcher for the loop to end on.

        Args:
            state: The execution's state.
            turn: The turn that ended.
            ending: Whether the harness declares an end this turn.
        """
        why = harness_verify_due(
            when=self.when,
            gate_present=self.mode == "run" and self.present(state.verify),
            tree_judged=state.verify.judged_and_untouched,
            changed_this_turn=turn.edit_since_verify_pass,
            finishing=ending or (turn.finish is not None and turn.finish.kind == "finish_session"),
        )
        if why is None:
            return
        self.log(f"LOOP: harness verify ({why}) at turn {turn.turn}")
        self.emit("loop.verify_harness", why=why, turn=turn.turn)
        try:
            scope = self.scope_paths(state.verify) if state.verify.scoped else ()
            result = self.dispatcher.run_verify(extra_argv=scope)
        except tool_errors.ToolDeniedError as exc:
            state.verify.withheld = True
            turn.tool_results.append(
                _conversation.Notice(gate_withheld_notice(f"[harness verify] {why}", exc))
            )
            return
        except tool_errors.ToolError as exc:
            turn.tool_results.append(
                _conversation.Notice(f"[harness verify] {why}: not run: {exc}")
            )
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
        turn.tool_results.append(_conversation.Notice(notice))

    def scope_paths(self, verdict: _verify_verdict.VerifyVerdict) -> tuple[str, ...]:
        """Return the tests nearest the run's diff, for a scoped gate run.

        Args:
            verdict: The run's verify bookkeeping.

        Returns:
            The test paths; () unless the gate is a bare pytest argv, or when nothing is near.
        """
        if not _nearest_tests.is_bare_pytest(self.command(verdict)):
            return ()
        return _nearest_tests.nearest_test_paths(
            self.chain.root,
            _nearest_tests.diff_changed_paths(self.chain.diff_since_base()),
            hide=self.hide,
        )

    def scoped_followup(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState
    ) -> results.ExecResult | None:
        """Re-run a gate that overran its budget over the tests nearest the run's diff.

        Arms `verdict.scoped`, so later harness gates skip the full run. The result is noted
        and noticed like any gate run.

        Args:
            state: The execution's state.
            turn: The turn the gate ran in.

        Returns:
            The scoped result; None when the gate cannot be scoped or the run was refused.
        """
        scope = self.scope_paths(state.verify)
        if not scope:
            return None
        state.verify.scoped = True
        self.log(f"LOOP: verify overran; gate scoped to {len(scope)} test files")
        self.emit("loop.verify_scoped", paths=list(scope), turn=turn.turn)
        try:
            result = self.dispatcher.run_verify(extra_argv=scope)
        except tool_errors.ToolDeniedError as exc:
            state.verify.withheld = True
            turn.tool_results.append(
                _conversation.Notice(gate_withheld_notice("[verify] scoped re-run", exc))
            )
            return None
        except tool_errors.ToolError as exc:
            turn.tool_results.append(_conversation.Notice(f"[verify] scoped re-run not run: {exc}"))
            return None
        self.note_result(state, turn, result)
        turn.tool_results.append(
            _conversation.Notice(
                scoped_verify_notice(result, timeout_s=self.timeout_s, paths=scope)
            )
        )
        return result

    def test_only_paths_since_red(self, red_tree: str) -> tuple[str, ...]:
        """Return the paths changed since the last red verify when every one is a test file.

        Asked of git, so a run_command edit counts like an apply_edit.

        Args:
            red_tree: The tree sha at the last red verify.

        Returns:
            The sorted test paths; () when a tree is unknown, nothing changed or a non-test did.
        """
        if not red_tree:
            return ()
        tree = self.chain.tree_sha()
        if not tree:
            return ()
        try:
            paths = git_ops.tree_diff_paths(self.chain.root, red_tree, tree)
        except (errors.GitError, OSError):
            return ()
        if paths and all(_nudges.is_test_path(p) for p in paths):
            return tuple(sorted(terminal.printable_name(p) for p in paths))
        return ()

    def tree_green(self, verdict: _verify_verdict.VerifyVerdict) -> bool | None:
        """Return whether the tree is verified green, None when no command is in force.

        Args:
            verdict: The run's verify bookkeeping.

        Returns:
            True when the last verify was green and nothing was edited since.
        """
        if not self.command(verdict):
            return None
        return verdict.green_and_untouched

    def verification(self, verdict: _verify_verdict.VerifyVerdict) -> _snapshot.Verification:
        """Return the gate's word for the session result.

        Only a run is gated. `failed` claims an observed red, so an execution where no verify
        ran or edits landed after the last green reads `unverified`.

        Args:
            verdict: The run's verify bookkeeping.

        Returns:
            The verification word.
        """
        if self.mode != "run":
            return "not_applicable"
        green = self.tree_green(verdict)
        if green is None:
            return "not_applicable"
        if green:
            return "passed"
        return "failed" if verdict.last_ok is False else "unverified"
