# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the verify gate's last verdict and whether the tree moved since.

Every reader of "is the run green" (the finish gates, the panel's grounding, the resume
snapshot, the turn notices) reads this one object; its transitions live here.
"""

from __future__ import annotations

import dataclasses

from agent6 import evidence


@dataclasses.dataclass(slots=True)
class VerifyVerdict:
    """Hold the verify bookkeeping of one execution.

    Attributes:
        command: The argv of the verify whose verdict or tree this holds, () before any.
        last_ok: The last verify's result; None before any verify.
        scoped: Whether the full gate overran its timeout, so gates run scoped until a full pass.
        red_tree: The tree sha at the last red verify, "" when git could not say.
        baseline_ok: Whether the gate passed on the unmodified tree; None when no such verify ran.
        last_tail: The last verify's output tail.
        fail_signature: The normalized signature of the current failure streak.
        fail_streak: How many consecutive failures share that signature.
        broken_warned: Whether the broken-gate notice was sent.
        edited_since: Whether the tree changed since the last verdict.
        ever_passed: Whether any verify passed this execution.
        ever_failed: Whether any verify failed this execution.
        adopted: The gate adopted mid-run, () when the gate is configured or absent.
        unadoptable: Every adopted argv that proved unrunnable, never re-adopted.
        declined: The last reason inference adopted nothing, logged when it changes.
    """

    final_verify_id: str | None = None
    command: tuple[str, ...] = ()
    verify_tree: str | None = None
    last_ok: bool | None = None
    scoped: bool = False
    red_tree: str = ""
    baseline_ok: bool | None = None
    last_tail: str = ""
    fail_signature: str = ""
    fail_streak: int = 0
    broken_warned: bool = False
    edited_since: bool = False
    ever_passed: bool = False
    ever_failed: bool = False
    adopted: tuple[str, ...] = ()
    unadoptable: set[tuple[str, ...]] = dataclasses.field(default_factory=set)
    declined: str = ""

    def observe(self, record: evidence.VerificationEvidence | None) -> bool:
        """Bind the verdict to a stable tree observation.

        Args:
            record: The dispatcher's evidence, absent in historical callers.

        Returns:
            Whether the command produced a verdict for an unchanged tree.
        """
        if record is None:
            return True
        self.final_verify_id = record.verify_id
        self.command = tuple(record.command)
        self.verify_tree = record.tree_after if record.stable_tree else None
        if not record.stable_tree or record.outcome not in ("passed", "failed"):
            self.last_ok = None
            self.edited_since = True
            return False
        return True

    def note_pass(self) -> None:
        """Record a green verify: the tree as it stands is verified and the streaks reset."""
        self.last_ok = True
        self.ever_passed = True
        self.edited_since = False
        self.fail_signature = ""
        self.fail_streak = 0

    def note_fail(self, signature: str) -> None:
        """Record a red verify, extending the streak on the same signature.

        Like a green, a red judges the tree as it stands: an untouched red tree is not re-run.

        Args:
            signature: The failure's normalized signature.
        """
        self.last_ok = False
        self.ever_failed = True
        self.edited_since = False
        if signature == self.fail_signature:
            self.fail_streak += 1
        else:
            self.fail_signature = signature
            self.fail_streak = 1

    def note_baseline(self, *, ok: bool) -> None:
        """Record the gate's verdict on the tree before the run's first turn.

        It judges the tree as it stands, as any verdict does, and counts as neither a pass
        nor a failure of the run's own: no streak starts and neither `ever_` flag moves. A
        green one ends a failure streak, as any green does.

        Args:
            ok: Whether the gate passed.
        """
        self.baseline_ok = ok
        self.last_ok = ok
        self.edited_since = False
        if ok:
            self.fail_signature = ""
            self.fail_streak = 0

    def note_edit(self) -> None:
        """Record that the tree changed since the last verdict."""
        self.edited_since = True

    def note_tree(self, tree: str) -> None:
        """Bind the edit flag to the tree as it stands: the verified tree is untouched.

        Args:
            tree: The worktree's tree sha; "" when git cannot say, which changes nothing.
        """
        if tree and self.verify_tree:
            self.edited_since = tree != self.verify_tree

    def fact(self) -> str:
        """Return the one sentence the worker reads at a resume or a context restart.

        Returns:
            The last full verification's command, verdict and tree, and whether the tree
            changed since; or that no verdict covers the tree as it stands.
        """
        command = f"`{' '.join(self.command)}`" if self.command else "the verify command"
        tree = f"tree {self.verify_tree[:12]}" if self.verify_tree else "an unrecorded tree"
        if self.last_ok is None:
            if not self.verify_tree:
                return "No verification verdict covers the tree as it stands."
            return (
                f"The last full verification ran {command} on {tree}; the tree as it stands"
                " is not that tree, so no verdict covers it."
            )
        verdict = "passed" if self.last_ok else "failed"
        since = "has changed since" if self.edited_since else "is unchanged since"
        return f"Last full verification: {command} {verdict} on {tree}; the tree {since}."

    @property
    def green_and_untouched(self) -> bool:
        """Whether the tree is verified green and untouched since."""
        return self.last_ok is True and not self.edited_since

    @property
    def judged_and_untouched(self) -> bool:
        """Whether a verdict, green or red, covers the tree as it stands."""
        return self.last_ok is not None and not self.edited_since
