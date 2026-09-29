# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the verify gate's last verdict and whether the tree moved since.

Every reader of "is the run green" (the finish gates, the panel's grounding, the resume
snapshot, the turn notices) reads this one object; its transitions live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VerifyVerdict:
    """Hold the verify bookkeeping of one execution.

    Attributes:
        last_ok: The last verify's result; None before any verify.
        denied: Whether an approval denied the gate; it is withheld for the rest of the run.
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
    """

    last_ok: bool | None = None
    denied: bool = False
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
    unadoptable: set[tuple[str, ...]] = field(default_factory=set)

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

    def note_edit(self) -> None:
        """Record that the tree changed since the last verdict."""
        self.edited_since = True

    @property
    def green_and_untouched(self) -> bool:
        """Whether the tree is verified green and untouched since."""
        return self.last_ok is True and not self.edited_since

    @property
    def judged_and_untouched(self) -> bool:
        """Whether a verdict, green or red, covers the tree as it stands."""
        return self.last_ok is not None and not self.edited_since
