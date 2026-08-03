# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Was the gate already red before this run touched anything?

A run that ends on a red gate reads as a failure, which is wrong when the tests
were broken to begin with or when the task WAS to change them. The only way to
know is to run the same gate against the commit the run started from.

Costs a second gate run, and only on red: when the gate is green the question
does not arise, and the answer would change nothing.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent6.git_ops import GitError, clone_repo, rollback_to_known_good
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.types import IsolationLevel, JailPolicy


@dataclass(frozen=True, slots=True)
class Baseline:
    """The gate's verdict on the base commit."""

    ran: bool
    returncode: int | None
    detail: str

    def line(self) -> str:
        if not self.ran:
            return f"could not check the base commit: {self.detail}"
        if self.returncode == 0:
            return "the gate passes on the base commit, so this run broke it"
        return f"the gate already failed on the base commit (exit {self.returncode})"


def gate_on_base(
    origin: Path,
    base_sha: str,
    *,
    argv: tuple[str, ...],
    isolation: IsolationLevel,
    timeout_s: float,
) -> Baseline:
    """Run *argv* against *base_sha* in a throwaway clone.

    A clone, not the live checkout: the run's own work must not be disturbed to
    answer a question about it, and the gate may write (caches, build output).
    """
    if not (argv and base_sha):
        return Baseline(ran=False, returncode=None, detail="no gate or no base commit recorded")
    work = Path(tempfile.mkdtemp(prefix="agent6-baseline-"))
    dest = work / "base"
    try:
        clone_repo(origin, dest)
        # A fresh clone has nothing to lose, and this is the sanctioned
        # primitive: the module refuses `reset --hard` outright.
        rollback_to_known_good(dest, base_sha)
    except GitError as exc:
        shutil.rmtree(work, ignore_errors=True)
        return Baseline(ran=False, returncode=None, detail=str(exc))
    try:
        res = run_in_jail(JailPolicy(cwd=dest, argv=argv, isolation=isolation, timeout_s=timeout_s))
    except JailUnavailableError as exc:
        return Baseline(ran=False, returncode=None, detail=str(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return Baseline(ran=True, returncode=res.returncode, detail="")
