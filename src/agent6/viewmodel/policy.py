# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's policy facts, folded from its dir.

The few things an operator wants to see without opening config or interrupting
the run: which model is driving it, whether commands ask, how it is sandboxed,
and what gate will judge it. One fold, so the CLI banner, the TUI composer and
the web header cannot drift apart -- the two front-ends are other processes and
have only the run dir to read.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from agent6.runs.manifest import ManifestError, read_manifest


@dataclass(frozen=True, slots=True)
class RunPolicy:
    """What a run was launched under. Empty strings where the dir says nothing."""

    model: str
    run_commands: str
    isolation: str
    verify_command: tuple[str, ...]
    verify_origin: str

    def gate(self) -> str:
        """The gate and whose it is: an operator's `configured` gate certifies
        differently from one `inferred` off a file the model can edit."""
        if not self.verify_command:
            return "no verify gate"
        return f"{shlex.join(self.verify_command)} ({self.verify_origin or 'unknown origin'})"

    def short(self) -> str:
        """The compact form for a border or header: the two facts a watching
        operator acts on. The model has its own place on those surfaces and the
        gate belongs with the run's outcome, not next to the composer."""
        parts = [
            p
            for p in (f"commands {self.run_commands}" if self.run_commands else "", self.isolation)
            if p
        ]
        return " · ".join(parts)

    def line(self) -> str:
        """The one-line form every surface shows."""
        parts = [p for p in (self.model, self.isolation) if p]
        if self.run_commands:
            parts.append(f"commands {self.run_commands}")
        parts.append(self.gate())
        return " · ".join(parts)


def run_policy(run_dir: Path) -> RunPolicy:
    """Fold *run_dir*'s manifest into its policy facts."""
    try:
        m = read_manifest(run_dir)
    except ManifestError:
        return RunPolicy("", "", "", (), "")
    driver = m.models.driver
    return RunPolicy(
        model=driver.model if driver else "",
        run_commands=m.policy.run_commands,
        isolation=m.policy.isolation,
        verify_command=tuple(m.workflow.verify_command),
        verify_origin=m.workflow.verify_origin,
    )
