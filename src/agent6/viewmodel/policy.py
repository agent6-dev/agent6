# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The session's policy facts, folded from its dir.

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

from agent6.sessions.manifest import ManifestError, read_manifest


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    """What a session was launched under. Empty strings where the dir says nothing."""

    model: str
    run_commands: str
    isolation: str
    verify_command: tuple[str, ...]
    verify_origin: str
    mode: str = ""  # run | plan | ask; an ask has no gate to name

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
        """The one-line form every surface shows; "" for a run whose manifest
        could not be read -- an all-empty policy must not claim "no verify
        gate" about a run it knows nothing of."""
        if not (self.model or self.isolation or self.run_commands or self.verify_command):
            return ""
        parts = [p for p in (self.model, self.isolation) if p]
        if self.run_commands:
            parts.append(f"commands {self.run_commands}")
        if self.mode != "ask":
            parts.append(self.gate())
        return " · ".join(parts)


def session_policy(session_dir: Path) -> SessionPolicy:
    """Fold *session_dir*'s manifest into its policy facts."""
    try:
        m = read_manifest(session_dir)
    except ManifestError:
        return SessionPolicy("", "", "", (), "")
    driver = m.models.driver
    return SessionPolicy(
        model=driver.model if driver else "",
        run_commands=m.policy.run_commands,
        isolation=m.policy.isolation,
        verify_command=tuple(m.workflow.verify_command),
        verify_origin=m.workflow.verify_origin,
        mode=m.mode,
    )
