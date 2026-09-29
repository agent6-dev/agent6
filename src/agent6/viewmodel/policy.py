# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's policy facts from its dir.

The driving model, whether commands ask, the isolation and the verify gate: one fold,
so the CLI banner, the TUI composer and the web header cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shlex

from agent6.sessions import manifest


@dataclasses.dataclass(frozen=True, slots=True)
class SessionPolicy:
    """What a session was launched under.

    Attributes:
        model: The driving model, "" when the manifest names none.
        run_commands: The commands mode ("allow", "ask", "deny"), "" when unknown.
        isolation: The sandbox isolation word, "" when unknown.
        verify_command: The verify gate's argv, empty when there is no gate.
        verify_origin: Where the gate came from: "configured" or "inferred".
        mode: "run", "plan" or "ask"; an ask has no gate to name.
    """

    model: str
    run_commands: str
    isolation: str
    verify_command: tuple[str, ...]
    verify_origin: str
    mode: str = ""

    def gate(self) -> str:
        """Return the gate and its origin.

        An operator's `configured` gate certifies differently from one `inferred`
        off a file the model can edit.

        Returns:
            The gate's shell line with its origin in parentheses, or "no verify gate".
        """
        if not self.verify_command:
            return "no verify gate"
        return f"{shlex.join(self.verify_command)} ({self.verify_origin or 'unknown origin'})"

    def short(self) -> str:
        """Return the compact form for a border or header: commands mode and isolation."""
        parts = [
            p
            for p in (f"commands {self.run_commands}" if self.run_commands else "", self.isolation)
            if p
        ]
        return " · ".join(parts)

    def line(self) -> str:
        """Return the one-line form every surface shows.

        Returns:
            The model, isolation, commands mode and gate joined by " · ", or "" for a
            run whose manifest could not be read (an all-empty policy must not claim
            "no verify gate" about a run it knows nothing of).
        """
        if not (self.model or self.isolation or self.run_commands or self.verify_command):
            return ""
        parts = [p for p in (self.model, self.isolation) if p]
        if self.run_commands:
            parts.append(f"commands {self.run_commands}")
        if self.mode != "ask":
            parts.append(self.gate())
        return " · ".join(parts)


def session_policy(session_dir: pathlib.Path) -> SessionPolicy:
    """Fold a session dir's manifest into its policy facts.

    Args:
        session_dir: The session's state dir.

    Returns:
        The policy, all-empty when the manifest cannot be read.
    """
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError:
        return SessionPolicy("", "", "", (), "")
    driver = m.models.driver
    return SessionPolicy(
        model=driver.model if driver else "",
        run_commands=m.policy.run_commands,
        isolation=m.policy.isolation,
        verify_command=tuple(m.harness.verify_command),
        verify_origin=m.harness.verify_origin,
        mode=m.mode,
    )
