# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's policy facts from its dir.

The driving model, the isolation, the network, whether commands ask and the verify gate:
one fold, so the CLI banner, the TUI composer and the web header cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shlex

from agent6 import evidence, kinds
from agent6.sessions import layout, manifest
from agent6.viewmodel import format, session_log, state_types


def model_lines(m: manifest.SessionManifest) -> tuple[str, ...]:
    """Return the recorded model lines, never consulting the current config."""
    driver_role = kinds.SESSION_KINDS[m.mode].role if m.mode in kinds.SESSION_KINDS else "driver"
    roles = {
        driver_role: m.models.driver,
        "reviewer": m.models.reviewer,
        "helper": m.models.helper,
        **m.models.review_seats,
    }
    lines: list[str] = []
    for role, brief in roles.items():
        if brief is None:
            continue
        # Old records did not capture effort; absence is not a provider default.
        effort = "not recorded"
        if brief.effort_source:
            effort = (
                "provider default"
                if brief.effort is None
                else f"{brief.effort} ({brief.effort_source})"
            )
        lines.append(f"{role} {format.format_model_route(brief)} effort {effort}")
    return tuple(lines)


@dataclasses.dataclass(frozen=True, slots=True)
class SessionPolicy:
    """What a session was launched under.

    Attributes:
        model: The driving model, "" when the manifest names none.
        run_commands: The commands mode ("allow", "deny", "ask"), "" when unknown.
        isolation: The sandbox isolation word, "" when unknown.
        network: The network the run's commands join ("session", "host"), "" when unknown.
        verify_gate: The pinned gate, its source and execution availability.
        mode: "run", "plan" or "ask"; an ask has no gate to name.
        models: The recorded per-role route, effort and origin lines.
        environment: The journal's current execution target, empty when unknown.
    """

    model: str
    run_commands: str
    isolation: str
    network: str
    verify_gate: manifest.GateStamp
    mode: str = ""
    models: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()

    def gate(self) -> str:
        """Return the gate and its origin.

        An operator's `configured` gate certifies differently from one `inferred`
        off a file the model can edit.

        Returns:
            `gate` plus the shell line with its origin in parentheses, `gate withheld:` plus
            the shell line, or "no verify gate".
        """
        if self.verify_gate.availability == "pending" and not self.verify_gate.command:
            return "verify gate not settled yet"
        if self.verify_gate.availability == "withheld" and self.verify_gate.command:
            return f"gate withheld: {shlex.join(self.verify_gate.command)}"
        if not self.verify_gate.command:
            return "no verify gate"
        origin = self.verify_gate.source or "unknown origin"
        if self.verify_gate.availability == "pending":
            origin += ", pending"
        return f"gate {shlex.join(self.verify_gate.command)} ({origin})"

    def short(self) -> str:
        """Return the compact form for a border or header: isolation, network, commands mode."""
        return short_form(self.isolation, self.network, self.run_commands)

    def line(self) -> str:
        """Return the one-line form every surface shows.

        Returns:
            The model, isolation, network, commands mode and gate joined by " · ", or ""
            for a run whose manifest could not be read (an all-empty policy must not claim
            "no verify gate" about a run it knows nothing of).
        """
        known = (
            self.model,
            self.isolation,
            self.network,
            self.run_commands,
            self.verify_gate.command,
            self.environment,
        )
        if not any(known):
            return ""
        models = self.models or (self.model,)
        parts = [p for p in (*models, self.short(), *self.environment) if p]
        if self.mode != "ask":
            parts.append(self.gate())
        return " · ".join(parts)


def short_form(isolation: str, network: str, run_commands: str) -> str:
    """Return the sandbox settings, each named before its resolved value.

    Args:
        isolation: The resolved `sandbox.isolation` word, "" when unknown.
        network: The resolved `sandbox.network` word, "" when unknown.
        run_commands: The `sandbox.run_commands` word, "" when unknown.

    Returns:
        `isolation full · network session · run_commands ask`, with an unknown setting left out.
    """
    parts = [
        f"isolation {isolation}" if isolation else "",
        f"network {network}" if network else "",
        f"run_commands {run_commands}" if run_commands else "",
    ]
    return " · ".join(p for p in parts if p)


def session_policy(
    session_dir: pathlib.Path,
    *,
    targets: evidence.ExecutionTargets | None = None,
    environment: state_types.EnvironmentView | None = None,
) -> SessionPolicy:
    """Fold a session's manifest and journal into its policy facts.

    Args:
        session_dir: The session's state dir.
        targets: A live view's target fold; None reads the journal's tail.
        environment: A live view's container; None reads the journal's tail when `targets` is
            None too.

    Returns:
        The policy, all-empty when the manifest cannot be read.
    """
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError:
        m = None
    logs = session_dir / layout.LOGS_NAME
    target = targets.current if targets is not None else session_log.current_target(logs)
    if targets is None and environment is None:
        environment = session_log.current_environment(logs)
    return manifest_policy(m, target=target, environment=environment)


def environment_lines(view: state_types.EnvironmentView | None) -> tuple[str, ...]:
    """Return the container line the policy row shows, () outside a container."""
    if view is None:
        return ()
    if not view.role:
        if not view.starting:
            return ()
        return (f"Container: starting ({shlex.join(view.starting)})",)
    line = f"Container: {view.role} ({view.reason})"
    if view.started and view.start_s is not None:
        line += f"; started by this run in {view.start_s:.0f}s"
    ends = {"left_running": "left running", "stopped": "stopped", "unknown": "lifecycle unknown"}
    if view.outcome in ends:
        line += f"; {ends[view.outcome]}"
    return (line,)


def manifest_policy(
    m: manifest.SessionManifest | None,
    *,
    target: evidence.ExecutionTargetEvidence | None = None,
    environment: state_types.EnvironmentView | None = None,
) -> SessionPolicy:
    """Return recorded launch policy with the journal's current target, never the mirror's last."""
    if m is None:
        return SessionPolicy("", "", "", "", manifest.GateStamp())
    driver = m.models.driver
    # A manifest an older agent6 wrote holds the configured word; `auto` names no network.
    network = m.policy.network if m.policy.network in ("session", "host") else ""
    return SessionPolicy(
        model=driver.model if driver else "",
        run_commands=m.policy.run_commands,
        isolation=m.policy.isolation,
        network=network,
        verify_gate=m.harness.gate,
        mode=m.mode,
        models=model_lines(m) if m.models.driver and m.models.driver.effort_source else (),
        environment=(
            *(evidence.target_lines(target.target_id, target.target) if target is not None else ()),
            *environment_lines(environment),
        ),
    )
