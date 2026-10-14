# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve one config per machine state, each overlay's sandbox bounded by the operator's.

The operator's layers alone (defaults, global, repo, preset, `--config FILE`) are the ceiling.
The machine's `[config]` and a state's `config` may set `sandbox` keys no wider than it, so
the machine narrows for every state and a state sets its own value back up to the operator's.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from collections.abc import Callable, Iterable, Mapping

from agent6 import kinds
from agent6.config import Config, ConfigError, SandboxConfig, layer
from agent6.machine import AgentState, MachineSpec, ToolState
from agent6.sandbox import detect
from agent6.tools import policy as tools_policy


class SandboxWidenedError(ConfigError):
    """An overlay sets a sandbox key wider than the operator's config."""


@dataclasses.dataclass(frozen=True, slots=True)
class StateConfig:
    """The config one state runs under.

    Attributes:
        config: The operator's layers, the machine's `[config]` and the state's `config`
            merged, with the invocation's sandbox flags applied last.
        explicit_leaves: The leaves a layer set, which refuse rather than degrade.
        isolation: The isolation level the state's jail runs at.
    """

    config: Config
    explicit_leaves: frozenset[str]
    isolation: kinds.IsolationLevel


@dataclasses.dataclass(frozen=True, slots=True)
class MachineConfigs:
    """Every config a machine run reads.

    Attributes:
        operator: The operator's layers alone: the ceiling, and where the allowlists are
            read.
        machine: The machine-level config: what the supervisor reads once, and the config
            of a state with no `config` table.
        states: The config of each agent and tool state, by name.
    """

    operator: StateConfig
    machine: StateConfig
    states: Mapping[str, StateConfig]


@dataclasses.dataclass(frozen=True, slots=True)
class _Side:
    """One side of a bound: a config and the isolation level it resolves to."""

    config: Config
    isolation: kinds.IsolationLevel

    @property
    def sandbox(self) -> SandboxConfig:
        """The side's `[sandbox]` table."""
        return self.config.sandbox

    @property
    def network(self) -> kinds.NetworkMode:
        """The network a jailed child of this side joins on this host."""
        return tools_policy.resolve_network(self.config, self.isolation)


def _covered(path: str, trees: Iterable[str]) -> bool:
    """Return whether a path is at or below one of the trees, each by its real path.

    Real paths, so a symlink inside a tree that points outside it is not covered.
    """
    real = pathlib.Path(os.path.realpath(path))
    return any(real.is_relative_to(os.path.realpath(tree)) for tree in trees)


_ISOLATION_RANK = {"none": 0, "hardened": 1, "strict": 2}
_RUN_COMMANDS_RANK = {"no": 0, "ask": 1, "yes": 2}

# Per sandbox key: whether the overlay's value is no wider than the operator's. A key with
# no rule must equal the operator's value.
_WITHIN: dict[str, Callable[[_Side, _Side], bool]] = {
    "isolation": lambda op, st: _ISOLATION_RANK[st.isolation] >= _ISOLATION_RANK[op.isolation],
    "network": lambda op, st: st.network != "host" or op.network == "host",
    "run_commands": lambda op, st: (
        _RUN_COMMANDS_RANK[st.sandbox.run_commands] <= _RUN_COMMANDS_RANK[op.sandbox.run_commands]
    ),
    "fetch_hosts": lambda op, st: (
        "*" in op.sandbox.fetch_hosts or set(st.sandbox.fetch_hosts) <= set(op.sandbox.fetch_hosts)
    ),
    "protect_git": lambda op, st: st.sandbox.protect_git or not op.sandbox.protect_git,
    "home": lambda op, st: st.sandbox.home in (op.sandbox.home, "tmp"),
    "memory_limit_mb": lambda op, st: (
        op.sandbox.memory_limit_mb == 0
        or 0 < st.sandbox.memory_limit_mb <= op.sandbox.memory_limit_mb
    ),
    # Write implies read, so a read entry may sit under the operator's write list.
    "extra_read_paths": lambda op, st: all(
        _covered(p, (*op.sandbox.extra_read_paths, *op.sandbox.extra_write_paths))
        for p in st.sandbox.extra_read_paths
    ),
    "extra_write_paths": lambda op, st: all(
        _covered(p, op.sandbox.extra_write_paths) for p in st.sandbox.extra_write_paths
    ),
    # A device is one node: the operator's own entry, whatever its spelling.
    "extra_device_paths": lambda op, st: (
        {os.path.normpath(p) for p in st.sandbox.extra_device_paths}
        <= {os.path.normpath(p) for p in op.sandbox.extra_device_paths}
    ),
    # Hiding more is narrower: each of the operator's entries stays under one of the state's.
    "hide_paths": lambda op, st: all(
        _covered(p, st.sandbox.hide_paths) for p in op.sandbox.hide_paths
    ),
}


def sandbox_widening(
    operator: Config,
    operator_isolation: kinds.IsolationLevel,
    state: Config,
    state_isolation: kinds.IsolationLevel,
) -> list[str]:
    """Return one line per sandbox key a config sets wider than the operator's.

    Args:
        operator: The operator's config, the ceiling.
        operator_isolation: The isolation level it resolves to on this host.
        state: The config with the overlays applied.
        state_isolation: The isolation level it resolves to on this host.

    Returns:
        The problems, in the model's field order; empty when the config is within bounds.
    """
    op, st = _Side(operator, operator_isolation), _Side(state, state_isolation)
    problems: list[str] = []
    for leaf in SandboxConfig.model_fields:
        within = _WITHIN.get(leaf)
        ours, theirs = getattr(op.sandbox, leaf), getattr(st.sandbox, leaf)
        if theirs == ours or (within is not None and within(op, st)):
            continue
        shown = list(theirs) if isinstance(theirs, tuple) else theirs
        ceiling = list(ours) if isinstance(ours, tuple) else ours
        problems.append(f"sandbox.{leaf} = {shown!r}, wider than your config's {ceiling!r}")
    return problems


def resolve_configs(
    spec: MachineSpec,
    cwd: pathlib.Path,
    env: detect.Environment,
    *,
    config_path: pathlib.Path | None = None,
    auto_approve: bool = False,
    no_commands: bool = False,
) -> MachineConfigs:
    """Resolve the operator's config, the machine's and each state's, and check the bounds.

    Args:
        spec: The machine.
        cwd: The repository whose config layers apply.
        env: The detected host environment.
        config_path: The `--config FILE` layer, part of the operator's ceiling.
        auto_approve: The invocation's `--auto-approve`.
        no_commands: The invocation's `--no-commands`.

    Returns:
        The configs.

    Raises:
        ConfigError: A layer or an overlay is invalid; the message names the overlay.
        SandboxWidenedError: An overlay sets a sandbox key wider than the operator's.
        IsolationUnavailableError: An explicit isolation level is unavailable on this host.
    """
    operator_eff = layer.load_effective(cwd, config_path)
    operator = operator_eff.config
    operator_isolation = detect.resolve_isolation(operator.sandbox.isolation, env)

    def one(state_overlay: dict[str, object] | None, where: str) -> StateConfig:
        try:
            eff = layer.load_effective_with_overlay(
                cwd, spec.config, explicit_path=config_path, state_overlay=state_overlay
            )
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
        isolation = detect.resolve_isolation(eff.config.sandbox.isolation, env)
        problems = sandbox_widening(operator, operator_isolation, eff.config, isolation)
        if problems:
            raise SandboxWidenedError(
                f"{where} sets {'; '.join(problems)}. An overlay may narrow the sandbox,"
                " never widen it: grant it in your own config (`agent6 config set --repo"
                " sandbox.<key> <value>`, or `agent6 --config FILE machine run` for this"
                " machine alone), or drop it from the machine file."
            )
        return StateConfig(
            eff.config.with_sandbox_overrides(auto_approve=auto_approve, no_commands=no_commands),
            eff.explicit_leaves,
            isolation,
        )

    machine = one(None, "the machine's [config]")
    states = {
        name: one(dict(state.config), f"[states.{name}.config]") if state.config else machine
        for name, state in spec.states.items()
        if isinstance(state, (AgentState, ToolState))
    }
    return MachineConfigs(
        StateConfig(
            operator.with_sandbox_overrides(auto_approve=auto_approve, no_commands=no_commands),
            operator_eff.explicit_leaves,
            operator_isolation,
        ),
        machine,
        states,
    )
