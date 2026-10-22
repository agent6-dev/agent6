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
import shlex
from collections.abc import Callable, Iterable, Iterator, Mapping

from agent6 import kinds
from agent6.config import Config, ConfigError, SandboxConfig, io, layer
from agent6.machine import AgentState, MachineSpec, ToolState
from agent6.sandbox import detect
from agent6.tools import policy as tools_policy


class SandboxWidenedError(ConfigError):
    """An overlay's sandbox cannot be honoured.

    A key set wider than the operator's config, or a grant the machine cannot bind safely: one
    below a region another state writes, or reached through a link inside one.

    Attributes:
        problems: One line per key or grant, each naming its table, the values and the remedy.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclasses.dataclass(frozen=True, slots=True)
class StateConfig:
    """The config one state runs under.

    Attributes:
        config: The operator's layers, the machine's `[config]` and the state's `config`
            merged, with the invocation's sandbox flags applied last.
        isolation: The isolation level the state's jail runs at.
    """

    config: Config
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

    Args:
        path: An overlay's grant.
        trees: The operator's grants.

    Returns:
        True when it is.
    """
    real = pathlib.Path(os.path.realpath(path))
    return any(real.is_relative_to(os.path.realpath(tree)) for tree in trees)


def _spelled_under(path: str, trees: Iterable[str]) -> bool:
    """Return whether a path is, as spelled, one of the trees or below one.

    A hidden path is masked where it is spelled, so a link to it elsewhere hides nothing.

    Args:
        path: One of the operator's hidden paths.
        trees: The overlay's hidden paths.

    Returns:
        True when it is.
    """
    spelled = pathlib.PurePath(os.path.normpath(path))
    return any(spelled.is_relative_to(os.path.normpath(tree)) for tree in trees)


_ISOLATION_RANK = {"none": 0, "partial": 1, "full": 2}
_RUN_COMMANDS_RANK = {"no": 0, "ask": 1, "yes": 2}
_PROTECT_GIT_RANK = {"off": 0, "auto": 1, "on": 2}
_HOME_RANK = {"cache": 0, "auto": 1, "tmp": 2}

# Per scalar sandbox key: whether the overlay's value is no wider than the operator's. A key
# with no rule here or in `_COVERS` must equal the operator's value.
_WITHIN: dict[str, Callable[[_Side, _Side], bool]] = {
    "isolation": lambda op, st: _ISOLATION_RANK[st.isolation] >= _ISOLATION_RANK[op.isolation],
    "network": lambda op, st: st.network != "host" or op.network == "host",
    "run_commands": lambda op, st: (
        _RUN_COMMANDS_RANK[st.sandbox.run_commands] <= _RUN_COMMANDS_RANK[op.sandbox.run_commands]
    ),
    "protect_git": lambda op, st: (
        _PROTECT_GIT_RANK[st.sandbox.protect_git] >= _PROTECT_GIT_RANK[op.sandbox.protect_git]
    ),
    "home": lambda op, st: _HOME_RANK[st.sandbox.home] >= _HOME_RANK[op.sandbox.home],
    "memory_limit_mib": lambda op, st: (
        op.sandbox.memory_limit_mib == 0
        or 0 < st.sandbox.memory_limit_mib <= op.sandbox.memory_limit_mib
    ),
}

# Per list sandbox key: whether a sandbox covers one entry. An overlay's grant is within the
# bound when the operator's sandbox covers each of its entries. `hide_paths` runs the other
# way, since hiding more is narrower: the overlay's sandbox covers each of the operator's.
_COVERS: dict[str, Callable[[SandboxConfig, str], bool]] = {
    "fetch_approved_hosts": lambda sandbox, host: (
        "*" in sandbox.fetch_approved_hosts or host in sandbox.fetch_approved_hosts
    ),
    # Write implies read, so a read entry may sit under the write list.
    "extra_read_paths": lambda sandbox, path: _covered(
        path, (*sandbox.extra_read_paths, *sandbox.extra_write_paths)
    ),
    "extra_write_paths": lambda sandbox, path: _covered(path, sandbox.extra_write_paths),
    # A device is one node: the operator's own entry, whatever its spelling.
    "extra_device_paths": lambda sandbox, path: (
        os.path.normpath(path) in {os.path.normpath(p) for p in sandbox.extra_device_paths}
    ),
    "hide_paths": lambda sandbox, path: _spelled_under(path, sandbox.hide_paths),
}


def _grant(verb: str, leaf: str, values: Iterable[str]) -> str:
    """Return the remedy for a key the operator grants: one config command per value.

    Args:
        verb: The `agent6 config` verb: `set` for a scalar key, `add` for list entries.
        leaf: The key under `[sandbox]`.
        values: The values to grant, spelled as `agent6 config` takes them.

    Returns:
        The remedy, unfinished: the caller appends where to drop it from.
    """
    commands = ", ".join(
        f"`agent6 config {verb} --repo sandbox.{leaf} {shlex.quote(value)}`" for value in values
    )
    return (
        f"Grant it in your own config ({commands}, or `agent6 --config FILE machine run` for"
        " this machine alone), or drop it"
    )


def sandbox_widening(
    operator: Config,
    operator_isolation: kinds.IsolationLevel,
    state: Config,
    state_isolation: kinds.IsolationLevel,
) -> dict[str, str]:
    """Return the problem with each sandbox key a config sets wider than the operator's.

    Each problem gives both values as TOML and the remedy: the config command that grants
    the value, or for `hide_paths` the entries the overlay has to keep. `protect_git` is
    compared by what it resolves to: an overlay's `git.control = "model"` turns `auto` off.

    Args:
        operator: The operator's config, the ceiling.
        operator_isolation: The isolation level it resolves to on this host.
        state: The config with the overlays applied.
        state_isolation: The isolation level it resolves to on this host.

    Returns:
        The problems by key, in the model's field order; empty when the config is within
        bounds.
    """
    op, st = _Side(operator, operator_isolation), _Side(state, state_isolation)
    problems: dict[str, str] = {}
    for leaf in SandboxConfig.model_fields:
        ours, theirs = getattr(op.sandbox, leaf), getattr(st.sandbox, leaf)
        if theirs == ours:
            continue
        shown = io.format_toml_value(theirs)
        remedy = _grant("set", leaf, [theirs if isinstance(theirs, str) else shown])
        if covers := _COVERS.get(leaf):
            hide = leaf == "hide_paths"
            bound, entries = (st.sandbox, ours) if hide else (op.sandbox, theirs)
            outside = [entry for entry in entries if not covers(bound, entry)]
            if not outside:
                continue
            kept = ", ".join(io.format_toml_value(entry) for entry in outside)
            remedy = (
                f"Keep {kept} in that list, or drop the key"
                if hide
                else _grant("add", leaf, outside)
            )
        elif (within := _WITHIN.get(leaf)) is not None and within(op, st):
            continue
        problems[leaf] = (
            f"sandbox.{leaf} = {shown}, wider than your config's {io.format_toml_value(ours)}."
            f" {remedy} from the machine file."
        )
    if "protect_git" not in problems and op.config.protects_git and not st.config.protects_git:
        # The same word on both sides, resolved off by the overlay's git.control.
        problems["protect_git"] = (
            f"git.control = {io.format_toml_value(st.config.git.control)} resolves"
            f" sandbox.protect_git = {io.format_toml_value(st.sandbox.protect_git)} to off, wider"
            " than your config's on. Grant it in your own config (`agent6 config set --repo"
            " git.control model`, or `agent6 --config FILE machine run` for this machine alone),"
            " or drop it from the machine file."
        )
    return problems


def resolve_configs(
    spec: MachineSpec,
    cwd: pathlib.Path,
    env: detect.Environment,
    *,
    config_paths: tuple[pathlib.Path, ...] = (),
    disable_sandbox: bool = False,
    auto_approve: bool = False,
    no_commands: bool = False,
) -> MachineConfigs:
    """Resolve the operator's config, the machine's and each state's, and check the bounds.

    The three flags land in every resolved config, so each agent state's overlay and each
    tool state's policy carry them; the supervisor's own environment stays as it was.

    Args:
        spec: The machine.
        cwd: The repository whose config layers apply.
        env: The detected host environment.
        config_paths: The `--config FILE` layers, part of the operator's ceiling.
        disable_sandbox: The invocation's `--disable-sandbox`.
        auto_approve: The invocation's `--auto-approve`.
        no_commands: The invocation's `--no-commands`.

    Returns:
        The configs.

    Raises:
        ConfigError: A layer or an overlay is invalid; the message names the overlay.
        SandboxWidenedError: An overlay sets a sandbox key wider than the operator's; it
            carries every table's problems.
        IsolationUnavailableError: An explicit isolation level is unavailable on this host.
    """

    def with_flags(cfg: Config) -> Config:
        return cfg.with_sandbox_overrides(
            disable_sandbox=disable_sandbox, auto_approve=auto_approve, no_commands=no_commands
        )

    operator_eff = layer.load_effective(cwd, config_paths)
    operator = operator_eff.config
    operator_isolation = detect.resolve_isolation(with_flags(operator).sandbox.isolation, env)

    def one(
        state_overlay: dict[str, object] | None, where: str
    ) -> tuple[StateConfig, dict[str, str]]:
        try:
            eff = layer.load_effective_with_overlay(
                cwd, spec.config, explicit_paths=config_paths, state_overlay=state_overlay
            )
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
        cfg = with_flags(eff.config)
        isolation = detect.resolve_isolation(cfg.sandbox.isolation, env)
        resolved = StateConfig(cfg, isolation)
        return resolved, sandbox_widening(operator, operator_isolation, eff.config, isolation)

    machine, machine_wide = one(None, "the machine's [config]")
    widened = [f"the machine's [config] sets {problem}" for problem in machine_wide.values()]
    states: dict[str, StateConfig] = {}
    for name, state in spec.states.items():
        if not isinstance(state, (AgentState, ToolState)):
            continue
        if not state.config:
            states[name] = machine
            continue
        where = f"[states.{name}.config]"
        states[name], wide = one(dict(state.config), where)
        own = state.config.get("sandbox", {})
        # A key the state's table leaves alone carries the machine's value, reported there.
        widened.extend(
            f"{where} sets {problem}"
            for leaf, problem in wide.items()
            if leaf in own or machine_wide.get(leaf) != problem
        )
    if widened:
        raise SandboxWidenedError(widened)
    configs = MachineConfigs(
        StateConfig(with_flags(operator), operator_isolation),
        machine,
        states,
    )
    if unsafe := unsafe_grants(configs, cwd):
        raise SandboxWidenedError(unsafe)
    return configs


def labelled_configs(configs: MachineConfigs) -> Iterator[tuple[str, StateConfig]]:
    """Yield each distinct config the run uses with the label a refusal names it by.

    Args:
        configs: The run's resolved configs.

    Yields:
        The label and the config, in order: the operator's and the machine's with no label,
        then each state config other than the machine's, labelled `[states.<name>.config]: `.
    """
    yield "", configs.operator
    yield "", configs.machine
    for name, resolved in configs.states.items():
        if resolved is not configs.machine:
            yield f"[states.{name}.config]: ", resolved


def shared_write_regions(configs: MachineConfigs, cwd: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return the regions any state of the machine may write.

    Every state's grants sit at or below the operator's, so the ceiling's write grants cover
    what any state's jailed command can plant a link in. A state off `full` writes the host's
    `/tmp` too; a `full` state's `/tmp` is private, yet its grants there are bound from the host.

    Args:
        configs: The run's resolved configs.
        cwd: The repository.

    Returns:
        The resolved paths.
    """
    ceiling = tools_policy.resolve_grants(configs.operator.config, cwd)
    regions = tuple(real for _p, real in ceiling.write)
    if any(sc.isolation != "full" for sc in (configs.machine, *configs.states.values())):
        regions += (tools_policy.HOST_TMP,)
    return regions


def unsafe_grants(configs: MachineConfigs, cwd: pathlib.Path) -> list[str]:
    """Return one line per state grant the machine cannot bind safely.

    A run drops such a grant with a warning at its start; a machine runs unattended, so it
    refuses before any state runs.

    Args:
        configs: The run's resolved configs.
        cwd: The repository.

    Returns:
        The lines, each labelled with the table that holds the grant; empty when every grant stands.
    """
    shared = shared_write_regions(configs, cwd)
    return [
        f"{label}{dropped.line()}"
        for label, resolved in labelled_configs(configs)
        if resolved is not configs.operator
        for dropped in tools_policy.resolve_grants(
            resolved.config, cwd, shared_writes=shared
        ).dropped
    ]
