# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The surface models.

`[skills]`, `[machine]`, `[web]`, `[notify]` and `[parallel]`.
"""

from __future__ import annotations

import ipaddress
from typing import Literal

import pydantic

from agent6.config import _base


class SkillsConfig(pydantic.BaseModel):
    """The `[skills]` table: operator-installed SKILL.md packs (agentskills.io).

    Skills live under `<data-dir>/skills/<name>/` plus any `extra_dirs`; installed means
    enabled, and `state` holds only the exceptions. A skill is trusted like config, and the
    loader executes nothing in it.
    """

    model_config = _base.MODEL_CONFIG

    enabled: bool = pydantic.Field(
        default=True,
        description=(
            "Master switch for skills: `false` means no skill index in the prompt, no `use_skill` "
            "tool, and no slash commands."
        ),
    )
    # Each entry may hold skill subdirectories or be a single skill dir itself.
    extra_dirs: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "Additional directories scanned for skills, before the installed skills dir; a skill "
            "of the same name in an earlier dir wins."
        ),
    )
    # One value per skill, so contradictory states are unrepresentable.
    state: dict[str, Literal["enabled", "disabled", "always"]] = pydantic.Field(
        default_factory=dict,
        description=(
            "Per-skill state by name: `enabled` (indexed, loaded on `use_skill`), `disabled` "
            "(dropped), or `always` (its full text sits in the system prompt). Layers merge key by "
            "key; `agent6 skills enable|disable [--repo]` writes it."
        ),
    )


class MachineNotifyConfig(pydantic.BaseModel):
    """The `[machine.notify]` table: a hook run on each `machine.notify` and at `machine.end`.

    The argv is operator-controlled, never carries LLM output, and runs on the host outside
    the jail like `[notify].on_complete`; a failed hook is logged and leaves the exit code.
    Its environment carries:

    - `AGENT6_MACHINE_ID`: the machine id
    - `AGENT6_MACHINE_DIR`: the absolute path of the instance dir
    - `AGENT6_MACHINE_EVENT`: `notify` or `end`
    - `AGENT6_MACHINE_STATE`: the state that emitted it
    - `AGENT6_MACHINE_MESSAGE`: the notify message, or the end reason
    - `AGENT6_MACHINE_LEVEL`: `info`, `warn` or `error` for notify; `ok` or `failed` for end
    """

    model_config = _base.MODEL_CONFIG

    on_event: _base.Argv = pydantic.Field(
        default=(),
        description=(
            "A command run on every machine notify event and at the machine's end, as argv (no "
            "shell), with the event in `AGENT6_MACHINE_*` variables. Empty: no hook."
        ),
    )
    timeout_s: float = pydantic.Field(
        gt=0.0,
        default=30.0,
        description="Seconds the hook may run before it is killed.",
    )


class MachineConfig(pydantic.BaseModel):
    """The `[machine]` table: the `agent6 machine run` runtime knobs."""

    model_config = _base.MODEL_CONFIG

    # Old snapshots are an audit convenience, not state: recovery reads the latest only.
    snapshot_keep: int = pydantic.Field(
        ge=0,
        default=5,
        description=(
            "How many blackboard snapshots a machine instance keeps (`machine status` reads the "
            "latest; recovery and `machine replay` fold the journal). `0` keeps all."
        ),
    )
    state_log_keep: int = pydantic.Field(
        ge=0,
        default=50,
        description=(
            "How many per-state log dirs a machine instance keeps under `<instance>/states/` (the "
            "watchable logs of each state's execution; the journal keeps the full transition "
            "history regardless). `0` keeps all."
        ),
    )
    notify: MachineNotifyConfig = pydantic.Field(default_factory=MachineNotifyConfig)
    pass_env: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "Environment variable names a machine's `tool` state may receive from the"
            " operator's environment when its own `pass_env` names them; a state naming one"
            " not listed here refuses the run at startup. Global/repo config only (a machine"
            " `[config]` overlay setting it is rejected); a provider's `api_key_env` is never"
            " allowed."
        ),
    )


def is_loopback_host(host: str) -> bool:
    """Return whether the host is a loopback bind.

    The one owner of the web UI's secure-by-default gate; a wildcard (`0.0.0.0`, `::`) is
    not loopback.

    Args:
        host: A hostname or address, brackets allowed around an IPv6 address.

    Returns:
        True for `localhost` and any loopback IP.
    """
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


class WebConfig(pydantic.BaseModel):
    """The `[web]` table: the server bind, loopback by default.

    Remote access goes behind `tailscale serve` in front of the loopback bind; the tailnet
    identity is the access control, so there is no app-level auth.
    """

    model_config = _base.MODEL_CONFIG

    host: str = pydantic.Field(
        default="127.0.0.1",
        description=(
            "Address `agent6 web` binds; a non-loopback address also needs `allow_non_loopback = "
            "true`."
        ),
    )
    port: int = pydantic.Field(
        ge=1,
        le=65535,
        default=7658,
        description="Port `agent6 web` listens on.",
    )
    allow_non_loopback: bool = pydantic.Field(
        default=False,
        description=(
            "Allow `host` to be a non-loopback address, so a typo can never silently expose the "
            "write surface (approvals, steers, config writes) beyond this machine."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _guard_non_loopback(self) -> WebConfig:
        """Refuse a non-loopback host without the opt-in.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `host` is not loopback and `allow_non_loopback` is off.
        """
        if not is_loopback_host(self.host) and not self.allow_non_loopback:
            raise ValueError(
                f"[web].host = {self.host!r} is not loopback. Binding a non-loopback"
                " address exposes the web UI's write surface; set [web]"
                " allow_non_loopback = true to opt in (and prefer `tailscale serve`"
                " in front of a 127.0.0.1 bind instead)."
            )
        return self


class NotifyConfig(pydantic.BaseModel):
    """The `[notify]` table: a hook run after a run or resume ends.

    The argv is operator-controlled, never carries LLM output, and runs outside the jail
    under `child_env.curated_env` (never a provider key); a failed hook is logged and leaves
    the exit code. Its environment carries:

    - `AGENT6_SESSION_ID`: the session id under the per-repo state dir
    - `AGENT6_SESSION_DIR`: the absolute path of the session dir
    - `AGENT6_SESSION_OK`: `1` when the harness finished cleanly, else `0`
    - `AGENT6_SESSION_REASON`: the end reason (`finish_session`, `budget_exhausted`, ...)
    - `AGENT6_SESSION_VERIFIED`: the gate's verdict, `passed`, `failed`, `unverified` or
      `not_applicable`; a hook wanting green reads this, not `OK`
    """

    model_config = _base.MODEL_CONFIG

    on_complete: _base.Argv = pydantic.Field(
        default=(),
        description=(
            "A command run when a run or resume ends, as argv (no shell), with "
            "`AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON` in its environment. Empty: no hook."
        ),
    )
    timeout_s: float = pydantic.Field(
        gt=0.0,
        default=30.0,
        description="Seconds the hook may run before it is killed.",
    )


class ParallelConfig(pydantic.BaseModel):
    """The `[parallel]` table: the bounds and placement of a `--parallel` fan-out."""

    model_config = _base.MODEL_CONFIG

    # `le` bounds the cap itself, or a huge max_lanes re-opens the allocation parse_spec refuses.
    # Static, not CPU-derived: lanes are I/O-bound and the same config must load on every box.
    max_lanes: int = pydantic.Field(
        ge=1,
        le=1024,
        default=4,
        description=(
            "The most lanes one `--parallel` fan-out may run, `1` to `1024`; a spec asking for "
            "more is refused before anything is cloned."
        ),
    )
    # A lane: `<workdir>/<repo-id>/<fanout-id>/lane-<i>`; a fork: `<workdir>/<repo-id>/<fork-id>`.
    workdir: str = pydantic.Field(
        default="",
        description=(
            "Base directory for the working trees lanes, machine run states, and forks work "
            "in, in a per-repo subdirectory. Empty: `<cache_dir>/parallel`. A lane's clone is "
            "removed after its work is imported; a fork's worktree by `sessions prune` once "
            "the fork is merged."
        ),
    )
