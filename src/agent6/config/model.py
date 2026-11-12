# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `Config` model: TOML to pydantic at the trust boundary.

Every field has a default and the security-sensitive ones default safe; push, `--force`
and history rewrites have no knob at all. A provider and key, which a run cannot guess,
are checked by `Config.require_runnable` rather than at load, so `config show` always works.
"""

from __future__ import annotations

import pathlib
import tomllib
from collections.abc import Callable
from typing import Annotated, Any, Literal

import pydantic

from agent6 import child_env, kinds
from agent6.config import (
    container_config,
    git_config,
    harness_config,
    io,
    model_roles,
    provider_config,
    sandbox_config,
    surface_config,
    validation,
)


def preset_name(name: str) -> str:
    """Refuse an empty preset selector.

    Args:
        name: The selected name.

    Returns:
        The non-empty name.

    Raises:
        ConfigError: The selector is empty.
    """
    if not name:
        raise validation.ConfigError('preset must not be empty; use "none" to select no preset.')
    return name


class Agent6Section(pydantic.BaseModel):
    """The `[agent6]` table."""

    model_config = validation.MODEL_CONFIG

    config_version: int = pydantic.Field(
        ge=1,
        le=1,
        default=1,
        description="Config schema version; only `1` is accepted.",
    )


class Config(pydantic.BaseModel):
    """The validated effective config, one immutable object per load.

    Frozen at the attribute level only; every derived config goes through a `with_*` copier,
    never in-place mutation.
    """

    model_config = validation.MODEL_CONFIG

    agent6: Agent6Section = pydantic.Field(default_factory=Agent6Section)
    providers: dict[str, provider_config.ProviderEntry] = pydantic.Field(
        default_factory=dict,
        description=(
            "provider endpoints by name (`[providers.<name>]`), referenced by `[models.*]` "
            "roles; names: one or more ASCII letters, digits, `-` or `_`; written by `agent6 "
            "connect`"
        ),
    )
    models: model_roles.ModelsConfig = pydantic.Field(default_factory=model_roles.ModelsConfig)
    sandbox: sandbox_config.SandboxConfig = pydantic.Field(
        default_factory=sandbox_config.SandboxConfig
    )
    container: container_config.ContainerConfig = pydantic.Field(
        default_factory=container_config.ContainerConfig
    )
    git: git_config.GitConfig = pydantic.Field(default_factory=git_config.GitConfig)
    harness: harness_config.HarnessConfig = pydantic.Field(
        default_factory=harness_config.HarnessConfig
    )
    review: harness_config.ReviewConfig = pydantic.Field(
        default_factory=harness_config.ReviewConfig
    )
    compaction: harness_config.CompactionConfig = pydantic.Field(
        default_factory=harness_config.CompactionConfig
    )
    prompt: harness_config.PromptConfig = pydantic.Field(
        default_factory=harness_config.PromptConfig
    )
    skills: surface_config.SkillsConfig = pydantic.Field(
        default_factory=surface_config.SkillsConfig
    )
    budget: harness_config.BudgetConfig = pydantic.Field(
        default_factory=harness_config.BudgetConfig
    )
    machine: surface_config.MachineConfig = pydantic.Field(
        default_factory=surface_config.MachineConfig
    )
    notify: surface_config.NotifyConfig = pydantic.Field(
        default_factory=surface_config.NotifyConfig
    )
    mcp: sandbox_config.MCPConfig = pydantic.Field(default_factory=sandbox_config.MCPConfig)
    web: surface_config.WebConfig = pydantic.Field(default_factory=surface_config.WebConfig)
    telemetry: surface_config.TelemetryConfig = pydantic.Field(
        default_factory=surface_config.TelemetryConfig
    )
    parallel: surface_config.ParallelConfig = pydantic.Field(
        default_factory=surface_config.ParallelConfig
    )
    workspace: surface_config.WorkspaceConfig = pydantic.Field(
        default_factory=surface_config.WorkspaceConfig
    )
    # The injection order and stacking rules live in config.layer_presets.apply_preset.
    preset: Annotated[str, pydantic.AfterValidator(preset_name)] = pydantic.Field(
        default="none",
        description=(
            'selected preset; `preset = "none"`: no preset, stop preset lookup; `preset = '
            '"no_review"`: panel off; `preset = "review"`: three seats able to block the '
            'finish; `preset = "deep_review"`: five repo-scope seats able to block the finish; '
            "custom preset: `[presets.<name>]`; overrides at the selecting layer; `--preset`: "
            "per run; `resume --preset`: from that resume on, recorded for later resumes; "
            "empty string refused"
        ),
    )

    @pydantic.field_validator("providers", mode="before")
    @classmethod
    def _refuse_old_backend_key(cls, entries: Any) -> Any:
        """Refuse the old discriminator with the replacement line.

        Args:
            entries: The raw provider table.

        Returns:
            The table unchanged.

        Raises:
            ValueError: An entry holds the old key.
        """
        if isinstance(entries, dict):
            for name, entry in entries.items():
                if isinstance(entry, dict) and "api_format" in entry:
                    replacement = io.format_toml_value(entry["api_format"])
                    raise ValueError(
                        f"[providers.{name}] api_format was renamed: backend = {replacement}"
                    )
        return entries

    @pydantic.field_validator("notify", mode="before")
    @classmethod
    def _refuse_old_notify_key(cls, table: Any) -> Any:
        """Refuse the renamed hook key with the replacement line.

        Args:
            table: The raw `[notify]` table.

        Returns:
            The table unchanged.

        Raises:
            ValueError: The table holds `on_complete`.
        """
        if isinstance(table, dict) and "on_complete" in table:
            replacement = io.format_toml_value(table["on_complete"])
            raise ValueError(f"[notify] on_complete was renamed: on_event = {replacement}")
        return table

    @pydantic.field_validator("machine", mode="before")
    @classmethod
    def _refuse_moved_machine_notify(cls, table: Any) -> Any:
        """Refuse the moved `[machine.notify]` table, naming its one replacement.

        Args:
            table: The raw `[machine]` table.

        Returns:
            The table unchanged.

        Raises:
            ValueError: The table holds `notify`.
        """
        if isinstance(table, dict) and "notify" in table:
            raise ValueError(
                "[machine.notify] moved: set [notify] on_event, the one hook for every notification"
            )
        return table

    @pydantic.field_validator("providers")
    @classmethod
    def _valid_provider_names(
        cls, entries: dict[str, provider_config.ProviderEntry]
    ) -> dict[str, provider_config.ProviderEntry]:
        """Refuse names the config editors cannot write as one bare key.

        Args:
            entries: The provider entries by name.

        Returns:
            The entries unchanged.

        Raises:
            ValueError: A name is empty or not a TOML bare key.
        """
        for name in entries:
            if refusal := provider_config.provider_name_refusal(name):
                raise ValueError(refusal)
        return entries

    def model_post_init(self, context: Any, /) -> None:
        """Register this config's provider-key variable names with `child_env`.

        A child spawned outside the jail never inherits one, whichever entry point built the
        config.

        Args:
            context: Pydantic's validation context, unused.
        """
        child_env.register_provider_keys(self.provider_key_env_names())

    def provider_key_env_names(self) -> frozenset[str]:
        """Return the `api_key_env` names the configured providers resolve keys from.

        Returns:
            The names; a Claude Code entry holds no key.
        """
        return frozenset(
            e.api_key_env
            for e in self.providers.values()
            if not isinstance(e, provider_config.ClaudeCodeProviderEntry) and e.api_key_env
        )

    @pydantic.model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        """Refuse a route naming a provider absent from a non-empty `[providers]`.

        An empty or partial config is valid at load; `require_runnable` checks completeness.

        Returns:
            The model unchanged.

        Raises:
            ValueError: A role or pinned reviewer names an unknown provider.
        """
        routes = {f"models.{role}.model": rm for role, rm in self.models.configured().items()}
        routes.update(
            (f"review.reviewers.{index}.model", route)
            for index, entry in enumerate(self.review.reviewers)
            if (route := entry.resolve(None)) is not None
        )
        for key, rm in routes.items():
            if self.providers and rm.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                raise ValueError(
                    f"{key} = {f'{rm.provider}/{rm.model}'!r} but"
                    f" [providers.{rm.provider}] is not configured."
                    f" Known providers: {known}."
                )
        return self

    @pydantic.model_validator(mode="after")
    def _model_git_control_needs_git_writes(self) -> Config:
        """Refuse `git.control = "model"` beside `protect_git = "on"`: the model must write .git.

        Returns:
            The model unchanged.

        Raises:
            ValueError: Both are set.
        """
        if self.git.control == "model" and self.sandbox.protect_git == "on":
            raise ValueError(
                'git.control = "model" needs the model to write .git;'
                ' set sandbox.protect_git = "auto" or "off" (or keep control = "agent6").'
            )
        return self

    @property
    def protects_git(self) -> bool:
        """Whether `.git` is kept from the model: `on`, or `auto` unless the model controls git."""
        word = self.sandbox.protect_git
        return word == "on" or (word == "auto" and self.git.control != "model")

    @pydantic.model_validator(mode="after")
    def _pass_env_excludes_provider_keys(self) -> Config:
        """Refuse a `pass_env` naming a provider's `api_key_env`.

        The invariant lives here so a direct config edit cannot bypass it; `mcp add`
        pre-checks the same rule for an earlier refusal.

        Returns:
            The model unchanged.

        Raises:
            ValueError: An MCP server's or the machine's `pass_env` names a provider key.
        """
        keys = self.provider_key_env_names()
        lists = [
            (f"[mcp.servers.{name}].pass_env", srv.pass_env, "an MCP server")
            for name, srv in self.mcp.servers.items()
        ]
        lists.append(("[machine].pass_env", self.machine.pass_env, "a machine's tool"))
        for where, names, who in lists:
            leaked = sorted(keys.intersection(names))
            if leaked:
                raise ValueError(
                    f"{where} names provider API key env var(s) {', '.join(leaked)};"
                    f" agent6 never passes a provider key to {who}."
                )
        return self

    @pydantic.model_validator(mode="after")
    def _check_review_model_quorum(self) -> Config:
        """Refuse a blocking threshold the resolved reviewer routes cannot reach.

        Returns:
            The validated config.

        Raises:
            ValueError: Too few distinct provider/model routes can block.
        """
        if self.review.block_on == "model_quorum":
            reviewer = self.models.resolve("reviewer")
            resolved = (
                [entry.resolve(reviewer) for entry in self.review.reviewers]
                if self.review.reviewers
                else [reviewer]
            )
            routes = {(rm.provider, rm.model) for rm in resolved if rm is not None}
            required = self.review.min_blocking_models
            if len(routes) < required:
                raise ValueError(
                    f"review.block_on='model_quorum' with min_blocking_models={required} "
                    f"needs >= {required} distinct provider/model routes; resolved {len(routes)}. "
                    "Set review.reviewers to distinct routes, or use block_on='any'."
                )
        return self

    def with_budget_overrides(
        self,
        *,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
        max_plan_points: float | None = None,
    ) -> Config:
        """Return a copy with the per-run budget flags applied.

        Args:
            max_usd: The `--max-usd` value, or None to keep the config's.
            max_tokens_fallback: The `--max-tokens-fallback` value, or None.
            max_plan_points: The `--max-plan-points` value, or None.

        Returns:
            The config with the given caps; self when none is given.
        """
        if max_usd is None and max_tokens_fallback is None and max_plan_points is None:
            return self
        data = self.model_dump(mode="python")
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        if max_plan_points is not None:
            budget["max_plan_points"] = max_plan_points
        return Config.model_validate(data)

    def model_route(self, role: kinds.RoleName, spec: str) -> kinds.ModelRoute:
        """Parse a `[provider/]model` value into the route it names for a role.

        A first segment naming a configured provider is the provider; otherwise the whole
        value is a model id on the role's current provider, so an OpenRouter id with its own
        slash stays one id.

        Args:
            role: The role whose current provider an unprefixed id lands on.
            spec: The value.

        Returns:
            The route.

        Raises:
            ConfigError: The value is empty, names no model, or names no provider while the
                role has none.
        """
        raw_spec = spec
        spec = spec.strip()
        if not spec:
            raise validation.ConfigError(f"{raw_spec!r}: no model id.")
        provider, slash, model = spec.partition("/")
        known = ", ".join(sorted(self.providers)) or "(none)"
        if slash and not provider:
            raise validation.ConfigError(
                f"{raw_spec!r}: name the provider as provider/model"
                f" (configured providers: {known})."
            )
        if slash and not model:
            raise validation.ConfigError(f"{raw_spec!r}: no model id after the slash.")
        if not slash or provider not in self.providers:
            current = self.models.resolve(role)
            if current is None:
                raise validation.ConfigError(
                    f"{raw_spec!r}: name the provider as provider/model"
                    f" (configured providers: {known})."
                )
            provider, model = current.provider, spec
        return kinds.ModelRoute(provider, model)

    def with_model_route(self, role: kinds.RoleName, route: kinds.ModelRoute) -> Config:
        """Return a copy whose role runs a route, keeping the role's effort and temperature.

        Args:
            role: The role.
            route: The provider and model.

        Returns:
            The config with the role rerouted.
        """
        base = getattr(self.models, role)
        entry = base.model_dump(mode="python", exclude_unset=True) if base is not None else {}
        entry["model"] = f"{route.provider}/{route.model}"
        data = self.model_dump(mode="python")
        data.setdefault("models", {})[role] = entry
        return Config.model_validate(data)

    def with_sandbox_overrides(
        self,
        *,
        disable_sandbox: bool = False,
        auto_approve: bool = False,
        no_commands: bool = False,
    ) -> Config:
        """Return a copy with the per-invocation sandbox flags applied.

        Every flag is operator-supplied; the model reaches none of them.

        Args:
            disable_sandbox: Force `sandbox.isolation = "none"`.
            auto_approve: Turn `run_commands` `ask` into `allow` and every MCP server's
                `approve` too; a configured `deny` stays, since a flag never grants what the
                standing policy denied.
            no_commands: Pin `run_commands` to `deny`; tightening needs no permission.

        Returns:
            The config with the flags applied; self when none is set.
        """
        if not disable_sandbox and not auto_approve and not no_commands:
            return self
        data = self.model_dump(mode="python")
        sandbox = data.setdefault("sandbox", {})
        if disable_sandbox:
            sandbox["isolation"] = "none"
        if auto_approve and self.sandbox.run_commands != "deny":
            sandbox["run_commands"] = "allow"
        if auto_approve:
            for server in data.get("mcp", {}).get("servers", {}).values():
                server["approve"] = "allow"
        if no_commands:
            sandbox["run_commands"] = "deny"
        return Config.model_validate(data)

    def with_machine_agent_overrides(
        self,
        *,
        effort: str | None = None,
        max_usd: float | None = None,
        remaining_usd: float | None = None,
        max_plan_points: float | None = None,
    ) -> Config:
        """Return a copy with a machine agent's effort and spend caps applied.

        Args:
            effort: The worker's effort, or None to keep the config's.
            max_usd: The state's dollar cap, which can only tighten the config's.
            remaining_usd: The machine's unspent budget, or None when uncapped.
            max_plan_points: The `budget.max_plan_points` cap, or None to keep the config's.

        Returns:
            The config, revalidated.
        """
        data = self.model_dump(mode="python")
        if effort is not None:
            worker = data.setdefault("models", {}).get("worker")
            if worker is None:
                worker = {}
                data["models"]["worker"] = worker
            worker["effort"] = effort
        caps = (self.budget.max_usd, max_usd, remaining_usd)
        data.setdefault("budget", {})["max_usd"] = min(
            (cap for cap in caps if cap is not None and cap != -1), default=-1.0
        )
        if max_plan_points is not None:
            data.setdefault("budget", {})["max_plan_points"] = max_plan_points
        return Config.model_validate(data)

    def with_verify_command(self, argv: tuple[str, ...]) -> Config:
        """Return a copy whose `harness.verify_command` is the argv, `()` for a gateless run.

        Runs never write config: the operator is shown what was picked and can pin it.

        Args:
            argv: The gate command.

        Returns:
            The config with the gate set.
        """
        data = self.model_dump(mode="python")
        data.setdefault("harness", {})["verify_command"] = list(argv)
        return Config.model_validate(data)

    def with_setup_command(self, argv: tuple[str, ...]) -> Config:
        """Return a copy whose `workspace.setup_command` is the argv AGENTS.md states.

        Args:
            argv: The setup command.

        Returns:
            The config with the setup command set.
        """
        data = self.model_dump(mode="python")
        data.setdefault("workspace", {})["setup_command"] = list(argv)
        return Config.model_validate(data)

    def cleartext_credential_endpoints(self) -> tuple[str, ...]:
        """Return the endpoints that send a credential over plain http off loopback.

        Returns:
            `[table] url` labels for the run-entry warning: every provider, and each MCP server
            a run starts.
        """
        out: list[str] = []
        for name, entry in sorted(self.providers.items()):
            if (
                not isinstance(entry, provider_config.ClaudeCodeProviderEntry)
                and sandbox_config.is_cleartext_url(entry.base_url)
                and entry.auth_style != "none"
                and not sandbox_config.is_loopback_url(entry.base_url)
            ):
                out.append(f"[providers.{name}] {entry.base_url}")
        for name, srv in sorted(self.mcp.enabled_servers.items()):
            if (
                srv.token_env
                and sandbox_config.is_cleartext_url(srv.url)
                and not sandbox_config.is_loopback_url(srv.url)
            ):
                out.append(f"[mcp.servers.{name}] {srv.url}")
        return tuple(out)

    def with_run_commands_clamped(self) -> Config:
        """Clamp `sandbox.run_commands` from `allow` to `ask` for an interactive mode.

        Ask and plan run with the operator present, often outside a repo, so nothing runs
        unwatched; `deny` stays, since a run never loosens a boundary the operator set.

        Returns:
            The clamped config; self when nothing changes.
        """
        if self.sandbox.run_commands != "allow":
            return self
        data = self.model_dump(mode="python")
        data.setdefault("sandbox", {})["run_commands"] = "ask"
        return Config.model_validate(data)

    def with_hidden_in_checkout(self, origin: pathlib.Path, checkout: pathlib.Path) -> Config:
        """Hide each `sandbox.hide_paths` entry under origin at its path under checkout too.

        A worktree holds its own copy of every tracked file, so an entry named by its path
        in the originating repository hides nothing there on its own.

        Args:
            origin: The repository the entries name.
            checkout: The worktree whose copies are hidden as well.

        Returns:
            The widened config; self when no entry lies under origin.
        """
        origin, checkout = origin.resolve(), checkout.resolve()
        added = [
            str(checkout / resolved.relative_to(origin))
            for resolved in (pathlib.Path(entry).resolve() for entry in self.sandbox.hide_paths)
            if resolved != origin and resolved.is_relative_to(origin)
        ]
        if not added:
            return self
        data = self.model_dump(mode="python")
        data.setdefault("sandbox", {})["hide_paths"] = list(
            dict.fromkeys((*self.sandbox.hide_paths, *added))
        )
        return Config.model_validate(data)

    def with_decompose(self, value: Literal["on", "off"]) -> Config:
        """Return a copy with `prompt.decompose` pinned, so the engine never sees `auto`.

        Args:
            value: The resolved setting.

        Returns:
            The config with the setting pinned.
        """
        data = self.model_dump(mode="python")
        data.setdefault("prompt", {})["decompose"] = value
        return Config.model_validate(data)

    def require_runnable(self, role: kinds.RoleName = "worker") -> None:
        """Refuse unless the role can run: a provider exists and the role resolves onto one.

        Each message names the command that fixes the gap. A verify command is not
        required; `agent6.tools.verify_infer` infers one.

        Args:
            role: The role the command uses.

        Raises:
            ConfigError: No provider, no model for the role, or the model's provider is
                unknown.
        """
        if not self.providers:
            raise validation.ConfigError(
                "No providers configured. Run `agent6 connect` to add one"
                " (stored in your global config), or add a [providers.*]"
                " block to the per-repo config."
            )
        if self.models.default is None or not self.models.default.model:
            legacy = (
                " [models.worker] is an optional override, not the default;"
                " rename [models.worker] to [models.default] to keep it as the baseline."
                if self.models.worker is not None
                else ""
            )
            raise validation.ConfigError(
                "No models.default.model configured. Run `agent6 model default PROVIDER/MODEL`"
                ' or add a [models.default] block with model = "PROVIDER/MODEL".' + legacy
            )
        rm = self.models.resolve(role)
        assert rm is not None  # default supplies every role's route
        if rm.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise validation.ConfigError(
                f"models.{role}.model = {f'{rm.provider}/{rm.model}'!r} but"
                f" [providers.{rm.provider}]"
                f" is not configured. Known providers: {known}."
            )


# pydantic's own message for a missing discriminator names neither the key nor its values.
_MISSING_BACKEND = (
    'set backend = "anthropic", "openai", "chatgpt", "responses" or "claude_code"'
    " (see docs/config.md)"
)


def _format_validation_error(
    err: pydantic.ValidationError,
    source: str,
    locate: Callable[[str, str], str | None] | None = None,
) -> str:
    """Render a validation error as one line per issue, with the locator's hint under each.

    Args:
        err: The validation error.
        source: The name of what was validated.
        locate: A function from (dotted leaf, error type) to a hint line, or None.

    Returns:
        The message.
    """
    lines = [f"Config validation failed: {source}"]
    for issue in err.errors():
        loc = io.render_key(str(part) for part in issue["loc"]) or "<root>"
        msg = issue["msg"]
        if issue["type"] == "union_tag_not_found" and loc.startswith("providers."):
            msg = _MISSING_BACKEND
        elif issue["type"] == "extra_forbidden" and loc == "context":
            msg = "rename the context table to [compaction]; keep its keys and values"
        elif issue["type"] == "extra_forbidden" and loc == "harness.verify_settled_stop_after":
            msg = "unknown config key; use harness.settled_stop_after with the same value"
        elif issue["type"] == "extra_forbidden" and loc == "prompt.revise_prompt":
            msg = "removed config key; delete prompt.revise_prompt"
        lines.append(f"  - {loc}: {msg} (type={issue['type']})")
        if locate is not None and (where := locate(loc, issue["type"])):
            lines.append(where)
    return "\n".join(lines)


def validate_config(
    raw: dict[str, object],
    *,
    source: str = "<config>",
    locate: Callable[[str, str], str | None] | None = None,
) -> Config:
    """Validate a parsed, possibly layer-merged, config table.

    `load_config` and the layered loader share it, so both surface identical errors.

    Args:
        raw: The table.
        source: The name errors report.
        locate: A function from (dotted leaf, pydantic error type) to a hint naming the
            file and the fix, appended under the error line; or None.

    Returns:
        The validated config.

    Raises:
        ConfigError: Validation failed; the message points at each field.
    """
    try:
        return Config.model_validate(raw)
    except pydantic.ValidationError as exc:
        raise validation.ConfigError(_format_validation_error(exc, source, locate)) from exc


def load_config(path: pathlib.Path) -> Config:
    """Load and validate one TOML config file.

    Args:
        path: The file.

    Returns:
        The validated config.

    Raises:
        ConfigError: The file is missing, unreadable, not TOML, or invalid.
    """
    if not path.is_file():
        raise validation.ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise validation.ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise validation.ConfigError(f"Config file cannot be read ({path}): {exc}") from exc
    return validate_config(raw, source=str(path))
