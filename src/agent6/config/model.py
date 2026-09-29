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
from typing import Literal

import pydantic

from agent6 import errors, kinds
from agent6.config import _base, _git, _harness, _providers, _sandbox, _surfaces


class ConfigError(errors.OperatorError):
    """The config file is missing, malformed or invalid: the operator's file, so a refusal."""


EffortLevel = Literal["off", "low", "medium", "high", "xhigh", "max"]


class RoleModel(pydantic.BaseModel):
    """One role's provider and model.

    `temperature` defaults to 0.0: high-temperature sampling degenerates on some open-weights
    models (an observed 15997 literal newline escapes in one `old_string`), and OpenRouter's
    per-model defaults vary, so pinning is what makes a bench reproducible.
    """

    model_config = _base.MODEL_CONFIG

    provider: str = pydantic.Field(
        min_length=1,
        description="A `[providers.<name>]` entry, by name.",
    )
    model: str = pydantic.Field(
        min_length=1,
        description="Model id as that provider names it (`agent6 model` lists them).",
    )
    temperature: float | None = pydantic.Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature pinned on every call, `0.0` to `2.0`. `0.0` keeps tool use "
            "stable. TOML omission uses `0.0`; only the Python API can pass `None` to leave the "
            "provider's default."
        ),
    )
    # Per wire: `reasoning.effort` on OpenAI-family models, `output_config.effort` or a thinking
    # budget on Anthropic ones.
    effort: EffortLevel | None = pydantic.Field(
        default=None,
        description=(
            "Reasoning effort: `off`, `low`, `medium`, `high`, `xhigh`, or `max` (the top tiers "
            "where the model offers them; Anthropic collapses them to its highest). Unset: what "
            "the wire applies, which `agent6 config show` prints resolved (`low` on "
            "openai-compatible reasoning models, no thinking on Anthropic)."
        ),
    )


class ModelsConfig(pydantic.BaseModel):
    """The `[models]` table: the provider and model per role.

    Every role is optional at load; `Config.require_runnable` requires the one a command
    uses. `planner` and `reviewer` fall back to `worker`.
    """

    model_config = _base.MODEL_CONFIG

    worker: RoleModel | None = pydantic.Field(
        default=None,
        description="The `(provider, model)` driving `agent6 run`/`resume`.",
    )
    reviewer: RoleModel | None = pydantic.Field(
        default=None,
        description=(
            "Drives `agent6 review`, the in-loop review panel, the context summariser and"
            " gister, and the prompt reviser. Unset falls back to `worker`."
        ),
    )
    planner: RoleModel | None = pydantic.Field(
        default=None,
        description="Drives `agent6 plan` (the planning pass). Unset falls back to `worker`.",
    )

    def configured(self) -> dict[str, RoleModel]:
        """Return the roles explicitly set.

        Returns:
            The set roles by name.
        """
        out: dict[str, RoleModel] = {}
        if self.worker is not None:
            out["worker"] = self.worker
        if self.reviewer is not None:
            out["reviewer"] = self.reviewer
        if self.planner is not None:
            out["planner"] = self.planner
        return out

    def resolve(self, role: kinds.RoleName) -> RoleModel | None:
        """Return the effective model for a role, with the worker fallback.

        Args:
            role: The role.

        Returns:
            The role's entry, the worker's for an unset planner or reviewer, or None.
        """
        if role == "worker":
            return self.worker
        if role == "planner":
            return self.planner or self.worker
        if role == "reviewer":
            return self.reviewer or self.worker
        return None

    def source_role(self, role: kinds.RoleName) -> kinds.RoleName:
        """Return the configured role `resolve` reads, so an error names the key written.

        Args:
            role: The role.

        Returns:
            The role itself when set, else `worker`.
        """
        return role if role in self.configured() else "worker"


class Agent6Section(pydantic.BaseModel):
    """The `[agent6]` table."""

    model_config = _base.MODEL_CONFIG

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

    model_config = _base.MODEL_CONFIG

    agent6: Agent6Section = pydantic.Field(default_factory=Agent6Section)
    providers: dict[str, _providers.ProviderEntry] = pydantic.Field(
        default_factory=dict,
        description=(
            "Provider endpoints by name (`[providers.<name>]`); a `[models.*]` role names one. "
            "`agent6 connect` writes them."
        ),
    )
    models: ModelsConfig = pydantic.Field(default_factory=ModelsConfig)
    sandbox: _sandbox.SandboxConfig = pydantic.Field(default_factory=_sandbox.SandboxConfig)
    git: _git.GitConfig = pydantic.Field(default_factory=_git.GitConfig)
    harness: _harness.HarnessConfig = pydantic.Field(default_factory=_harness.HarnessConfig)
    review: _harness.ReviewConfig = pydantic.Field(default_factory=_harness.ReviewConfig)
    context: _harness.ContextConfig = pydantic.Field(default_factory=_harness.ContextConfig)
    prompt: _harness.PromptConfig = pydantic.Field(default_factory=_harness.PromptConfig)
    skills: _surfaces.SkillsConfig = pydantic.Field(default_factory=_surfaces.SkillsConfig)
    budget: _harness.BudgetConfig = pydantic.Field(default_factory=_harness.BudgetConfig)
    machine: _surfaces.MachineConfig = pydantic.Field(default_factory=_surfaces.MachineConfig)
    notify: _surfaces.NotifyConfig = pydantic.Field(default_factory=_surfaces.NotifyConfig)
    mcp: _sandbox.MCPConfig = pydantic.Field(default_factory=_sandbox.MCPConfig)
    web: _surfaces.WebConfig = pydantic.Field(default_factory=_surfaces.WebConfig)
    parallel: _surfaces.ParallelConfig = pydantic.Field(default_factory=_surfaces.ParallelConfig)
    # The injection order and stacking rules live in config.layer._apply_preset.
    preset: str = pydantic.Field(
        default="",
        description=(
            "The strategy preset in force: `standard` (plain defaults), `quick` (no review panel), "
            "`ultra` (a three-seat panel that advises and vetoes before finish), `paranoid` (five "
            "explore-tier seats), or a `[presets.<name>]` of your own. Fills many settings at once "
            "and overrides every section of the layer that selects it; `--preset` overrides per "
            "run, `resume --preset` per resumed execution. Empty: no preset."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        """Refuse a configured role naming a provider absent from a non-empty `[providers]`.

        An empty or partial config is valid at load; `require_runnable` checks completeness.

        Returns:
            The model unchanged.

        Raises:
            ValueError: A role names an unknown provider.
        """
        for role, rm in self.models.configured().items():
            if self.providers and rm.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                raise ValueError(
                    f"models.{role}.provider = {rm.provider!r} but"
                    f" [providers.{rm.provider}] is not configured."
                    f" Known providers: {known}."
                )
        return self

    @pydantic.model_validator(mode="after")
    def _model_git_control_needs_git_writes(self) -> Config:
        """Refuse `git.control = "model"` beside `sandbox.protect_git`: the model must write .git.

        Returns:
            The model unchanged.

        Raises:
            ValueError: Both are set.
        """
        if self.git.control == "model" and self.sandbox.protect_git:
            raise ValueError(
                'git.control = "model" needs the model to write .git;'
                ' set sandbox.protect_git = false (or keep control = "agent6").'
            )
        return self

    @pydantic.model_validator(mode="after")
    def _pass_env_excludes_provider_keys(self) -> Config:
        """Refuse a `pass_env` naming a provider's `api_key_env`.

        The invariant lives here so a direct config edit cannot bypass it; `mcp connect`
        pre-checks the same rule for an earlier refusal.

        Returns:
            The model unchanged.

        Raises:
            ValueError: An MCP server's or the machine's `pass_env` names a provider key.
        """
        keys = {
            e.api_key_env
            for e in self.providers.values()
            if not isinstance(e, _providers.ClaudeCodeProviderEntry) and e.api_key_env
        }
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

    def with_budget_overrides(
        self,
        *,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
        max_percent: float | None = None,
    ) -> Config:
        """Return a copy with the per-run budget flags applied.

        Args:
            max_usd: The `--max-usd` value, or None to keep the config's.
            max_tokens_fallback: The `--max-tokens-fallback` value, or None.
            max_percent: The `--max-percent` value, or None.

        Returns:
            The config with the given caps; self when none is given.
        """
        if max_usd is None and max_tokens_fallback is None and max_percent is None:
            return self
        data = self.model_dump(mode="python")
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        if max_percent is not None:
            budget["max_percent"] = max_percent
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
            raise ConfigError(f"{raw_spec!r}: no model id.")
        provider, slash, model = spec.partition("/")
        known = ", ".join(sorted(self.providers)) or "(none)"
        if slash and not provider:
            raise ConfigError(
                f"{raw_spec!r}: name the provider as provider/model"
                f" (configured providers: {known})."
            )
        if slash and not model:
            raise ConfigError(f"{raw_spec!r}: no model id after the slash.")
        if not slash or provider not in self.providers:
            current = self.models.resolve(role)
            if current is None:
                raise ConfigError(
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
        base = self.models.resolve(role)
        entry = base.model_dump(mode="python") if base is not None else {}
        entry["provider"], entry["model"] = route.provider, route.model
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
            auto_approve: Turn `run_commands` `ask` into `yes` and every MCP server's
                `approve` too; a configured `no` stays, since a flag never grants what the
                standing policy denied.
            no_commands: Pin `run_commands` to `no`; tightening needs no permission.

        Returns:
            The config with the flags applied; self when none is set.
        """
        if not disable_sandbox and not auto_approve and not no_commands:
            return self
        data = self.model_dump(mode="python")
        sandbox = data.setdefault("sandbox", {})
        if disable_sandbox:
            sandbox["isolation"] = "none"
        if auto_approve and self.sandbox.run_commands != "no":
            sandbox["run_commands"] = "yes"
        if auto_approve:
            for server in data.get("mcp", {}).get("servers", {}).values():
                server["approve"] = "yes"
        if no_commands:
            sandbox["run_commands"] = "no"
        return Config.model_validate(data)

    def with_machine_agent_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
    ) -> Config:
        """Return a copy with a machine `agent` state's knobs on the worker role and the budget.

        Args:
            provider: The worker's provider, or None to inherit.
            model: The worker's model, or None.
            effort: The worker's effort, or None.
            temperature: The worker's temperature, or None.
            max_usd: The `budget.max_usd` cap, or None.
            max_tokens_fallback: The `budget.max_tokens_fallback` cap, or None.

        Returns:
            The config, revalidated so the provider-name checks run on the merged result.
        """
        data = self.model_dump(mode="python")
        worker = data.setdefault("models", {}).get("worker")
        if worker is None:
            worker = {}
            data["models"]["worker"] = worker
        if provider is not None:
            worker["provider"] = provider
        if model is not None:
            worker["model"] = model
        if effort is not None:
            worker["effort"] = effort
        if temperature is not None:
            worker["temperature"] = temperature
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
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

    def cleartext_credential_endpoints(self) -> tuple[str, ...]:
        """Return the endpoints that send a credential over plain http off loopback.

        Returns:
            `[table] url` labels for the run-entry warning and the `mcp connect` confirmation.
        """
        out: list[str] = []
        for name, entry in sorted(self.providers.items()):
            if (
                not isinstance(entry, _providers.ClaudeCodeProviderEntry)
                and _sandbox.is_cleartext_url(entry.base_url)
                and entry.auth_style != "none"
                and not _sandbox.is_loopback_url(entry.base_url)
            ):
                out.append(f"[providers.{name}] {entry.base_url}")
        for name, srv in sorted(self.mcp.servers.items()):
            if (
                srv.token_env
                and _sandbox.is_cleartext_url(srv.url)
                and not _sandbox.is_loopback_url(srv.url)
            ):
                out.append(f"[mcp.servers.{name}] {srv.url}")
        return tuple(out)

    def with_run_commands_clamped(self) -> Config:
        """Return a copy with `sandbox.run_commands` `yes` clamped to `ask` for an interactive mode.

        Ask and plan run with the operator present, often outside a repo, so nothing runs
        unwatched; `no` stays, since a run never loosens a boundary the operator set.

        Returns:
            The clamped config; self when nothing changes.
        """
        if self.sandbox.run_commands != "yes":
            return self
        data = self.model_dump(mode="python")
        data.setdefault("sandbox", {})["run_commands"] = "ask"
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
        required; `agent6.verify_infer` infers one.

        Args:
            role: The role the command uses.

        Raises:
            ConfigError: No provider, no model for the role, or the model's provider is
                unknown.
        """
        if not self.providers:
            raise ConfigError(
                "No providers configured. Run `agent6 connect` to add one"
                " (stored in your global config), or add a [providers.*]"
                " block to the per-repo config."
            )
        rm = self.models.resolve(role)
        if rm is None:
            raise ConfigError(
                f"No model configured for the {role!r} role. Run `agent6 model`"
                " to set it, or add a [models.worker] block to your config."
            )
        if rm.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise ConfigError(
                f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}]"
                f" is not configured. Known providers: {known}."
            )


# pydantic's own message for a missing discriminator names neither the key nor its values.
_MISSING_API_FORMAT = (
    'set api_format = "anthropic", "openai", "chatgpt", or "claude_code" (see docs/config.md)'
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
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        msg = issue["msg"]
        if issue["type"] == "union_tag_not_found" and loc.startswith("providers."):
            msg = _MISSING_API_FORMAT
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
        raise ConfigError(_format_validation_error(exc, source, locate)) from exc


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
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file cannot be read ({path}): {exc}") from exc
    return validate_config(raw, source=str(path))
