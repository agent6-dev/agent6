# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading — TOML to pydantic.

This is a trust boundary (untrusted text -> structured types), so we use
pydantic and surface field-pointing errors.

Field policy: security-sensitive fields (``sandbox.*``, ``providers.*``,
``models.*``, ``budget.max_*_tokens``, ``git.allow_*``,
``workflow.verify_command``) are required — they must reflect explicit
operator intent. Operational fields with safe defaults
(``agent6.config_version``, ``git.require_clean_worktree``, ``git.auto_stash``,
``git.branch_per_run``, ``git.commit_strategy``, ``workflow.verify_timeout_s``,
and the Anthropic ``prompt_caching`` toggle) provide sane defaults so a
fresh config does not need to know every knob.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, ValidationError, model_validator

from agent6.budget import usd_budget_to_tokens


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or fails validation."""


_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

ProviderKind = Literal["anthropic", "openai"]
RoleName = Literal["worker", "reviewer"]


class AnthropicProviderEntry(BaseModel):
    """`kind = "anthropic"` — the Anthropic Messages endpoint.

    Model names live in `[models.<role>]`, not here — this block only
    carries auth and Anthropic-specific knobs.
    """

    model_config = _BASE_MODEL_CONFIG

    kind: Literal["anthropic"]
    api_key_env: str = Field(min_length=1)
    prompt_caching: bool = True
    # per-HTTP-call timeout (connect + read) for this provider in
    # seconds. Default 600s is generous enough for a slow provider streaming
    # a long response but tight enough that a stuck connection fails fast
    # rather than burning the whole budget window. Reasoning models on
    # OpenRouter have been observed to stall for >10min before erroring;
    # set this lower on benches that should fail fast.
    http_timeout_s: float = Field(gt=0.0, default=600.0)


class OpenAIProviderEntry(BaseModel):
    """`kind = "openai"` — any OpenAI Chat Completions-compatible endpoint.

    Works against OpenAI itself, OpenRouter, Ollama (`/v1`), vLLM, LM
    Studio, llama.cpp's server, etc. Each [providers.<name>] block is one
    endpoint; configure as many as you want under whatever names you like
    (e.g. `[providers.openai]`, `[providers.openrouter]`,
    `[providers.ollama]`) and reference them by name from `[models.<role>]`.

    `api_key_env` is optional: leave it unset (or point at an unset env var)
    for unauthenticated local endpoints like Ollama. `extra_headers` is
    forwarded verbatim — OpenRouter, for example, asks for `HTTP-Referer`
    and `X-Title`. Header names are sent lowercased by httpx.
    """

    model_config = _BASE_MODEL_CONFIG

    kind: Literal["openai"]
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Env var holding the API key. Set to None / omit for unauthenticated"
            " local endpoints (Ollama, llama.cpp)."
        ),
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        min_length=1,
        description="Base URL of an OpenAI Chat Completions-compatible endpoint.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers to attach to every request (e.g. OpenRouter's).",
    )
    # per-HTTP-call timeout in seconds. See AnthropicProviderEntry
    # for the rationale. OpenRouter heartbeats are handled at the streaming
    # SSE layer; this timeout is the underlying httpx ceiling.
    http_timeout_s: float = Field(gt=0.0, default=600.0)


ProviderEntry = Annotated[
    AnthropicProviderEntry | OpenAIProviderEntry,
    Discriminator("kind"),
]


class RoleModel(BaseModel):
    """One role's `(provider, model)` assignment.

    `provider` is the name (TOML table key) of an entry in `[providers.*]`.

    `temperature` is the sampling temperature agent6 will pin on every
    call for this role. Defaults to ``0.0`` — agent6's tool-use loop is a
    search-and-act feedback loop and high-temperature sampling causes
    observable degeneration on some open-weights models (caught
    Kimi K2.6 emitting 15997 literal ``\\n`` escapes in a single
    ``old_string`` argument before hitting the completion-tokens cap).
    Anthropic and OpenAI models are tuned to behave well at any
    temperature; OpenRouter routes to provider defaults that vary by
    model, so pinning is the only way to make benches reproducible.
    Set to ``null`` (TOML: omit-and-rely-on-default doesn't apply here;
    use ``temperature = nan`` is invalid — explicitly write the value)
    only if you specifically want the provider's default behaviour.
    """

    model_config = _BASE_MODEL_CONFIG

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)


class ModelsConfig(BaseModel):
    """Per-role provider + model routing.

    there are only two live roles:

    - ``worker`` drives the single-loop agent (``agent6 run`` / ``agent6
      resume``); pricing for this model also drives the USD → token
      budget conversion.
    - ``reviewer`` is used by the one-shot ``agent6 review`` subcommand.

    Any of the configured providers may serve either role.
    """

    model_config = _BASE_MODEL_CONFIG

    worker: RoleModel
    reviewer: RoleModel

    def all(self) -> dict[RoleName, RoleModel]:
        return {
            "worker": self.worker,
            "reviewer": self.reviewer,
        }


class SandboxConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    profile: Literal["auto", "strict", "hardened"]
    # `provider_only` confines the agent process to an empty network
    # namespace whose only route out is a per-endpoint unix socket served
    # by a trusted broker (see agent6.sandbox.broker): egress is structurally
    # limited to the host:port of every configured `[providers.*]` block.
    # Requires the strict profile (unprivileged user namespaces); the run is
    # refused on hosts that only support `hardened`. `no` keeps the process
    # in the host netns with no egress confinement at the process level;
    # `allow` is identical to `no` (egress unrestricted).
    network: Literal["no", "provider_only", "allow"]
    run_commands: Literal["yes", "no", "ask"]
    # Make `.git/` read-only from the child's view so a worker that gains
    # `run_command` (e.g. `run_commands = "ask"` + user approval) cannot
    # `rm -rf .git`, rewrite history, or otherwise corrupt the repository
    # from inside a child process. The workflow's own commits go through
    # `git_ops.py` from the agent process (outside the jail) and are
    # unaffected. Strict re-binds it RO; hardened switches Landlock from
    # "RW on cwd" to "R on cwd + RW on each top-level entry except the
    # protect set". Hardened-mode side effect: writes to NEW top-level
    # entries created at the cwd root after launch are denied.
    protect_git: bool
    # Same idea, for `agent6.toml` and `.agent6/` (run state, transcripts,
    # graph). The curator subprocess has its own jail policy that does
    # grant `.agent6/` write access; worker children do not.
    protect_agent6: bool


class GitCommitConfig(BaseModel):
    """Optional overrides for the author/committer identity on agent6 commits.

    All three fields default to None, meaning "use whatever the project's
    `git config user.name` / `user.email` already resolves to". The startup
    check in `agent6 run` refuses to proceed if neither an override nor a
    resolvable git-config identity is present — we will not silently commit
    as `(no author) <(none)>`.

    Set `name` / `email` to override the identity on commits made by this
    agent (e.g. to commit as `agent6 <agent6@local>`). Set
    `coauthor` to append a `Co-authored-by:` trailer naming the human
    operator (e.g. `"Alice <alice@example.com>"`).
    """

    model_config = _BASE_MODEL_CONFIG

    name: str | None = None
    email: str | None = None
    coauthor: str | None = None


class GitConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    require_clean_worktree: bool = True
    auto_stash: bool = False
    branch_per_run: bool = True
    commit_strategy: Literal["per_step", "squash", "stage", "none"] = "per_step"
    allow_push: bool
    allow_force: bool
    allow_history_rewrite: bool
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)


class MetricConfig(BaseModel):
    """Optional continuous-score metric for tasks that have a measurable goal
    (cycles, wall time, kB, bench score) distinct from binary verify pass/fail.

    When configured, ``run_metric_command`` (the metric tool) runs ``command``
    in the jail (same env as ``verify_command``) and parses ``pattern``'s
    first capture group as a number. ``goal = "minimize"`` for things like
    cycles/time; ``"maximize"`` for bench scores. ``pattern`` is a Python
    regex; the FIRST capture group must be a base-10 integer or float. If
    the pattern does not match in the command's combined stdout+stderr the
    metric is treated as missing.
    """

    model_config = _BASE_MODEL_CONFIG

    command: tuple[str, ...] = Field(min_length=1)
    pattern: str = Field(min_length=1)
    goal: Literal["minimize", "maximize"]


class WorkflowConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    verify_command: tuple[str, ...] = Field(min_length=1)
    # per-call timeout for verify_command (and metric_command) in
    # seconds. Defaults to the jail's general 600s but should be cranked
    # MUCH lower for benches where the verify is a fast correctness test
    # (perf-takehome's CorrectnessTests run in ~2s; a 30s cap detects
    # infinite-loop / quadratic edits 20x faster than the 600s default).
    # Setting too low for slow legitimate tests will cause false-positive
    # failures, so leave at 600 unless the verify is reliably fast.
    verify_timeout_s: float = Field(gt=0.0, default=600.0)
    # Optional. None means "no metric; ``run_metric_command`` is unavailable".
    metric: MetricConfig | None = None
    # critic-in-loop. When != "off", Workflow runs the
    # ``reviewer`` model as a critic at the chosen trigger and injects
    # its critique as a user message the worker sees next turn.
    #   off              - never (default; behaviour unchanged).
    #   on_verify_fail   - after every verify failure.
    #   before_finish    - intercept ``finish_run``; reject if critic
    #                      is not satisfied and inject critique.
    #   periodic         - every ``critic_period`` iterations.
    # The reviewer provider must already be configured in
    # ``[models.reviewer]`` (same one ``agent6 review`` uses).
    critic: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    critic_period: int = Field(ge=1, default=10)
    # one-shot task prompt revision before the worker loop starts.
    # Reuses the reviewer model, takes no tools, and is budget-tracked like
    # any other provider call. Default off keeps crisp prompts/frontier-model
    # runs on the old path.
    revise_prompt: Literal["off", "auto", "interactive"] = "off"


class BudgetConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    # Optional: operator-friendly USD cap. When set AND the token
    # ceilings are zero / unset / lower than the converted-from-USD
    # values, the loader replaces them with the converted ceilings
    # using the worker model's pricing. This is a config-time
    # convenience; the runtime ceiling is still per-token. Set 0 to
    # disable. See `agent6.budget.usd_budget_to_tokens` for the
    # conversion math.
    max_usd: float = Field(ge=0.0, default=0.0)


class Agent6Section(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    config_version: int = Field(ge=1, le=1, default=1)


class NotifyConfig(BaseModel):
    """Optional post-run notification hook.

    When ``on_complete`` is set, agent6 runs the argv tuple after the
    workflow returns (``agent6 run`` or ``agent6 resume``). The argv is
    operator-controlled — it never includes LLM output — and runs in the
    user's shell environment, NOT in the jail, with these env vars:

    - ``AGENT6_RUN_ID``       — run id under ``.agent6/runs/``
    - ``AGENT6_RUN_OK``       — ``1`` if the workflow finished cleanly, ``0`` otherwise
    - ``AGENT6_RUN_REASON``   — workflow termination reason (e.g. ``finish_run``,
                                 ``budget_exhausted``, ``provider_error``)
    - ``AGENT6_RUN_DIR``      — absolute path to the run dir

    Use cases: desktop notification (``notify-send``), shell-bell, ssh
    push notification, mailx, etc. A failure of the notify command is
    logged but does not change the agent6 exit code.
    """

    model_config = _BASE_MODEL_CONFIG

    on_complete: tuple[str, ...] = Field(default=(), description="argv to run on completion")
    timeout_s: float = Field(gt=0.0, default=30.0)


class MCPServerEntry(BaseModel):
    """One MCP (Model Context Protocol) server to spawn at run start.

    The server runs as a long-lived subprocess speaking JSON-RPC 2.0
    over stdio. Its ``command`` (argv) is operator-controlled and never
    contains LLM output. The server runs OUTSIDE the agent6 jail with
    the user's environment - same trust model as ``[notify].on_complete``.

    The LLM sees each MCP-server tool as
    ``mcp__<name>__<server-side-tool-name>`` and can call it through
    the normal tool surface. The MCP server itself is responsible for
    validating the arguments the LLM passes; agent6 forwards them
    verbatim.

    A misbehaving server (crash, hang, malformed output) surfaces as
    a clean tool failure, not an agent crash.
    """

    model_config = _BASE_MODEL_CONFIG

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    command: tuple[str, ...] = Field(min_length=1)
    enabled: bool = True
    # Time budget for the initialize + tools/list handshake. If the
    # server doesn't respond in this window we log and skip it.
    startup_timeout_s: float = Field(gt=0.0, default=10.0)
    # Per-call timeout for ``tools/call`` requests. Surfaces as a tool
    # failure (ToolError) if exceeded.
    call_timeout_s: float = Field(gt=0.0, default=60.0)


class MCPConfig(BaseModel):
    """``[mcp]`` section. Empty / absent / ``enabled = false`` means no
    MCP servers are spawned and the LLM sees zero ``mcp__*`` tools."""

    model_config = _BASE_MODEL_CONFIG

    enabled: bool = False
    servers: tuple[MCPServerEntry, ...] = ()


class Config(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    agent6: Agent6Section = Field(default_factory=Agent6Section)
    providers: dict[str, ProviderEntry] = Field(min_length=1)
    models: ModelsConfig
    sandbox: SandboxConfig
    git: GitConfig
    workflow: WorkflowConfig
    budget: BudgetConfig
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        for role, rm in self.models.all().items():
            if rm.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                raise ValueError(
                    f"models.{role}.provider = {rm.provider!r} but"
                    f" [providers.{rm.provider}] is not configured."
                    f" Known providers: {known}."
                )
        return self

    @model_validator(mode="after")
    def _apply_usd_budget_override(self) -> Config:
        """When `[budget].max_usd > 0`, convert to token ceilings via the
        worker model's pricing and apply as a TIGHTER upper bound on top
        of any explicit `max_input_tokens` / `max_output_tokens`. The
        smaller of (operator-set, USD-converted) wins per axis - both
        are valid ceilings; the lower one is the effective cap.
        Operators who want USD only can set the token ceilings to large
        placeholder values (e.g. 999_999_999) and the USD conversion
        will dominate."""
        if self.budget.max_usd <= 0:
            return self
        worker_model = self.models.worker.model
        usd_in, usd_out = usd_budget_to_tokens(self.budget.max_usd, worker_model=worker_model)
        new_in = min(self.budget.max_input_tokens, usd_in)
        new_out = min(self.budget.max_output_tokens, usd_out)
        if new_in == self.budget.max_input_tokens and new_out == self.budget.max_output_tokens:
            return self
        new_budget = BudgetConfig(
            max_input_tokens=new_in,
            max_output_tokens=new_out,
            max_usd=self.budget.max_usd,
        )
        return self.model_copy(update={"budget": new_budget})


def _format_validation_error(err: ValidationError, path: Path) -> str:
    lines = [f"Config validation failed: {path}"]
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']} (type={issue['type']})")
    return "\n".join(lines)


def load_config(path: Path) -> Config:
    """Load and strictly validate the TOML config at *path*.

    Raises ConfigError on any problem; never returns a partially valid config.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, path)) from exc
