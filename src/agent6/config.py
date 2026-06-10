# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading — TOML to pydantic.

This is a trust boundary (untrusted text -> structured types), so we use
pydantic and surface field-pointing errors.

Field policy: **secure by default, fully auditable**. Every field has a
default, and security-sensitive fields default to the *safe* value
(``sandbox.agent_network = "providers"``, ``sandbox.tool_network = "block"``,
``sandbox.run_commands = "ask"``,
``sandbox.protect_* = true``, ``git.allow_push/force/history_rewrite =
false``). This means a config can be layered (global ``$XDG_CONFIG_HOME``
defaults, per-repo ``./.agent6/config.toml`` overrides) and a repo can be
zero-config when the global config supplies providers + models. Use
``agent6 config show`` to audit the *effective* value of every field and
exactly where it came from (default / global / repo / flag). The few
things a run genuinely cannot guess — a provider+key and the repo's
``verify_command`` — are checked by :meth:`Config.require_runnable` with a
friendly pointer to ``agent6 connect`` / ``agent6 init`` rather than a
load-time failure, so ``config show`` always works.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agent6.budget import usd_budget_to_tokens
from agent6.paths import validate_workspace_subdir


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or fails validation."""


_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

ProviderKind = Literal["anthropic", "openai"]
# The three live roles. ``planner`` drives ``agent6 plan`` and ``reviewer``
# drives ``agent6 review`` + the in-loop critic; both fall back to
# ``worker`` when unset (see ModelsConfig.resolve).
RoleName = Literal["worker", "reviewer", "planner"]
ThinkingLevel = Literal["off", "low", "medium", "high"]


def _validate_base_url(url: str) -> None:
    """Reject a ``[providers.*].base_url`` that is not an http(s) URL with a host.

    Unlike ``sandbox.allow_urls`` (which accepts a bare ``host``), a
    ``kind = "openai"`` ``base_url`` is the full endpoint the HTTP client posts
    to (``base_url + "/chat/completions"``), so it must carry an explicit
    ``http://`` / ``https://`` scheme and a host. The common paste error this
    catches is dropping an API key (or a bare host) into the field, which would
    otherwise be accepted and only fail much later as an opaque HTTP error.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # urlsplit raises ValueError on an out-of-range port
    except ValueError as exc:
        raise ValueError(f"invalid base_url {url!r}: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"base_url {url!r} must start with http:// or https://")
    if not parts.hostname:
        raise ValueError(f"base_url {url!r} has no host")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"base_url {url!r} has an invalid port")


def _validate_allow_url(entry: str) -> None:
    """Reject a `sandbox.allow_urls` entry that has no usable host:port.

    Accepts a bare ``host``, ``host:port``, or full URL; a missing scheme
    implies ``https://``. Only the host:port is meaningful for the egress
    broker, so the body/path is ignored. Kept in lock-step with the egress
    folding in ``cli._allow_url_endpoints`` — both prepend ``https://`` when
    the entry omits a scheme, then parse with ``urlsplit``.
    """
    if not entry or not entry.strip():
        raise ValueError("sandbox.allow_urls entries must be non-empty")
    normalized = entry if "://" in entry else f"https://{entry}"
    try:
        parts = urlsplit(normalized)
        port = parts.port  # urlsplit raises ValueError on an out-of-range port
    except ValueError as exc:
        raise ValueError(f"invalid sandbox.allow_urls entry {entry!r}: {exc}") from exc
    if not parts.hostname:
        raise ValueError(f"sandbox.allow_urls entry {entry!r} has no host")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"sandbox.allow_urls entry {entry!r} has an invalid port")


class AnthropicProviderEntry(BaseModel):
    """`kind = "anthropic"` — the Anthropic Messages endpoint.

    Model names live in `[models.<role>]`, not here — this block only
    carries auth and Anthropic-specific knobs.
    """

    model_config = _BASE_MODEL_CONFIG

    kind: Literal["anthropic"]
    # Name of the env var holding the API key. Optional: leave it unset to
    # let `agent6 connect` store the key in secrets.toml instead. Either
    # source works; the env var (when set and non-empty) takes precedence.
    api_key_env: str | None = Field(default=None, min_length=1)
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

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        _validate_base_url(v)
        return v

    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers to attach to every request (e.g. OpenRouter's).",
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra JSON merged into every request body (keys override computed"
            " fields). Provider-specific — e.g. OpenRouter routing: set"
            ' extra_body = { provider = { sort = "throughput" } } to prefer the'
            " fastest backend, pin one with { order = [...] }, or cap price with"
            " { max_price = { ... } }. Pay-for-speed lives here."
        ),
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
    # Reasoning/thinking effort for this role. ``None`` leaves the
    # provider default; ``off`` disables it explicitly. Mapped per
    # provider: OpenAI-compatible reasoning models receive a
    # ``reasoning.effort`` knob, Anthropic models receive an
    # ``extended_thinking`` budget. Non-reasoning models ignore it.
    thinking: ThinkingLevel | None = None


class ModelsConfig(BaseModel):
    """Per-role provider + model routing.

    Three roles, all optional:

    - ``worker`` drives the single-loop agent (``agent6 run`` / ``agent6
      resume``); its pricing also drives the USD -> token budget
      conversion.
    - ``planner`` drives ``agent6 plan`` (the read-only planning pass).
      Unset -> falls back to ``worker`` (set it to a frontier model + high
      thinking for careful up-front planning).
    - ``reviewer`` drives the one-shot ``agent6 review`` subcommand and the
      optional in-loop critic. Unset -> falls back to ``worker``.

    Any configured provider may serve any role. Leaving every role unset is
    valid (e.g. a global config that only declares providers); a role is
    only *required* for the command that uses it, checked by
    :meth:`Config.require_runnable`.
    """

    model_config = _BASE_MODEL_CONFIG

    worker: RoleModel | None = None
    reviewer: RoleModel | None = None
    planner: RoleModel | None = None

    def configured(self) -> dict[str, RoleModel]:
        """Only the roles explicitly set (used for validation/key checks)."""
        out: dict[str, RoleModel] = {}
        if self.worker is not None:
            out["worker"] = self.worker
        if self.reviewer is not None:
            out["reviewer"] = self.reviewer
        if self.planner is not None:
            out["planner"] = self.planner
        return out

    def resolve(self, role: RoleName) -> RoleModel | None:
        """The effective model for *role*, applying worker fallbacks."""
        if role == "worker":
            return self.worker
        if role == "planner":
            return self.planner or self.worker
        if role == "reviewer":
            return self.reviewer or self.worker
        return None


class SandboxConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    profile: Literal["auto", "strict", "hardened"] = "auto"
    # Where the agent PROCESS (its own LLM/provider HTTP) may connect:
    #  - `providers`: only the configured `[providers.*]` endpoints, plus any
    #    `allow_urls`. On `strict` this is structural — a trusted broker (see
    #    agent6.sandbox.broker) confines the agent to an empty netns whose only
    #    routes are per-endpoint unix sockets; on `hardened` it is Landlock
    #    TCP-port confinement to the provider ports.
    #  - `local`: only loopback providers (local models, e.g. Ollama). `strict`-
    #    only; refused if a configured provider is non-local or `allow_urls` is
    #    set (there is nothing external to allow-list when offline).
    #  - `open`: unconfined egress.
    agent_network: Literal["providers", "local", "open"] = "providers"
    # Whether JAILED commands (`run_command`, `verify`, `metric`, and machine
    # `tool` states) may reach the network. A jailed child can never out-reach
    # the process that launches it, so:
    #  - `block`: no jailed command gets the network.
    #  - `only_explicit_states`: blocked, EXCEPT machine `tool` states that opt
    #    in with `allow_network = "allow"` (audited, deterministic commands);
    #    `run_command` stays blocked. `strict`-only — singling one tool out needs
    #    a per-child network namespace, which only `strict` provides.
    #  - `allow`: `run_command` reaches the network too. Because `run_command`
    #    runs inside the (possibly confined) agent process, this requires
    #    `agent_network = "open"`.
    tool_network: Literal["block", "only_explicit_states", "allow"] = "block"
    run_commands: Literal["yes", "no", "ask"] = "ask"
    # Make `.git/` read-only from the child's view so a worker that gains
    # `run_command` (e.g. `run_commands = "ask"` + user approval) cannot
    # `rm -rf .git`, rewrite history, or otherwise corrupt the repository
    # from inside a child process. The workflow's own commits go through
    # `git_ops.py` from the agent process (outside the jail) and are
    # unaffected. Strict re-binds it RO; hardened switches Landlock from
    # "RW on cwd" to "R on cwd + RW on each top-level entry except the
    # protect set". Hardened-mode side effect: writes to NEW top-level
    # entries created at the cwd root after launch are denied.
    protect_git: bool = True
    # Same idea, for the `.agent6/` directory (config + run state: transcripts,
    # graph, logs). The curator subprocess has its own jail policy that does
    # grant `.agent6/` write access; worker children do not.
    protect_agent6: bool = True
    # Extra egress destinations the AGENT process may reach under
    # `agent_network = "providers"`, on top of the configured provider
    # endpoints. Each entry is a `host`, `host:port`, or full URL (a missing
    # scheme implies https / port 443); only the host:port is used to open a
    # broker socket. Secure default empty — no destination beyond the
    # providers is reachable. MERGE: last-overlay-wins (the most-specific tier
    # that sets the key replaces it wholesale, like every other list field);
    # provider endpoints always UNION in regardless of tier. Effective egress
    # = union(provider endpoints) + allow_urls(winning tier). Only meaningful
    # under `agent_network = "providers"`; ignored under `local`/`open`. It
    # widens only the agent path, never a jailed `tool`/`run_command`.
    allow_urls: tuple[str, ...] = ()

    @field_validator("allow_urls")
    @classmethod
    def _check_allow_urls(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for entry in v:
            _validate_allow_url(entry)
        return v

    @model_validator(mode="after")
    def _check_network_combo(self) -> SandboxConfig:
        # A jailed child can never out-reach the process that launches it, and
        # `run_command` runs inside the agent process. So letting `run_command`
        # reach the network (`tool_network = "allow"`) requires the agent to be
        # unconfined. `only_explicit_states` is exempt: machine `tool` states are
        # jailed by the host-netns engine, not the (possibly confined) agent.
        if self.tool_network == "allow" and self.agent_network != "open":
            raise ValueError(
                "sandbox.tool_network = 'allow' requires sandbox.agent_network"
                " = 'open' — run_command runs inside the agent process and cannot"
                " reach the network while the agent is confined. Use"
                " 'only_explicit_states' for audited per-tool egress, or set"
                " agent_network = 'open'."
            )
        if self.agent_network == "local" and self.allow_urls:
            # The docstring promises `local` refuses allow_urls; enforce it rather
            # than silently ignoring the list. `local` confines egress to loopback
            # providers, so an external allow-list can never take effect.
            raise ValueError(
                "sandbox.agent_network = 'local' (loopback providers only) cannot"
                " be combined with sandbox.allow_urls — there is nothing external"
                " to allow-list when offline. Remove allow_urls, or use"
                " agent_network = 'providers'."
            )
        return self


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
    # Whether the repo's own git hooks (`.git/hooks/*`) run during agent6's
    # OWN git operations (notably the per-step auto-commit). Default false:
    # secure-by-default (a hook is repo-controlled code that would execute on
    # the HOST, outside the jail, when agent6 commits -- a host-RCE vector for
    # an adversarial repo) and also avoids re-running a slow pre-commit hook on
    # every micro-commit. The verify_command is agent6's real success gate, not
    # git hooks. Set true to honor the repo's hooks (trust the repo). Either
    # way `core.fsmonitor`/`diff.external` stay neutralized (those fire on
    # status/diff and have no legitimate use here).
    run_repo_hooks: bool = False
    # Security-sensitive: default to the safe (disabled) value. agent6's
    # git_ops layer refuses push / force / history rewrite unconditionally
    # regardless of these toggles; they exist for the few workflows that
    # legitimately need them and must be opted into explicitly.
    allow_push: bool = False
    allow_force: bool = False
    allow_history_rewrite: bool = False
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

    # The command agent6 runs to decide whether a step "succeeded". This is
    # inherently repo-specific, so it has no useful global default and
    # defaults to empty; `Config.require_runnable` requires it before a run
    # (with a pointer to `agent6 init`). `agent6 plan` / `agent6 review` do
    # not need it.
    verify_command: tuple[str, ...] = ()
    # per-call timeout for verify_command (and metric_command) in
    # seconds. Defaults to the jail's general 600s but should be cranked
    # MUCH lower for benches where the verify is a fast correctness test
    # (perf-takehome's CorrectnessTests run in ~2s; a 30s cap detects
    # infinite-loop / quadratic edits 20x faster than the 600s default).
    # Setting too low for slow legitimate tests will cause false-positive
    # failures, so leave at 600 unless the verify is reliably fast.
    verify_timeout_s: float = Field(gt=0.0, default=600.0)
    # Tiered context-compaction thresholds (approximate chars; tokens ~=
    # chars/4). When cumulative *tool_result* content grows past
    # ``compact_drop_at_chars`` the oldest tool_results are replaced by a
    # short placeholder (the worker can re-call the tool to refetch). When the
    # *whole* context (text + tool_use inputs + surviving tool_results) grows
    # past ``compact_summarise_at_chars`` -- which must be > drop, so tier-2
    # escalates above tier-1 -- the conversation is summarized and restarted
    # (the durable task DAG survives; the restart notice points the worker at
    # ``dag_list_tasks`` to recover task-level state).
    # ``context_summary_max_tokens`` caps the summarizer's output. Raise the
    # thresholds for big-context models; lower them to compact sooner.
    compact_drop_at_chars: int = Field(gt=0, default=256_000)
    compact_summarise_at_chars: int = Field(gt=0, default=768_000)
    context_summary_max_tokens: int = Field(gt=0, default=2048)
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

    @model_validator(mode="after")
    def _check_compaction_thresholds(self) -> WorkflowConfig:
        # Tier 2 (summarise + restart) must escalate ABOVE tier 1 (drop old
        # tool_results). If summarise <= drop, tier 2 fires at or before tier 1
        # -- the inverted ordering that historically left tier 2 unreachable.
        if self.compact_summarise_at_chars <= self.compact_drop_at_chars:
            raise ValueError(
                "workflow.compact_summarise_at_chars"
                f" ({self.compact_summarise_at_chars}) must be greater than"
                f" compact_drop_at_chars ({self.compact_drop_at_chars}): tier-2"
                " summarise must escalate above tier-1 elision."
            )
        return self


class BudgetConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    # Hard stops on token spend. Defaults are generous safety ceilings (the
    # run is resumable from the persistent task graph if hit); tighten them
    # per-repo or use `max_usd` for a dollar cap.
    max_input_tokens: int = Field(gt=0, default=2_000_000)
    max_output_tokens: int = Field(gt=0, default=200_000)
    # Optional: operator-friendly USD cap. When set, the loader converts it to
    # token ceilings (worker-model pricing) and lowers max_input/output_tokens
    # to them, AND the runtime additionally enforces an exact dollar ceiling
    # that counts cache_read/cache_creation cost (which the token caps do not),
    # so a heavily-cached run cannot blow past it. Set 0 to disable. See
    # `agent6.budget.usd_budget_to_tokens` for the conversion math.
    max_usd: float = Field(ge=0.0, default=0.0)


class Agent6Section(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    config_version: int = Field(ge=1, le=1, default=1)
    # Rename the in-repo agent6 directory (default ``.agent6``) that holds
    # this config + run state. A BARE directory name only (no path
    # separators, no ``..``, not absolute). Can ONLY be set in the GLOBAL
    # config: the per-repo config lives inside this dir, so it cannot name
    # the dir that contains it. Setting it in a repo/flag config is an error.
    workspace_subdir: str | None = None

    @field_validator("workspace_subdir")
    @classmethod
    def _check_workspace_subdir(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_workspace_subdir(v)


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

    @field_validator("name")
    @classmethod
    def _no_double_underscore(cls, v: str) -> str:
        # The LLM-visible tool name is ``mcp__<name>__<tool>`` and routing
        # recovers the server by splitting on the FIRST ``__`` after the prefix.
        # A ``__`` inside the server name makes that split ambiguous and routes
        # to the wrong (or no) server. (pydantic v2 patterns use a regex engine
        # without lookahead, so this is a validator rather than a pattern.)
        if "__" in v:
            raise ValueError(
                f"[mcp] server name must not contain '__' (it separates server"
                f" from tool in mcp__<server>__<tool>): {v!r}"
            )
        return v


class MCPConfig(BaseModel):
    """``[mcp]`` section. Empty / absent / ``enabled = false`` means no
    MCP servers are spawned and the LLM sees zero ``mcp__*`` tools."""

    model_config = _BASE_MODEL_CONFIG

    enabled: bool = False
    servers: tuple[MCPServerEntry, ...] = ()


class Config(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    agent6: Agent6Section = Field(default_factory=Agent6Section)
    providers: dict[str, ProviderEntry] = Field(default_factory=dict)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        # Only configured roles are checked here, and only when their
        # provider is actually present; an empty/partial config is valid
        # at load time (require_runnable enforces completeness per command).
        for role, rm in self.models.configured().items():
            if self.providers and rm.provider not in self.providers:
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
        worker = self.models.resolve("worker")
        if worker is None:
            # No worker model to price against yet; the conversion is
            # applied once a runnable config is assembled.
            return self
        usd_in, usd_out = usd_budget_to_tokens(self.budget.max_usd, worker_model=worker.model)
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

    def with_budget_overrides(
        self,
        *,
        max_usd: float | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Config:
        """Return a copy with budget fields overridden (e.g. from CLI flags).

        Re-validates through ``model_validate`` so the USD->token conversion
        in ``_apply_usd_budget_override`` runs again on the new values.
        ``None`` means "keep the existing value".
        """
        if max_usd is None and max_input_tokens is None and max_output_tokens is None:
            return self
        data = self.model_dump(mode="python")
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_input_tokens is not None:
            budget["max_input_tokens"] = max_input_tokens
        if max_output_tokens is not None:
            budget["max_output_tokens"] = max_output_tokens
        return Config.model_validate(data)

    def with_machine_agent_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        temperature: float | None = None,
        max_usd: float | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Config:
        """Return a copy with a machine ``agent`` state's per-state knobs applied.

        Overrides the ``worker`` role (the role machine agent loops run as)
        and the budget caps. ``None`` means "inherit the effective config".
        Re-validates so the USD->token conversion and provider-name checks
        run against the merged result.
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
        if thinking is not None:
            worker["thinking"] = thinking
        if temperature is not None:
            worker["temperature"] = temperature
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_input_tokens is not None:
            budget["max_input_tokens"] = max_input_tokens
        if max_output_tokens is not None:
            budget["max_output_tokens"] = max_output_tokens
        return Config.model_validate(data)

    def require_runnable(self, role: RoleName = "worker", *, need_verify: bool = True) -> None:
        """Raise ConfigError unless *role* can actually run.

        Checks (in order) that a provider is configured, the role resolves
        to a model whose provider exists, and — for execution roles — that
        ``verify_command`` is set. Messages point at the command that fixes
        the gap so a fresh user is never stuck.
        """
        if not self.providers:
            raise ConfigError(
                "No providers configured. Run `agent6 connect` to add one"
                " (stored in your global config), or add a [providers.*]"
                " block to .agent6/config.toml."
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
        if need_verify and not self.workflow.verify_command:
            raise ConfigError(
                "workflow.verify_command is empty — agent6 needs to know what"
                " 'a step succeeded' means in this repo. Run `agent6 init` (it"
                " writes a starter) or set workflow.verify_command in"
                " .agent6/config.toml."
            )


def _format_validation_error(err: ValidationError, source: str) -> str:
    lines = [f"Config validation failed: {source}"]
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']} (type={issue['type']})")
    return "\n".join(lines)


def validate_config(raw: dict[str, object], *, source: str = "<config>") -> Config:
    """Validate an already-parsed (and possibly layer-merged) config dict.

    Shared by :func:`load_config` and the layered loader
    (``agent6.config_layer``) so both surface identical field-pointing
    errors.
    """
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source)) from exc


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
    return validate_config(raw, source=str(path))
