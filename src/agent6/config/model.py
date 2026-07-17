# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading, TOML to pydantic.

This is a trust boundary (untrusted text -> structured types), so we use
pydantic and surface field-pointing errors.

Field policy: **secure by default, fully auditable**. Every field has a
default, and security-sensitive fields default to the *safe* value
(``sandbox.agent_network = "providers"``, ``sandbox.tool_network = "block"``,
``sandbox.run_commands = "ask"``,
``sandbox.protect_git = true``, ``git.allow_push/force/history_rewrite =
false``). This means a config can be layered (global ``$XDG_CONFIG_HOME``
defaults, per-repo config (out of the workspace, under the state dir) overrides)
and a repo can be
zero-config when the global config supplies providers + models. Use
``agent6 config show`` to audit the *effective* value of every field and
exactly where it came from (default / global / repo / flag). The one thing a
run genuinely cannot guess, a provider+key, is checked by
:meth:`Config.require_runnable` with a friendly pointer to ``agent6 connect``
rather than a load-time failure, so ``config show`` always works. The repo's
``verify_command`` is optional: `agent6 run`/`plan` infer one per run when it
is unset (see :mod:`agent6.verify_infer`), else run gateless.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from ipaddress import ip_address
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


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or fails validation."""


_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

ApiFormat = Literal["anthropic", "openai"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]
# The three live roles. ``planner`` drives ``agent6 plan`` and ``reviewer``
# drives ``agent6 review`` + the in-loop critic; both fall back to
# ``worker`` when unset (see ModelsConfig.resolve).
RoleName = Literal["worker", "reviewer", "planner"]
ThinkingLevel = Literal["off", "low", "medium", "high"]
# The review-seat depth (`[review].tier`); ReviewSeat.tier mirrors this, so the
# vocabulary has one owner.
ReviewTier = Literal["diff", "explore"]


def validate_base_url(url: str) -> None:
    """Reject a ``[providers.*].base_url`` that is not an http(s) URL with a host.

    Unlike ``sandbox.allow_urls`` (which accepts a bare ``host``), a provider's
    ``base_url`` is the host+path prefix the HTTP client posts to (the
    deployment profile appends ``/chat/completions``, ``/messages``, etc.), so
    it must carry an explicit
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
    folding in ``cli._allow_url_endpoints``, both prepend ``https://`` when
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


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _default_base_url(api_format: str, deployment: str) -> str | None:
    """Default ``base_url`` for a (format, deployment), or None if required.

    Only the ``direct`` deployment has a sensible fixed endpoint; vertex/azure
    (and future bedrock) carry project/resource/region in the URL, so the
    operator must supply ``base_url``.
    """
    if deployment != "direct":
        return None
    return _ANTHROPIC_DEFAULT_BASE_URL if api_format == "anthropic" else _OPENAI_DEFAULT_BASE_URL


def _default_auth_style(api_format: str, deployment: str) -> str:
    """Default ``auth_style`` for a (format, deployment)."""
    if deployment == "azure":
        return "api_key_header"
    if deployment == "vertex":
        return "bearer"
    return "x_api_key" if api_format == "anthropic" else "bearer"


class _ProviderBase(BaseModel):
    """Transport + auth fields shared by every provider, independent of format.

    Three orthogonal concerns: ``api_format`` (the discriminator, on each
    subclass) selects the wire dialect; ``deployment`` selects the URL /
    model-placement profile; and the auth fields (``auth_style`` + a static
    ``api_key_env`` or a refreshable ``token_command``) select the credential.
    They compose freely -- e.g. Claude-on-Vertex and Gemini-on-Vertex differ
    only in ``api_format`` (both ``deployment = "vertex"``). ``base_url`` and
    ``auth_style`` default from (api_format, deployment) in ``_fill_defaults`` so
    a minimal entry behaves exactly like the old fixed providers. Each block is
    one endpoint; configure as many as you like under any names and reference
    them from ``[models.*]``.
    """

    model_config = _BASE_MODEL_CONFIG

    deployment: Deployment = "direct"
    # Resolved by _fill_defaults from (api_format, deployment) when omitted;
    # never empty post-validation. The host also feeds the egress allow-list.
    base_url: str = ""
    # Auth header style; defaults from (api_format, deployment) in _fill_defaults.
    auth_style: AuthStyle = "bearer"
    # Static key: env var name (falls back to secrets.toml by provider name).
    # Secrets live here, never in base_url/extra_headers/extra_query.
    api_key_env: str | None = Field(default=None, min_length=1)
    token_command: list[str] | None = Field(
        default=None,
        description=(
            "Command (argv) whose stdout is a bearer token, run instead of a"
            " static key for endpoints behind a short-lived, refreshable token"
            " (cloud OAuth access tokens, OIDC/STS gateways). Cached on a TTL"
            " and re-minted on a 401/403; takes precedence over api_key_env."
        ),
    )
    token_command_ttl_s: float = Field(
        gt=0.0,
        default=300.0,
        description="Seconds to cache token_command output before re-running it.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers attached to every request (e.g. OpenRouter's).",
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra JSON merged into every request body (load-bearing"
            " messages/model/stream keys are filtered). E.g. OpenRouter routing:"
            ' extra_body = { provider = { sort = "throughput" } }.'
        ),
    )
    extra_query: dict[str, str] = Field(
        default_factory=dict,
        description="Extra URL query params (e.g. Azure's api-version). No secrets here.",
    )
    # per-HTTP-call timeout (connect + read) in seconds. Default 600s streams a
    # long response yet fails a stuck connection before it burns the budget
    # window; lower it on benches that should fail fast.
    http_timeout_s: float = Field(gt=0.0, default=600.0)

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fmt = data.get("api_format")
        dep = data.get("deployment", "direct")
        if fmt == "anthropic" and dep == "azure":
            raise ValueError("deployment 'azure' requires api_format 'openai'")
        if not data.get("base_url"):
            default = _default_base_url(fmt, dep) if isinstance(fmt, str) else None
            if default is None:
                raise ValueError(f"base_url is required for deployment {dep!r}")
            data["base_url"] = default
        if not data.get("auth_style") and isinstance(fmt, str):
            data["auth_style"] = _default_auth_style(fmt, dep)
        if dep == "azure" and "api-version" not in (data.get("extra_query") or {}):
            raise ValueError("deployment 'azure' requires extra_query['api-version']")
        return data

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        if v:
            validate_base_url(v)
        return v

    @field_validator("token_command")
    @classmethod
    def _check_token_command(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and (not v or any(not arg.strip() for arg in v)):
            raise ValueError("token_command must be a non-empty argv of non-empty strings")
        return v


class AnthropicProviderEntry(_ProviderBase):
    """``api_format = "anthropic"`` -- the Anthropic Messages wire format.

    ``deployment = "direct"`` (default) hits api.anthropic.com; ``"vertex"``
    is Claude-on-Vertex (model id in the URL, ``anthropic_version`` in the body,
    a Google-OAuth bearer via ``token_command``).
    """

    api_format: Literal["anthropic"]
    prompt_caching: bool = True


class OpenAIProviderEntry(_ProviderBase):
    """``api_format = "openai"`` -- any OpenAI Chat Completions wire format.

    ``deployment = "direct"`` works against OpenAI, OpenRouter, Ollama, vLLM,
    LM Studio, llama.cpp, Gemini's OpenAI-compatible endpoint, GitHub Copilot,
    etc.; ``"vertex"`` is Gemini's Vertex OpenAPI endpoint; ``"azure"`` is Azure
    OpenAI (deployment-name in the URL, api-version query param, ``api-key``
    header).
    """

    api_format: Literal["openai"]


ProviderEntry = Annotated[
    AnthropicProviderEntry | OpenAIProviderEntry,
    Discriminator("api_format"),
]


class RoleModel(BaseModel):
    """One role's `(provider, model)` assignment.

    `provider` is the name (TOML table key) of an entry in `[providers.*]`.

    `temperature` is the sampling temperature agent6 will pin on every
    call for this role. Defaults to ``0.0``, agent6's tool-use loop is a
    search-and-act feedback loop and high-temperature sampling causes
    observable degeneration on some open-weights models (caught
    Kimi K2.6 emitting 15997 literal ``\\n`` escapes in a single
    ``old_string`` argument before hitting the completion-tokens cap).
    Anthropic and OpenAI models are tuned to behave well at any
    temperature; OpenRouter routes to provider defaults that vary by
    model, so pinning is the only way to make benches reproducible.
    Set to ``null`` only if you specifically want the provider's default
    behaviour. TOML has no null literal and ``temperature = nan`` fails the
    0.0-2.0 bounds, so null is reachable only via the Python API; omitting the
    key leaves the ``0.0`` default, not the provider's default.
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

    # "none" is the explicit UNSANDBOXED opt-out (no Landlock/seccomp/namespaces),
    # self-authorizing: an operator-only, LLM-unreachable config value, so writing
    # it is the consent (the loud run-startup warning is the safety net). The
    # per-invocation forms are `--dangerously-disable-sandbox` /
    # AGENT6_DANGEROUSLY_DISABLE_SANDBOX. `auto` never resolves to none on Linux
    # (only by detection on a non-Linux host) -- see detect.select_profile.
    profile: Literal["auto", "strict", "hardened", "none"] = "auto"
    # Where the agent PROCESS (its own LLM/provider HTTP) may connect:
    #  - `providers`: only the configured `[providers.*]` endpoints, plus any
    #    `allow_urls`. On `strict` this is structural, a trusted broker (see
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
    #    `run_command` stays blocked. `strict`-only, singling one tool out needs
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
    # unaffected. STRICT-ONLY: it is a read-only bind-remount, which needs a
    # mount namespace. On hardened the cwd is blanket read-write (no namespace
    # to carve with, and carving .git read-only would also deny new top-level
    # entries and break toolchains), so .git is writable there: recoverable,
    # gated by run_commands, and run state lives out of the workspace.
    protect_git: bool = True
    # Per-process memory cap in MiB for every JAILED child (`run_command`,
    # verify, metric, machine `tool` states, offline script tests), applied as
    # RLIMIT_DATA by the launcher and inherited by the child's descendants.
    # RLIMIT_DATA (heap + private writable anonymous mappings) rather than
    # RLIMIT_AS so runtimes that reserve large address space without
    # committing it (V8, JVM, ASAN) keep working. A runaway allocation fails
    # with ENOMEM (Python MemoryError) that the agent sees as an ordinary
    # failed command, instead of driving the host to the OOM killer. The cap
    # is per PROCESS, not per tree; it bounds the common single-runaway case,
    # not a fork bomb. An operational guardrail, not a security control. 0
    # disables. No effect under profile `none` (no confinement at all there).
    # Raise it when a legitimate build or test suite needs more than 4 GiB in
    # one process.
    memory_limit_mb: int = Field(default=4096, ge=0)
    # Extra egress destinations the AGENT process may reach under
    # `agent_network = "providers"`, on top of the configured provider
    # endpoints. Each entry is a `host`, `host:port`, or full URL (a missing
    # scheme implies https / port 443); only the host:port is used to open a
    # broker socket. Secure default empty, no destination beyond the
    # providers is reachable. MERGE: last-overlay-wins (the most-specific tier
    # that sets the key replaces it wholesale, like every other list field);
    # provider endpoints always UNION in regardless of tier. Effective egress
    # = union(provider endpoints) + allow_urls(winning tier). Only meaningful
    # under `agent_network = "providers"`; ignored under `local`/`open`. It
    # widens only the agent path, never a jailed `tool`/`run_command`.
    allow_urls: tuple[str, ...] = ()
    # Extra filesystem paths a JAILED command may READ and EXECUTE, on top of
    # the system defaults (/usr /bin /lib /lib64 /etc /dev) and the workspace.
    # For projects whose toolchain or interpreter lives outside the repo — a
    # system conda/virtualenv, a language toolchain (Go/Rust/Node), a shared
    # data dir. Each entry is an absolute path; it is granted read+execute
    # (not write) under `hardened`/`strict`. This LOOSENS confinement (the child
    # can read more of the host), so list only what the build/test actually
    # needs. Empty by default. Has no effect under the `none` profile.
    extra_read_paths: tuple[str, ...] = ()

    @field_validator("allow_urls")
    @classmethod
    def _check_allow_urls(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for entry in v:
            _validate_allow_url(entry)
        return v

    @field_validator("extra_read_paths")
    @classmethod
    def _check_extra_read_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for p in v:
            if not p.startswith("/"):
                raise ValueError(f"sandbox.extra_read_paths must be absolute: {p!r}")
            # These paths are bind-mounted read+execute into the jail, so a `..`
            # component would let an entry traverse outside its apparent target.
            # Reject any `..` segment outright (absolute + no traversal).
            if ".." in Path(p).parts:
                raise ValueError(f"sandbox.extra_read_paths must not contain '..': {p!r}")
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
                " = 'open': run_command runs inside the confined agent process."
                " Use 'only_explicit_states' for audited per-tool egress, or set"
                " agent_network = 'open'."
            )
        if self.agent_network == "local" and self.allow_urls:
            # The docstring promises `local` refuses allow_urls; enforce it rather
            # than silently ignoring the list. `local` confines egress to loopback
            # providers, so an external allow-list can never take effect.
            raise ValueError(
                "sandbox.agent_network = 'local' (loopback providers only) cannot"
                " be combined with sandbox.allow_urls: offline has nothing"
                " external to allow-list. Remove allow_urls, or use"
                " agent_network = 'providers'."
            )
        return self


class GitCommitConfig(BaseModel):
    """Optional overrides for the author/committer identity on agent6 commits.

    All three fields default to None, meaning "use whatever the project's
    `git config user.name` / `user.email` already resolves to". The startup
    check in `agent6 run` refuses to proceed if neither an override nor a
    resolvable git-config identity is present, we will not silently commit
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
    # When auto_stash stashed pre-run changes, restore them at run end. Default
    # off (safe): the run-end reporter always prints how to pop the stash; with
    # this on, agent6 also pops it for you when it can do so cleanly (switching
    # back to the base branch first under branch_per_run), and otherwise leaves
    # the stash with a message rather than risk a conflicted auto-apply.
    auto_stash_pop: bool = False
    branch_per_run: bool = True
    # Where a run's branch is cut from when you are NOT on the base branch (e.g.
    # you are still on a previous run's `agent6/*` branch, having not merged it):
    #   "current" (default) -- cut from HEAD, STACKING the new run on the current
    #      branch's work. Serial runs pile up; deliberate if you are iterating.
    #   "base" -- cut from the base branch (the nearest non-run branch this branch
    #      descends from), so each run starts from a clean line, not the last run.
    #   "ask" -- prompt when you are not on the base branch (stack / from base /
    #      abort); non-interactive falls back to "base" (the un-surprising choice).
    # No effect when you are already on the base branch (nothing to stack on).
    branch_from: Literal["current", "base", "ask"] = "current"
    # Default strategy for `agent6 runs merge`: how the run branch lands on
    # your branch. `squash` (one combined commit), `merge` (a
    # --no-ff merge keeping the per-step history), or `ff` (fast-forward only).
    # The per-step commits always happen on the run branch during the run; this
    # only governs how they are consolidated when you merge.
    merge_strategy: Literal["squash", "merge", "ff"] = "squash"
    # After a successful run, automatically run `merge_strategy` to land the run
    # branch on its base (what `agent6 runs merge` does, run for you). Default off:
    # the run branch is kept until you choose to merge. Requires branch_per_run
    # (without a run branch there is nothing to merge). With auto_stash_pop the
    # merge lands first, then your stashed pre-run changes go back on top.
    auto_merge: bool = False
    # After auto_merge, delete the run branch when it is safely deletable
    # (`git branch -d`: reachable-merged, so merge/ff strategies). A squash-merged
    # branch is unreachable and is reported with the `git branch -D` to remove it by
    # hand, never force-deleted. Requires auto_merge. With both on, run branches
    # stop accumulating, so agent6 looks like a direct-to-branch agent while keeping
    # the per-step commits during the run. Default off.
    auto_prune: bool = False
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
    # regardless of these toggles; they are reserved (nothing honors them
    # today) and `agent6 check` flags allow_push=True as a misconfiguration.
    allow_push: bool = False
    allow_force: bool = False
    allow_history_rewrite: bool = False
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)

    @model_validator(mode="after")
    def _check_auto_merge(self) -> GitConfig:
        if self.auto_merge and not self.branch_per_run:
            raise ValueError(
                "git.auto_merge requires git.branch_per_run: with no run branch there is "
                "nothing to merge (the run commits straight onto your branch)."
            )
        if self.auto_prune and not self.auto_merge:
            raise ValueError(
                "git.auto_prune requires git.auto_merge: pruning a run branch only makes "
                "sense once it has been merged."
            )
        return self


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
    # inherently repo-specific, so it has no useful global default and defaults
    # to empty. Optional: `agent6 run`/`plan` infer one per run when it is unset
    # (AGENTS.md -> repo signals -> a cheap LLM call; see agent6.verify_infer),
    # falling back to a gateless run. `agent6 init` can pin one.
    verify_command: tuple[str, ...] = ()
    # per-call timeout for verify_command (and metric_command) in
    # seconds. Defaults to the jail's general 600s but should be cranked
    # MUCH lower for benches where the verify is a fast correctness test
    # (perf-takehome's CorrectnessTests run in ~2s; a 30s cap detects
    # infinite-loop / quadratic edits 20x faster than the 600s default).
    # Setting too low for slow legitimate tests will cause false-positive
    # failures, so leave at 600 unless the verify is reliably fast.
    verify_timeout_s: float = Field(gt=0.0, default=600.0)
    # When true, finish_run is refused while the last verify is red (or a verify
    # command is configured but was never run): the worker must get verify green
    # or explicitly stop. Default false keeps finish_run always honorable, but
    # even then a finish over a red verify is reported honestly (run.end
    # all_passed=False -> "finished", never "passed"); this flag turns the honest
    # signal into a hard gate for operators who want it.
    require_verify_to_finish: bool = False
    # Opt-in: bounce the FIRST finish_run over a green verify once, with a
    # directive to re-check every spec requirement (the committed suite may
    # cover a subset). Targets the finish-on-green-but-incomplete failure
    # mode measured on bench/coreagent's eventflow task; costs about one
    # extra turn per run when on. See docs/config.md for the measurements.
    spec_recheck_on_finish: bool = False
    # Optional. None means "no metric; ``run_metric_command`` is unavailable".
    metric: MetricConfig | None = None


class ContextConfig(BaseModel):
    """``[context]`` section: tiered context-compaction thresholds."""

    model_config = _BASE_MODEL_CONFIG

    # Tiered context-compaction thresholds (approximate chars; tokens ~=
    # chars/4). When cumulative *tool_result* content grows past
    # ``drop_at_chars`` the oldest tool_results are replaced by a
    # short placeholder (the worker can re-call the tool to refetch). When the
    # *whole* context (text + tool_use inputs + surviving tool_results) grows
    # past ``summarise_at_chars`` -- which must be > drop, so tier-2
    # escalates above tier-1 -- the conversation is summarized and restarted
    # (the durable task DAG survives; the restart notice points the worker at
    # ``list_tasks`` to recover task-level state).
    # ``summary_max_tokens`` caps the summarizer's output.
    #
    # Default ``None`` == ADAPTIVE: agent6 sizes both thresholds from the worker
    # model's context window (tier-1 at ~45%, tier-2 at ~80% of it), resolving
    # the window from a bundled table of tested models + the live model cache
    # (see ``models.registry.compaction_thresholds``). Pin them by setting BOTH
    # explicitly (e.g. a self-hosted model agent6 can't size); leave BOTH unset
    # to stay adaptive. When the window is unknown the historical 256k/768k
    # fixed defaults apply.
    drop_at_chars: int | None = Field(default=None, gt=0)
    summarise_at_chars: int | None = Field(default=None, gt=0)
    summary_max_tokens: int = Field(gt=0, default=2048)
    # Tier-1 gist elision: a large read_file result about to be elided decays
    # to a placeholder carrying a model-written gist of the file first (one
    # batched reviewer-model call per drop event), then to the bare marker
    # under continued pressure. Measured on the longhorizon bench: bare
    # elision of reference docs halves a retention task's score under a small
    # window. False = straight to bare markers (no distiller calls).
    elision_gists: bool = True

    @model_validator(mode="after")
    def _check_compaction_thresholds(self) -> ContextConfig:
        drop, summarise = self.drop_at_chars, self.summarise_at_chars
        # Both-or-neither: a lone value is ambiguous (is the other adaptive or
        # fixed?). Neither set == adaptive from the model's context window.
        if (drop is None) != (summarise is None):
            raise ValueError(
                "set BOTH context.drop_at_chars and"
                " summarise_at_chars, or NEITHER (neither == adaptive,"
                " sized from the worker model's context window)."
            )
        # Tier 2 (summarise + restart) must escalate ABOVE tier 1 (drop old
        # tool_results). If summarise <= drop, tier 2 fires at or before tier 1
        # -- the inverted ordering that historically left tier 2 unreachable.
        if drop is not None and summarise is not None and summarise <= drop:
            raise ValueError(
                "context.summarise_at_chars"
                f" ({summarise}) must be greater than"
                f" drop_at_chars ({drop}): tier-2"
                " summarise must escalate above tier-1 elision."
            )
        return self


class PromptConfig(BaseModel):
    """``[prompt]`` section: system-prompt override, structural priors, and
    one-shot task-prompt revision."""

    model_config = _BASE_MODEL_CONFIG

    # ADVANCED: replace run-mode's static base system prompt (role + edit/tool-use/
    # dag/scope rules) with the contents of this file. The dynamic blocks (verify,
    # metric, budget, repo-priors + AGENTS.md) still append, so repo context and
    # the budget cap are preserved. Empty = the built-in default. You own keeping
    # the tool contracts intact (apply_edit/apply_patch, run_verify_command,
    # finish_run); run startup warns if the override omits them. Inspect the
    # assembled result with `agent6 prompt show`.
    system_prompt_file: str = ""
    # Include the structural-prior blocks in the run-mode <repo-priors>: hot
    # symbols (cross-file reference ranking), git co-change pairs, and the
    # tree-sitter symbol outline. Default on. Set false for a leaner/cheaper
    # prompt that relies purely on on-demand exploration (outline/find_definition)
    # -- the base repo map + AGENTS.md still ship.
    structural_priors: bool = True
    # one-shot task prompt revision before the worker loop starts.
    # Reuses the reviewer model, takes no tools, and is budget-tracked like
    # any other provider call. Default off keeps crisp prompts/frontier-model
    # runs on the old path.
    revise_prompt: Literal["off", "auto", "interactive"] = "off"
    # Front-load task decomposition (run mode). When on the worker's system
    # prompt swaps the "DAG is optional" guidance for a "decompose first"
    # directive: lay the task out as ordered subtasks before editing, then work
    # one focused subtask at a time (the existing surface-current-task and
    # finish-gate machinery walks the frontier). Helps small/open models that
    # lose track of multi-part tasks; a capable model decomposes implicitly and
    # only pays the 2-4x turn overhead. "auto" (default) enables it ONLY for
    # worker models with a measured win in the capability registry
    # (models.registry.decompose_default); the CLI pins auto to on/off at run
    # start via ``with_decompose``, and the engine treats any value other than
    # "on" as off. No effect on plan/ask/machine/agent modes. See
    # docs/config.md for the measured per-model effect.
    decompose: Literal["auto", "on", "off"] = "auto"

    @model_validator(mode="after")
    def _check_system_prompt_file(self) -> PromptConfig:
        # Fail loud at config time if the override path is set but missing, rather
        # than silently falling back to the default prompt at run start.
        if self.system_prompt_file:
            p = Path(self.system_prompt_file).expanduser()
            if not p.is_file():
                raise ValueError(f"prompt.system_prompt_file: not a readable file: {p}")
        return self


class SkillsConfig(BaseModel):
    """``[skills]`` section: operator-installed SKILL.md packs (agentskills.io).

    Skills live under ``<data-dir>/skills/<name>/`` (``agent6 skills install``)
    plus any ``extra_dirs``. Installed means enabled: the run-mode system
    prompt lists each enabled skill's name + description and the worker loads
    content on demand; the ``state`` map holds only the exceptions. Skills are
    trusted like config (operator-chosen prompt content); nothing in a skill
    is ever executed by the loader.
    """

    model_config = _BASE_MODEL_CONFIG

    # Master switch for the whole subsystem. Off = no index block, no
    # use_skill tool, slash commands don't register.
    enabled: bool = True
    # Additional skill directories scanned BEFORE the installed dir (a local
    # checkout during skill development wins over an installed copy). Each may
    # hold skill subdirectories or be a single skill dir itself.
    extra_dirs: tuple[str, ...] = ()
    # Per-skill exceptions, one value per skill so contradictory states are
    # unrepresentable: "disabled" drops it from the index; "always" injects
    # the full SKILL.md text into the system prompt instead of indexing it.
    # Absent = "enabled". Layered configs merge this map key-wise, so a repo
    # config can flip one skill without restating the rest.
    state: dict[str, Literal["enabled", "disabled", "always"]] = Field(default_factory=dict)


class ReviewConfig(BaseModel):
    """``[review]`` section: critic-in-loop trigger + the adversarial review panel."""

    model_config = _BASE_MODEL_CONFIG

    # critic-in-loop. When != "off", Workflow runs the
    # ``reviewer`` model as a critic at the chosen trigger and injects
    # its critique as a user message the worker sees next turn.
    #   off              - never (default; behaviour unchanged).
    #   on_verify_fail   - after every verify failure.
    #   before_finish    - intercept ``finish_run``; reject if critic
    #                      is not satisfied and inject critique.
    #   periodic         - every ``period`` iterations.
    # The reviewer provider must already be configured in
    # ``[models.reviewer]`` (same one ``agent6 review`` uses).
    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    period: int = Field(ge=1, default=10)
    # Adversarial review panel (opt-in). Shared by `agent6 review --reviewers N`
    # and, once validated, the in-loop critic trigger. ``decision`` is only
    # a GATE in-loop; "advisory" (default) just injects findings as guidance and
    # never blocks. ``seats`` (flat "persona@provider/model" strings, e.g.
    # "security@openrouter/moonshotai/kimi-k2") overrides size/personas for
    # distinct models per seat; empty = ``panel_size`` seats on the
    # ``reviewer`` model with ``personas`` cycled across them.
    panel_size: int = Field(ge=1, default=1)
    personas: tuple[str, ...] = ()
    decision: Literal["advisory", "veto", "quorum", "all"] = "advisory"
    quorum: int = Field(ge=1, default=2)
    # Per-run cap on total panel blocks before the gate auto-downgrades to
    # advisory for the rest of the run (so a gating panel can never stall forever).
    max_total_rejections: int = Field(ge=1, default=4)
    # Budget floor: the in-loop review panel is SKIPPED (approve-and-proceed) once
    # the run's remaining token budget falls below this fraction -- reviewing costs
    # most exactly when budget is scarcest. Default 0.25 = skip the panel in the
    # last quarter of the budget.
    budget_fraction: float = Field(gt=0.0, le=1.0, default=0.25)
    seats: tuple[str, ...] = ()
    # Seat concurrency for the in-loop panel (1 = sequential). The post-hoc
    # `agent6 review` runs all seats in parallel regardless (fast one-shot).
    concurrency: int = Field(ge=1, default=1)
    # Reviewer tier: "diff" (one grounded call over the diff) or "explore" (a
    # read-only tool-using mini-loop that reads the broader repo first to catch
    # cross-file impact). explore is more thorough but costs several calls/seat.
    tier: ReviewTier = "diff"

    @model_validator(mode="after")
    def _check_review_seats(self) -> ReviewConfig:
        # Each seats entry is "persona", "persona@provider/model", or
        # "@provider/model"; an "@" form must name BOTH a provider and a model so
        # a typo doesn't silently degrade to the reviewer route.
        for spec in self.seats:
            if not spec.strip():
                raise ValueError("review.seats entries must be non-empty")
            _persona, sep, route = spec.partition("@")
            if sep:
                provider, slash, model = route.partition("/")
                if not (provider.strip() and slash and model.strip()):
                    raise ValueError(
                        f"review.seats: {spec!r} must be"
                        " 'persona@provider/model' (both provider and model required)"
                    )
        return self

    @model_validator(mode="after")
    def _check_review_quorum(self) -> ReviewConfig:
        # The quorum gate counts one block per DISTINCT model, so quorum > 1 needs
        # at least that many distinct models -- a same-model panel can reach only 1
        # and would never gate. Catch the footgun at load time.
        if self.decision == "quorum" and self.quorum > 1:
            models = {(s.partition("@")[2].strip() if "@" in s else "") for s in self.seats}
            if len(models) < self.quorum:
                raise ValueError(
                    f"review.decision='quorum' with quorum={self.quorum}"
                    f" needs >= {self.quorum} DISTINCT models (the gate counts one block per"
                    " distinct model). Provide them via seats"
                    " ('persona@provider/model'), or use decision='veto'."
                )
        return self


class BudgetConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    # Hard stops on token spend. Defaults are generous safety ceilings (the
    # run is resumable from the persistent task graph if hit); tighten them
    # per-repo or use `best_effort_usd_limit` for a dollar-denominated bound.
    max_input_tokens: int = Field(gt=0, default=2_000_000)
    max_output_tokens: int = Field(gt=0, default=200_000)
    # Optional best-effort dollar limit (0 = off). The token ceilings above are
    # the authoritative constraint; this field sizes and bounds them when price
    # data exists. At load it converts to token ceilings (worker-model pricing,
    # the lower of the two wins per axis); at runtime it stops the run when the
    # ESTIMATED spend (provider-reported cost, else price x tokens, including
    # cache cost the token caps omit) crosses it. With no price data and no
    # reported cost it does nothing, hence best effort. The `--max-usd` flag
    # writes this field and, because an explicit flag is a promise, refuses to
    # start when the worker model has no price data.
    best_effort_usd_limit: float = Field(ge=0.0, default=0.0)


class MachineNotifyConfig(BaseModel):
    """Optional out-of-band notify hook for a running machine.

    When ``on_event`` is set, `agent6 machine run` runs the argv tuple on each
    `machine.notify` (a state's ``notify`` message) and on the terminal
    `machine.end`, on the host OUTSIDE the jail (mirror of
    ``[notify].on_complete``). The argv is operator-controlled and never
    includes LLM output. Env vars passed:

    - ``AGENT6_MACHINE_ID``      , the machine id
    - ``AGENT6_MACHINE_DIR``     , absolute path to the instance dir
    - ``AGENT6_MACHINE_EVENT``   , ``notify`` or ``end``
    - ``AGENT6_MACHINE_STATE``   , the state that emitted it
    - ``AGENT6_MACHINE_MESSAGE`` , the notify message (or the end reason)
    - ``AGENT6_MACHINE_LEVEL``   , ``info``/``warn``/``error`` for notify, or the
                                   ``ok``/``failed`` status for end

    Use it to fan out to a phone (ntfy/Pushover/Telegram/email); agent6 owns no
    push infra. A failed hook is logged and does not change the exit code.
    """

    model_config = _BASE_MODEL_CONFIG

    on_event: tuple[str, ...] = Field(default=(), description="argv to run on a notify/end event")
    timeout_s: float = Field(gt=0.0, default=30.0)


class MachineConfig(BaseModel):
    """State-machine runtime knobs (`agent6 machine run`)."""

    model_config = _BASE_MODEL_CONFIG

    # How many recent blackboard snapshots to keep per machine instance.
    # Recovery only reads the latest and `machine replay` rebuilds from the
    # journal, so old snapshots are an audit convenience, not state. 0 keeps
    # every snapshot (one file per transition; budget disk accordingly for
    # long-running machines).
    snapshot_keep: int = Field(ge=0, default=5)
    notify: MachineNotifyConfig = Field(default_factory=MachineNotifyConfig)


def is_loopback_host(host: str) -> bool:
    """True iff *host* is a loopback bind (the one source of truth for the web
    UI's secure-by-default gate; a wildcard like 0.0.0.0/:: is NOT loopback)."""
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized.lower() == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class WebConfig(BaseModel):
    """`agent6 web` server bind. Secure by default: loopback only.

    Remote access is expected behind `tailscale serve` (HTTPS + WireGuard) in
    front of the loopback bind; the tailnet identity is the access control, so
    there is no app-level auth. Binding a non-loopback address exposes the write
    surface (spawn runs, answer prompts) to anyone who can reach the port, so it
    is gated behind `allow_non_loopback = true` and carries no default.
    """

    model_config = _BASE_MODEL_CONFIG

    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535, default=7658)
    # Opt-in required to bind a non-loopback host. Off by default so a typo or a
    # copied config can never silently expose the agent to the local network.
    allow_non_loopback: bool = False

    @model_validator(mode="after")
    def _guard_non_loopback(self) -> WebConfig:
        if not is_loopback_host(self.host) and not self.allow_non_loopback:
            raise ValueError(
                f"[web].host = {self.host!r} is not loopback. Binding a non-loopback"
                " address exposes the web UI's write surface; set [web]"
                " allow_non_loopback = true to opt in (and prefer `tailscale serve`"
                " in front of a 127.0.0.1 bind instead)."
            )
        return self


class Agent6Section(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    config_version: int = Field(ge=1, le=1, default=1)
    # Absolute base directory for per-repo agent6 state (this per-repo config +
    # all run state), which lives OUT of the workspace under ``<base>/<repo-id>/``
    # (default ``$XDG_STATE_HOME/agent6``; see ``agent6.paths.state_base``). Can
    # ONLY be set in the GLOBAL config: it locates the per-repo config, so a
    # per-repo/flag value would be chicken-and-egg. Must be absolute. Point it
    # at a persisted, out-of-cwd path (e.g. a mounted volume) to keep run state
    # across devcontainer rebuilds.
    state_dir: str | None = None

    @field_validator("state_dir")
    @classmethod
    def _check_state_dir(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not Path(v).expanduser().is_absolute():
            raise ValueError(f"[agent6].state_dir must be an absolute path, got {v!r}")
        return v


class NotifyConfig(BaseModel):
    """Optional post-run notification hook.

    When ``on_complete`` is set, agent6 runs the argv tuple after the
    workflow returns (``agent6 run`` or ``agent6 resume``). The argv is
    operator-controlled, it never includes LLM output, and runs in the
    user's shell environment, NOT in the jail, with these env vars:

    - ``AGENT6_RUN_ID``      , run id under the per-repo run-state dir
    - ``AGENT6_RUN_OK``      , ``1`` if the workflow finished cleanly, ``0`` otherwise
    - ``AGENT6_RUN_REASON``  , workflow termination reason (e.g. ``finish_run``,
                                 ``budget_exhausted``, ``provider_error``)
    - ``AGENT6_RUN_DIR``     , absolute path to the run dir

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


class ParallelConfig(BaseModel):
    """``[parallel]`` section: fan-out defaults for `agent6 run --parallel`.

    ``--parallel N`` (or a comma-separated model list) runs N isolated lanes,
    each a disposable clone of the repo, and auto-compares the results. These
    knobs bound and place that fan-out; nothing here mutates the origin repo.
    """

    model_config = _BASE_MODEL_CONFIG

    # Hard cap on lanes per fan-out. `--parallel` over this refuses up front so a
    # typo (or a long model list) can't spawn an unbounded pile of clones+runs.
    max_lanes: int = Field(ge=1, default=4)
    # Base directory for lane workspaces (each fan-out gets `<workdir>/<fanout-id>/
    # lane-<i>`). "" resolves to `<cache_dir>/parallel`, a regenerable cache the
    # orchestrator cleans up after importing each lane. Point it at a fast disk
    # for large repos.
    workdir: str = ""


class Config(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    agent6: Agent6Section = Field(default_factory=Agent6Section)
    providers: dict[str, ProviderEntry] = Field(default_factory=dict)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    machine: MachineConfig = Field(default_factory=MachineConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    # Named config PROFILE: a preset that fills in many settings at once (see
    # agent6.config BUILTIN_PROFILES + user [profiles.<name>] tables). Injected
    # just ABOVE the config layer that selected it, so the profile OVERRIDES that
    # config; a more-specific config layer (repo over global, an explicit
    # --config FILE) or the --profile flag still overrides the profile. Only the
    # most-specific source's profile applies -- global and repo presets do not
    # stack. "" / "standard" = the plain defaults. The --profile CLI flag selects
    # a profile that overrides all config except an explicit --config FILE.
    profile: str = ""

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
        """When `[budget].best_effort_usd_limit > 0`, convert to token ceilings via the
        worker model's pricing and apply as a TIGHTER upper bound on top
        of any explicit `max_input_tokens` / `max_output_tokens`. The
        smaller of (operator-set, USD-converted) wins per axis - both
        are valid ceilings; the lower one is the effective cap.
        Operators who want USD only can set the token ceilings to large
        placeholder values (e.g. 999_999_999) and the USD conversion
        will dominate."""
        if self.budget.best_effort_usd_limit <= 0:
            return self
        worker = self.models.resolve("worker")
        if worker is None:
            # No worker model to price against yet; the conversion is
            # applied once a runnable config is assembled.
            return self
        converted = usd_budget_to_tokens(
            self.budget.best_effort_usd_limit, worker_model=worker.model
        )
        if converted is None:
            # No cached price for the worker model (pricing comes from the
            # provider's models endpoint; anthropic publishes none). The
            # operator token ceilings stand; the runtime estimated-USD ceiling
            # still applies where per-call cost is reported or priced.
            return self
        usd_in, usd_out = converted
        new_in = min(self.budget.max_input_tokens, usd_in)
        new_out = min(self.budget.max_output_tokens, usd_out)
        if new_in == self.budget.max_input_tokens and new_out == self.budget.max_output_tokens:
            return self
        new_budget = BudgetConfig(
            max_input_tokens=new_in,
            max_output_tokens=new_out,
            best_effort_usd_limit=self.budget.best_effort_usd_limit,
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
            budget["best_effort_usd_limit"] = max_usd
        if max_input_tokens is not None:
            budget["max_input_tokens"] = max_input_tokens
        if max_output_tokens is not None:
            budget["max_output_tokens"] = max_output_tokens
        return Config.model_validate(data)

    def with_sandbox_overrides(
        self,
        *,
        disable_sandbox: bool = False,
        auto_approve: bool = False,
    ) -> Config:
        """Return a copy with per-invocation sandbox overrides from CLI flags.

        ``disable_sandbox`` forces ``sandbox.profile = "none"`` (unconfined).
        ``auto_approve`` upgrades ``run_commands`` ``"ask" -> "yes"`` but never
        resurrects a withheld ``"no"`` (a per-invocation flag must not grant a
        capability the standing policy denied). Both are operator-supplied
        (flag/env); the LLM can reach neither.
        """
        if not disable_sandbox and not auto_approve:
            return self
        data = self.model_dump(mode="python")
        sandbox = data.setdefault("sandbox", {})
        if disable_sandbox:
            sandbox["profile"] = "none"
        if auto_approve and self.sandbox.run_commands != "no":
            sandbox["run_commands"] = "yes"
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
            budget["best_effort_usd_limit"] = max_usd
        if max_input_tokens is not None:
            budget["max_input_tokens"] = max_input_tokens
        if max_output_tokens is not None:
            budget["max_output_tokens"] = max_output_tokens
        return Config.model_validate(data)

    def with_inferred_verify(self, argv: tuple[str, ...]) -> Config:
        """Return a copy with an inferred ``workflow.verify_command``.

        Used by `agent6 run`/`plan` to inject a verify command inferred at run
        start when none is configured. IN-MEMORY only -- runs never write config;
        the operator is shown the inferred command and can pin it explicitly.
        Re-validates through ``model_validate``. A no-op for empty ``argv``.
        """
        if not argv:
            return self
        data = self.model_dump(mode="python")
        data.setdefault("workflow", {})["verify_command"] = list(argv)
        return Config.model_validate(data)

    def with_decompose(self, value: Literal["on", "off"]) -> Config:
        """Return a copy with ``prompt.decompose`` pinned to *value*.

        Used by the CLI to resolve ``"auto"`` (from the model-capability
        registry) before the workflow starts, so the engine only ever sees
        on/off. IN-MEMORY only, like ``with_inferred_verify``.
        """
        data = self.model_dump(mode="python")
        data.setdefault("prompt", {})["decompose"] = value
        return Config.model_validate(data)

    def require_runnable(self, role: RoleName = "worker") -> None:
        """Raise ConfigError unless *role* can actually run.

        Checks (in order) that a provider is configured and the role resolves
        to a model whose provider exists. Messages point at the command that
        fixes the gap so a fresh user is never stuck. ``verify_command`` is NOT
        required: `agent6 run`/`plan` infer one when unset (and fall back to a
        gateless run if even that fails) -- see ``agent6.verify_infer``.
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


def _format_validation_error(
    err: ValidationError, source: str, locate: Callable[[str], str | None] | None = None
) -> str:
    lines = [f"Config validation failed: {source}"]
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']} (type={issue['type']})")
        if locate is not None and (where := locate(loc)):
            lines.append(where)
    return "\n".join(lines)


def validate_config(
    raw: dict[str, object],
    *,
    source: str = "<config>",
    locate: Callable[[str], str | None] | None = None,
) -> Config:
    """Validate an already-parsed (and possibly layer-merged) config dict.

    Shared by :func:`load_config` and the layered loader
    (``agent6.config.layer``) so both surface identical field-pointing errors.
    ``locate`` maps a dotted leaf to a "which file, how to fix" hint appended to
    its error line, so a stale value in a layered config names its own source.
    """
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source, locate)) from exc


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
