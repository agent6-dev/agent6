# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading, TOML to pydantic.

This is a trust boundary (untrusted text -> structured types), so we use
pydantic and surface field-pointing errors.

Field policy: **secure by default, fully auditable**. Every field has a
default, and security-sensitive fields default to the *safe* value
(``sandbox.network = "auto"``,
``sandbox.run_commands = "ask"``, ``sandbox.protect_git = true``; git push,
``--force``, and history rewrites are refused unconditionally by ``git_ops``,
with no config override at all). This means a config can be layered (global ``$XDG_CONFIG_HOME``
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

import ipaddress
import math
import re
import string
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

from agent6.errors import OperatorError
from agent6.paths import private_dirs
from agent6.types import RoleName


class ConfigError(OperatorError):
    """Raised when the config file is missing, malformed, or fails validation.

    An OperatorError: the config is the operator's file, so ``cli_main``
    presents it as a refusal, never a crash report.
    """


_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

ApiFormat = Literal["anthropic", "openai"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]
ThinkingLevel = Literal["off", "low", "medium", "high"]
# The review-seat depth (`[review].tier`); ReviewSeat.tier mirrors this, so the
# vocabulary has one owner.
ReviewTier = Literal["diff", "explore"]


def validate_base_url(url: str) -> None:
    """Reject a ``[providers.*].base_url`` that is not an http(s) URL with a host.

    A provider's
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
    a minimal entry (just ``api_format``) is fully usable. Each block is
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

    def source_role(self, role: RoleName) -> RoleName:
        """The configured entry ``resolve(role)`` reads: *role* itself when
        explicitly set, else the worker fallback. Lets an error message name
        the config key the user actually wrote (mirrors ``resolve`` above)."""
        return role if role in self.configured() else "worker"


class SandboxConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    # "none" is the explicit UNSANDBOXED opt-out (no Landlock/seccomp/namespaces),
    # self-authorizing: an operator-only, LLM-unreachable config value, so writing
    # it is the consent (the loud run-startup warning is the safety net). The
    # per-invocation forms are `--dangerously-disable-sandbox` /
    # AGENT6_DANGEROUSLY_DISABLE_SANDBOX. `auto` resolves to none only when the
    # host offers no confinement mechanism at all (non-Linux, or a Linux kernel
    # with neither userns nor Landlock) -- see detect.resolve_isolation.
    isolation: Literal["auto", "strict", "hardened", "none"] = "auto"
    # Which network JAILED commands (`run_command`, `verify`, `metric`, and
    # machine `tool` states) join. A jailed child can never out-reach the
    # process that launches it, so:
    #  - `auto` (default): the run's PRIVATE network where the environment can
    #    give one, DEGRADED WITH A WARNING where it cannot. On `strict` that is
    #    a real network namespace with no route out; on `hardened`/`none` there
    #    is no netns, so the child shares the host network and a once-per-run
    #    warning says so. The secure-by-default option that still runs
    #    everywhere (see AGENTS.md "Secure by default, degrade or refuse").
    #  - `session`: ENFORCE the run's own network -- the commands see each
    #    other (a dev server one starts answers the next) and nothing off the
    #    box. Refuses to run where there is no netns, naming what is
    #    unsupported and how to change it, never silently ineffective.
    #  - `only_explicit_states`: private, EXCEPT machine `tool` states that opt
    #    in with `network = "host"` (audited, deterministic commands);
    #    `run_command` stays private. `strict`-only, refused elsewhere.
    #  - `host`: the machine's own network (a package install, a real service).
    # There is no per-command `none`: the run's commands share one launcher,
    # and isolating them from each other costs the dev server for no security
    # -- the model can chain them into a single script anyway.
    network: Literal["auto", "session", "only_explicit_states", "host"] = "auto"
    run_commands: Literal["yes", "no", "ask"] = "ask"
    # Hosts the `fetch` tool may read WITHOUT asking. Empty (the default) means
    # none: every fetch is a prompt. `"*"` allows any host, written down so the
    # opt-out reads as a choice in `config show` rather than as an absent
    # setting. A leading dot allows subdomains (`.readthedocs.io`). Hosts, not
    # URL prefixes: a prefix invites `evil.com/docs.python.org`.
    #
    # `fetch` exists because a jailed command has no network; it is hidden when
    # `network = "host"`, where the worker can already run curl. It is
    # still an egress channel a model drives -- a GET can encode data in its
    # path -- so a host not listed here is asked about, and an absent operator
    # is a no.
    fetch_hosts: tuple[str, ...] = ()
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
    # committing it (V8, JVM, ASAN) keep working. Per PROCESS, not per tree.
    # An operational guardrail, never a security control: a memory bomb is a
    # denial of service against your own machine, and the kernel already
    # handles that. DEFAULT 0 (off) because a cap costs real builds (a large
    # link, a test matrix) more than it buys; set one when a specific task
    # needs bounding. No effect under profile `none`.
    memory_limit_mb: int = Field(default=0, ge=0)
    # Extra filesystem paths a JAILED command may READ and EXECUTE, on top of
    # the system defaults (/usr /bin /lib /lib64 /etc /dev) and the workspace.
    # For projects whose toolchain or interpreter lives outside the repo — a
    # system conda/virtualenv, a language toolchain (Go/Rust/Node), a shared
    # data dir. Each entry is an absolute path; it is granted read+execute
    # (not write) under `hardened`/`strict`. This LOOSENS confinement (the child
    # can read more of the host), so list only what the build/test actually
    # needs. Empty by default. Has no effect under the `none` profile.
    extra_read_paths: tuple[str, ...] = ()

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

    # Extra absolute paths a jailed command may READ AND WRITE, mounted at
    # their real locations: a build cache, an output dir, a sibling checkout
    # the task legitimately edits. Write implies read (a writable bind mount
    # is readable). This loosens confinement further than extra_read_paths,
    # so list only what the task actually writes. Empty by default; no effect
    # under `none`.
    extra_write_paths: tuple[str, ...] = ()

    @field_validator("extra_write_paths")
    @classmethod
    def _check_extra_write_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for p in v:
            if not p.startswith("/"):
                raise ValueError(f"sandbox.extra_write_paths must be absolute: {p!r}")
            if ".." in Path(p).parts:
                raise ValueError(f"sandbox.extra_write_paths must not contain '..': {p!r}")
        return v

    # Absolute paths hidden from jailed commands even when a broader grant
    # covers them (a dir masks as an empty tmpfs, a file reads empty). agent6's
    # own private dirs (config + state) are ALWAYS hidden -- secrets never
    # enter the jail, even through an explicit extra_read_paths grant of $HOME --
    # and this list adds to that set. Needs the mount namespace: on `hardened`
    # a hide inside a granted region refuses to run (see docs/security.md).
    hide_paths: tuple[str, ...] = ()

    @field_validator("hide_paths")
    @classmethod
    def _check_hide_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for p in v:
            if not p.startswith("/"):
                raise ValueError(f"sandbox.hide_paths must be absolute: {p!r}")
            if ".." in Path(p).parts:
                raise ValueError(f"sandbox.hide_paths must not contain '..': {p!r}")
        return v

    @model_validator(mode="after")
    def _extra_paths_never_target_private_dirs(self) -> SandboxConfig:
        # An extra grant AT or INSIDE an agent6-private dir would mount secrets,
        # transcripts, or installed skills into the jail by name; there is no
        # legitimate case. A grant CONTAINING one (e.g. $HOME) is allowed on
        # strict, where the private dirs are masked out of it.
        for p in (*self.extra_read_paths, *self.extra_write_paths):
            for d in private_dirs():
                if Path(p).is_relative_to(d):
                    raise ValueError(
                        f"sandbox extra path {p!r} is inside the agent6-private dir"
                        f" {str(d)!r} (secrets/state); it never enters the jail."
                        " Grant a different directory."
                    )
        return self


class GitCommitCheckpointConfig(BaseModel):
    """Message style for the per-step commits a run makes on its branch."""

    model_config = _BASE_MODEL_CONFIG

    # agent6: the `agent6 iter N:` subject. conventional: a `type(scope): subject`
    # derived from the diff without a model call. model: the model writes the
    # message from git facts, degrading to agent6 with a warning on any failure.
    message: Literal["agent6", "conventional", "model"] = "agent6"


class GitCommitSquashConfig(BaseModel):
    """Message style for the one commit a squash merge produces."""

    model_config = _BASE_MODEL_CONFIG

    # As checkpoint's styles, plus combine: git's own squash message (the
    # concatenated per-step log).
    message: Literal["agent6", "conventional", "combine", "model"] = "agent6"


class GitCommitConfig(BaseModel):
    """Overrides for the author/committer identity on agent6 commits, the
    provenance trailer, and the per-kind message styles.

    `name`/`email` default to None = the project's own `git config` identity;
    `agent6 run` refuses at startup when neither an override nor a resolvable
    identity exists, rather than committing as `(no author) <(none)>`.
    """

    model_config = _BASE_MODEL_CONFIG

    name: str | None = None
    email: str | None = None
    # Appended to every commit agent6 makes when non-empty, e.g.
    # "Assisted-by: agent6:{model}". {model} = the model(s) that wrote the
    # code, first worker first, ", "-joined when several contributed.
    trailer: str = ""
    checkpoint: GitCommitCheckpointConfig = GitCommitCheckpointConfig()
    squash: GitCommitSquashConfig = GitCommitSquashConfig()

    @field_validator("trailer")
    @classmethod
    def _trailer_is_a_trailer_line(cls, v: str) -> str:
        if not v:
            return v
        fields = {f for _, f, _, _ in string.Formatter().parse(v) if f is not None}
        unknown = fields - {"model"}
        if unknown:
            raise ValueError(
                f"unknown placeholder {sorted(unknown)} in git.commit.trailer (known: {{model}})"
            )
        rendered = v.format(model="m")
        if not re.fullmatch(r"[A-Za-z][A-Za-z-]*: .+", rendered, re.DOTALL):
            raise ValueError(
                'git.commit.trailer must be a git trailer line, "Key: value"'
                ' (e.g. "Assisted-by: agent6:{model}")'
            )
        return v


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
    # Per-step commits land on the run's own detached chain
    # (refs/agent6/<session>), parented on HEAD at run start; HEAD, the
    # operator's index, and the checkout are never touched. branch_per_run
    # additionally advances a visible agent6/<slug> branch ref to the chain
    # tip (off = the hidden ref only). Forced on for --parallel lanes (work
    # is imported by branch).
    branch_per_run: bool = True
    # Off = no per-step commits at all: sessions diff/commits/merge, fork
    # rollback, and the compare judge honestly degrade to "no step history";
    # resume still works from snapshots.
    commit_per_step: bool = True
    # Default strategy for `agent6 sessions merge`: how the run branch lands on
    # your branch. `squash` (one combined commit), `merge` (a
    # --no-ff merge keeping the per-step history), or `ff` (fast-forward only).
    # The per-step commits always happen on the run branch during the run; this
    # only governs how they are consolidated when you merge.
    merge_strategy: Literal["squash", "merge", "ff"] = "squash"
    # After a successful run, automatically run `merge_strategy` to land the
    # run's work on its base (what `agent6 sessions merge` does, run for you).
    # Default off: the run's refs are kept until you choose to merge. Works
    # with branch_per_run off too (the hidden chain ref is merged). With
    # auto_stash_pop the merge lands first, then your stashed pre-run changes
    # go back on top.
    auto_merge: bool = False
    # After auto_merge, delete the run branch when it is safely deletable
    # (`git branch -d`: reachable-merged, so merge/ff strategies). A squash-merged
    # branch is unreachable and is reported with the `git branch -D` to remove it by
    # hand, never force-deleted. Requires auto_merge; no-op when branch_per_run
    # is off (there is no branch, and the hidden chain ref stays as the run's
    # record until `sessions rm`). With both on, run branches stop
    # accumulating, so agent6 looks like a direct-to-branch agent while keeping
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
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)

    @model_validator(mode="after")
    def _check_auto_merge(self) -> GitConfig:
        if self.auto_stash_pop and not self.auto_stash:
            raise ValueError(
                "git.auto_stash_pop requires git.auto_stash: with nothing stashed "
                "pre-run there is nothing to restore at run end."
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
    # When true, finish_session is refused while the last verify is red (or a verify
    # command is configured but was never run): the worker must get verify green
    # or explicitly stop. Default false keeps finish_session always honorable, but
    # even then a finish over a red verify is reported honestly (session.end
    # all_passed=False -> "finished", never "passed"); this flag turns the honest
    # signal into a hard gate for operators who want it.
    require_verify_to_finish: bool = False
    # Opt-in: bounce the FIRST finish_session over a green verify once, with a
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
        if (drop is None) != (summarise is None):
            raise ValueError(
                "set BOTH context.drop_at_chars and"
                " summarise_at_chars, or NEITHER (neither == adaptive,"
                " sized from the worker model's context window). Both at once:"
                " agent6 config set context"
                " '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'"
            )
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
    # finish_session); run startup warns if the override omits them. Inspect the
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
    # any other provider call. Default off: crisp prompts and frontier models
    # do not need revision.
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
    #   before_finish    - intercept ``finish_session``; reject if critic
    #                      is not satisfied and inject critique.
    #   periodic         - every ``period`` iterations.
    # The reviewer provider must already be configured in
    # ``[models.reviewer]`` (same one ``agent6 review`` uses).
    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    period: int = Field(ge=1, default=10)
    # Adversarial review panel (opt-in). ``seats`` is THE roster: flat
    # "persona[@provider/model]" strings (e.g. "security" routes via
    # [models.reviewer]; "security@openrouter/moonshotai/kimi-k2" pins a
    # model). The `agent6 review --reviewers N`/`--personas` flags synthesize
    # an in-memory equivalent. ``decision`` is only a GATE in-loop; "advisory"
    # (default) just injects findings as guidance and never blocks.
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
    """``[budget]``: every provider call is bounded in exactly ONE currency.

    A call the runtime can meter (provider-reported cost, else price x tokens
    at the model's fetched rates, cache-aware) counts against ``max_usd``; a
    call it cannot price counts its input+output tokens against
    ``max_tokens_fallback``. Both fields share one rule: ``-1`` = unlimited,
    ``0`` = refuse calls in that ledger up front (``max_tokens_fallback = 0``
    means never run an unmeterable model), ``> 0`` = the cap. Hitting a cap
    ends the run resumably (``budget_exhausted``); each resumed leg gets a
    fresh budget. The ``--max-usd`` / ``--max-tokens-fallback`` flags override
    per run."""

    model_config = _BASE_MODEL_CONFIG

    max_usd: float = 10.0
    max_tokens_fallback: int = Field(ge=-1, default=2_000_000)

    @field_validator("max_usd")
    @classmethod
    def _usd_unlimited_is_exactly_minus_one(cls, v: float) -> float:
        # Non-finite never binds (nan fails every comparison; inf exceeds any
        # spend), which would silently disable the hard budget.
        if not math.isfinite(v) or (v < 0 and v != -1):
            raise ValueError("max_usd is a finite cap >= 0, or exactly -1 for unlimited")
        return v


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

    - ``AGENT6_SESSION_ID``      , session id under the per-repo state dir
    - ``AGENT6_SESSION_OK``      , ``1`` if the workflow finished cleanly, ``0`` otherwise
    - ``AGENT6_SESSION_REASON``  , workflow termination reason (e.g. ``finish_session``,
                                 ``budget_exhausted``, ``provider_error``)
    - ``AGENT6_SESSION_DIR``     , absolute path to the session dir

    Use cases: desktop notification (``notify-send``), shell-bell, ssh
    push notification, mailx, etc. A failure of the notify command is
    logged but does not change the agent6 exit code.
    """

    model_config = _BASE_MODEL_CONFIG

    on_complete: tuple[str, ...] = Field(default=(), description="argv to run on completion")
    timeout_s: float = Field(gt=0.0, default=30.0)


class MCPSandbox(BaseModel):
    """What ONE spawned MCP server gets, on top of the sandbox a jailed
    command gets.

    A server is spawned by agent6 and fed model input, so it is confined the
    same way and by the same launcher: the workspace, the system dirs, the
    operator's tool dirs, a writable /tmp as HOME. This block names only what
    is EXTRA -- which is why there is nothing to name for most servers, and
    why nobody has to know where their interpreter lives.

    Absent block: exactly those defaults. `unconfined = true` is the escape
    hatch for a server that genuinely needs the host (a shell, a docker
    driver); it contradicts every other field here, so setting both is
    refused rather than silently half-applied.
    """

    model_config = _BASE_MODEL_CONFIG

    # Readable+executable, and writable, BEYOND the command sandbox. `~` expands.
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    # Which network this server joins -- per-server because servers differ from
    # commands and from each other: a browser server exists to reach something,
    # a memory server does not.
    #   auto    (default) a network of its own where the host can give one,
    #                     degrading to the host's with a warning where it cannot
    #   none              a network of its own, alone; refuses where impossible
    #   private           the run's network: the dev server a `run_background`
    #                     started answers this server too, and still nothing
    #                     off the box (a browser server driving the app under
    #                     test is the case this exists for)
    #   host              the machine's network
    network: Literal["auto", "none", "session", "host"] = "auto"
    # No confinement at all: the server runs as the operator, with their whole
    # filesystem and network. For a server whose job IS arbitrary host access.
    unconfined: bool = False

    @model_validator(mode="after")
    def _escape_hatch_is_exclusive(self) -> MCPSandbox:
        if not self.unconfined:
            for group in (self.read_paths, self.write_paths):
                for raw in group:
                    if not Path(raw).expanduser().is_absolute():
                        raise ValueError(
                            f"sandbox paths must be absolute (or start with ~): {raw!r}."
                            " A relative one would be resolved against whatever"
                            " directory agent6 happened to start in."
                        )
            for raw in (*self.read_paths, *self.write_paths):
                for private in private_dirs():
                    if Path(raw).expanduser().is_relative_to(private):
                        raise ValueError(
                            f"sandbox path {raw!r} is inside the agent6-private dir"
                            f" {str(private)!r} (secrets/state); it never enters a"
                            " jail. Grant a different directory."
                        )
            return self
        stated = [
            name
            for name, value in (
                ("read_paths", self.read_paths),
                ("write_paths", self.write_paths),
                ("network", self.network != "auto"),
            )
            if value
        ]
        if stated:
            raise ValueError(
                f"unconfined = true means no sandbox at all, so {', '.join(stated)}"
                " cannot also apply. Drop unconfined, or drop the rest."
            )
        return self


class MCPServerEntry(BaseModel):
    """One MCP (Model Context Protocol) server to spawn at run start.

    The server runs as a long-lived subprocess speaking JSON-RPC 2.0
    over stdio. Its ``command`` (argv) is operator-controlled and never
    contains LLM output. The server runs OUTSIDE the agent6 jail, with the
    same curated environment a ``[notify]`` hook gets -- never the agent6
    process's full ``os.environ``, which carries the provider API keys -- plus
    whatever ``pass_env`` names.

    The LLM sees each MCP-server tool as
    ``mcp__<name>__<server-side-tool-name>`` and can call it through
    the normal tool surface. The MCP server itself is responsible for
    validating the arguments the LLM passes; agent6 forwards them
    verbatim.

    A misbehaving server (crash, hang, malformed output) surfaces as
    a clean tool failure, not an agent crash.
    """

    model_config = _BASE_MODEL_CONFIG

    # Exactly one of these. `command` spawns the server (agent6 owns its env,
    # lifetime and confinement); `url` connects to one the OPERATOR runs, in
    # whatever container or sandbox they chose -- which is how anyone actually
    # runs a server that wants a browser or a device.
    command: tuple[str, ...] = ()
    url: str = ""
    # The env var holding the bearer token for `url`. Named, never inlined: a
    # secret in a config file is a secret in a backup.
    token_env: str = ""
    enabled: bool = True
    # Environment variables this server needs, BY NAME (e.g. ["GITHUB_TOKEN"]).
    # Everything else comes from the curated base agent6 gives any child it
    # spawns outside the jail. Naming each one is the point: a provider key is
    # never among them, because nobody would write it down.
    pass_env: tuple[str, ...] = ()
    # Filesystem confinement for a SPAWNED server. A `url` one is the
    # operator's own process; they confine it where they start it.
    sandbox: MCPSandbox | None = None
    # Ask before each of this server's tool calls ("ask"), or never ("yes").
    # A server's tools do arbitrary things agent6 cannot classify, so the
    # default is the same as a command's: ask. There is no "no" -- withholding
    # a server's tools is what `enabled = false` already says.
    approve: Literal["ask", "yes"] = "ask"
    # Time budget for the initialize + tools/list handshake. If the
    # server doesn't respond in this window we log and skip it.
    startup_timeout_s: float = Field(gt=0.0, default=10.0)
    # Per-call timeout for ``tools/call`` requests. Surfaces as a tool
    # failure (ToolError) if exceeded.
    call_timeout_s: float = Field(gt=0.0, default=60.0)

    @model_validator(mode="after")
    def _one_transport(self) -> MCPServerEntry:
        if bool(self.command) == bool(self.url):
            raise ValueError("set exactly one of `command` (spawn) or `url` (connect)")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError(f"url must be http(s), got {self.url!r}")
        if self.token_env and not self.url:
            raise ValueError("token_env is for `url` servers; a spawned one uses pass_env")
        if self.sandbox is not None and self.url:
            raise ValueError(
                "a [sandbox] block confines a server agent6 SPAWNS; a `url` one"
                " is your own process, so confine it where you start it"
            )
        if self.pass_env and self.url:
            # Nothing is spawned, so there is no environment to pass. Refusing
            # loudly beats accepting a setting that can never take effect.
            raise ValueError("pass_env is for spawned servers; a `url` one uses token_env")
        if self.token_env and self.url.startswith("http://") and not _is_loopback(self.url):
            raise ValueError(
                "a token over plain http would cross the network in cleartext;"
                " use https, or drop token_env for a loopback server"
            )
        return self


def _is_loopback(url: str) -> bool:
    """Whether *url*'s host is this machine. The operator dialling their own
    server is the normal case for `url`, and the only one where plain http
    with a token is not a cleartext secret on the wire."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        # Parsed, never prefix-matched: `127.evil.com` and `127.0.0.1.nip.io`
        # both start with "127." and both resolve wherever their owner points
        # them, so a string test sent the bearer token across the network in
        # cleartext while claiming it never left the machine.
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def mcp_server_name_refusal(name: str) -> str:
    """Why *name* cannot be an MCP server key, or "".

    The LLM-visible tool name is ``mcp__<name>__<tool>`` and routing recovers
    the server by splitting on the FIRST ``__`` after the prefix, so the key
    must be identifier-shaped and ``__``-free.

    Shared with `agent6 mcp connect`, which must refuse BEFORE it writes: the
    name becomes a TOML table header, and validating only at load meant a
    name carrying `]` and a newline could close the table and open one of its
    own choosing.
    """
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        return f"[mcp.servers.<name>] keys must be [A-Za-z0-9_-]+: {name!r}"
    if "__" in name:
        return (
            f"[mcp] server name must not contain '__' (it separates server"
            f" from tool in mcp__<server>__<tool>): {name!r}"
        )
    return ""


class MCPConfig(BaseModel):
    """``[mcp]`` section. Empty / absent / ``enabled = false`` means no
    MCP servers are spawned and the LLM sees zero ``mcp__*`` tools.

    ``servers`` is a name-keyed map (``[mcp.servers.<name>]``), like
    ``[providers.<name>]``: duplicates are unrepresentable, a repo overlay can
    flip one server without restating the rest, and ``config set`` reaches the
    leaves."""

    model_config = _BASE_MODEL_CONFIG

    enabled: bool = False
    servers: dict[str, MCPServerEntry] = Field(default_factory=dict)

    @field_validator("servers")
    @classmethod
    def _valid_server_names(cls, v: dict[str, MCPServerEntry]) -> dict[str, MCPServerEntry]:
        for name in v:
            refusal = mcp_server_name_refusal(name)
            if refusal:
                raise ValueError(refusal)
        return v


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
    # Named strategy PRESET: fills in many settings at once (BUILTIN_PRESETS +
    # user `[presets.<name>]`). "" / "standard" = plain defaults; injection
    # order and stacking rules: `config.layer._apply_preset`.
    preset: str = ""

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

    def with_budget_overrides(
        self,
        *,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
    ) -> Config:
        """Return a copy with budget fields overridden (the per-run CLI flags,
        each writing the config field of the same name). ``None`` keeps the
        existing value."""
        if max_usd is None and max_tokens_fallback is None:
            return self
        data = self.model_dump(mode="python")
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        return Config.model_validate(data)

    def with_sandbox_overrides(
        self,
        *,
        disable_sandbox: bool = False,
        auto_approve: bool = False,
        no_commands: bool = False,
    ) -> Config:
        """Return a copy with per-invocation sandbox overrides from CLI flags.

        ``disable_sandbox`` forces ``sandbox.isolation = "none"`` (unconfined).
        ``auto_approve`` upgrades ``run_commands`` ``"ask" -> "yes"`` but never
        resurrects a withheld ``"no"`` (a per-invocation flag must not grant a
        capability the standing policy denied); it covers every MCP server's
        ``approve`` too, because "do not prompt me this run" that still prompted
        would not be that. ``no_commands`` pins ``run_commands`` to ``"no"`` and
        always may: tightening needs no permission. All are operator-supplied
        (flag/env); the LLM can reach none of them.
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
        thinking: str | None = None,
        temperature: float | None = None,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
    ) -> Config:
        """Return a copy with a machine ``agent`` state's per-state knobs applied.

        Overrides the ``worker`` role (the role machine agent loops run as)
        and the budget ledgers. ``None`` means "inherit the effective config".
        Re-validates so the provider-name checks run against the merged result."""
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
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        return Config.model_validate(data)

    def with_verify_command(self, argv: tuple[str, ...]) -> Config:
        """Return a copy whose ``workflow.verify_command`` is *argv*, `()` for
        a gateless run.

        How `agent6 run`/`plan` inject a verify command inferred at run start,
        and how a run whose policy withholds command tools drops the gate it
        could never execute. IN-MEMORY only -- runs never write config; the
        operator is shown what was picked and can pin it explicitly.
        """
        data = self.model_dump(mode="python")
        data.setdefault("workflow", {})["verify_command"] = list(argv)
        return Config.model_validate(data)

    def clamped_for_ask(self) -> Config:
        """Return a copy with ``sandbox.run_commands`` clamped for `agent6 ask`.

        An ask is a question with the operator sitting there, often in a
        directory that is not even a repo, so it must never execute anything
        unwatched: ``"yes"`` becomes ``"ask"``. Only ever tightens -- ``"no"``
        stays refused, because a run can never loosen a boundary the operator
        set. IN-MEMORY only, like ``with_verify_command``: `config show` keeps
        reporting what the operator actually configured.
        """
        if self.sandbox.run_commands != "yes":
            return self
        data = self.model_dump(mode="python")
        data.setdefault("sandbox", {})["run_commands"] = "ask"
        return Config.model_validate(data)

    def with_decompose(self, value: Literal["on", "off"]) -> Config:
        """Return a copy with ``prompt.decompose`` pinned to *value*.

        Used by the CLI to resolve ``"auto"`` (from the model-capability
        registry) before the workflow starts, so the engine only ever sees
        on/off. IN-MEMORY only, like ``with_verify_command``.
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
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file cannot be read ({path}): {exc}") from exc
    return validate_config(raw, source=str(path))
