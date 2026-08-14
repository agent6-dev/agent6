# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[sandbox]` and `[mcp]` models bounding what a jailed child, and a
spawned MCP server on top of one, may reach."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG
from agent6.paths import private_dirs


class SandboxConfig(BaseModel):
    model_config = MODEL_CONFIG

    # "none" is the explicit UNSANDBOXED opt-out (no Landlock/seccomp/namespaces),
    # self-authorizing: an operator-only, LLM-unreachable config value, so writing
    # it is the consent (the loud run-startup warning is the safety net). The
    # per-invocation forms are `--dangerously-disable-sandbox` /
    # AGENT6_DANGEROUSLY_DISABLE_SANDBOX. `auto` resolves to none only when the
    # host offers no confinement mechanism at all (non-Linux, or a Linux kernel
    # with neither userns nor Landlock) -- see detect.resolve_isolation.
    isolation: Literal["auto", "strict", "hardened", "none"] = Field(
        default="auto",
        description=(
            "`auto` picks the strongest the host supports (`strict`, else `hardened`; `none` only "
            "when the host offers no confinement, loudly). Explicit `strict`/`hardened` refuse "
            "where unsupported, never downgrade. Explicit `none` runs UNSANDBOXED (also "
            "`--dangerously-disable-sandbox` / `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`)."
        ),
    )
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
    network: Literal["auto", "session", "only_explicit_states", "host"] = Field(
        default="auto",
        description=(
            "Which network jailed commands join. `auto`: the run's PRIVATE network (commands "
            "reach each other, nothing off the box, nothing outside reaches in), enforced on "
            "`strict`, degraded to the host's with a warning on `hardened`/`none`. `session`: "
            "the same, refusing where unenforceable. `only_explicit_states`: strict-only; "
            "machine `tool` states opt in. `host`: the machine's network. No per-command "
            "`none`: a run's commands share one launcher."
        ),
    )
    run_commands: Literal["yes", "no", "ask"] = Field(
        default="ask",
        description=(
            "May the LLM run commands (`run_command`, `run_verify_command`, `stop_background`: "
            "one decision for all three): `yes` auto-approves, `no` withholds the tools (and the "
            "verify gate with them), `ask` prompts per call with session-wide allow/deny "
            "answers. `agent6 ask`/`plan` clamp `yes` to `ask`. Per-invocation: `--auto-approve` "
            "(never over a configured `no`), `--no-commands`. A run that cannot ask anyone "
            "refuses to start."
        ),
    )
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
    fetch_hosts: tuple[str, ...] = Field(
        default=(),
        description=(
            "Hosts the `fetch` tool reads WITHOUT asking; any other host prompts, and an absent "
            'operator is a no. Empty = every fetch prompts; `["*"]` = any host, written down as '
            "a choice; a leading dot allows subdomains (`.readthedocs.io`). HOSTS, not URL "
            "prefixes. The rest of fetch is fixed (https only, 1 MiB cap, redirects returned "
            "not followed: security.md, Fixed tool surface). Hidden when "
            '`network = "host"`; withheld from '
            "machine/agent states. A GET can still encode data in its path, hence the empty "
            "default."
        ),
    )
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
    protect_git: bool = Field(
        default=True,
        description=(
            "Keep `.git/` unwritable by jailed commands (else one can plant a git filter that "
            "agent6's host-side auto-commit executes). STRICT-ONLY: a read-only bind needs a "
            "mount namespace (security.md, Git invariants). On `hardened` the "
            "default degrades with a warning; "
            "an explicit `true` refuses. The in-process edit tools refuse `.git` writes "
            "everywhere regardless."
        ),
    )
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
    # needs bounding. No effect under `isolation = "none"`.
    memory_limit_mb: int = Field(
        default=0,
        ge=0,
        description=(
            "`RLIMIT_DATA` cap (MiB) per jailed process (inherited). Off by default: the kernel "
            "already handles a memory bomb, and a cap costs real builds more than it buys. Set one "
            "to bound a specific task; a runaway then fails as an ordinary command error."
        ),
    )
    # Extra filesystem paths a JAILED command may READ and EXECUTE, on top of
    # the system defaults (/usr /bin /lib /lib64 /etc /dev) and the workspace.
    # For projects whose toolchain or interpreter lives outside the repo — a
    # system conda/virtualenv, a language toolchain (Go/Rust/Node), a shared
    # data dir. Each entry is an absolute path; it is granted read+execute
    # (not write) under `hardened`/`strict`. This LOOSENS confinement (the child
    # can read more of the host), so list only what the build/test actually
    # needs. Empty by default. No effect under `isolation = "none"`.
    extra_read_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Extra absolute paths **the run** may **read + execute**, at their real locations: a "
            "toolchain or interpreter outside the repo (conda, Go/Rust/Node, a shared data dir). "
            "Mounted for jailed commands, readable by the in-process tools (name one with an "
            "absolute path). Loosens confinement; list only what the build needs."
        ),
    )

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
    extra_write_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Extra absolute paths **the run** may **read + write**, at their real locations: a "
            "build cache, an output dir, a sibling checkout the task edits. Write implies read. "
            "List only what the task writes."
        ),
    )

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
    hide_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Paths **the run** may never read or write, even under a broader grant. agent6's "
            "config dir and state base are always hidden, so an `extra_read_paths` grant of "
            "`$HOME` never exposes `secrets.toml` or your run history (the data dir and cache "
            "are not: installed skills stay usable). Enforced twice: the in-process tools refuse "
            "them at **every** isolation level (`none` included), and jailed commands see them "
            "masked (a dir appears empty, a file reads empty). Masking needs the mount "
            "namespace: on `hardened` an entry it cannot mask refuses the run, and a grant "
            "exposing the always-hidden dirs warns loudly instead."
        ),
    )

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

    model_config = MODEL_CONFIG

    # Readable+executable, and writable, BEYOND the command sandbox. `~` expands.
    read_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Read+execute paths for this server BEYOND the sandbox a jailed command gets "
            "(absolute or `~`). The workspace, system dirs, tool dirs and a writable `/tmp` as "
            "`HOME` are already there, so a block names only the server's own data."
        ),
    )
    write_paths: tuple[str, ...] = Field(
        default=(),
        description="Paths it may write, likewise additive.",
    )
    # Which network this server joins -- per-server because servers differ from
    # commands and from each other: a browser server exists to reach something,
    # a memory server does not.
    #   auto    (default) a network of its own where the host can give one,
    #                     degrading to the host's with a warning where it cannot
    #   none              a network of its own, alone; refuses where impossible
    #   private           the run's network: the dev server a background command
    #                     started answers this server too, and still nothing
    #                     off the box (a browser server driving the app under
    #                     test is the case this exists for)
    #   host              the machine's network
    network: Literal["auto", "none", "session", "host"] = Field(
        default="auto",
        description=(
            "Which network this server joins. `auto`: one of its own where the host can give a "
            "namespace, degrading to the host's with a warning. `none`: the same, refusing "
            "rather than running connected. `session`: the RUN's network, so a dev server a "
            "background command started answers this server too (a browser server driving the "
            "app under test), and still nothing off the box. `host`: the machine's network."
        ),
    )
    # No confinement at all: the server runs as the operator, with their whole
    # filesystem and network. For a server whose job IS arbitrary host access.
    unconfined: bool = Field(
        default=False,
        description=(
            "No sandbox at all, for a server whose job IS arbitrary host access. Contradicts every "
            "other field here, so setting both is refused rather than half-applied."
        ),
    )

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
    over stdio. Its `command` (argv) is operator-controlled and never
    contains LLM output. The server runs OUTSIDE the agent6 jail, with the
    same curated environment a `[notify]` hook gets -- never the agent6
    process's full `os.environ`, which carries the provider API keys -- plus
    whatever `pass_env` names.

    The LLM sees each MCP-server tool as
    `mcp__<name>__<server-side-tool-name>` and can call it through
    the normal tool surface. The MCP server itself is responsible for
    validating the arguments the LLM passes; agent6 forwards them
    verbatim.

    A misbehaving server (crash, hang, malformed output) surfaces as
    a clean tool failure, not an agent crash.
    """

    model_config = MODEL_CONFIG

    # Exactly one of these. `command` spawns the server (agent6 owns its env,
    # lifetime and confinement); `url` connects to one the OPERATOR runs, in
    # whatever container or sandbox they chose -- which is how anyone actually
    # runs a server that wants a browser or a device.
    command: tuple[str, ...] = Field(
        default=(),
        description="argv for a stdio server agent6 spawns. Exactly one of this or `url`.",
    )
    url: str = Field(
        default="",
        description=(
            "An http(s) endpoint the OPERATOR runs; agent6 only connects, owning none of its "
            "environment or confinement."
        ),
    )
    # The env var holding the bearer token for `url`. Named, never inlined: a
    # secret in a config file is a secret in a backup.
    token_env: str = Field(
        default="",
        description=(
            "For a `url` server: env var holding the bearer. Named, never inlined; never logged."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Per-server toggle.",
    )
    # Environment variables this server needs, BY NAME (e.g. ["GITHUB_TOKEN"]).
    # Everything else comes from the curated base agent6 gives any child it
    # spawns outside the jail. Naming each one is the point: a provider key is
    # never among them, because nobody would write it down.
    pass_env: tuple[str, ...] = Field(
        default=(),
        description="Env vars the server needs, BY NAME. Everything else is the curated base.",
    )
    sandbox: MCPSandbox | None = Field(
        default=None,
        description=(
            "Extra grants for a SPAWNED server beyond the sandbox a jailed command gets; absent "
            "= exactly that sandbox. A `url` server is your own process: confine it where you "
            "start it."
        ),
    )
    # Ask before each of this server's tool calls ("ask"), or never ("yes").
    # A server's tools do arbitrary things agent6 cannot classify, so the
    # default is the same as a command's: ask. There is no "no" -- withholding
    # a server's tools is what `enabled = false` already says.
    approve: Literal["ask", "yes"] = Field(
        default="ask",
        description=(
            "Ask before each of this server's tool calls, showing the arguments the model "
            'chose; `yes` never asks. The session answers are per server: "allow all" covers '
            'THIS server for the run (not the command tools, not a sibling server), "deny all" '
            "withdraws its tools from the next turn. `--auto-approve` sets `yes` for the run. "
            "No `no`: withholding a server's tools is what `enabled = false` says."
        ),
    )
    # Time budget for the initialize + tools/list handshake. If the
    # server doesn't respond in this window the manager logs and skips it.
    startup_timeout_s: float = Field(
        gt=0.0,
        default=10.0,
        description="`initialize` + `tools/list` budget.",
    )
    # Per-call timeout for `tools/call` requests. Surfaces as a tool
    # failure (ToolError) if exceeded.
    call_timeout_s: float = Field(
        gt=0.0,
        default=60.0,
        description="Per `tools/call` timeout.",
    )
    httpx_trust_env: bool = Field(
        default=False,
        description=(
            "For a `url` server: forward httpx's `trust_env`, so the connection honors the "
            "ambient HTTP(S)_PROXY (and .netrc / SSL_CERT_FILE). Off by default so a local "
            "server's bearer token never routes to a proxy; set it for a server reachable only "
            "through the environment's proxy."
        ),
    )

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
        if self.httpx_trust_env and not self.url:
            raise ValueError(
                "httpx_trust_env is for `url` servers; a spawned one has no http client"
            )
        if self.token_env and self.url.startswith("http://") and not _is_loopback(self.url):
            raise ValueError(
                "a token over plain http would cross the network in cleartext;"
                " use https, or drop token_env for a loopback server"
            )
        return self

    @property
    def effective_network(self) -> Literal["auto", "none", "session", "host"]:
        """The network this server joins, resolving an absent `[sandbox]` table
        to the same `auto` default a present table's field carries. One value
        for the spawn policy, the degrade warning, and the refusal, so they
        cannot disagree on the table-less case."""
        return self.sandbox.network if self.sandbox else "auto"


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

    The LLM-visible tool name is `mcp__<name>__<tool>` and routing recovers
    the server by splitting on the FIRST `__` after the prefix, so the key
    must be identifier-shaped and `__`-free.

    Shared with `agent6 mcp connect`, which must refuse BEFORE it writes: the
    name becomes a TOML table header, and validating only at load meant a
    name carrying `]` and a newline could close the table and open one of its
    own choosing.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        # ASCII fullmatch: no Unicode look-alikes, no trailing newline.
        return f"[mcp.servers.<name>] keys must be [A-Za-z0-9_-]+: {name!r}"
    if "__" in name:
        return (
            f"[mcp] server name must not contain '__' (it separates server"
            f" from tool in mcp__<server>__<tool>): {name!r}"
        )
    return ""


class MCPConfig(BaseModel):
    """`[mcp]` section. Empty / absent / `enabled = false` means no
    MCP servers are spawned and the LLM sees zero `mcp__*` tools.

    `servers` is a name-keyed map (`[mcp.servers.<name>]`), like
    `[providers.<name>]`: duplicates are unrepresentable, a repo overlay can
    flip one server without restating the rest, and `config set` reaches the
    leaves."""

    model_config = MODEL_CONFIG

    enabled: bool = Field(
        default=False,
        description="Master switch; `false` = zero `mcp__*` tools.",
    )
    servers: dict[str, MCPServerEntry] = Field(
        default_factory=dict,
        description="MCP servers by name (`[mcp.servers.<name>]`).",
    )

    @field_validator("servers")
    @classmethod
    def _valid_server_names(cls, v: dict[str, MCPServerEntry]) -> dict[str, MCPServerEntry]:
        for name in v:
            refusal = mcp_server_name_refusal(name)
            if refusal:
                raise ValueError(refusal)
        return v
