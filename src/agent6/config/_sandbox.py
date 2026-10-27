# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[sandbox]` and `[mcp]` models.

They bound what a jailed child, and a spawned MCP server on top of one, may reach.
"""

from __future__ import annotations

import pathlib
import re
from typing import Literal
from urllib import parse

import pydantic

from agent6 import paths
from agent6.config import _base, _surfaces


class SandboxConfig(pydantic.BaseModel):
    """The `[sandbox]` table."""

    model_config = _base.MODEL_CONFIG

    # `none` is operator-only and LLM-unreachable, so writing it is the consent; the resolution
    # lives in detect.resolve_isolation.
    isolation: Literal["auto", "full", "partial", "none"] = pydantic.Field(
        default="auto",
        description=(
            'jailed-command confinement; `sandbox.isolation = "full"`: user and mount '
            "namespaces, seccomp, Landlock where supported by the kernel; `sandbox.isolation = "
            '"partial"`: Landlock and seccomp, no namespaces; `sandbox.isolation = "none"`: '
            "unconfined, also via `--disable-sandbox` or `AGENT6_DISABLE_SANDBOX=1`; "
            '`sandbox.isolation = "auto"`: strongest host-supported level, warning below full '
            "isolation; explicit full or partial isolation refused on an unsupported host"
        ),
    )
    # A jailed child never out-reaches the process that launches it.
    # No per-command `none`: isolating commands from each other costs the dev server for no
    # security, and the model can chain them into one script anyway.
    network: Literal["auto", "session", "host"] = pydantic.Field(
        default="auto",
        description=(
            'jailed-command network; `sandbox.network = "session"`: private to the run, '
            "commands mutually reachable, no off-box access or outside ingress, refused if "
            'unenforceable; `sandbox.network = "host"`: machine\'s network; `sandbox.network = '
            '"auto"`: session network with `sandbox.isolation = "full"`, host network with a '
            "warning under partial or no isolation; one launcher shared by the run's commands, "
            "no per-command network isolation; machine-state override in its `config` table, "
            "never wider than this setting"
        ),
    )
    run_commands: Literal["allow", "deny", "ask"] = pydantic.Field(
        default="ask",
        description=(
            "one permission for `run_command`, `run_verify_command`, `run_metric_command` and "
            '`stop_background`; `sandbox.run_commands = "allow"`: commands allowed; '
            '`sandbox.run_commands = "deny"`: tools and verify gate withheld; '
            '`sandbox.run_commands = "ask"`: per-call prompt with allow-for-this-session '
            "answers, startup refused without someone to answer; allow clamped to ask in `ask` "
            "and `plan` modes; per-invocation flags: `--auto-approve` (never over configured "
            "deny), `--no-commands`"
        ),
    )
    # Hosts, never URL prefixes: a prefix invites `evil.com/docs.python.org`.
    # A GET can encode data in its path, so an unlisted host is asked about.
    fetch_approved_hosts: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "hosts readable by `fetch` without approval; other hosts: prompt, refusal without "
            'an operator; empty: every fetch prompts; `sandbox.fetch_approved_hosts = ["*"]`: '
            "any host; leading dot: subdomains allowed (`.readthedocs.io`); host entries only, "
            "never URL prefixes; fixed fetch limits: https only, 1 MiB, redirects returned "
            "without following; tool hidden when jailed commands have host networking "
            '(`sandbox.network = "host"` or `sandbox.isolation` other than `"full"`); tool '
            "withheld from machine and agent states"
        ),
    )
    # A read-only bind-remount; the harness's own commits run outside the jail and are unaffected.
    protect_git: Literal["auto", "on", "off"] = pydantic.Field(
        default="auto",
        description=(
            "`.git/` write protection against jailed commands planting filters for agent6's "
            'host-side commits to execute; `sandbox.protect_git = "on"`: read-only '
            'bind-remount, requires `sandbox.isolation = "full"`, startup refused elsewhere or '
            'with `git.control = "model"`; `sandbox.protect_git = "auto"`: bind under full '
            "isolation, warning below it, off under model git control; `sandbox.protect_git = "
            '"off"`: writable; in-process `.git` writes refused at every isolation level with '
            '`sandbox.protect_git = "on"` or `"auto"`'
        ),
    )
    # The cache dir is paths.jail_cache_home, keyed on the repository.
    home: Literal["auto", "tmp", "cache"] = pydantic.Field(
        default="auto",
        description=(
            'home directory (HOME) for jailed commands; `sandbox.home = "tmp"`: '
            "`/tmp/agent6-home` in the run's private tmpfs, removed with the run, requires "
            '`sandbox.isolation = "full"`, startup refused elsewhere; `sandbox.home = "cache"`: '
            "persistent `$XDG_CACHE_HOME/agent6/homes/<repo-id>`, one directory per repo, "
            "created `0700`, refused if loosened, read-write bind at its real path; "
            '`sandbox.home = "auto"`: tmp under full isolation, cache under partial or no '
            "isolation (no private tmpfs); persistent HOME: cross-run channel inside the jail, "
            "poisoned caches or `~/.gitconfig` aliases reaching later jailed runs in the same "
            "repo, never other repos or the operator's tools; cached HOME for Go builds: "
            "compiled-package cache retained across runs without a host-cache grant"
        ),
    )
    # RLIMIT_DATA rather than RLIMIT_AS so V8, the JVM and ASAN, which reserve address space
    # without committing it, keep working. A guardrail, never a security control; set before
    # exec at every isolation level.
    memory_limit_mib: int = pydantic.Field(
        default=0,
        ge=0,
        description=(
            "per-process `RLIMIT_DATA` data-segment cap in MiB (1024 * 1024 bytes) for jailed "
            "processes and their children; `sandbox.memory_limit_mib = 0`: no cap; explicit cap "
            "to bound a task; cap exceeded: ordinary command error"
        ),
    )
    # On top of /usr /bin /lib /lib64 /etc /dev and the workspace; no effect under `none`.
    extra_read_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "absolute paths outside the repo readable and executable at their real locations "
            "(toolchains, conda/Go/Rust/Node interpreters, shared data dirs); mounted for "
            "jailed commands, readable by in-process tools; wider sandbox, only paths needed by "
            "the build"
        ),
    )

    @pydantic.field_validator("extra_read_paths")
    @classmethod
    def _check_extra_read_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a relative path or one with a `..` segment.

        Args:
            v: The configured paths.

        Returns:
            The paths unchanged.

        Raises:
            ValueError: A path is relative or traverses with `..`.
        """
        for p in v:
            if not p.startswith("/"):
                raise ValueError(f"sandbox.extra_read_paths must be absolute: {p!r}")
            # A `..` segment would let a bind mount traverse outside its apparent target.
            if ".." in pathlib.Path(p).parts:
                raise ValueError(f"sandbox.extra_read_paths must not contain '..': {p!r}")
        return v

    # No effect under `none`.
    extra_write_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "absolute paths outside the repo readable and writable at their real locations "
            "(build caches, output dirs, sibling checkouts edited by the task); read access "
            "included with write access; wider sandbox, only paths written by the task"
        ),
    )

    @pydantic.field_validator("extra_write_paths")
    @classmethod
    def _check_extra_write_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a relative path or one with a `..` segment.

        Args:
            v: The configured paths.

        Returns:
            The paths unchanged.

        Raises:
            ValueError: A path is relative or traverses with `..`.
        """
        for p in v:
            if not p.startswith("/"):
                raise ValueError(f"sandbox.extra_write_paths must be absolute: {p!r}")
            if ".." in pathlib.Path(p).parts:
                raise ValueError(f"sandbox.extra_write_paths must not contain '..': {p!r}")
        return v

    extra_device_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "device nodes under `/dev` exposed read-write (GPU compute: `/dev/nvidiactl`, "
            "`/dev/nvidia0`, `/dev/nvidia-uvm`); default device access only to `/dev/null`, "
            "`/dev/zero`, `/dev/urandom`, `/dev/random` and `/dev/full`; existing character or "
            "block devices required at run start, otherwise startup refused; wider sandbox, "
            "direct hardware access"
        ),
    )

    @pydantic.field_validator("extra_device_paths")
    @classmethod
    def _check_extra_device_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a path outside /dev or one with a `..` segment.

        Args:
            v: The configured paths.

        Returns:
            The paths unchanged.

        Raises:
            ValueError: A path is not under /dev or traverses with `..`.
        """
        for p in v:
            if not p.startswith("/dev/") or ".." in pathlib.Path(p).parts:
                raise ValueError(f"sandbox.extra_device_paths must live under /dev: {p!r}")
        return v

    # Adds to the always-hidden private dirs; the masking rules are in docs/security.md.
    hide_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "absolute paths never readable or writable by the run, even under a broader grant; "
            "an entry inside the repository stays out of every tree agent6 commits or compares; "
            "agent6 config dir and state base always hidden, `secrets.toml` and run history "
            "excluded even from `sandbox.extra_read_paths` granting `$HOME`; data dir and cache "
            "readable for installed skills; in-process refusal at every isolation level; "
            "jailed-command masking: empty dirs and files, mount namespace required; under "
            '`sandbox.isolation = "partial"`: run refused for an unmaskable entry, loud warning '
            "instead for a grant exposing always-hidden dirs"
        ),
    )

    @pydantic.field_validator("hide_paths")
    @classmethod
    def _check_hide_paths(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a relative path or one with a `..` segment.

        Args:
            v: The configured paths.

        Returns:
            The paths unchanged.

        Raises:
            ValueError: A path is relative or traverses with `..`.
        """
        for p in v:
            if not p.startswith("/"):
                raise ValueError(f"sandbox.hide_paths must be absolute: {p!r}")
            if ".." in pathlib.Path(p).parts:
                raise ValueError(f"sandbox.hide_paths must not contain '..': {p!r}")
        return v

    @pydantic.model_validator(mode="after")
    def _extra_paths_never_target_private_dirs(self) -> SandboxConfig:
        """Refuse an extra grant at or inside an agent6-private dir.

        A grant containing one (`$HOME`) is allowed, since `full` `sandbox.isolation` masks
        the private dirs out of it and a `partial` run warns.

        Returns:
            The model unchanged.

        Raises:
            ValueError: An extra read or write path resolves inside a private dir.
        """
        for p in (*self.extra_read_paths, *self.extra_write_paths):
            # Resolved on both sides: the launcher binds a symlink's target, as jail_home_refusal.
            resolved = pathlib.Path(p).resolve()
            for d in paths.private_dirs():
                if resolved.is_relative_to(d.resolve()):
                    raise ValueError(
                        f"sandbox extra path {p!r} is inside the agent6-private dir"
                        f" {str(d)!r} (secrets/state); it never enters the jail."
                        " Grant a different directory."
                    )
        return self


class MCPSandbox(pydantic.BaseModel):
    """The extra grants one spawned MCP server gets beyond a jailed command's sandbox.

    A server is fed model input, so the same launcher confines it the same way; an absent
    block means exactly that sandbox.
    """

    model_config = _base.MODEL_CONFIG

    extra_read_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "read and execute paths for this server beyond the run's sandbox (absolute or `~`); "
            "workspace, system dirs, tool dirs and writable `HOME` included without extra "
            "grants; only the server's own data needed here"
        ),
    )
    extra_write_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description="Paths this server may write beyond the run's sandbox (absolute or `~`).",
    )
    # Per server because servers differ: a browser server exists to reach something.
    network: Literal["auto", "none", "session", "host"] = pydantic.Field(
        default="auto",
        description=(
            'server network; `network = "auto"`: private namespace where supported, otherwise '
            'host network with a warning; `network = "none"`: private network or refusal, never '
            'a connected fallback; `network = "session"`: run\'s network, background dev servers '
            "reachable by this server (such as a browser testing the app), no off-box access; "
            '`network = "host"`: machine\'s network'
        ),
    )
    unconfined: bool = pydantic.Field(
        default=False,
        description=(
            "no sandbox for a server needing arbitrary host access; refused alongside any other "
            "field in this block, no partial application"
        ),
    )

    @pydantic.model_validator(mode="after")
    def _escape_hatch_is_exclusive(self) -> MCPSandbox:
        """Refuse a relative or private-dir path, and any grant beside `unconfined`.

        Returns:
            The model unchanged.

        Raises:
            ValueError: A path is relative or resolves inside an agent6-private dir, or
                `unconfined` is set with paths or a network.
        """
        if not self.unconfined:
            for group in (self.extra_read_paths, self.extra_write_paths):
                for raw in group:
                    if not pathlib.Path(raw).expanduser().is_absolute():
                        raise ValueError(
                            f"sandbox paths must be absolute (or start with ~): {raw!r}."
                            " A relative one would be resolved against whatever"
                            " directory agent6 happened to start in."
                        )
            for raw in (*self.extra_read_paths, *self.extra_write_paths):
                # Resolved on both sides: a symlink named here still mounts its target.
                resolved = pathlib.Path(raw).expanduser().resolve()
                for private in paths.private_dirs():
                    if resolved.is_relative_to(private.resolve()):
                        raise ValueError(
                            f"sandbox path {raw!r} is inside the agent6-private dir"
                            f" {str(private)!r} (secrets/state); it never enters a"
                            " jail. Grant a different directory."
                        )
            return self
        stated = [
            name
            for name, value in (
                ("extra_read_paths", self.extra_read_paths),
                ("extra_write_paths", self.extra_write_paths),
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


class MCPServerEntry(pydantic.BaseModel):
    """One MCP (Model Context Protocol) server, spawned at run start or connected to.

    A spawned server speaks JSON-RPC 2.0 over stdio as a jailed child with the curated
    environment a `[notify]` hook gets, never the agent6 process's own environ; its argv is
    operator-controlled and never carries LLM output. The model sees each of its tools as
    `mcp__<name>__<tool>` and the server validates the arguments itself: agent6 forwards
    them verbatim. A crash, hang or malformed reply is a tool failure, not an agent crash.
    """

    model_config = _base.MODEL_CONFIG

    command: _base.Argv = pydantic.Field(
        default=(),
        description=(
            "argv of a spawned stdio MCP server; jailed like a command, plus `sandbox` grants; "
            "exactly one of `command` or `url` required"
        ),
    )
    url: str = pydantic.Field(
        default="",
        description=(
            "operator-run http(s) MCP endpoint; connection only, environment and confinement "
            "outside agent6's control; exactly one of `command` or `url` required"
        ),
    )
    # Named, never inlined: a secret in a config file is a secret in a backup.
    token_env: str = pydantic.Field(
        default="",
        description=(
            "bearer-token environment variable for a `url` server, named rather than inlined, "
            "never logged; token readable on the wire over plaintext `http://` to a "
            "non-loopback host; approval in `mcp add`, warning on every run starting it"
        ),
    )
    enabled: bool = pydantic.Field(
        default=True,
        description=(
            "`false` withholds this server's tools from the model without deleting the entry."
        ),
    )
    # A provider key reaches a server only when named here.
    pass_env: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "Environment variables a spawned server needs, by name; everything else is agent6's "
            "curated base environment."
        ),
    )
    sandbox: MCPSandbox | None = pydantic.Field(
        default=None,
        description=(
            "extra grants for a spawned server beyond a jailed command's sandbox; unset: "
            "exactly that sandbox; `url` server: operator's process, confinement at its launch "
            "site"
        ),
    )
    # A server's tools do things agent6 cannot classify, so the default is a command's: ask.
    approve: Literal["ask", "allow"] = pydantic.Field(
        default="ask",
        description=(
            'approval policy for this server\'s tools; `approve = "ask"`: prompt before each '
            'call with the model\'s arguments; `approve = "allow"`: no prompt; session answer '
            '"allow all": this server only for the run, not command tools or sibling servers; '
            '"deny all": tools withdrawn from the next turn; `--auto-approve`: `approve = '
            '"allow"` for the run; no deny mode, tools withheld with `enabled = false`'
        ),
    )
    startup_timeout_s: float = pydantic.Field(
        gt=0.0,
        default=10.0,
        description=(
            "Seconds the server gets to answer `initialize` and `tools/list` before it is given up "
            "on."
        ),
    )
    call_timeout_s: float = pydantic.Field(
        gt=0.0,
        default=60.0,
        description=(
            "Seconds one `tools/call` may take before it fails; a spawned server is restarted "
            "after a timeout."
        ),
    )
    http_trust_env: bool = pydantic.Field(
        default=False,
        description=(
            "ambient `HTTP(S)_PROXY`, `.netrc` and `SSL_CERT_FILE` for a `url` server; "
            "`http_trust_env = false`: no proxy routing of a local server's bearer token; "
            "opt-in for a server reachable only through the environment's proxy"
        ),
    )

    @pydantic.model_validator(mode="after")
    def _one_transport(self) -> MCPServerEntry:
        """Refuse an entry with both or neither transport, or a field of the other transport.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `command` and `url` both set or both empty, a non-http(s) `url`, or a
                `url`-only field on a spawned server and the reverse.
        """
        if bool(self.command) == bool(self.url):
            raise ValueError("set exactly one of `command` (spawn) or `url` (connect)")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError(f"url must be http(s), got {self.url!r}")
        if self.token_env and not self.url:
            raise ValueError("token_env is for `url` servers; a spawned one uses pass_env")
        if self.sandbox is not None and self.url:
            raise ValueError(
                "a [sandbox] block confines a server agent6 spawns; a `url` one"
                " is your own process, so confine it where you start it"
            )
        if self.pass_env and self.url:
            # Nothing is spawned, so there is no environment to pass.
            raise ValueError("pass_env is for spawned servers; a `url` one uses token_env")
        if self.http_trust_env and not self.url:
            raise ValueError(
                "http_trust_env is for `url` servers; a spawned one has no http client"
            )
        return self

    @property
    def effective_network(self) -> Literal["auto", "none", "session", "host"]:
        """The network this server joins; an absent `[sandbox]` table reads as `auto`."""
        return self.sandbox.network if self.sandbox else "auto"

    @property
    def jailed(self) -> bool:
        """Whether agent6 spawns this server in its jail: never a `url` or `unconfined` one."""
        return not self.url and not (self.sandbox is not None and self.sandbox.unconfined)


def is_cleartext_url(url: str) -> bool:
    """Return whether the URL dials plain http.

    The parsed scheme is compared, never a prefix: `HTTP://` would evade a prefix match while
    the client still dialled cleartext.

    Args:
        url: The URL.

    Returns:
        True when the scheme is http.
    """
    return parse.urlsplit(url).scheme == "http"


def is_loopback_url(url: str) -> bool:
    """Return whether the URL's host is this machine.

    The parsed hostname is checked, never a prefix: `127.evil.com` resolves wherever its owner
    points it. Loopback is the one case where plain http with a credential is not readable on
    the wire.

    Args:
        url: The URL.

    Returns:
        True when the host is loopback.
    """
    return _surfaces.is_loopback_host(parse.urlsplit(url).hostname or "")


def mcp_server_name_refusal(name: str) -> str:
    """Return why the name cannot be an MCP server key, or "".

    Routing splits `mcp__<name>__<tool>` on the first `__` after the prefix, so the key is
    identifier-shaped and `__`-free. `agent6 mcp add` refuses on this before it writes:
    the name becomes a TOML table header, and one carrying `]` and a newline could open a
    table of its own choosing.

    Args:
        name: The proposed key.

    Returns:
        The refusal, or "" when the name is valid.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        # An ASCII fullmatch: no Unicode look-alikes, no trailing newline.
        return f"[mcp.servers.<name>] keys must be [A-Za-z0-9_-]+: {name!r}"
    if "__" in name:
        return (
            f"[mcp] server name must not contain '__' (it separates server"
            f" from tool in mcp__<server>__<tool>): {name!r}"
        )
    return ""


class MCPConfig(pydantic.BaseModel):
    """The `[mcp]` table.

    `servers` is keyed by name like `[providers.<name>]`: duplicates are unrepresentable, a
    repo overlay can flip one server without restating the rest, and `config set` reaches
    the leaves.
    """

    model_config = _base.MODEL_CONFIG

    enabled: bool = pydantic.Field(
        default=False,
        description=(
            "Master switch for MCP servers: `false` means no `mcp__*` tools reach the model, "
            "whatever `[mcp.servers]` lists."
        ),
    )
    servers: dict[str, MCPServerEntry] = pydantic.Field(
        default_factory=dict,
        description=(
            "MCP servers by name (`[mcp.servers.<name>]`); their tools reach the model as "
            "`mcp__<name>__<tool>` in run mode when `enabled` is on."
        ),
    )

    @pydantic.field_validator("servers")
    @classmethod
    def _valid_server_names(cls, v: dict[str, MCPServerEntry]) -> dict[str, MCPServerEntry]:
        """Refuse a server key `mcp_server_name_refusal` rejects.

        Args:
            v: The servers by name.

        Returns:
            The map unchanged.

        Raises:
            ValueError: A key is not identifier-shaped or contains `__`.
        """
        for name in v:
            refusal = mcp_server_name_refusal(name)
            if refusal:
                raise ValueError(refusal)
        return v

    @property
    def enabled_servers(self) -> dict[str, MCPServerEntry]:
        """The servers a run starts: every `enabled` entry while the table's `enabled` is on."""
        if not self.enabled:
            return {}
        return {name: srv for name, srv in self.servers.items() if srv.enabled}
