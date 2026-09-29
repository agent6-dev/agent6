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
    isolation: Literal["auto", "strict", "hardened", "none"] = pydantic.Field(
        default="auto",
        description=(
            "How jailed commands are confined: `strict` (user + mount namespaces, Landlock, "
            "seccomp), `hardened` (Landlock + seccomp, no namespaces), or `none` (unconfined). "
            "`auto` picks the strongest the host supports and says so when that is `none`. An "
            "explicit `strict` or `hardened` refuses to start where the host cannot honor it. "
            "`none` also via `--dangerously-disable-sandbox` or "
            "`AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`."
        ),
    )
    # A jailed child never out-reaches the process that launches it.
    # No per-command `none`: isolating commands from each other costs the dev server for no
    # security, and the model can chain them into one script anyway.
    network: Literal["auto", "session", "only_explicit_states", "host"] = pydantic.Field(
        default="auto",
        description=(
            "Which network jailed commands join. `session`: the run's private network (commands "
            "reach each other, nothing off the box, nothing outside reaches in), refused where it "
            "cannot be enforced. `host`: the machine's network. `only_explicit_states`: strict "
            "only, machine `tool` states opt in. `auto`: `session` under `strict`, degraded to the "
            "host's network with a warning under `hardened` or `none`. A run's commands share one "
            "launcher, so there is no per-command `none`."
        ),
    )
    run_commands: Literal["yes", "no", "ask"] = pydantic.Field(
        default="ask",
        description=(
            "Whether the model may run commands (`run_command`, `run_verify_command`, "
            "`run_metric_command`, `stop_background`, one decision for all four): `yes` runs "
            "them, `no` withholds the "
            "tools (and the verify gate with them), `ask` prompts per call with "
            "allow-for-this-session answers. `ask` and `plan` clamp `yes` to `ask`. Per "
            "invocation: `--auto-approve` (never over a configured `no`), `--no-commands`. A run "
            "set to `ask` with nobody to answer refuses to start."
        ),
    )
    # Hosts, never URL prefixes: a prefix invites `evil.com/docs.python.org`.
    # A GET can encode data in its path, so an unlisted host is asked about.
    fetch_hosts: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "Hosts the `fetch` tool reads without asking; any other host prompts, and an absent "
            'operator is a no. Empty: every fetch prompts. `["*"]`: any host. A leading dot allows '
            "subdomains (`.readthedocs.io`). Each entry is a host, never a URL prefix; the rest of "
            "fetch is fixed (https only, 1 MiB cap, redirects returned, not followed). Hidden when "
            'a jailed command already has the host network (`network = "host"`, or any isolation '
            "but `strict`); withheld from machine and agent states."
        ),
    )
    # A read-only bind-remount; the harness's own commits run outside the jail and are unaffected.
    # Under hardened the cwd is blanket read-write: carving .git out would deny new top-level names.
    protect_git: bool = pydantic.Field(
        default=True,
        description=(
            "Keep `.git/` unwritable by jailed commands, so a command cannot plant a git filter "
            "that agent6's host-side commits would execute. Needs a mount namespace: `strict` "
            "only. Under `hardened` the default `true` degrades with a warning; an explicit `true` "
            "refuses to start. The in-process edit tools refuse `.git` writes at every level "
            "regardless."
        ),
    )
    # The cache dir is paths.jail_cache_home.
    home: Literal["tmp", "cache"] = pydantic.Field(
        default="tmp",
        description=(
            "The HOME jailed commands get under `strict`: `tmp` is `/tmp/agent6-home` inside the "
            "run's private tmpfs, gone with the run; `cache` is the persistent "
            "`$XDG_CACHE_HOME/agent6/home` (created `0700`, refused once loosened), bind-mounted "
            "read-write at its real "
            "path. `hardened` and `none` have no private tmpfs and always use the cache dir; an "
            "explicit `tmp` refuses to start there. Persistence is a cross-run channel inside the "
            "jail's world: a poisoned cache or a `~/.gitconfig` alias written by one run reaches "
            "the next jailed run, never your own tools."
        ),
    )
    # RLIMIT_DATA rather than RLIMIT_AS so V8, the JVM and ASAN, which reserve address space
    # without committing it, keep working. A guardrail, never a security control; set before
    # exec at every isolation level.
    memory_limit_mb: int = pydantic.Field(
        default=0,
        ge=0,
        description=(
            "`RLIMIT_DATA` cap in MiB on each jailed process (inherited by its children). `0`: no "
            "cap. Set one to bound a specific task; a process over it fails as an ordinary command "
            "error."
        ),
    )
    # On top of /usr /bin /lib /lib64 /etc /dev and the workspace; no effect under `none`.
    extra_read_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "Absolute paths outside the repo the run may read and execute, at their real "
            "locations: a toolchain, an interpreter (conda, Go, Rust, Node), a shared data dir. "
            "Mounted for jailed commands and readable by the in-process tools. Widens the sandbox; "
            "list only what the build needs."
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
            "Absolute paths outside the repo the run may read and write, at their real locations: "
            "a build cache, an output dir, a sibling checkout the task edits. Write implies read. "
            "Widens the sandbox; list only what the task writes."
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
            "Device nodes under /dev the jail exposes read-write (GPU compute: /dev/nvidiactl, "
            "/dev/nvidia0, /dev/nvidia-uvm). Empty (the default) keeps the device wall: strict's "
            "/dev holds only null/zero/urandom/random/full. Each path must be an existing "
            "character or block device at run start, or the run refuses. Widens the sandbox: a "
            "device node is direct hardware access."
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
            "Absolute paths the run may never read or write, even under a broader grant. agent6's "
            "config dir and state base are always hidden, so an `extra_read_paths` grant of "
            "`$HOME` never exposes `secrets.toml` or run history (the data dir and cache stay "
            "readable: installed skills work). Enforced twice: the in-process tools refuse them at "
            "every isolation level, and jailed commands see them masked (a dir reads empty, a file "
            "reads empty). Masking needs the mount namespace: under `hardened` an entry it cannot "
            "mask refuses the run, and a grant exposing the always-hidden dirs warns loudly "
            "instead."
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

        A grant containing one (`$HOME`) is allowed on strict, where the private dirs are
        masked out of it.

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

    read_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            "Read+execute paths for this server beyond the sandbox a jailed command gets "
            "(absolute or `~`). The workspace, system dirs, tool dirs and a writable `HOME` are "
            "already there, so a block names only the server's own data."
        ),
    )
    write_paths: _base.StrTuple = pydantic.Field(
        default=(),
        description="Paths it may write, likewise additive.",
    )
    # Per server because servers differ: a browser server exists to reach something.
    network: Literal["auto", "none", "session", "host"] = pydantic.Field(
        default="auto",
        description=(
            "Which network this server joins. `auto`: one of its own where the host can give a "
            "namespace, degrading to the host's with a warning. `none`: the same, refusing "
            "rather than running connected. `session`: the run's network, so a dev server a "
            "background command started answers this server too (a browser server driving the "
            "app under test), and still nothing off the box. `host`: the machine's network."
        ),
    )
    unconfined: bool = pydantic.Field(
        default=False,
        description=(
            "No sandbox at all, for a server whose job is arbitrary host access. Contradicts every "
            "other field here, so setting both is refused rather than half-applied."
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
            for group in (self.read_paths, self.write_paths):
                for raw in group:
                    if not pathlib.Path(raw).expanduser().is_absolute():
                        raise ValueError(
                            f"sandbox paths must be absolute (or start with ~): {raw!r}."
                            " A relative one would be resolved against whatever"
                            " directory agent6 happened to start in."
                        )
            for raw in (*self.read_paths, *self.write_paths):
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
            "argv of a stdio MCP server agent6 spawns (jailed like a command, plus `sandbox`). "
            "Exactly one of `command` or `url`."
        ),
    )
    url: str = pydantic.Field(
        default="",
        description=(
            "An http(s) MCP endpoint you run yourself; agent6 only connects, owning none of its "
            "environment or confinement. Exactly one of `command` or `url`."
        ),
    )
    # Named, never inlined: a secret in a config file is a secret in a backup.
    token_env: str = pydantic.Field(
        default="",
        description=(
            "For a `url` server: the environment variable holding its bearer token, named here and "
            "never inlined or logged. Over plaintext `http://` to a non-loopback host the token is "
            "readable on the wire: `mcp connect` asks first, and every run warns."
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
            "Extra grants for a spawned server beyond the sandbox a jailed command gets; unset "
            "means exactly that sandbox. A `url` server is your own process: confine it where you "
            "start it."
        ),
    )
    # A server's tools do things agent6 cannot classify, so the default is a command's: ask.
    approve: Literal["ask", "yes"] = pydantic.Field(
        default="ask",
        description=(
            "`ask` prompts before each of this server's tool calls, showing the arguments the "
            'model chose; `yes` never asks. The session answers are per server: "allow all" covers '
            'this server for the run (not the command tools, not a sibling server), "deny all" '
            "withdraws its tools from the next turn. `--auto-approve` sets `yes` for the run. "
            "There is no `no`: `enabled = false` is how a server's tools are withheld."
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
    httpx_trust_env: bool = pydantic.Field(
        default=False,
        description=(
            "For a `url` server: honor the ambient `HTTP(S)_PROXY`, `.netrc`, and `SSL_CERT_FILE` "
            "(httpx's `trust_env`). `false` so a local server's bearer token never routes to a "
            "proxy; set it for a server reachable only through the environment's proxy."
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
        if self.httpx_trust_env and not self.url:
            raise ValueError(
                "httpx_trust_env is for `url` servers; a spawned one has no http client"
            )
        return self

    @property
    def effective_network(self) -> Literal["auto", "none", "session", "host"]:
        """The network this server joins; an absent `[sandbox]` table reads as `auto`."""
        return self.sandbox.network if self.sandbox else "auto"


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
    identifier-shaped and `__`-free. `agent6 mcp connect` refuses on this before it writes:
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
