# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 mcp connect`: add an MCP server after proving it works.

The order is the point: handshake, list the tools, show them, and only then write
config. A server named in config that does not answer is a run that starts, logs
"failed to start", and quietly has fewer tools than the operator thinks.

Nothing the server returns is ever executed. Its tool names and descriptions are printed
as text and stored nowhere.
"""

from __future__ import annotations

import pathlib
import shlex
import shutil
import sys

import pydantic

from agent6 import kinds, paths
from agent6.app import _setup
from agent6.config import (
    Config,
    MCPServerEntry,
    io,
    is_cleartext_url,
    is_loopback_url,
    mcp_server_name_refusal,
    write,
)
from agent6.config import layer as config_layer
from agent6.sandbox import detect, jail
from agent6.tools import mcp_client
from agent6.ui.cli import _common

# Long enough for a cold `npx` to fetch and boot a server; the per-run default stays 10s.
_CONNECT_TIMEOUT_S = 60.0


def _probe(spec: mcp_client.MCPServerSpec) -> tuple[tuple[mcp_client.MCPToolDescriptor, ...], str]:
    """Start the server as a run does, take its tool list, stop it.

    Returns:
        The tools and the failure text; one of them is empty.
    """
    manager = mcp_client.MCPManager.start([spec])
    try:
        return manager.descriptors(), manager.failures[0].error if manager.failures else ""
    finally:
        manager.close()


def _refuse_bad_flags(
    *, name: str, command: list[str], url: str, token_env: str, pass_env: list[str], cfg: Config
) -> str:
    """Return why this invocation cannot be acted on, or "".

    Each transport owns one env flag, so the wrong pairing is named rather than silently
    ignored.

    Args:
        name: The server name.
        command: The stdio command.
        url: The HTTP URL.
        token_env: The `--token-env` flag.
        pass_env: The `--pass-env` names.
        cfg: The effective config.
    """
    if bool(command) == bool(url):
        return (
            "give exactly one of a command to spawn or --url to connect to.\n"
            "  spawn:   agent6 mcp connect files -- npx -y"
            " @modelcontextprotocol/server-filesystem .\n"
            "  connect: agent6 mcp connect browser --url http://127.0.0.1:8931/mcp"
        )
    if token_env and not url:
        return "--token-env is for --url servers; a spawned one uses --pass-env"
    if pass_env and url:
        return "--pass-env is for spawned servers; a --url one uses --token-env"
    name_refusal = mcp_server_name_refusal(name)
    if name_refusal:
        # Before the write: the name becomes a TOML table header.
        return name_refusal
    keys = {str(getattr(e, "api_key_env", "")) for e in cfg.providers.values()} - {""}
    leaked = sorted(keys.intersection(pass_env))
    if leaked:
        # A provider key stays out of every child agent6 spawns; `--pass-env` would hand one over.
        return (
            f"{', '.join(leaked)} holds a provider API key; agent6 does not pass one"
            " to an MCP server.\n"
            "  If the server needs its own credential, give it a different variable."
        )
    return ""


def _cleartext_token_go_ahead(url: str, token_env: str) -> bool:
    """Return whether to proceed past the plaintext non-loopback confirmation.

    Explicit but discouraged, so the cost is named and never refused (an internal-network or
    VPN endpoint is a real case): interactive asks, default no; headless warns and proceeds.

    Args:
        url: The server URL.
        token_env: The variable holding the token.
    """
    if not (token_env and is_cleartext_url(url) and not is_loopback_url(url)):
        return True
    _common.warn(
        f"{url} is plaintext http to a non-loopback host: the token"
        f" from ${token_env} will be readable on the network path."
    )
    if not sys.stdin.isatty():
        return True
    return input("Connect anyway? [y/N]: ").strip().lower() in ("y", "yes")


def _describe(spec: mcp_client.MCPServerSpec) -> str:
    """Return the server's transport and target as one phrase."""
    if spec.http is not None:
        return f"connecting to {spec.http.url}"
    return f"spawning {shlex.join(spec.command)}"


def _report_no_answer(
    name: str, command: list[str], isolation: kinds.IsolationLevel, failure: str
) -> None:
    """Print the failure and the one hint that applies.

    A binary missing on the host needs no sandbox grant; one present here but not in the
    jail does.

    Args:
        name: The server name.
        command: The stdio command.
        isolation: The isolation the probe ran under.
        failure: The probe's failure text.
    """
    _common.error(f"{name} did not answer: {failure}")
    if command and shutil.which(command[0]) is None:
        head = command[0]
        what = (
            "not executable"
            if pathlib.Path(head).exists()
            else "no such file"
            if "/" in head
            else "not on PATH"
        )
        print(f"       {head}: {what} on this host.", file=sys.stderr)
    elif command and isolation != "none":
        print(
            f"       (probed under the run's {isolation} sandbox: a server outside the"
            f" workspace needs [mcp.servers.{name}.sandbox] read_paths, or"
            " unconfined = true)",
            file=sys.stderr,
        )
    print("       nothing was written to config.", file=sys.stderr)


def cmd_mcp_connect(
    name: str,
    *,
    command: list[str],
    url: str,
    token_env: str,
    pass_env: list[str],
    to_repo: bool,
    config_path: pathlib.Path | None = None,
) -> int:
    """Prove the server answers, then write it into config.

    Args:
        name: The server name, a TOML table header.
        command: The stdio command, or empty for HTTP.
        url: The HTTP URL, or "" for stdio.
        token_env: The variable holding the bearer token.
        pass_env: The variables the server receives by name.
        to_repo: Write to the repo config instead of the global one.
        config_path: The `--config` file, if any.

    Returns:
        The exit code; 2 on a refusal or a server that gave no proof.
    """
    effective = config_layer.load_effective(pathlib.Path.cwd(), config_path)
    cfg = effective.config
    refusal = _refuse_bad_flags(
        name=name, command=command, url=url, token_env=token_env, pass_env=pass_env, cfg=cfg
    )
    if refusal:
        _common.error(f"{refusal}")
        return 2
    if not _cleartext_token_go_ahead(url, token_env):
        print("nothing was written to config.", file=sys.stderr)
        return 1

    try:
        entry = MCPServerEntry.model_validate(
            {
                "command": command,
                "url": url,
                "token_env": token_env,
                "pass_env": pass_env,
                "startup_timeout_s": _CONNECT_TIMEOUT_S,
            }
        )
    except pydantic.ValidationError as exc:
        # The entry's own rules (the URL shape above all) reach the operator as one line each.
        detail = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'entry'}: {issue['msg']}"
            for issue in exc.errors()
        )
        _common.error(f"{name}: {detail}")
        return 2
    env = _setup.detect_env()
    isolation = detect.resolve_isolation(cfg.sandbox.isolation, env)
    if command and isolation == "none":
        # No jail, no read-only workspace to probe under: the entry is written unproved, said so.
        _common.warn(
            f"{name} not probed: no jail ({_setup.no_jail_cause(cfg, env)}); a run starts "
            "it unconfined."
        )
    else:
        rc = _prove(cfg, name, entry, isolation)
        if rc is not None:
            return rc

    # Values, not TOML text: a pre-quoted argv would validate as a tuple of characters.
    fields: dict[str, io.ConfigLeafValue] = {"enabled": True}
    if command:
        fields["command"] = command
    else:
        fields["url"] = url
    if token_env:
        fields["token_env"] = token_env
    if pass_env:
        fields["pass_env"] = pass_env
    written = write.set_config_leaves(
        pathlib.Path.cwd(), f"mcp.servers.{name}", fields, to_repo=to_repo
    )
    if written is not None:
        _common.error(f"{written}")
        return 2
    print(f"\n{_written_line(effective, name, to_repo)}")
    # The master switch is a security default: named, never flipped on the operator's behalf.
    print(f"enable MCP for runs with:  {_enable_command(to_repo)}")
    return 0


def _prove(
    cfg: Config, name: str, entry: MCPServerEntry, isolation: kinds.IsolationLevel
) -> int | None:
    """Start the server as a run would and print its tools.

    The probe runs under the run's sandbox with the workspace bound read-only; a probe
    never writes the repository.

    Args:
        cfg: The effective config.
        name: The server name.
        entry: The config entry.
        isolation: The isolation to probe under.

    Returns:
        The exit code when the server gave no proof, else None.
    """
    try:
        spec = _setup.mcp_server_spec(
            cfg, pathlib.Path.cwd(), isolation, name, entry, readonly=True
        )
    except jail.JailUnavailableError as exc:
        _common.error(f"{exc}")
        return 2
    print(f"[agent6] {_describe(spec)} ...", file=sys.stderr)
    tools, failure = _probe(spec)
    if failure:
        _report_no_answer(name, list(entry.command), isolation, failure)
        return 1
    if not tools:
        _common.error(f"{name} started but exposed no tools; nothing was written.")
        return 1
    print(f"\n{name}: {mcp_client.tool_count(len(tools))}")
    for tool in tools:
        # Server-chosen text: no forged extra line, no ESC sequence repainting the terminal.
        summary = "".join(c for c in " ".join(tool.description.split()) if c.isprintable())[:80]
        print(f"  mcp__{name}__{tool.tool_name}{'  ' + summary if summary else ''}")
    return None


def _repo_flag(to_repo: bool) -> str:
    """Return the `--repo ` a command line needs to target the repo config, or ""."""
    return "--repo " if to_repo else ""


def _layers_holding(effective: config_layer.EffectiveConfig, name: str) -> set[str]:
    """Return the config layers whose own file declares `[mcp.servers.<name>]`."""
    return {
        layer.name
        for layer in effective.layers
        if name in layer.data.get("mcp", {}).get("servers", {})
    }


def _written_line(effective: config_layer.EffectiveConfig, name: str, to_repo: bool) -> str:
    """Return where the entry went, and what that means beside the other layer's entry.

    The repo layer wins over the global one.
    """
    holders = _layers_holding(effective, name)
    target, other = ("repo", "global") if to_repo else ("global", "repo")
    if target in holders:
        note = f", replacing {name}"
    elif other in holders and to_repo:
        note = f"; it shadows the global config's entry for {name}"
    elif other in holders:
        note = f"; the repo config's entry for {name} keeps winning"
    else:
        note = ""
    return f"written to the {target} config{note}."


def _enable_command(to_repo: bool) -> str:
    """Return the command that flips `[mcp].enabled` on the layer the entry went to."""
    return f"agent6 config set {_repo_flag(to_repo)}mcp.enabled true"


def cmd_mcp_remove(
    name: str, *, to_repo: bool = False, config_path: pathlib.Path | None = None
) -> int:
    """Drop `[mcp.servers.<name>]` from the global (or `--repo`) config.

    The inverse of `connect`, and the only way to drop a server: the entry is a table, so
    `config unset` cannot name it, and unsetting its `command` or `url` is refused because
    an entry needs exactly one of them.

    Args:
        name: The server name.
        to_repo: Edit the repo config instead of the global one.
        config_path: The `--config` file, if any.

    Returns:
        The exit code; 2 when the layer does not hold the entry.
    """
    effective = config_layer.load_effective(pathlib.Path.cwd(), config_path)
    holders = _layers_holding(effective, name)
    target, other = ("repo", "global") if to_repo else ("global", "repo")
    if target not in holders:
        elsewhere = (
            f"the {other} config declares it: agent6 mcp remove {_repo_flag(not to_repo)}{name}"
            if other in holders
            else "agent6 mcp list shows the configured servers"
        )
        _common.error(f"no {name!r} in the {target} config ({elsewhere}).")
        return 2
    res = write.unset_config_table(pathlib.Path.cwd(), f"mcp.servers.{name}", to_repo=to_repo)
    if res.error is not None:
        _common.error(f"removing {name} left an invalid config:\n{res.error}")
        return 2
    if not res.removed:
        # A dotted key or inline table the line surgery cannot delete: still live, and said so.
        path = paths.repo_config_path(pathlib.Path.cwd()) if to_repo else paths.global_config_path()
        _common.error(
            f"{name} is not written as a [mcp.servers.{name}] table in {path};"
            " it lives in a dotted key or an inline table, which this verb does not"
            " rewrite. Edit that file by hand."
        )
        return 2
    print(f"removed {name} from the {target} config")
    if other in holders:
        print(f"note: the {other} config's entry for {name} applies now")
    return 0


def cmd_mcp_list(config_path: pathlib.Path | None = None) -> int:
    """Print the configured servers and how each is reached.

    Reads config only: it never starts anything, so it says nothing about whether a server
    currently works (`agent6 check mcp` does that).

    Returns:
        The exit code, 0.
    """
    effective = config_layer.load_effective(pathlib.Path.cwd(), config_path)
    cfg = effective.config
    if not cfg.mcp.servers:
        print("no MCP servers configured. Add one with `agent6 mcp connect <name> ...`.")
        return 0
    # The switch goes where the servers are: a repo-only setup never enables every repository.
    layers = {
        layer
        for leaf, layer in effective.sources.items()
        if leaf.startswith("mcp.servers.") and layer != "default"
    }
    state = "enabled" if cfg.mcp.enabled else f"DISABLED ({_enable_command(layers == {'repo'})})"
    print(f"MCP is {state}\n")
    for name, srv in sorted(cfg.mcp.servers.items()):
        how = f"connect {srv.url}" if srv.url else f"spawn   {shlex.join(srv.command)}"
        off = "" if srv.enabled else "  [disabled]"
        print(f"  {name:<16} {how}{off}")
        if srv.token_env:
            print(f"  {'':<16} token from ${srv.token_env}")
        if srv.pass_env:
            print(f"  {'':<16} env {' '.join(srv.pass_env)}")
    return 0
