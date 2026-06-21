# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 connect`, add a provider + API key."""

from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path

from agent6.cli.toml_io import _upsert_toml_table
from agent6.config import _validate_base_url
from agent6.config_layer import (
    repo_config_path_for,
)
from agent6.paths import (
    chown_to_real_user,
    global_config_path,
)
from agent6.secrets import SecretsError, save_secret

# Known provider presets for `agent6 connect`. api_format + default base_url;
# the table key (provider name) is what [models.<role>].provider references and
# what the key is stored under in secrets.toml. connect handles the common
# `direct` deployment with a stored key; auth.style and deployment default from
# api_format (see config), and advanced deployments (vertex/azure/token_command)
# are documented in CONFIG.md for hand-editing.
_CONNECT_PRESETS: dict[str, dict[str, str]] = {
    "anthropic": {"api_format": "anthropic"},
    "openai": {"api_format": "openai", "base_url": "https://api.openai.com/v1"},
    "openrouter": {"api_format": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "ollama": {"api_format": "openai", "base_url": "http://localhost:11434/v1"},
}


def _prompt_api_key(name: str) -> str:
    """Prompt for an API key without leaking it.

    On Python 3.14+ ``getpass`` accepts ``echo_char`` so we mask each
    keystroke with ``*``, live feedback that the paste landed, without ever
    revealing the key. On 3.12/3.13 input stays fully hidden and we print a
    post-entry summary (length + last four chars) so the operator can still
    tell a partial/garbled paste from a clean one. The key itself is never
    logged.
    """
    prompt = f"API key for {name} (input hidden, blank for none): "
    masked = False
    try:
        api_key = getpass.getpass(prompt, echo_char="*").strip()  # type: ignore[call-arg]
        masked = True
    except TypeError:
        # Python < 3.14: no echo_char parameter.
        api_key = getpass.getpass(prompt).strip()
    except EOFError:
        return ""
    if api_key and not masked:
        tail = f", ending …{api_key[-4:]}" if len(api_key) >= 8 else ""
        print(f"Captured key: {len(api_key)} chars{tail}.")
    return api_key


def _prompt_base_url(default_url: str) -> str:
    """Prompt for an OpenAI-compatible base URL and validate it.

    Validates before any secret/config write so a scheme-less value (e.g. an
    API key pasted into the wrong prompt) is rejected up front rather than
    persisted and surfaced later as an opaque HTTP error. Raises ``ValueError``
    on an invalid URL (same check as the ``OpenAIProviderEntry.base_url``
    validator).
    """
    try:
        url = input(f"Base URL [{default_url}]: ").strip() or default_url
    except EOFError:
        url = default_url
    _validate_base_url(url)
    return url


def _resolve_provider_name(provider: str) -> str | None:
    """Resolve + validate the provider name; print an error and return None if bad.

    The name becomes a TOML table key ``[providers.<name>]``; a non-bare-key
    name (space, dot, bracket, …) would be written verbatim and corrupt the
    whole config file, which ``connect`` -- unlike ``model``/``config set`` --
    does not re-validate after writing. So reject it before any write.
    """
    name = provider.strip()
    if not name:
        print("Known presets: " + ", ".join(sorted(_CONNECT_PRESETS)) + " (or any custom name).")
        try:
            name = input("Provider name [anthropic]: ").strip() or "anthropic"
        except EOFError:
            print("ERROR: no input.", file=sys.stderr)
            return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        print(
            f"ERROR: provider name {name!r} is not a valid TOML bare key"
            " (use only letters, digits, '-', '_').",
            file=sys.stderr,
        )
        return None
    return name


def _cmd_connect(*, provider: str, to_repo: bool) -> int:
    """Interactively add a provider + API key.

    Security: this command NEVER executes anything supplied by a remote. It
    only prompts locally (key via getpass, hidden, or masked with ``*`` on
    Python 3.14+), stores the key in the 0600 secrets file, and writes a
    minimal ``[providers.<name>]`` block.
    """
    print("agent6 connect — add a provider + API key.\n")
    name = _resolve_provider_name(provider)
    if name is None:
        return 2
    preset = _CONNECT_PRESETS.get(name)
    api_format = preset["api_format"] if preset else ""
    if not api_format:
        try:
            api_format = (
                input(f"API format for {name!r} [anthropic/openai]: ").strip() or "anthropic"
            )
        except EOFError:
            return 2
    if api_format not in ("anthropic", "openai"):
        print(
            f"ERROR: unknown api_format {api_format!r} (expected anthropic or openai).",
            file=sys.stderr,
        )
        return 2
    base_url = (preset or {}).get("base_url", "")
    if api_format == "openai":
        try:
            base_url = _prompt_base_url(base_url or "https://api.openai.com/v1")
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        api_key = _prompt_api_key(name)
    except EOFError:
        api_key = ""
    if api_key:
        try:
            saved = save_secret(name, api_key)
        except SecretsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Saved key to {saved} (0600).")
    else:
        print("No key entered; assuming an unauthenticated/local endpoint.")

    target = repo_config_path_for(Path.cwd()) if to_repo else global_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, str | bool | None] = {"api_format": api_format}
    if api_format == "openai" and base_url and base_url != "https://api.openai.com/v1":
        fields["base_url"] = base_url
    _upsert_toml_table(target, f"providers.{name}", fields)
    chown_to_real_user(target.parent)
    chown_to_real_user(target)
    print(f"Wrote [providers.{name}] to {target}.")
    print(
        "\nNext: `agent6 model worker "
        f"{name} <model>` to route a role here, then `agent6 config show`."
    )
    return 0
