# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 connect`, add a provider + API key."""

from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from agent6.config import (
    AnthropicProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
    validate_base_url,
)
from agent6.config.layer import (
    PROVIDER_PRESETS,
    repo_config_path_for,
    set_config_table,
)
from agent6.models.cache import probe_provider_key
from agent6.paths import global_config_path
from agent6.secrets import SecretsError, save_secret


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
    if not sys.stdin.isatty():
        # No controlling terminal (piped/scripted connect): getpass would fall
        # back to an unmasked read AND print a GetPassWarning about echo. Read a
        # plain line instead -- echo is moot without a terminal, and the scary
        # warning is suppressed.
        try:
            return input(prompt).strip()
        except EOFError:
            return ""
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
    validate_base_url(url)
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
        print("Known presets: " + ", ".join(sorted(PROVIDER_PRESETS)) + " (or any custom name).")
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


def _verify_key(*, api_format: str, base_url: str, api_key: str) -> None:
    """Probe the provider's /models endpoint to confirm the key authenticates.

    A read-only GET, so it does not violate connect's no-remote-execution rule.
    Prints the outcome; never raises (a probe failure must not fail connect, the
    key is already saved). Skipped for offline/local endpoints via --no-verify.
    """
    try:
        entry: ProviderEntry = (
            AnthropicProviderEntry(api_format="anthropic")
            if api_format == "anthropic"
            else OpenAIProviderEntry(
                api_format="openai", base_url=base_url or "https://api.openai.com/v1"
            )
        )
    except ValidationError as exc:
        print(f"  (skipped key check: {exc})", file=sys.stderr)
        return
    print("Checking the key against the provider...")
    result = probe_provider_key(entry, api_key)
    if result.status == "ok":
        print(f"  Key validated: {result.detail}.")
    elif result.status == "auth_failed":
        print(
            f"  WARNING: the provider REJECTED this key ({result.detail}). It was saved anyway;\n"
            "  re-run `agent6 connect` with the correct key (or pass --no-verify for a local"
            " endpoint).",
            file=sys.stderr,
        )
    elif result.status == "unsupported":
        print(f"  (key check skipped: {result.detail})")
    else:  # unreachable
        print(
            f"  NOTE: could not reach the provider to validate the key ({result.detail}); saved"
            " anyway.",
            file=sys.stderr,
        )


def _cmd_connect(*, provider: str, to_repo: bool, verify: bool = True) -> int:  # noqa: PLR0911, PLR0912
    """Interactively add a provider + API key.

    Security: this command NEVER executes anything supplied by a remote. It
    only prompts locally (key via getpass, hidden, or masked with ``*`` on
    Python 3.14+), stores the key in the 0600 secrets file, writes a minimal
    ``[providers.<name>]`` block, and (unless ``verify`` is False) makes one
    read-only GET to the provider's ``/models`` endpoint to confirm the key
    authenticates.
    """
    print("agent6 connect: add a provider and API key.\n")
    name = _resolve_provider_name(provider)
    if name is None:
        return 2
    preset = PROVIDER_PRESETS.get(name)
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
        if verify:
            _verify_key(api_format=api_format, base_url=base_url, api_key=api_key)
    elif api_format == "anthropic":
        # The Anthropic api_format always sends a key; a keyless block is
        # unusable and `agent6 run` would later fail with "no API key". Say so
        # now rather than contradicting ourselves one command later.
        print(
            f"WARNING: no key entered, but the Anthropic API format requires one.\n"
            f"  [providers.{name}] is written but not usable yet -- rerun"
            " `agent6 connect`\n  (or set the api_key_env var) before `agent6 run`."
        )
    else:
        print("No key entered; assuming an unauthenticated/local endpoint.")

    target = repo_config_path_for(Path.cwd()) if to_repo else global_config_path()
    fields: dict[str, str | bool | None] = {"api_format": api_format}
    if api_format == "openai" and base_url and base_url != "https://api.openai.com/v1":
        fields["base_url"] = base_url
    # Shared edit path: persist [providers.<name>], re-validate the merged config,
    # and roll the file back on failure so a bad endpoint never leaves config.toml
    # broken (the key, saved above, is a harmless orphan until a valid retry).
    err = set_config_table(Path.cwd(), f"providers.{name}", fields, to_repo=to_repo)
    if err is not None:
        print(f"Refusing: that would make the config invalid:\n{err}", file=sys.stderr)
        return 2
    print(f"Wrote [providers.{name}] to {target}.")
    print(
        "\nNext: `agent6 model worker "
        f"{name} <model>` to route a role here, then `agent6 config show`."
    )
    return 0
