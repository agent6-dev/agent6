# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 connect`: add a provider and its key or sign-in."""

from __future__ import annotations

import dataclasses
import getpass
import pathlib
import sys
import time

from agent6 import paths, secret_store
from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ConfigError,
    OpenAIProviderEntry,
    ProviderEntry,
    ResponsesProviderEntry,
    _layer_data,
    io,
    layer,
    provider_name_refusal,
    validate_base_url,
    write,
)
from agent6.models import cache
from agent6.providers import _headers, _oauth_protocol, chatgpt_oauth, claude_code, oauth, types
from agent6.ui.cli import _common, _connect_status, _oauth


def _prompt_api_key(name: str) -> str:
    """Prompt for an API key without leaking it.

    On Python 3.14+ each keystroke is masked with `*`; on 3.12 and 3.13 input stays hidden
    and a summary prints the key's length, which still tells a garbled paste from a clean
    one. The key itself is never logged.

    Args:
        name: The provider name, for the prompt.

    Returns:
        The key as entered.
    """
    prompt = f"API key for {name} (input hidden, blank for none): "
    if not sys.stdin.isatty():
        # Without a terminal getpass warns and falls back to an unmasked read; echo is moot anyway.
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
        print(f"Captured key: {len(api_key)} chars.")
    return api_key


def _current_entry(name: str) -> ProviderEntry | None:
    """Return the effective `[providers.<name>]`; None when absent or the config does not load."""
    try:
        return layer.load_effective(pathlib.Path.cwd()).config.providers.get(name)
    except ConfigError:
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class _Leaves:
    """The two leaves connect writes, as one file or the merged config holds them.

    Attributes:
        api_format: The API format; "" when absent.
        base_url: The base URL; "" when absent.
    """

    api_format: str
    base_url: str


@dataclasses.dataclass(frozen=True, slots=True)
class _Current:
    """What a run here reads for a provider, beside what the target file holds.

    Attributes:
        effective: The merged config's leaves, the values the prompts offer.
        target: The target file's own leaves, which Enter leaves alone when they match.
        shadowed: The layer above the target that sets each leaf, by leaf name.
    """

    effective: _Leaves
    target: _Leaves
    shadowed: dict[str, _layer_data.Layer]


def _target_leaves(name: str, target: pathlib.Path) -> _Leaves:
    """Return the target file's own `[providers.<name>]` leaves.

    Args:
        name: The provider name.
        target: The config file connect will edit; absent reads as empty.

    Returns:
        The leaves; "" for each the file does not hold.

    Raises:
        ConfigError: The target file cannot be read as TOML.
    """
    providers = io.read_toml_file(target).get("providers")
    block = providers.get(name) if isinstance(providers, dict) else None
    if not isinstance(block, dict):
        return _Leaves("", "")
    return _Leaves(str(block.get("api_format", "")), str(block.get("base_url", "")))


def _current(name: str, target: pathlib.Path, to_repo: bool) -> _Current:
    """Read the provider's effective leaves and the target file's own.

    Args:
        name: The provider name.
        target: The config file connect will edit.
        to_repo: The target is the repo config.

    Returns:
        The current state; when the config does not load, the target's own leaves stand in
        for the effective ones.

    Raises:
        ConfigError: The target file cannot be read as TOML.
    """
    held = _target_leaves(name, target)
    try:
        eff = layer.load_effective(pathlib.Path.cwd())
    except ConfigError:
        return _Current(held, held, {})
    entry = eff.config.providers.get(name)
    effective = _Leaves(
        entry.api_format if entry is not None else "",
        entry.base_url if isinstance(entry, (OpenAIProviderEntry, ResponsesProviderEntry)) else "",
    )
    target_layer = "repo" if to_repo else "global"
    shadowed = {
        leaf: lyr
        for leaf in ("api_format", "base_url")
        if (lyr := layer.shadowing_layer(eff, f"providers.{name}.{leaf}", target_layer)) is not None
    }
    return _Current(effective, held, shadowed)


def _shadow_notice(current: _Current, leaf: str, to_repo: bool, written: str) -> None:
    """Say when a layer above the target keeps a run on a value other than the one written.

    Args:
        current: The provider's current state.
        leaf: `api_format` or `base_url`.
        to_repo: The target is the repo config.
        written: The value the target holds after this command.
    """
    above = current.shadowed.get(leaf)
    value = getattr(current.effective, leaf)
    if above is None or value == written:
        return
    target = "repo" if to_repo else "global"
    print(
        f"  note: {above.described} sets {leaf} = {value}, above the {target} config this"
        " command writes; a run here keeps reading it."
    )


def _prompt_base_url(offered: str) -> str:
    """Prompt for an OpenAI-compatible base URL and validate it.

    Validated before any secret or config write, so a scheme-less value (an API key pasted
    into the wrong prompt) is rejected up front rather than surfaced later as an opaque HTTP
    error.

    Args:
        offered: The URL Enter keeps: the one a run reads, else the preset's or the default.

    Returns:
        The URL typed, or `offered`.

    Raises:
        ValueError: The URL is invalid, by the `OpenAIProviderEntry.base_url` check.
    """
    try:
        url = input(f"Base URL [{offered}]: ").strip() or offered
    except EOFError:
        url = offered
    validate_base_url(url)
    return url


def _write_provider_block(
    name: str, fields: dict[str, io.ConfigLeafValue], *, to_repo: bool
) -> str | None:
    """Write the block's leaves, before any credential is stored.

    Leaf surgery keeps hand-added sibling keys and revalidates the file.

    Args:
        name: The provider name.
        fields: The leaves to set.
        to_repo: Write the per-repo config instead of the global one.

    Returns:
        The refusal, or None once written: a file the write would make invalid, or an entry
        that is not a plain table.
    """
    try:
        err = write.set_config_leaves(
            pathlib.Path.cwd(), f"providers.{name}", fields, to_repo=to_repo
        )
    except ConfigError as exc:
        return str(exc)
    return None if err is None else f"that would make the config invalid:\n{err}"


def _resolve_provider_name(provider: str) -> str | None:
    """Validate the provider name and print an error when it is not a bare TOML key.

    The name is checked before any config or credential write.

    Args:
        provider: The name; blank prompts for one.

    Returns:
        The name, or None after the error.
    """
    name = provider.strip()
    if not name:
        print(
            "Known presets: "
            + ", ".join(sorted(write.PROVIDER_DEFAULTS))
            + " (or any custom name)."
        )
        try:
            name = input("Provider name [anthropic]: ").strip() or "anthropic"
        except EOFError:
            _common.error("no input.")
            return None
    if refusal := provider_name_refusal(name):
        _common.error(refusal)
        return None
    return name


def _verify_key(entry: ProviderEntry | None, api_key: str) -> None:
    """Probe the provider's /models endpoint to confirm the key authenticates.

    A read-only GET, within connect's no-remote-execution rule. Prints the outcome and never
    raises: the key is already saved, so a probe failure must not fail connect.

    Args:
        entry: The provider as the config holds it after the write, so the probe dials the
            host a run dials; None when the config does not load.
        api_key: The key to test.
    """
    if entry is None:
        print(
            "  (key check skipped: the config does not load; `agent6 check config` says why)",
            file=sys.stderr,
        )
        return
    print("Checking the key against the provider...")
    result = cache.probe_provider_key(entry, api_key)
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


def _verify_sign_in(
    name: str, entry: AnthropicProviderEntry | OpenAIProviderEntry | ResponsesProviderEntry
) -> None:
    """Use the stored grant once: the configured exchange, then a read-only model listing.

    Prints the outcome and never raises: the grant is already saved.

    Args:
        name: The provider name.
        entry: The provider as the config holds it.
    """
    print("Checking the sign-in against the provider...")
    try:
        oauth.OAuthCredential(name, entry).token()
    except types.ProviderError as exc:
        _common.warn(f"the sign-in is saved but not usable yet: {exc}")
        return
    listed = cache.fetch_models_live(name, entry, None, timeout_s=10.0)
    if listed:
        print(f"  Sign-in validated: {len(listed)} models listed.")
    else:
        print("  (could not list models with the sign-in; saved anyway)")


def _sign_in(
    name: str,
    entry: AnthropicProviderEntry
    | ChatGPTProviderEntry
    | OpenAIProviderEntry
    | ResponsesProviderEntry,
    flow: str,
    verify: bool,
) -> int:
    """Sign in through the provider's profile and store the bound grant.

    Nothing the issuer returns is executed; the inputs read back are the state-checked
    authorization code, the device poll's answer and the token JSON.

    Args:
        name: The provider name.
        entry: The provider as the config holds it.
        flow: The `--flow` value.
        verify: Use the grant once after storing it.

    Returns:
        The exit code.
    """
    profile = oauth.profile_for(entry)
    headers = _headers.HeaderSet.from_config(entry.extra_headers)
    try:
        grant = _oauth.sign_in_grant(name, profile, headers, flow)
        if grant is None:
            return 2
        stored = oauth.stored_grant_for(grant, entry, now=time.time())
        saved = secret_store.save_oauth_grant(name, stored)
    except (types.ProviderError, secret_store.SecretsError) as exc:
        _common.error(f"{exc}")
        return 2
    plan = chatgpt_oauth.plan_type_of(grant) if isinstance(entry, ChatGPTProviderEntry) else ""
    print(
        f"Signed in{f' ({plan} plan)' if plan else ''}; the grant is saved to {saved} (0600),"
        f" bound to [providers.{name}]."
    )
    if isinstance(entry, ChatGPTProviderEntry):
        if not stored.account_id:
            _common.warn(
                "the sign-in carried no ChatGPT account id; runs will refuse until a"
                " sign-in with a ChatGPT plan succeeds."
            )
        print(
            "\nNote: whether these conversations train OpenAI's models follows the ChatGPT\n"
            "account's own data controls (Settings > Data controls > 'Improve the model for\n"
            "everyone'); agent6 cannot change that setting, and it never sends feedback or\n"
            "ratings, which would opt those turns in regardless of it."
        )
    elif verify:
        _verify_sign_in(name, entry)
    return 0


def _claude_code_check(name: str) -> None:
    """Name a signed-out `claude` install now rather than at the first run.

    No secret to store: the binary carries the operator's own login. `connect` checks
    `claude` on PATH; the run preflight and `agent6 model` check `[providers.<name>].binary`.

    Args:
        name: The provider name, for the warning.
    """
    err = claude_code.login_status("claude")
    if err is None:
        print("Claude Code (`claude` on PATH): signed in.")
        return
    _common.warn(f"{err}\n  [providers.{name}] is written but not usable yet.")


def _prompt_api_format(name: str, preset_format: str, current_format: str) -> str | None:
    """Return the api_format for the provider: the block's, the preset's, or the answer.

    An existing block's format wins over a preset's; the prompt appears when the two differ
    or there is no preset, and Enter keeps the block's.

    Args:
        name: The provider name, for the prompt.
        preset_format: The preset's format; "" for a custom name.
        current_format: The existing block's format; "" for a new block.

    Returns:
        The format, or None (after printing why) on no input or an unknown value.
    """
    default = current_format or preset_format or "anthropic"
    api_format = default if default == preset_format else ""
    if not api_format:
        try:
            api_format = (
                input(
                    f"API format for {name!r} (anthropic/openai/chatgpt/responses/claude_code)"
                    f" [{default}]: "
                ).strip()
                or default
            )
        except EOFError:
            _common.error("no input.")
            return None
    if api_format not in ("anthropic", "openai", "chatgpt", "responses", "claude_code"):
        _common.error(
            f"unknown api_format {api_format!r}"
            " (expected anthropic, openai, chatgpt, responses or claude_code)."
        )
        return None
    return api_format


def _revoke(name: str, entry: ProviderEntry | None) -> bool:
    """Revoke a stored OAuth grant at its issuer, best effort; a dead token revokes fine.

    Args:
        name: The provider name.
        entry: The provider as the config holds it; None when the config does not load.

    Returns:
        True only when the issuer accepted the revocation.
    """
    if (
        entry is None
        or not isinstance(
            entry,
            (
                AnthropicProviderEntry,
                ChatGPTProviderEntry,
                OpenAIProviderEntry,
                ResponsesProviderEntry,
            ),
        )
        or (not isinstance(entry, ChatGPTProviderEntry) and entry.oauth is None)
    ):
        return False
    try:
        grant = oauth.load_bound_grant(name, entry)
    except types.ProviderError as exc:
        _common.warn(f"{exc} Not revoked.")
        return False
    profile = oauth.profile_for(entry)
    where = profile.revocation_endpoint
    if where is None:
        return False
    err = _oauth_protocol.revoke_tokens(
        where,
        profile.client_id,
        access_token=grant.access_token,
        refresh_token=grant.refresh_token,
        encoding=profile.revocation_encoding,
        headers=_headers.HeaderSet.from_config(entry.extra_headers).renderer("oauth"),
    )
    if err is None:
        print(f"Revoked the sign-in for {name!r} at {where}.")
        return True
    _common.warn(f"revocation failed ({err}); removing the local grant anyway.")
    return False


def _cmd_logout(name: str, entry: ProviderEntry | None, api_format: str) -> int:
    """Remove a provider's stored credentials (`connect --logout`).

    An OAuth grant is revoked at its issuer first, best effort: local removal proceeds
    regardless. The `[providers.<name>]` block stays. A claude_code provider holds no
    credentials here.

    Args:
        name: The provider name.
        entry: The provider as the config holds it; None when the config does not load.
        api_format: Its API format.

    Returns:
        The exit code: 0, or 2 for a claude_code provider or a secrets file that cannot be
        written.
    """
    if api_format == "claude_code":
        print(
            "agent6 stores no Claude Code credentials; `claude auth logout` signs out.",
            file=sys.stderr,
        )
        return 2
    revoked = _revoke(name, entry)
    try:
        removed = secret_store.delete_provider_secrets(name)
    except secret_store.SecretsError as exc:
        _common.error(f"{exc}")
        return 2
    print(
        f"Removed stored credentials for {name!r} from secrets.toml"
        f"{'.' if revoked else ' (locally only).'}"
        if removed
        else f"No stored credentials for {name!r}."
    )
    return 0


def _write_shown_values(
    name: str,
    api_format: str,
    current: _Current,
    preset_url: str,
    target: pathlib.Path,
    to_repo: bool,
) -> int:
    """Prompt for the base URL when the format takes one, then write what the prompts showed.

    Enter writes the shown value to the target unless the target already holds it.

    Args:
        name: The provider name.
        api_format: The format the prompt settled.
        current: The provider's current state.
        preset_url: The preset's URL; "" for a custom name.
        target: The config file connect edits.
        to_repo: Write the per-repo config instead of the global one.

    Returns:
        0 once written or already held; 2 after an invalid URL or a refused edit.
    """
    fields: dict[str, io.ConfigLeafValue] = {
        "api_format": api_format if api_format != current.target.api_format else None
    }
    if api_format in ("openai", "responses"):
        # A loaded block always carries a URL; the target's own leaves may not.
        held = current.effective.base_url if current.effective.api_format == api_format else ""
        vendor = OpenAIProviderEntry(api_format="openai").base_url if api_format == "openai" else ""
        offered = held or preset_url or vendor  # a Responses endpoint has no vendor default
        try:
            url = _prompt_base_url(offered)
        except ValueError as exc:
            _common.error(f"{exc}")
            return 2
        _shadow_notice(current, "base_url", to_repo, url)
        fields["base_url"] = url if url != current.target.base_url else None
    if all(value is None for value in fields.values()):
        print(f"[providers.{name}] in {target} already holds these values.")
        return 0
    if (refusal := _write_provider_block(name, fields, to_repo=to_repo)) is not None:
        _common.refuse(refusal)
        return 2
    print(f"Wrote [providers.{name}] to {target}.")
    return 0


def _credential_step(name: str, api_format: str, flow: str, verify: bool) -> int:
    """Store what the provider signs requests with: a sign-in, a key, or nothing.

    Args:
        name: The provider name.
        api_format: The format the prompt settled.
        flow: The `--flow` value for a configured issuer.
        verify: Use the key or grant once after storing it.

    Returns:
        The exit code.
    """
    entry = _current_entry(name)
    if api_format == "claude_code":
        _claude_code_check(name)
        return 0
    if isinstance(entry, ChatGPTProviderEntry) or (
        isinstance(entry, (AnthropicProviderEntry, OpenAIProviderEntry, ResponsesProviderEntry))
        and entry.oauth is not None
    ):
        return _sign_in(name, entry, flow, verify)
    if api_format == "chatgpt":
        _common.error("the provider config does not load; run `agent6 check config`.")
        return 2
    try:
        api_key = _prompt_api_key(name)
    except EOFError:
        api_key = ""
    if api_key:
        try:
            saved = secret_store.save_secret(name, api_key)
        except secret_store.SecretsError as exc:
            _common.error(f"{exc}")
            return 2
        print(f"Saved key to {saved} (0600).")
        if verify:
            _verify_key(entry, api_key)
    elif api_format == "anthropic":
        # The Anthropic api_format always sends a key, so a keyless block fails at the first run.
        _common.warn(
            f"no key entered, but the Anthropic API format requires one.\n"
            f"  [providers.{name}] is written but not usable yet; rerun"
            " `agent6 connect`\n  (or set the api_key_env var) before `agent6 run`."
        )
    else:
        print("No key entered; assuming an unauthenticated/local endpoint.")
    return 0


def _cmd_connect(
    *,
    provider: str,
    to_repo: bool,
    verify: bool | None = None,
    logout: bool = False,
    flow: str | None = None,
    status: bool = False,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> int:
    """Interactively add a provider and its key or sign-in.

    Nothing a remote supplies is executed. The command prompts locally (the key hidden or
    masked), writes a minimal `[providers.<name>]` block, stores the key or grant in the
    0600 secrets file, and uses it once with a read-only GET when `verify` is on and the
    provider supports the check.

    Args:
        provider: The provider name or preset.
        to_repo: Write the block to the repo config instead of the global one.
        verify: Use the key or grant once after storing it; None defaults to on.
        logout: Remove the credentials instead.
        flow: How a configured issuer signs in; None defaults to auto.
        status: List local credential state and free live checks instead.
        config_paths: Explicit config overlays for the status listing.

    Returns:
        The exit code; 2 on a refused name, format or config edit.
    """
    if status:
        conflicts = [
            flag
            for flag, set_ in (
                ("--logout", logout),
                ("--verify" if verify else "--no-verify", verify is not None),
                ("--flow", flow is not None),
                ("--repo", to_repo),
            )
            if set_
        ]
        if conflicts:
            _common.error(f"--status cannot combine with {', '.join(conflicts)}.")
        return 2 if conflicts else _connect_status.show(provider, config_paths)
    name = _resolve_provider_name(provider)
    if name is None:
        return 2
    preset = write.PROVIDER_DEFAULTS.get(name)
    preset_format = preset["api_format"] if preset else ""
    target = paths.repo_config_path(pathlib.Path.cwd()) if to_repo else paths.global_config_path()
    try:
        current = _current(name, target, to_repo)
    except ConfigError as exc:
        if not logout:
            _common.error(str(exc))
            return 2
        current = _Current(_Leaves("", ""), _Leaves("", ""), {})
    if logout:
        return _cmd_logout(
            name, _current_entry(name), current.effective.api_format or preset_format
        )
    print("agent6 connect: set up a model provider.\n")
    api_format = _prompt_api_format(name, preset_format, current.effective.api_format)
    if api_format is None:
        return 2
    _shadow_notice(current, "api_format", to_repo, api_format)
    preset_url = (preset or {}).get("base_url", "")
    rc = _write_shown_values(name, api_format, current, preset_url, target, to_repo)
    if rc == 0:
        rc = _credential_step(name, api_format, flow or "auto", verify is not False)
    if rc == 0:
        print(
            "\nNext: `agent6 model worker "
            f"{name} <model>` to route a role here, then `agent6 config show`."
        )
    return rc
