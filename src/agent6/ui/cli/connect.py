# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 connect`: add a provider and its key or sign-in."""

from __future__ import annotations

import dataclasses
import getpass
import pathlib
import secrets
import sys
import time
from urllib import parse

from agent6 import paths, secret_store
from agent6.config import (
    AnthropicProviderEntry,
    ConfigError,
    OAuthConfig,
    OpenAIProviderEntry,
    ProviderEntry,
    _layer_data,
    io,
    layer,
    provider_name_refusal,
    validate_base_url,
    write,
)
from agent6.models import cache
from agent6.providers import _oauth_protocol, chatgpt_oauth, claude_code, oauth, types
from agent6.ui.cli import _common, _oauth


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
        entry.base_url if isinstance(entry, OpenAIProviderEntry) else "",
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


def _grant_via_device_code(
    issuer: str, client_id: str, provider: str
) -> _oauth_protocol.TokenGrant | None:
    """Show a short code and poll while the person enters it at the issuer's device page.

    The no-display path: nothing to forward over SSH.

    Args:
        issuer: The OAuth issuer.
        client_id: The client registration.
        provider: The provider name, for the reconnect hint.

    Returns:
        The grant, or None when the issuer has the flow disabled, on an error or on Ctrl-C;
        the caller falls back to pasting the callback URL.
    """
    try:
        device = chatgpt_oauth.start_device_auth(issuer, client_id)
    except types.ProviderError as exc:
        print(f"(device sign-in unavailable: {exc})")
        return None
    if device is None:
        return None
    print(f"On any device, open  {issuer.rstrip('/')}{chatgpt_oauth.DEVICE_VERIFY_PATH}")
    print(f"and enter the code:  {device.user_code}")
    print("(waiting; Ctrl-C to paste the callback URL by hand instead)")
    try:
        return chatgpt_oauth.poll_device_auth(issuer, client_id, device, provider=provider)
    except KeyboardInterrupt:
        print()
        return None
    except types.ProviderError as exc:
        print(f"(device sign-in failed: {exc})")
        return None


def _chatgpt_sign_in(name: str) -> int:
    """Run the ChatGPT OAuth sign-in.

    Three ways in, picked by the environment: a GUI machine gets the browser and localhost
    callback, a display-less terminal gets the device flow, and pasting the callback URL
    always works (the whole flow for a piped stdin). Nothing the remote returns is executed;
    the only inputs read back are the state-checked authorization code and the token JSON.

    Args:
        name: The provider name.

    Returns:
        The exit code.
    """
    issuer, client_id = chatgpt_oauth.CHATGPT_ISSUER, chatgpt_oauth.CHATGPT_CLIENT_ID
    verifier, challenge = _oauth_protocol.pkce_pair()
    state = secrets.token_urlsafe(24)
    url = chatgpt_oauth.authorize_url(issuer, client_id, challenge=challenge, state=state)
    print("Open this URL to sign in with your ChatGPT account:\n\n  " + url + "\n")

    grant: _oauth_protocol.TokenGrant | None = None
    code: str | None = None
    if sys.stdin.isatty() and _oauth.gui_browser_available():
        code = _oauth.code_via_callback_server(url, state, port=chatgpt_oauth.CALLBACK_PORT)
    elif sys.stdin.isatty():
        grant = _grant_via_device_code(issuer, client_id, name)
    if grant is None and code is None:
        code = _oauth.paste_callback(state, redirect_uri=chatgpt_oauth.REDIRECT_URI)
        if code is None:
            return 2

    if grant is None:
        assert code is not None
        try:
            grant = chatgpt_oauth.exchange_code(
                issuer, client_id, code=code, verifier=verifier, provider=name
            )
        except types.ProviderError as exc:
            _common.error(f"{exc}")
            return 2
    tokens = chatgpt_oauth.tokens_from_grant(grant)
    try:
        saved = secret_store.save_oauth_tokens(name, tokens)
    except secret_store.SecretsError as exc:
        _common.error(f"{exc}")
        return 2
    plan = chatgpt_oauth.plan_type_of(grant)
    print(f"Signed in{f' ({plan} plan)' if plan else ''}; tokens saved to {saved} (0600).")
    if not tokens.account_id:
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
    return 0


def _configured_issuer(entry: ProviderEntry | None) -> OAuthConfig | None:
    """Return the entry's `oauth` table; None for an entry without one."""
    if isinstance(entry, (AnthropicProviderEntry, OpenAIProviderEntry)):
        return entry.oauth
    return None


def _pick_flow(flow: str, issuer: OAuthConfig) -> str | None:
    """Return the configured issuer's way in: browser, device or paste.

    Args:
        flow: The `--flow` value.
        issuer: The `[providers.<name>].oauth` table.

    Returns:
        The flow, or None after printing why the issuer's settings cannot run it.
    """
    has_code = issuer.authorization_endpoint is not None
    fixed_port = bool(parse.urlsplit(issuer.redirect_uri).port)
    if flow == "auto":
        if has_code and sys.stdin.isatty() and _oauth.gui_browser_available():
            return "browser"
        if issuer.device_authorization_endpoint is not None:
            return "device"
        flow = "paste"
    if flow == "device" and issuer.device_authorization_endpoint is None:
        _common.error("--flow device needs oauth.device_authorization_endpoint in the provider.")
        return None
    if flow in ("browser", "paste") and not has_code:
        _common.error(f"--flow {flow} needs oauth.authorization_endpoint in the provider.")
        return None
    if flow == "paste" and not fixed_port:
        _common.error(
            "pasting the callback needs a fixed port in oauth.redirect_uri (port 0 serves the"
            " local listener only); set one, or add oauth.device_authorization_endpoint."
        )
        return None
    return flow


def _with_port(uri: str, port: int) -> str:
    """Return the redirect with the listener's bound port in place of port 0."""
    parts = parse.urlsplit(uri)
    host = parts.hostname or ""
    netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    return parse.urlunsplit(parts._replace(netloc=netloc))


def _code_grant(
    name: str, issuer: OAuthConfig, *, browser: bool
) -> _oauth_protocol.TokenGrant | None:
    """Run authorization code with PKCE: the loopback redirect, else the pasted callback.

    Args:
        name: The provider name.
        issuer: The `[providers.<name>].oauth` table.
        browser: Open the browser and listen for the redirect first.

    Returns:
        The grant, or None after printing why none was taken.

    Raises:
        ProviderError: The token endpoint was unreachable or refused the code.
    """
    assert issuer.authorization_endpoint is not None
    verifier, challenge = _oauth_protocol.pkce_pair()
    state = secrets.token_urlsafe(24)
    parts = parse.urlsplit(issuer.redirect_uri)
    server: _oauth.CallbackServer | None = None
    if browser:
        try:
            server = _oauth.CallbackServer(
                state, port=parts.port or 0, path=parts.path, host=parts.hostname or "127.0.0.1"
            )
        except OSError as exc:
            print(f"(no local callback: {exc}; paste the callback URL instead)")
    if server is None and not parts.port:
        _common.error(
            "pasting the callback needs a fixed port in oauth.redirect_uri; set one and retry."
        )
        return None
    redirect = (
        issuer.redirect_uri if server is None else _with_port(issuer.redirect_uri, server.port)
    )
    url = _oauth_protocol.authorization_url(
        issuer.authorization_endpoint,
        issuer.client_id,
        redirect_uri=redirect,
        scopes=issuer.scopes,
        state=state,
        challenge=challenge,
        extra=issuer.authorization_params,
    )
    print(f"Open this URL to sign in at {issuer.issuer}:\n\n  {url}\n")
    code = _oauth.wait_for_redirect(server, url) if server is not None else None
    if code is None:
        code = _oauth.paste_callback(state, redirect_uri=redirect)
        if code is None:
            return None
    return _oauth_protocol.exchange_code(
        issuer.token_endpoint,
        issuer.client_id,
        code=code,
        verifier=verifier,
        redirect_uri=redirect,
        provider=name,
        label=name,
        encoding=issuer.request_encoding,
    )


def _device_grant(name: str, issuer: OAuthConfig) -> _oauth_protocol.TokenGrant | None:
    """Show the issuer's user code and poll while the person enters it on any device.

    Args:
        name: The provider name.
        issuer: The `[providers.<name>].oauth` table.

    Returns:
        The grant, or None on Ctrl-C.

    Raises:
        ProviderError: The issuer was unreachable, refused, or the code expired.
    """
    assert issuer.device_authorization_endpoint is not None
    started = _oauth_protocol.start_device_grant(
        issuer.device_authorization_endpoint,
        issuer.client_id,
        scopes=issuer.scopes,
        encoding=issuer.request_encoding,
        label=name,
    )
    print(f"On any device, open  {started.verification_uri}")
    print(f"and enter the code:  {started.user_code}")
    if started.verification_uri_complete:
        print(f"(or open {started.verification_uri_complete}, which carries the code)")
    print("(waiting; Ctrl-C to stop)")
    try:
        return _oauth_protocol.poll_device_grant(
            issuer.token_endpoint,
            issuer.client_id,
            started,
            encoding=issuer.request_encoding,
            label=name,
        )
    except KeyboardInterrupt:
        print()
        return None


def _verify_sign_in(name: str, entry: AnthropicProviderEntry | OpenAIProviderEntry) -> None:
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


def _configured_sign_in(
    name: str,
    entry: AnthropicProviderEntry | OpenAIProviderEntry,
    issuer: OAuthConfig,
    flow: str,
    verify: bool,
) -> int:
    """Sign in at the provider's configured issuer and store the grant bound to the entry.

    Nothing the issuer returns is executed; the inputs read back are the state-checked
    authorization code, the device poll's answer and the token JSON.

    Args:
        name: The provider name.
        entry: The provider as the config holds it.
        issuer: Its `oauth` table.
        flow: The `--flow` value.
        verify: Use the grant once after storing it.

    Returns:
        The exit code.
    """
    chosen = _pick_flow(flow, issuer)
    if chosen is None:
        return 2
    try:
        grant = (
            _device_grant(name, issuer)
            if chosen == "device"
            else _code_grant(name, issuer, browser=chosen == "browser")
        )
    except types.ProviderError as exc:
        _common.error(f"{exc}")
        return 2
    if grant is None:
        return 2
    try:
        saved = secret_store.save_oauth_grant(
            name, oauth.stored_grant_for(grant, entry, now=time.time())
        )
    except secret_store.SecretsError as exc:
        _common.error(f"{exc}")
        return 2
    print(f"Signed in; the grant is saved to {saved} (0600), bound to [providers.{name}].")
    if verify:
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
                    f"API format for {name!r} (anthropic/openai/chatgpt/claude_code) [{default}]: "
                ).strip()
                or default
            )
        except EOFError:
            _common.error("no input.")
            return None
    if api_format not in ("anthropic", "openai", "chatgpt", "claude_code"):
        _common.error(
            f"unknown api_format {api_format!r}"
            " (expected anthropic, openai, chatgpt, or claude_code)."
        )
        return None
    return api_format


def _revoke(name: str, entry: ProviderEntry | None) -> None:
    """Revoke a stored OAuth grant at its issuer, best effort; a dead token revokes fine.

    Args:
        name: The provider name.
        entry: The provider as the config holds it; None when the config does not load.
    """
    issuer = _configured_issuer(entry)
    if issuer is not None:
        grant = secret_store.load_oauth_grant(name)
        if grant is None or issuer.revocation_endpoint is None:
            return
        where = issuer.revocation_endpoint
        err = _oauth_protocol.revoke_tokens(
            where,
            issuer.client_id,
            access_token=grant.access_token,
            refresh_token=grant.refresh_token,
            encoding=issuer.request_encoding,
        )
    else:
        tokens = secret_store.load_oauth_tokens(name)
        if tokens is None:
            return
        where = chatgpt_oauth.CHATGPT_ISSUER
        err = _oauth_protocol.revoke_tokens(
            chatgpt_oauth.revocation_endpoint(where),
            chatgpt_oauth.CHATGPT_CLIENT_ID,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            encoding="json",
        )
    if err is None:
        print(f"Revoked the sign-in for {name!r} at {where}.")
    else:
        _common.warn(f"revocation failed ({err}); removing the local grant anyway.")


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
    _revoke(name, entry)
    try:
        removed = secret_store.delete_provider_secrets(name)
    except secret_store.SecretsError as exc:
        _common.error(f"{exc}")
        return 2
    print(
        f"Removed stored credentials for {name!r} from secrets.toml."
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
    if api_format == "openai":
        # A loaded block always carries a URL; the target's own leaves may not.
        held = current.effective.base_url if current.effective.api_format == "openai" else ""
        offered = held or preset_url or OpenAIProviderEntry(api_format="openai").base_url
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
    if api_format == "chatgpt":
        return _chatgpt_sign_in(name)
    issuer = _configured_issuer(entry)
    if issuer is not None:
        assert isinstance(entry, (AnthropicProviderEntry, OpenAIProviderEntry))
        return _configured_sign_in(name, entry, issuer, flow, verify)
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
    verify: bool = True,
    logout: bool = False,
    flow: str = "auto",
) -> int:
    """Interactively add a provider and its key or sign-in.

    Nothing a remote supplies is executed. The command prompts locally (the key hidden or
    masked), writes a minimal `[providers.<name>]` block, stores the key or grant in the
    0600 secrets file, and uses it once with a read-only GET when `verify` is on and the
    provider supports the check.

    Args:
        provider: The provider name or preset.
        to_repo: Write the block to the repo config instead of the global one.
        verify: Use the key or grant once after storing it.
        logout: Remove the credentials instead.
        flow: How a configured issuer signs in: auto, browser, device or paste.

    Returns:
        The exit code; 2 on a refused name, format or config edit.
    """
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
        rc = _credential_step(name, api_format, flow, verify)
    if rc == 0:
        print(
            "\nNext: `agent6 model worker "
            f"{name} <model>` to route a role here, then `agent6 config show`."
        )
    return rc
