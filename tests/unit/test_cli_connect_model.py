# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 connect` / `agent6 model` / `agent6 config` CLI flows."""

from __future__ import annotations

import dataclasses
import io
import pathlib
import stat
import subprocess
import time
import tomllib
from collections.abc import Callable

import httpx2
import pytest

from agent6 import paths, secret_store
from agent6.config import ChatGPTProviderEntry, OpenAIProviderEntry
from agent6.models import cache
from agent6.providers import _oauth_protocol, _transport, chatgpt_oauth, oauth, types
from agent6.ui.cli import _common, _oauth, main
from agent6.ui.cli import model as modelmod

_REAL_PROBE = cache.probe_provider_key


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    # An interactive terminal by default: the masked-input path runs only when stdin is a TTY.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # The post-save key probe is stubbed so no test makes a network call; probe tests re-patch it.
    monkeypatch.setattr(
        "agent6.models.cache.probe_provider_key",
        lambda *a, **k: cache.KeyProbeResult(  # type: ignore[misc]
            ok=True, status="ok", detail="provider returned 1 models"
        ),
    )
    return tmp_path / "g"


def test_connect_stores_key_and_provider_and_never_execs(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-FAKE\n"))
    # Security: connect must NEVER run a subprocess (no remote-supplied command).
    calls: list[object] = []

    def _record_run(*args: object, **kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr("subprocess.run", _record_run)

    rc = main(["connect", "anthropic"])
    assert rc == 0

    sp = tmp_path / "g" / "agent6" / "secrets.toml"
    assert sp.is_file()
    assert stat.S_IMODE(sp.stat().st_mode) == 0o600
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-FAKE"

    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.anthropic]" in gc
    assert calls == []


def test_connect_preserves_hand_edited_provider_keys(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Re-running connect for a key rotation keeps the operator's hand-added sibling keys."""
    gc = tmp_path / "g" / "agent6" / "config.toml"
    gc.parent.mkdir(parents=True, exist_ok=True)
    gc.write_text(
        "[providers.openrouter]\n"
        'api_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        "# hand-added\n"
        "http_timeout_s = 120\n"
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-or-FAKE\n"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # accept the preset base_url
    rc = main(["connect", "openrouter"])
    assert rc == 0
    text = gc.read_text(encoding="utf-8")
    assert "[providers.openrouter]" in text
    assert 'api_format = "openai"' in text
    assert "http_timeout_s = 120" in text
    assert "extra_body" in text
    assert "# hand-added" in text


_AZURE_BLOCK = (
    "[providers.az]\n"
    'api_format = "openai"\n'
    'deployment = "azure"\n'
    'base_url = "https://res.openai.azure.com"\n'
    'extra_query = { "api-version" = "2024-10-21" }\n'
)


def _scripted_input(monkeypatch: pytest.MonkeyPatch, *answers: str) -> list[str]:
    """Answer each `input()` prompt in turn.

    Args:
        monkeypatch: The test's patcher.
        *answers: The replies, one per prompt.

    Returns:
        The prompts as asked, filled in as the run goes.
    """
    prompts: list[str] = []
    replies = iter(answers)

    def answer(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr("builtins.input", answer)
    return prompts


@pytest.mark.parametrize(
    ("block", "host"),
    [
        (_AZURE_BLOCK, "https://res.openai.azure.com"),
        (_AZURE_BLOCK + "[budget]\nbogus_key = 1\n", "https://res.openai.azure.com"),
        ("", "https://api.openai.com/v1"),
        (
            '[providers.az]\napi_format = "anthropic"\nbase_url = "https://proxy.example/v1"\n',
            "https://api.openai.com/v1",
        ),
    ],
    ids=["existing", "broken-config", "new", "format-change"],
)
def test_enter_at_the_base_url_prompt_keeps_the_url_it_shows(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    block: str,
    host: str,
) -> None:
    """Enter at the base URL prompt keeps the URL the prompt shows, and the key check probes it.

    Over an existing block the prompt showed `https://api.openai.com/v1` while Enter kept the
    block's own host, and the key check probed the generic host.
    """
    if block:
        gc = iso / "agent6" / "config.toml"
        gc.parent.mkdir(parents=True)
        gc.write_text(block, encoding="utf-8")
    prompts = _scripted_input(monkeypatch, "openai", "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-az\n"))
    probed: list[str] = []

    def probe(entry: OpenAIProviderEntry, api_key: str) -> cache.KeyProbeResult:
        probed.append(entry.base_url)
        return cache.KeyProbeResult(ok=True, status="ok", detail="provider accepted the key")

    monkeypatch.setattr("agent6.models.cache.probe_provider_key", probe)
    assert main(["connect", "az"]) == 0
    saved = tomllib.loads(paths.global_config_path().read_text())["providers"]["az"]
    assert (prompts[-1], saved.get("base_url", "https://api.openai.com/v1"), probed) == (
        f"Base URL [{host}]: ",
        host,
        [] if "bogus_key" in block else [host],
    )


@pytest.mark.parametrize("broken", [False, True])
def test_enter_at_the_api_format_prompt_keeps_an_existing_blocks_format(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, broken: bool
) -> None:
    """Enter at the API format prompt keeps an existing block's format, which the prompt shows.

    The prompt offered `anthropic`, so Enter refused the azure block.
    """
    gc = iso / "agent6" / "config.toml"
    gc.parent.mkdir(parents=True)
    gc.write_text(_AZURE_BLOCK + ("[budget]\nbogus_key = 1\n" if broken else ""), encoding="utf-8")
    prompts = _scripted_input(monkeypatch, "", "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-az\n"))
    rc = main(["connect", "az"])
    api_format = tomllib.loads(gc.read_text())["providers"]["az"]["api_format"]
    assert (rc, api_format, prompts[0]) == (
        0,
        "openai",
        "API format for 'az' (anthropic/openai/chatgpt/responses/claude_code) [openai]: ",
    )


@pytest.mark.parametrize("to_repo", [False, True])
@pytest.mark.parametrize("replacement", ["", "https://api.openai.com/v1"])
def test_connect_checks_effective_host_after_target_layer_edit(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    to_repo: bool,
    replacement: str,
) -> None:
    """The prompt offers the URL a run reads; a value the repo layer keeps over the target is said.

    Enter writes the shown URL to the target unless the target holds it; the probe follows the
    effective value. Writing the global file under a repo override wrote nothing and said nothing.
    """
    global_path = paths.global_config_path()
    repo_path = paths.repo_config_path(tmp_path)
    for path, host in ((global_path, "global"), (repo_path, "repo")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'[providers.custom]\napi_format = "openai"\nbase_url = "https://{host}.example/v1"\n'
        )
    prompts = _scripted_input(monkeypatch, "", replacement)
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-test\n"))
    monkeypatch.setattr(cache, "probe_provider_key", _REAL_PROBE)
    requested: list[str] = []

    def get(method: str, url: str, **kwargs: object) -> httpx2.Response:
        requested.append(url)
        return httpx2.Response(200, json={"data": [{"id": "model"}]})

    monkeypatch.setattr(_transport, "http_request", get)
    assert main(["connect", "custom", *(["--repo"] if to_repo else [])]) == 0
    out = capsys.readouterr().out
    assert prompts[-1] == "Base URL [https://repo.example/v1]: "
    notice = (
        "note: the repo config sets base_url = https://repo.example/v1, above the global config"
        " this command writes; a run here keeps reading it."
    )
    assert (notice in out) is bool(replacement and not to_repo)
    shown = replacement or "https://repo.example/v1"
    for path, host in ((global_path, "global"), (repo_path, "repo")):
        expected = shown if (path == repo_path) == to_repo else f"https://{host}.example/v1"
        assert tomllib.loads(path.read_text())["providers"]["custom"] == {
            "api_format": "openai",
            "base_url": expected,
        }
    assert ("already holds" in out) is (to_repo and not replacement)
    effective_url = replacement if replacement and to_repo else "https://repo.example/v1"
    assert requested == [effective_url + "/models"]


def test_connect_says_when_a_typed_url_is_kept_out_by_the_repo_layer(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Global and repo held the same URL, so no notice said the typed one never reaches a run."""
    same = '[providers.custom]\napi_format = "openai"\nbase_url = "https://same.example/v1"\n'
    for path in (paths.global_config_path(), paths.repo_config_path(tmp_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(same)
    _scripted_input(monkeypatch, "", "https://typed.example/v1")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-test\n"))
    assert main(["connect", "custom"]) == 0
    out = capsys.readouterr().out
    assert (
        "note: the repo config sets base_url = https://same.example/v1, above the global config"
        " this command writes; a run here keeps reading it." in out
    )
    saved = tomllib.loads(paths.global_config_path().read_text())["providers"]["custom"]
    assert saved["base_url"] == "https://typed.example/v1"


def test_a_broken_config_still_offers_the_default_url_for_a_bare_openai_block(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the config not loading, a block without base_url offered `Base URL []`; Enter failed."""
    gc = paths.global_config_path()
    gc.parent.mkdir(parents=True)
    gc.write_text('[providers.az]\napi_format = "openai"\n[budget]\nbogus_key = 1\n')
    prompts = _scripted_input(monkeypatch, "", "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-az\n"))
    assert main(["connect", "az"]) == 0
    assert prompts[-1] == "Base URL [https://api.openai.com/v1]: "
    saved = tomllib.loads(gc.read_text())["providers"]["az"]
    assert saved["base_url"] == "https://api.openai.com/v1"


def test_connect_writes_a_repo_only_providers_url_to_the_global_file(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A provider only the repo config held got a global block with no URL: Enter wrote nothing."""
    repo_path = paths.repo_config_path(tmp_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.write_text(
        '[providers.custom]\napi_format = "openai"\nbase_url = "https://repo.example/v1"\n'
    )
    _scripted_input(monkeypatch, "", "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-test\n"))
    assert main(["connect", "custom"]) == 0
    saved = tomllib.loads(paths.global_config_path().read_text())["providers"]["custom"]
    assert saved == {"api_format": "openai", "base_url": "https://repo.example/v1"}


def test_a_preset_name_keeps_an_existing_blocks_other_format(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`connect claude` over an anthropic-format `[providers.claude]` rewrote it to claude_code."""
    gc = paths.global_config_path()
    gc.parent.mkdir(parents=True)
    gc.write_text('[providers.claude]\napi_format = "anthropic"\n')
    prompts = _scripted_input(monkeypatch, "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant\n"))
    assert main(["connect", "claude"]) == 0
    assert prompts == [
        "API format for 'claude' (anthropic/openai/chatgpt/responses/claude_code) [anthropic]: "
    ]
    assert tomllib.loads(gc.read_text())["providers"]["claude"] == {"api_format": "anthropic"}
    assert "already holds these values" in capsys.readouterr().out


def test_logout_reads_the_blocks_format_not_the_preset_name(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--logout` on a custom-named claude_code block said "No stored credentials"."""
    gc = paths.global_config_path()
    gc.parent.mkdir(parents=True)
    gc.write_text('[providers.cc]\napi_format = "claude_code"\n')
    assert main(["connect", "cc", "--logout"]) == 2
    assert "claude auth logout" in capsys.readouterr().err


def test_connect_over_a_non_table_provider_entry_does_not_crash(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `providers.az` that is a string, not a table, crashed the broken-config fallback."""
    gc = paths.global_config_path()
    gc.parent.mkdir(parents=True)
    gc.write_text('[providers]\naz = "bad"\n')
    _scripted_input(monkeypatch, "openai", "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-az\n"))
    rc = main(["connect", "az"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "providers.az" in err and "Traceback" not in err


def test_connect_auth_none_skips_key_check(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An auth-none block cannot validate a key it never sends."""
    gc = paths.global_config_path()
    gc.parent.mkdir(parents=True)
    gc.write_text('[providers.local]\napi_format = "openai"\nauth_style = "none"\n')
    _scripted_input(monkeypatch, "", "")
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-test\n"))
    monkeypatch.setattr(cache, "probe_provider_key", _REAL_PROBE)
    requested: list[str] = []

    def get(method: str, url: str, **kwargs: object) -> httpx2.Response:
        requested.append(url)
        return httpx2.Response(200, json={"data": [{"id": "model"}]})

    monkeypatch.setattr(_transport, "http_request", get)
    assert main(["connect", "local"]) == 0
    out = capsys.readouterr().out
    assert requested == []
    assert "Key validated" not in out
    assert "(key check skipped: auth_style 'none' sends no key)" in out


def test_connect_validates_key_and_reports_ok(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-REAL\n"))
    rc = main(["connect", "anthropic"])  # iso stubs the probe -> ok
    assert rc == 0
    out = capsys.readouterr().out
    assert "Checking the key against the provider" in out
    assert "Key validated" in out


def test_connect_warns_when_provider_rejects_key(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-BAD\n"))
    monkeypatch.setattr(
        "agent6.models.cache.probe_provider_key",
        lambda *a, **k: cache.KeyProbeResult(ok=False, status="auth_failed", detail="HTTP 401"),  # type: ignore[misc]
    )
    rc = main(["connect", "anthropic"])
    assert rc == 0  # the key is saved anyway; the warning is advisory
    err = capsys.readouterr().err
    assert "REJECTED this key" in err
    assert "HTTP 401" in err
    # The key was still written (the user may fix it later).
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-BAD"


def test_connect_no_verify_skips_the_probe(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-REAL\n"))

    def _boom(*_a: object, **_k: object) -> cache.KeyProbeResult:
        raise AssertionError("--no-verify must not probe the provider")

    monkeypatch.setattr("agent6.models.cache.probe_provider_key", _boom)
    rc = main(["connect", "anthropic", "--no-verify"])
    assert rc == 0
    assert "Checking the key" not in capsys.readouterr().out


def test_connect_non_tty_reads_plain_input_without_getpass(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # With no controlling terminal the key is one line of stdin, never getpass.
    def _boom(*_a: object, **_k: object) -> str:
        raise AssertionError("getpass must not be called on the non-TTY path")

    monkeypatch.setattr("getpass.getpass", _boom)
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-PIPED\n"))

    rc = main(["connect", "anthropic"])
    assert rc == 0
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-PIPED"


def test_connect_rejects_non_bare_key_provider_name(
    iso: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A name with a space would corrupt `[providers.<name>]`; rejected before any write.
    rc = main(["connect", "my provider"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a valid TOML bare key" in err
    # Nothing was written.
    assert not (tmp_path / "g" / "agent6" / "config.toml").exists()
    assert not (tmp_path / "g" / "agent6" / "secrets.toml").exists()


def test_connect_prints_post_entry_key_summary(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Off a terminal nothing shows while typing, so the helper prints the length captured.
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-0123456789wxyz\n"))
    rc = main(["connect", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key: 21 chars." in out
    assert "wxyz" not in out  # no fragment of the key reaches the terminal
    # The key itself is never echoed in full.
    assert "sk-ant-0123456789wxyz" not in out


def test_connect_short_key_summary_omits_tail(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    rc = main(["connect", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key: 5 chars." in out
    assert "ending" not in out


def test_connect_masked_echo_skips_summary(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # On a terminal the key shows as stars while typing, so no summary follows.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "agent6.ui.cli.connect._common.read_secret", lambda prompt="": "sk-ant-0123456789wxyz"
    )
    rc = main(["connect", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key:" not in out


def test_connect_local_endpoint_no_key(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # accept default base_url
    rc = main(["connect", "ollama"])
    assert rc == 0
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.ollama]" in gc
    # No key entered -> no secrets file required.
    assert not (tmp_path / "g" / "agent6" / "secrets.toml").is_file()


def test_model_set_and_show(
    iso: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["model", "worker", "anthropic/claude-x", "--effort", "medium"])
    assert rc == 0
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[models.worker]" in gc
    assert "claude-x" in gc
    assert "medium" in gc

    capsys.readouterr()
    rc = main(["model"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "worker" in out
    assert "claude-x" in out


def test_model_all_sets_every_role(
    iso: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # "all" is a pseudo-role: one command writes planner/worker/reviewer alike.
    rc = main(["model", "all", "anthropic/claude-x"])
    assert rc == 0
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[models.planner]" in gc
    assert "[models.worker]" in gc
    assert "[models.reviewer]" in gc
    assert gc.count("claude-x") == 3
    assert "all roles" in capsys.readouterr().out


def test_model_invalid_provider_refuses_and_rolls_back(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A role on an unconfigured provider refuses at rc 2 and leaves config.toml byte for byte.
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-FAKE\n"))
    assert main(["connect", "anthropic"]) == 0
    assert main(["model", "worker", "anthropic/good-x"]) == 0
    capsys.readouterr()
    cfg = tmp_path / "g" / "agent6" / "config.toml"
    before = cfg.read_text(encoding="utf-8")

    def _invalid(*_a: object, **_k: object) -> str:
        return "models.worker.provider: no such provider"

    monkeypatch.setattr("agent6.config.write.set_config_table", _invalid)
    assert main(["model", "worker", "anthropic/bad-x"]) == 2
    assert capsys.readouterr().err.startswith("REFUSING:")
    after = cfg.read_text(encoding="utf-8")
    assert after == before  # rolled back exactly
    assert "bad-x" not in after
    assert main(["model"]) == 0  # config still loads (not bricked)


def test_connect_config_rollback_uses_the_shared_refusal_without_saving_the_key(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-ant-FAKE\n"))

    def bad_combination(*_args: object, **_kwargs: object) -> str:
        return "bad combination"

    monkeypatch.setattr("agent6.config.write.set_config_leaves", bad_combination)

    assert main(["connect", "anthropic", "--no-verify"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("REFUSING:") and "bad combination" in err
    assert secret_store.resolve_api_key("anthropic", None) is None


def test_connect_config_rollback_precedes_oauth_sign_in(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The provider block is validated and written before the ChatGPT sign-in stores any token."""
    signed_in: list[str] = []

    def bad_combination(*_args: object, **_kwargs: object) -> str:
        return "bad combination"

    def _sign_in(name: str, *_args: object) -> int:
        signed_in.append(name)
        return 0

    monkeypatch.setattr("agent6.config.write.set_config_leaves", bad_combination)
    monkeypatch.setattr("agent6.ui.cli.connect._sign_in", _sign_in)

    assert main(["connect", "chatgpt"]) == 2
    assert "bad combination" in capsys.readouterr().err
    assert signed_in == []


def test_model_rejects_unknown_role(iso: pathlib.Path) -> None:
    # argparse `choices` validates the role positional (and feeds argcomplete).
    with pytest.raises(SystemExit) as exc:
        main(["model", "bogus", "anthropic", "claude-x"])
    assert exc.value.code == 2


def _models_stub(models: list[str]) -> Callable[..., list[str]]:
    """A typed list_models stand-in (strict pyright rejects bare lambdas here)."""

    def _list(*_a: object, **_k: object) -> list[str]:
        return models

    return _list


def _key_stub(key: str | None) -> Callable[..., str | None]:
    """A typed resolve_api_key stand-in returning a fixed key (or None)."""

    def _resolve(*_a: object, **_k: object) -> str | None:
        return key

    return _resolve


def test_model_piped_without_model_lists_the_catalog(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No tty and no model named: a listing, one id per line at exit 0, with the hint on stderr.
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.models.cache.list_models", _models_stub(["claude-a", "claude-b"]))
    rc = main(["model", "worker", "anthropic"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == "claude-a\nclaude-b\n"
    assert "set one with: agent6 model worker anthropic/<model>" in captured.err
    # Nothing was written: the listing never touches config.
    assert "[models.worker]" not in (tmp_path / "g" / "agent6" / "config.toml").read_text(
        encoding="utf-8"
    )


def test_model_set_warns_when_the_provider_has_no_key(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A configured but keyless provider is valid config; the warning names the first run's refusal.
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("agent6.secret_store.resolve_api_key", _key_stub(None))
    rc = main(["model", "worker", "anthropic/claude-x"])
    assert rc == 0
    err = capsys.readouterr().err
    assert err.startswith("[agent6] WARNING: provider 'anthropic' has no stored API key")
    assert "agent6 connect" in err


def test_model_set_stays_quiet_when_the_key_resolves(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("agent6.secret_store.resolve_api_key", _key_stub("sk-x"))
    rc = main(["model", "worker", "anthropic/claude-x"])
    assert rc == 0
    assert "note:" not in capsys.readouterr().err


def test_model_set_keeps_an_openrouter_slug_on_the_roles_provider(
    iso: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A slash inside a model id splits only when its head names a configured provider."""
    config = tmp_path / "g" / "agent6" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.openrouter]\napi_format = "openai"\nbase_url = "https://x/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "old"\n',
        encoding="utf-8",
    )

    assert main(["model", "worker", "moonshotai/kimi-k2.6"]) == 0
    text = config.read_text(encoding="utf-8")
    assert 'provider = "openrouter"' in text
    assert 'model = "moonshotai/kimi-k2.6"' in text


def test_model_piped_unknown_provider_errors(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.models.cache.list_models", _models_stub([]))
    rc = main(["model", "worker", "nosuch"])
    assert rc == 2
    assert "name one as provider/model" in capsys.readouterr().err


def test_model_stdout_piped_lists_even_with_a_tty_stdin(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Stdin stays a tty while stdout is piped: the listing triggers on the piped stdout.
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("agent6.models.cache.list_models", _models_stub(["claude-a"]))
    rc = main(["model", "worker", "anthropic"])
    assert rc == 0
    assert capsys.readouterr().out == "claude-a\n"


def test_model_piped_without_provider_errors_without_prompt_dump(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The provider prompt is interactive-only; piped, it is a listing.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["model", "worker"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "no provider given" in captured.err
    assert "Connected providers" not in captured.out


def test_model_whitespace_route_names_the_value(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["model", "worker", "   "]) == 2
    assert "'   ': no model id" in capsys.readouterr().err


def test_model_piped_listing_notes_an_ignored_thinking_flag(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.models.cache.list_models", _models_stub(["claude-a"]))
    rc = main(["model", "worker", "anthropic", "--effort", "high"])
    assert rc == 0
    assert "--effort ignored" in capsys.readouterr().err


def test_model_aborts_without_provider(iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Role given, provider omitted, none connected: refuse instead of writing bad config.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert main(["model", "worker"]) == 2


def test_model_interactive_prefill(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # The operator picks the provider by default and the model by number from a mocked live list.
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )

    def fake_list_models(*a: object, **k: object) -> list[str]:
        return ["claude-a", "claude-b"]

    monkeypatch.setattr("agent6.models.cache.list_models", fake_list_models)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)  # interactive = both ttys
    answers = iter(["", "2"])  # provider default, then model #2
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    rc = main(["model", "worker"])
    assert rc == 0
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[models.worker]" in gc
    assert "claude-b" in gc


def test_model_all_interactive_prompts_once(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # "all" prompts ONCE for provider/model (not per role), then applies to each.
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )

    def _models(*_a: object, **_k: object) -> list[str]:
        return ["claude-a", "claude-b"]

    monkeypatch.setattr("agent6.models.cache.list_models", _models)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)  # interactive = both ttys
    calls = {"n": 0}
    answers = iter(["", "2"])  # provider default, model #2: once, not 3x

    def _input(prompt: str = "") -> str:
        calls["n"] += 1
        return next(answers)

    monkeypatch.setattr("builtins.input", _input)
    rc = main(["model", "all"])
    assert rc == 0
    assert calls["n"] == 2  # one provider prompt + one model prompt, total
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    for role in ("planner", "worker", "reviewer"):
        assert f"[models.{role}]" in gc
    assert gc.count("claude-b") == 3


def test_model_repo_scope_writes_repo(iso: pathlib.Path, tmp_path: pathlib.Path) -> None:
    rc = main(["model", "reviewer", "anthropic/claude-o", "--repo"])
    assert rc == 0
    repo_cfg = (paths.state_dir(tmp_path) / "config.toml").read_text(encoding="utf-8")
    assert "[models.reviewer]" in repo_cfg


def test_config_fill_writes_global(iso: pathlib.Path, tmp_path: pathlib.Path) -> None:
    rc = main(["config", "fill"])
    assert rc == 0
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[sandbox]" in gc
    assert "[budget]" in gc


def test_config_show_runs(iso: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "[sandbox]" in out
    assert "source:" in out


def test_config_path_runs(iso: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "path"]) == 0
    out = capsys.readouterr().out
    assert "global config" in out
    assert "secrets" in out


def _grant_jwt() -> str:
    import base64
    import json

    claims = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acct-7",
            "chatgpt_plan_type": "plus",
        }
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"h.{payload}.s"


@pytest.mark.parametrize("filtered", [False, True])
def test_connect_status_lists_local_credentials_without_secrets(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filtered: bool,
) -> None:
    """Status checks in config order, filters names and never exposes or writes credentials."""
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.keyed]\napi_format = "anthropic"\napi_key_env = "CST_API_KEY"\n'
        '[providers.chatgpt]\napi_format = "chatgpt"\n'
        '[providers.legacy]\napi_format = "chatgpt"\n'
        '[providers.issuer]\napi_format = "openai"\n'
        '[providers.issuer.oauth]\nissuer = "https://login.example"\n'
        'client_id = "agent6"\nauthorization_endpoint = "https://login.example/auth"\n'
        'token_endpoint = "https://login.example/token"\n'
        '[providers.claude]\napi_format = "claude_code"\n'
        '[providers.stored]\napi_format = "anthropic"\n'
        '[providers.command]\napi_format = "anthropic"\ntoken_command = ["mint-token"]\n'
        '[providers.local]\napi_format = "openai"\nauth_style = "none"\n'
        '[providers.unset]\napi_format = "anthropic"\napi_key_env = "CST_UNSET_KEY"\n'
    )
    monkeypatch.setenv("CST_API_KEY", "sk-env-secret")
    monkeypatch.delenv("CST_UNSET_KEY", raising=False)
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    grant = oauth.stored_grant_for(
        _oauth_protocol.TokenGrant(_grant_jwt(), "refresh-secret", 42 * 60),
        ChatGPTProviderEntry(api_format="chatgpt"),
        now=1000.0,
    )
    secret_store.save_oauth_grant("chatgpt", grant)
    secret_store.save_secret("legacy", "", extra={"oauth_access_token": "legacy-secret"})
    secret_store.save_secret("stored", "sk-stored-secret")
    before = (config.read_bytes(), paths.secrets_path().read_bytes())

    checked: list[str] = []

    def request(method: str, url: str, **kwargs: object) -> httpx2.Response:
        checked.append(url)
        assert method == "GET" and kwargs["timeout"] == 7.0
        return httpx2.Response(
            200, json={"data": [], "models": []}, request=httpx2.Request(method, url)
        )

    def run(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        checked.append(argv[0])
        assert kwargs["timeout"] == 7.0
        if argv == ["mint-token"]:
            return subprocess.CompletedProcess(argv, 0, stdout="command-secret\n", stderr="")
        assert argv == ["claude", "auth", "status", "--json"]
        return subprocess.CompletedProcess(argv, 0, stdout=b'{"loggedIn": true}', stderr=b"")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("status must not prompt or read the model cache")

    monkeypatch.setattr(cache, "probe_provider_key", _REAL_PROBE)
    monkeypatch.setattr(_transport, "http_request", request)
    monkeypatch.setattr(cache, "list_models", forbidden)
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr("getpass.getpass", forbidden)
    assert main(["connect", "--status", *(["chatgpt"] if filtered else [])]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    rows = [line.split() for line in lines[1:] if line.strip()]
    expected = [
        ["keyed", "anthropic", "CST_API_KEY", "ok"],
        ["chatgpt", "chatgpt", "ChatGPT", "sign-in", "ok", "acct-7", "expires", "in", "42", "min"],
        ["legacy", "chatgpt", "ChatGPT", "sign-in", "legacy"],
        ["issuer", "openai", "login.example", "missing"],
        ["claude", "claude_code", "none", "ok"],
        ["stored", "anthropic", "secrets.toml", "ok"],
        ["command", "anthropic", "token", "command", "ok"],
        ["local", "openai", "none", "ok"],
        ["unset", "anthropic", "CST_UNSET_KEY", "key", "unset"],
    ]
    for row, live in zip(
        expected,
        (
            "verified",
            "verified",
            "skipped: legacy",
            "skipped: missing",
            "verified",
            "verified",
            "verified",
            "verified",
            "skipped: key unset",
        ),
        strict=True,
    ):
        row.extend(live.split())
    assert rows == ([expected[1]] if filtered else expected)
    endpoints = [
        "https://api.anthropic.com/v1/models",
        "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
        "claude",
        "https://api.anthropic.com/v1/models",
        "mint-token",
        "https://api.anthropic.com/v1/models",
        "https://api.openai.com/v1/models",
    ]
    assert checked == ([endpoints[1]] if filtered else endpoints)
    for secret in (
        "sk-env-secret",
        "sk-stored-secret",
        _grant_jwt(),
        "refresh-secret",
        "legacy-secret",
        "command-secret",
    ):
        assert secret not in captured.out + captured.err
    assert (config.read_bytes(), paths.secrets_path().read_bytes()) == before


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("rebound", "rebound"),
        ("contradiction", "account mismatch"),
        ("expired", "ok acct-7 expired"),
        ("unrefreshable", "expired acct-7"),
    ],
)
def test_connect_status_names_bound_grant_causes(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
    expected: str,
) -> None:
    """Status shares binding and account checks; an expired refreshable grant remains usable."""
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[providers.work]\napi_format = "chatgpt"\n')
    monkeypatch.setattr(time, "time", lambda: 5000.0)
    grant = oauth.stored_grant_for(
        _oauth_protocol.TokenGrant(_grant_jwt(), "refresh-secret", 3600),
        ChatGPTProviderEntry(api_format="chatgpt"),
        now=1000.0,
    )
    secret_store.save_oauth_grant(
        "work",
        dataclasses.replace(
            grant,
            binding="other" if state == "rebound" else grant.binding,
            account_id="other-account" if state == "contradiction" else grant.account_id,
            refresh_token="" if state == "unrefreshable" else grant.refresh_token,
        ),
    )
    requests: list[str] = []

    def request(method: str, url: str, **kwargs: object) -> httpx2.Response:
        requests.append(url)
        assert state == "expired" and method == "GET"
        return httpx2.Response(401, request=httpx2.Request(method, url))

    monkeypatch.setattr(_transport, "http_request", request)
    assert main(["connect", "--status", "work"]) == 0
    output = capsys.readouterr()
    assert expected in " ".join(output.out.split())
    live = f"skipped: {expected}" if state != "expired" else "refused: HTTP 401"
    assert output.out.splitlines()[1].endswith(live)
    assert len(requests) == (1 if state == "expired" else 0)
    assert _grant_jwt() not in output.out + output.err
    assert "refresh-secret" not in output.out + output.err


@pytest.mark.parametrize("option", ["--logout", "--flow", "--verify", "--no-verify", "--repo"])
def test_connect_status_refuses_sign_in_options(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str], option: str
) -> None:
    """Status refuses every write or sign-in option, including an explicit default flow."""
    assert main(["connect", "--status", option, *(["auto"] if option == "--flow" else [])]) == 2
    error = capsys.readouterr().err
    assert "--status" in error and option in error and "cannot combine" in error


def test_connect_status_reads_explicit_config(
    iso: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-only status mode honors the same explicit config layers as model listings."""
    config = tmp_path / "extra.toml"
    config.write_text('[providers.extra]\napi_format = "openai"\nauth_style = "none"\n')

    def request(method: str, url: str, **kwargs: object) -> httpx2.Response:
        assert kwargs["headers"] == {}
        return httpx2.Response(200, json={"data": []}, request=httpx2.Request(method, url))

    monkeypatch.setattr(_transport, "http_request", request)
    assert main(["--config", str(config), "connect", "--status"]) == 0
    assert "extra openai none ok verified" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("state", ["legacy", "rebound"])
def test_logout_unbound_chatgpt_is_local_only(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
) -> None:
    """Logout cannot send a legacy or rebound bearer and keeps the provider config."""
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[providers.work]\napi_format = "chatgpt"\n')
    if state == "legacy":
        secret_store.save_secret(
            "work",
            "",
            extra={
                "oauth_access_token": "AT",
                "oauth_refresh_token": "RT",
                "oauth_expires_at": "9999999999",
            },
        )
    else:
        grant = oauth.stored_grant_for(
            _oauth_protocol.TokenGrant("AT", "RT", 3600),
            ChatGPTProviderEntry(api_format="chatgpt"),
            now=time.time(),
        )
        secret_store.save_oauth_grant("work", dataclasses.replace(grant, binding="other"))
    requests: list[str] = []

    def request(method: str, url: str, **kwargs: object) -> httpx2.Response:
        requests.append(url)
        return httpx2.Response(200)

    monkeypatch.setattr(_transport, "http_request", request)
    assert main(["connect", "work", "--logout"]) == 0
    assert requests == []
    output = capsys.readouterr()
    assert "locally only" in output.out + output.err
    assert "Revoked" not in output.out
    assert secret_store.load_secrets().get("providers", {}) == {}
    assert config.read_text() == '[providers.work]\napi_format = "chatgpt"\n'


def test_chatgpt_connect_replaces_legacy_atomically_with_bound_grant(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The published replacement is bound and contains no legacy-only fields or id token."""
    secret_store.save_secret(
        "chatgpt",
        "",
        extra={
            "oauth_access_token": "old",
            "oauth_refresh_token": "old-refresh",
            "legacy_only": "discard",
        },
    )
    secret_store.save_secret("sibling", "keep")
    old = paths.secrets_path().read_bytes()
    published: list[str] = []
    real_write = secret_store.portable.atomic_write

    def publish(path: pathlib.Path, text: str, **kwargs: object) -> None:
        if path == paths.secrets_path():
            assert path.read_bytes() == old
            published.append(text)
        real_write(path, text, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(secret_store.portable, "atomic_write", publish)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def paste(*_args: object, **_kwargs: object) -> str:
        return "code"

    def exchange(*_args: object, **_kwargs: object) -> _TokenResp:
        return _TokenResp()

    monkeypatch.setattr(_oauth, "paste_callback", paste)
    monkeypatch.setattr(_oauth_protocol, "post_form", exchange)
    assert main(["connect", "chatgpt", "--no-verify"]) == 0
    assert len(published) == 1 and "legacy_only" not in published[0]
    grant = secret_store.load_oauth_grant("chatgpt")
    assert grant is not None
    assert grant.binding == oauth.binding_fingerprint(ChatGPTProviderEntry(api_format="chatgpt"))
    assert secret_store.resolve_api_key("sibling", None) == "keep"
    output = capsys.readouterr().out
    assert "(0600), bound to [providers.chatgpt]" in output
    assert "plus plan" in output and "Data controls" in output
    assert "validated" not in output and "Checking" not in output


def test_logout_without_config_reports_local_removal(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A legacy grant without a provider entry can only be removed locally."""
    secret_store.save_secret("chatgpt", "", extra={"oauth_access_token": "old"})
    assert main(["connect", "chatgpt", "--logout"]) == 0
    output = capsys.readouterr()
    assert "locally only" in output.out
    assert "Revoked" not in output.out


class _TokenResp:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict[str, object]:
        return {
            "access_token": _grant_jwt(),
            "refresh_token": "RT1",
            "expires_in": 3600,
            "id_token": "",
        }


def test_connect_chatgpt_paste_flow_signs_in_and_writes_config(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The headless ChatGPT paste flow signs in and writes the config.

    Paste the callback URL, exchange the code, store tokens 0600, write the provider block, print
    the training-data notice; no subprocess, no browser.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.ui.cli._oauth.secrets.token_urlsafe", lambda n=24: "STATE1")
    exchanges: list[dict[str, str]] = []

    def fake_post(url: str, data: dict[str, str], timeout_s: float, **_kw: object) -> _TokenResp:
        exchanges.append(data)
        return _TokenResp()

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_post)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("http://localhost:1455/auth/callback?code=C1&state=STATE1\n")
    )
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    execs: list[object] = []

    def record_run(*args: object, **kwargs: object) -> None:
        execs.append(args)

    monkeypatch.setattr("subprocess.run", record_run)

    rc = main(["connect", "chatgpt"])
    assert rc == 0
    assert opened == [] and execs == []
    assert exchanges[0]["grant_type"] == "authorization_code"
    assert exchanges[0]["code"] == "C1"

    sp = tmp_path / "g" / "agent6" / "secrets.toml"
    assert stat.S_IMODE(sp.stat().st_mode) == 0o600
    tokens = secret_store.load_oauth_grant("chatgpt")
    assert tokens is not None and tokens.account_id == "acct-7"

    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.chatgpt]" in gc and 'api_format = "chatgpt"' in gc
    out = capsys.readouterr().out
    assert "Signed in (plus plan)" in out
    assert "Data controls" in out and "never sends feedback" in out


def test_connect_claude_writes_the_format_only_and_stores_no_secret(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`connect claude` writes `api_format` and no secret.

    A signed-out binary warns with the remedy; `--logout` refuses, since agent6 holds no Claude Code
    credential.
    """

    def signed_in(binary: str) -> str | None:
        return None

    def signed_out(binary: str) -> str | None:
        return "not signed in; run `claude auth login`"

    monkeypatch.setattr("agent6.providers.claude_code.login_status", signed_in)
    assert main(["connect", "claude"]) == 0
    gc = (tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.claude]" in gc and 'api_format = "claude_code"' in gc
    assert not (tmp_path / "g" / "agent6" / "secrets.toml").exists()
    assert "Claude Code (`claude` on PATH): signed in." in capsys.readouterr().out

    monkeypatch.setattr("agent6.providers.claude_code.login_status", signed_out)
    assert main(["connect", "claude"]) == 0
    err = capsys.readouterr().err
    assert "WARNING: not signed in; run `claude auth login`" in err and "not usable yet" in err

    assert main(["connect", "claude", "--logout"]) == 2
    assert "claude auth logout" in capsys.readouterr().err


def test_connect_chatgpt_state_mismatch_refuses(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.ui.cli._oauth.secrets.token_urlsafe", lambda n=24: "STATE1")
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("http://localhost:1455/auth/callback?code=C1&state=EVIL\n")
    )
    rc = main(["connect", "chatgpt"])
    assert rc == 2
    assert secret_store.load_oauth_grant("chatgpt") is None


def test_oauth_callback_server_round_trip() -> None:
    """The localhost receiver answers the redirect, hands over the code and 404s every other path.

    A state-mismatch hit is a 400, not a capture.
    """
    import urllib.error
    import urllib.request

    srv = _oauth.CallbackServer("S1", port=0)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        with pytest.raises(urllib.error.HTTPError) as e404:
            urllib.request.urlopen(f"{base}/favicon.ico", timeout=5)
        assert e404.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e400:
            urllib.request.urlopen(f"{base}/auth/callback?code=X&state=EVIL", timeout=5)
        assert e400.value.code == 400
        assert srv.wait(timeout_s=0.3) is None
        with urllib.request.urlopen(f"{base}/auth/callback?code=OK&state=S1", timeout=5) as resp:
            assert resp.status == 200
        assert srv.wait(timeout_s=5.0) == "OK"
    finally:
        srv.close()


def test_connect_logout_revokes_and_removes_tokens(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--logout revokes a ChatGPT grant at the issuer and removes the secrets entry."""
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[providers.chatgpt]\napi_format = "chatgpt"\n')
    secret_store.save_oauth_grant(
        "chatgpt",
        oauth.stored_grant_for(
            _oauth_protocol.TokenGrant("AT", "RT", 3600),
            ChatGPTProviderEntry(api_format="chatgpt"),
            now=time.time(),
        ),
    )
    revoked: list[dict[str, object]] = []

    class _RevokeResp:
        status_code = 200
        text = ""

    def fake_request(
        method: str, url: str, *, json: dict[str, object] | None = None, **_kw: object
    ) -> _RevokeResp:
        revoked.append({"url": url, **(json or {})})
        return _RevokeResp()

    monkeypatch.setattr("agent6.providers._transport.http_request", fake_request)
    rc = main(["connect", "chatgpt", "--logout"])
    assert rc == 0
    assert revoked[0]["url"] == "https://auth.openai.com/oauth/revoke"
    assert revoked[0]["token"] == "RT" and revoked[0]["token_type_hint"] == "refresh_token"
    assert secret_store.load_oauth_grant("chatgpt") is None
    assert "Removed stored credentials" in capsys.readouterr().out

    rc = main(["connect", "chatgpt", "--logout"])
    assert rc == 0
    assert "No stored credentials" in capsys.readouterr().out


def test_connect_chatgpt_headless_terminal_uses_the_device_flow(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tty without a display signs in by device code: no browser, no localhost server.

    No paste.
    """
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    from agent6.providers import _oauth_protocol, chatgpt_oauth

    def fake_start(issuer: str, client_id: str) -> chatgpt_oauth.DeviceAuth:
        return chatgpt_oauth.DeviceAuth(device_auth_id="da_1", user_code="AB-12", interval_s=5.0)

    def fake_poll(
        issuer: str, client_id: str, device: chatgpt_oauth.DeviceAuth, *, provider: str
    ) -> _oauth_protocol.TokenGrant:
        assert provider == "chatgpt"
        return _oauth_protocol.TokenGrant(_grant_jwt(), "RT9", 3600.0)

    monkeypatch.setattr("agent6.providers.chatgpt_oauth.start_device_auth", fake_start)
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.poll_device_auth", fake_poll)

    def no_input(prompt: str = "") -> str:
        pytest.fail("device path must not prompt for a paste")

    monkeypatch.setattr("builtins.input", no_input)
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    rc = main(["connect", "chatgpt"])
    assert rc == 0 and opened == []
    tokens = secret_store.load_oauth_grant("chatgpt")
    assert tokens is not None and tokens.refresh_token == "RT9"
    out = capsys.readouterr().out
    assert "enter the code:  AB-12" in out and "/codex/device" in out


def test_connect_chatgpt_format_under_another_name_signs_in_as_itself(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chatgpt-format provider under another name signs in and stores tokens as itself."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    from agent6.providers import _oauth_protocol, chatgpt_oauth

    def fake_start(issuer: str, client_id: str) -> chatgpt_oauth.DeviceAuth:
        return chatgpt_oauth.DeviceAuth(device_auth_id="da_1", user_code="AB-12", interval_s=5.0)

    told: list[str] = []

    def fake_poll(
        issuer: str, client_id: str, device: chatgpt_oauth.DeviceAuth, *, provider: str
    ) -> _oauth_protocol.TokenGrant:
        told.append(provider)
        return _oauth_protocol.TokenGrant(_grant_jwt(), "RT9", 3600.0)

    monkeypatch.setattr("agent6.providers.chatgpt_oauth.start_device_auth", fake_start)
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.poll_device_auth", fake_poll)

    def api_format(prompt: str = "") -> str:
        return "chatgpt"

    def no_browser(_url: str) -> None:
        return

    monkeypatch.setattr("builtins.input", api_format)
    monkeypatch.setattr("webbrowser.open", no_browser)

    assert main(["connect", "codex"]) == 0
    assert told == ["codex"]
    tokens = secret_store.load_oauth_grant("codex")
    assert tokens is not None and tokens.refresh_token == "RT9"


@pytest.mark.parametrize("failure", ["disabled", "start-error", "poll-error", "interrupt"])
def test_connect_chatgpt_device_failure_falls_back_to_paste(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    def start(issuer: str, client_id: str) -> chatgpt_oauth.DeviceAuth | None:
        if failure == "disabled":
            return None
        if failure == "start-error":
            raise types.ProviderError("unavailable")
        return chatgpt_oauth.DeviceAuth("device", "AB-CD", 5)

    def poll(*_args: object, **_kwargs: object) -> _oauth_protocol.TokenGrant:
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise types.ProviderError("poll refused")

    def no_listener(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a failed device flow must fall back to paste, not a listener")

    monkeypatch.setattr(_oauth, "CallbackServer", no_listener)
    monkeypatch.setattr(chatgpt_oauth, "start_device_auth", start)
    monkeypatch.setattr(chatgpt_oauth, "poll_device_auth", poll)
    monkeypatch.setattr("agent6.ui.cli._oauth.secrets.token_urlsafe", lambda n=24: "STATE1")

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _TokenResp:
        return _TokenResp()

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_post)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("http://localhost:1455/auth/callback?code=C1&state=STATE1\n")
    )
    rc = main(["connect", "chatgpt"])
    assert rc == 0
    assert secret_store.load_oauth_grant("chatgpt") is not None


def test_connect_eof_at_the_api_format_prompt_says_why(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """EOF at the api_format prompt names the abort: a custom provider name has no preset format."""

    def _eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    rc = main(["connect", "mycustom"])
    assert rc == 2
    assert "no input." in capsys.readouterr().err


_WORKER_ONLY = """\
[providers.anthropic]
api_format = "anthropic"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"
"""


def test_a_role_that_falls_back_to_the_worker_says_so(
    iso: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unset planner or reviewer shows the worker's model with an origin naming the fallback."""
    gpath = tmp_path / "g" / "agent6" / "config.toml"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(_WORKER_ONLY, encoding="utf-8")

    assert main(["model"]) == 0
    lines = capsys.readouterr().out.splitlines()
    worker = next(line for line in lines if line.strip().startswith("worker"))
    planner = next(line for line in lines if line.strip().startswith("planner"))
    assert "anthropic/claude-sonnet-4-5" in worker and worker.endswith("[global]")
    assert "anthropic/claude-sonnet-4-5" in planner and planner.endswith("[worker's, global]")


def test_a_number_outside_the_model_list_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """A number past the picker's list is refused, not written as a model id."""
    typed = ["7", "2"]

    def models(config_paths: tuple[pathlib.Path, ...], provider: str) -> list[str]:
        return ["a", "b"]

    def answer(prompt: str) -> str:
        return typed.pop(0)

    monkeypatch.setattr(modelmod, "_models_for", models)
    monkeypatch.setattr(_common, "safe_input", answer)
    assert modelmod._prompt_for_model((), "p") is None  # pyright: ignore[reportPrivateUsage]
    assert "no model 7: the list has 2" in capsys.readouterr().err
    assert modelmod._prompt_for_model((), "p") == "b"  # pyright: ignore[reportPrivateUsage]

    # The verb prints the picker's one refusal, never a second line contradicting it.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    typed[:] = ["7"]
    # A provider name alone is that provider with the model still to pick.
    cfg = tmp_path / "custom.toml"
    cfg.write_text('[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8")
    assert main(["--config", str(cfg), "model", "worker", "anthropic"]) == 2
    err = capsys.readouterr().err
    assert err.count("ERROR") == 1 and "no model 7" in err
