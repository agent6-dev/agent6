# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config secret target writes telemetry headers without exposing their values."""

from __future__ import annotations

import argparse
import pathlib
import stat
import tomllib

import pytest

from agent6 import paths, secret_store
from agent6.ui.cli import completers, main


def _run(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


@pytest.mark.parametrize("value", ["Bearer private-telemetry-token", "123", 'a"quoted"value'])
def test_secret_set_round_trips_without_touching_config_or_printing_the_value(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    """A secret is a literal string in a private file, not an effective-config leaf."""
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "telemetry.headers", '["Authorization"]']) == 0
    config = paths.global_config_path().read_bytes()
    capsys.readouterr()
    assert _run(["config", "set", "--secret", "telemetry.headers.Authorization", value]) == 0
    result = capsys.readouterr()
    assert value not in result.out + result.err
    path = paths.secrets_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert tomllib.loads(path.read_text())["telemetry"]["headers"] == {"Authorization": value}
    assert secret_store.telemetry_headers(("Authorization",)) == {"Authorization": value}
    assert paths.global_config_path().read_bytes() == config
    assert not paths.repo_config_path(tmp_path).exists()
    for flags in ([], ["--json"]):
        assert main(["config", "show", "telemetry", *flags]) == 0
        shown = capsys.readouterr()
        assert "Authorization" in shown.out
        assert value not in shown.out + shown.err


def test_secret_unset_preserves_other_headers_and_provider_credentials(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing one quoted header leaves its siblings and provider keys intact."""
    monkeypatch.chdir(tmp_path)
    path = secret_store.save_secret("anthropic", "private-provider-key")
    path.write_text(
        path.read_text()
        + '[telemetry.headers]\n"X.Scope" = "private-scope"\nAuthorization = "Bearer sibling"\n'
    )
    assert _run(["config", "unset", "--secret", 'telemetry.headers."X.Scope"']) == 0
    assert secret_store.telemetry_headers(("Authorization",)) == {"Authorization": "Bearer sibling"}
    assert secret_store.resolve_api_key("anthropic", None) == "private-provider-key"
    assert "X.Scope" not in tomllib.loads(path.read_text())["telemetry"]["headers"]
    result = capsys.readouterr()
    assert "private-scope" not in result.out + result.err
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_secret_unset_on_a_fresh_home_creates_no_file_or_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-op unset does not create an open config directory through the lock writer."""
    monkeypatch.chdir(tmp_path)
    assert _run(["config", "unset", "--secret", "telemetry.headers.Authorization"]) == 0
    assert not paths.secrets_path().parent.exists()


@pytest.mark.parametrize("verb", ["set", "unset"])
@pytest.mark.parametrize(
    "key",
    [
        "sandbox.network",
        "providers.anthropic.api_key",
        "telemetry.headers",
        "telemetry.headers.Authorization.extra",
        'telemetry.headers.""',
        "telemetry.headers.Content-Type",
    ],
)
def test_secret_target_refuses_other_keys_without_writes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    key: str,
) -> None:
    """The secret target cannot redirect an arbitrary config key into the secrets file."""
    monkeypatch.chdir(tmp_path)
    argv = ["config", verb, "--secret", key, *(["private-value"] if verb == "set" else [])]
    assert _run(argv) == 2
    error = capsys.readouterr().err
    assert "--secret accepts only telemetry.headers.<name>" in error
    assert "private-value" not in error
    assert not paths.secrets_path().exists()
    assert not paths.global_config_path().exists()


@pytest.mark.parametrize("value", ["", "a\r\nInjected: yes", "non-ascii-\u00e9"])
def test_secret_set_rejects_unusable_header_values_without_echoing_them(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    """An unusable HTTP header is refused before any secrets file is written."""
    monkeypatch.chdir(tmp_path)
    assert _run(["config", "set", "--secret", "telemetry.headers.Authorization", value]) == 2
    error = capsys.readouterr().err
    assert "telemetry header" in error
    if value:
        assert value not in error
    assert not paths.secrets_path().exists()


@pytest.mark.parametrize(
    ("barrier", "reason"), [("mode", "unsafe permissions"), ("link", "symlink"), ("lock", "lock")]
)
@pytest.mark.parametrize("verb", ["set", "unset"])
def test_secret_target_uses_the_secrets_files_security_checks(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    barrier: str,
    reason: str,
    verb: str,
) -> None:
    """Both edits refuse an unsafe secrets file or an unavailable lock without changing it."""
    monkeypatch.chdir(tmp_path)
    path = secret_store.save_secret("anthropic", "keep-provider-key")
    path.write_text(path.read_text() + '[telemetry.headers]\nAuthorization = "Bearer keep"\n')
    before = path.read_bytes()
    if barrier == "mode":
        path.chmod(0o644)
    elif barrier == "link":
        real = tmp_path / "real-secrets.toml"
        path.rename(real)
        path.symlink_to(real)
    else:
        path.with_name(path.name + ".lock").mkdir()
    argv = [
        "config",
        verb,
        "--secret",
        "telemetry.headers.Authorization",
        *(["private-new-value"] if verb == "set" else []),
    ]
    assert _run(argv) == 2
    error = capsys.readouterr().err
    assert reason in error
    assert "private-new-value" not in error
    assert path.read_bytes() == before


def test_secret_completion_offers_configured_and_saved_header_names_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secret completion offers quoted header keys, never ordinary keys or credential values."""
    monkeypatch.chdir(tmp_path)
    overlay = tmp_path / "headers.toml"
    overlay.write_text('[telemetry]\nheaders = ["Authorization", "X.Scope"]\n')
    path = secret_store.save_secret("anthropic", "hidden-provider-value")
    path.write_text(path.read_text() + '[telemetry.headers]\nX-Saved = "hidden-header-value"\n')
    for verb in ("set", "unset"):
        args = argparse.Namespace(config_command=verb, secret=True, config_after=[overlay])
        keys = completers._complete_config_keys("", parsed_args=args)  # pyright: ignore[reportPrivateUsage]
        assert keys == [
            'telemetry.headers."X.Scope"',
            "telemetry.headers.Authorization",
            "telemetry.headers.X-Saved",
        ]
        assert completers._complete_config_keys("telemetry.headers.A", parsed_args=args) == [
            "telemetry.headers.Authorization"
        ]  # pyright: ignore[reportPrivateUsage]
        for key in ("telemetry.headers.Authorization", "sandbox.isolation"):
            args.key = key
            assert completers._complete_config_values("", args) == []  # pyright: ignore[reportPrivateUsage]
