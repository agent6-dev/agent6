from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tomllib

import pytest

from agent6 import paths
from agent6.ui.cli import entry


def _invoke(argv: list[str]) -> int:
    try:
        return entry.cli_main(["--allow-root", *argv])
    except SystemExit as exc:
        return int(exc.code or 0)


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.chdir(root)
    return root


def test_fold_copies_explicit_leaves_and_keeps_repo_values(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fold carries explicit global leaves without copying defaults or replacing repo values."""
    global_config = _write(
        paths.global_config_path(),
        '[providers.demo]\nbackend = "openai"\nbase_url = "http://localhost:11434/v1"\n'
        '[models.default]\nmodel = "demo/example"\n'
        "[budget]\nmax_usd = 5\n"
        "[compaction]\nsummary_max_tokens = 1024\n"
        '[[review.reviewers]]\nfocus = "security"\n',
    )
    before = global_config.read_bytes()
    target = _write(paths.repo_config_path(repo), "# Keep the repo limit.\n[budget]\nmax_usd = 9\n")

    assert _invoke(["pack", "--repo-config"]) == 0

    expected = {
        "providers": {"demo": {"backend": "openai", "base_url": "http://localhost:11434/v1"}},
        "models": {"default": {"model": "demo/example"}},
        "budget": {"max_usd": 9},
        "compaction": {"summary_max_tokens": 1024},
        "review": {"reviewers": [{"focus": "security"}]},
    }
    assert tomllib.loads(target.read_text()) == expected
    assert tomllib.loads((repo / ".agent6/repo/config.toml").read_text()) == expected
    assert "# Keep the repo limit." in target.read_text()
    assert global_config.read_bytes() == before
    lines = capsys.readouterr().out.splitlines()
    moved = [line for line in lines if line.startswith("Moved to repo config: ")]
    assert moved == [
        "Moved to repo config: compaction.summary_max_tokens",
        "Moved to repo config: models.default.model",
        "Moved to repo config: providers.demo.backend",
        "Moved to repo config: providers.demo.base_url",
        "Moved to repo config: review.reviewers",
    ]


@pytest.mark.parametrize(
    "repo_settings",
    [
        '[skills.state]\nshared = "disabled"\nlocal = "always"\n',
        '[skills]\nstate = { shared = "disabled", local = "always" }\n',
    ],
)
def test_fold_merges_mapping_entries_and_keeps_the_repo_list(
    repo: pathlib.Path, repo_settings: str
) -> None:
    """Mappings merge by key; a repo list replaces the global list."""
    _write(
        paths.global_config_path(),
        '[skills.state]\nshared = "always"\n"lint.md" = "disabled"\n'
        '[sandbox]\nextra_read_paths = ["/global"]\n',
    )
    target = _write(
        paths.repo_config_path(repo), repo_settings + '[sandbox]\nextra_read_paths = ["/repo"]\n'
    )

    assert _invoke(["pack", "--repo-config"]) == 0

    assert tomllib.loads(target.read_text()) == {
        "skills": {"state": {"shared": "disabled", "local": "always", "lint.md": "disabled"}},
        "sandbox": {"extra_read_paths": ["/repo"]},
    }


def test_repo_config_repack_omits_global_settings_but_keeps_private_secrets(
    repo: pathlib.Path,
) -> None:
    """A repack removes global settings files but keeps private secrets."""
    _write(paths.global_config_path(), "[budget]\nmax_usd = 5\n")
    _write(paths.global_config_dir() / "ui.toml", '[tui]\ntheme = "dark"\n')
    _write(paths.secrets_path(), '[demo]\napi_key = "private"\n')
    assert _invoke(["pack"]) == 0

    assert _invoke(["pack", "--repo-config"]) == 0
    assert _invoke(["pack", "--repo-config"]) == 0

    directory = repo / ".agent6"
    manifest = json.loads((directory / "pack.json").read_text())
    assert set(manifest["files"]) == {"config/secrets.toml", "repo/config.toml"}
    assert not (directory / "config/config.toml").exists()
    assert not (directory / "config/ui.toml").exists()
    assert (directory / "config/secrets.toml").read_text() == '[demo]\napi_key = "private"\n'
    assert (directory / "config/secrets.toml").stat().st_mode & 0o777 == 0o600
    assert (directory / "config").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("home_has_config", [False, True])
def test_unpack_folded_config_leaves_global_settings_untouched_and_shows_repo_origins(
    repo: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    home_has_config: bool,
) -> None:
    """A folded pack restores repo settings without creating or changing global settings."""
    _write(
        paths.global_config_path(),
        '[budget]\nmax_usd = 5\n[skills.state]\n"lint.md" = "disabled"\n',
    )
    _write(paths.global_config_dir() / "ui.toml", '[tui]\ntheme = "dark"\n')
    assert _invoke(["pack", "--repo-config"]) == 0
    destination = tmp_path / "destination"
    destination.mkdir()
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    shutil.move(repo / ".agent6", destination / ".agent6")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "new-config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "new-state"))
    monkeypatch.chdir(destination)
    global_config = paths.global_config_path()
    ui_config = paths.global_config_dir() / "ui.toml"
    if home_has_config:
        _write(global_config, "[budget]\nmax_usd = 99\n")
        _write(ui_config, '[tui]\ntheme = "light"\n')
    capsys.readouterr()

    assert _invoke(["unpack"]) == 0

    assert "No global config.toml or ui.toml written." in capsys.readouterr().out
    assert tomllib.loads(paths.repo_config_path(destination).read_text()) == {
        "budget": {"max_usd": 5},
        "skills": {"state": {"lint.md": "disabled"}},
    }
    if home_has_config:
        assert global_config.read_text() == "[budget]\nmax_usd = 99\n"
        assert ui_config.read_text() == '[tui]\ntheme = "light"\n'
    else:
        assert not global_config.exists()
        assert not ui_config.exists()
    assert _invoke(["config", "show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["budget.max_usd"]["value"] == 5
    assert shown["budget.max_usd"]["source"] == "repo"
    assert shown['skills.state."lint.md"']["source"] == "repo"


@pytest.mark.parametrize("section", ["sandbox", "presets.unused.sandbox"])
def test_invalid_fold_keeps_repo_file_unchanged(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str], section: str
) -> None:
    """A bad global leaf cannot leave a partially folded repo config or a pack."""
    _write(
        paths.global_config_path(),
        f'[budget]\nmax_usd = 5\n[{section}]\nrun_commands = "invalid"\n',
    )
    target = _write(paths.repo_config_path(repo), "# Keep this exact file.\n[web]\nport = 8001\n")
    before = target.read_bytes()

    assert _invoke(["pack", "--repo-config"]) == 2

    output = capsys.readouterr()
    assert "sandbox.run_commands" in output.err
    assert target.read_bytes() == before
    assert not (repo / ".agent6").exists()
    assert "Moved to repo config:" not in output.out


def test_fold_refuses_without_the_config_writer_lock(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fold must not write when a rollback cannot be protected from another writer."""
    _write(paths.global_config_path(), "[budget]\nmax_usd = 5\n")
    target = _write(paths.repo_config_path(repo), "[web]\nport = 8001\n")
    target.with_name(target.name + ".lock").symlink_to(repo / "missing")
    before = target.read_bytes()

    assert _invoke(["pack", "--repo-config"]) == 2

    assert "cannot take writer lock" in capsys.readouterr().err
    assert target.read_bytes() == before
    assert not (repo / ".agent6").exists()
