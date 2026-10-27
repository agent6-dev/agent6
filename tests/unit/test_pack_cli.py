from __future__ import annotations

import json
import pathlib
import socket
import subprocess

import pytest

from agent6 import pack, paths
from agent6.ui.cli import entry


def _invoke(argv: list[str]) -> int:
    try:
        return entry.cli_main(["--allow-root", *argv])
    except SystemExit as exc:
        return int(exc.code or 0)


def test_pack_and_unpack_cli(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("kept")
    state = paths.state_dir(tmp_path)
    state.mkdir(parents=True)
    (state / "note").write_text("note")
    assert _invoke(["pack"]) == 0
    assert ".agent6/" in (tmp_path / ".gitignore").read_text().splitlines()
    assert (tmp_path / ".gitignore").read_text().startswith("kept\n")
    assert ".gitignore" in capsys.readouterr().out
    assert _invoke(["pack"]) == 0
    assert (tmp_path / ".gitignore").read_text().count(".agent6/") == 1
    (state / "note").unlink()
    assert _invoke(["unpack"]) == 0
    out = capsys.readouterr().out
    assert "moved" in out and "kept" in out
    assert str(tmp_path) in out
    assert (state / "note").read_text() == "note"
    assert not (tmp_path / ".agent6").exists()


def test_pack_refuses_negated_ignore_rule(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ignore entry overridden by a negation must not let pack publish credentials."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".agent6/\n!.agent6/\n")
    paths.global_config_dir().mkdir(parents=True)
    paths.secrets_path().write_text("private")
    assert _invoke(["pack"]) == 2
    err = capsys.readouterr().err
    assert ".gitignore" in err and ".agent6/" in err and "ignored" in err
    assert not (tmp_path / ".agent6").exists()


def test_pack_refuses_any_tracked_pack_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tracked payload bypasses gitignore even when the manifest itself is untracked."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    state = paths.state_dir(tmp_path)
    state.mkdir(parents=True)
    (state / "note").write_text("note")
    pack.pack(tmp_path)
    subprocess.run(["git", "add", ".agent6/repo/note"], check=True)
    paths.global_config_dir().mkdir(parents=True)
    paths.secrets_path().write_text("private")
    assert _invoke(["pack"]) == 2
    err = capsys.readouterr().err
    assert "tracked" in err and ".agent6" in err
    assert not (tmp_path / ".agent6/config/secrets.toml").exists()


@pytest.mark.parametrize("argv", [["run", "task"], ["sessions", "list"]])
@pytest.mark.parametrize("foreign_field", [None, "host", "home"])
def test_present_pack_refuses_startup(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    foreign_field: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    pack.pack(tmp_path)
    file = tmp_path / ".agent6/pack.json"
    manifest = json.loads(file.read_text())
    if foreign_field is not None:
        manifest["origin"][foreign_field] = "/foreign" if foreign_field == "home" else "other-host"
    file.write_text(json.dumps(manifest))
    assert _invoke(argv) == 2
    err = capsys.readouterr().err
    assert manifest["origin"]["host"] in err
    assert manifest["origin"]["home"] in err
    assert manifest["packed_at"] in err
    assert "agent6 unpack" in err and "delete" in err


def test_foreign_pack_allows_unpack(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    pack.pack(tmp_path)
    file = tmp_path / ".agent6/pack.json"
    manifest = json.loads(file.read_text())
    manifest["origin"]["host"] = socket.gethostname() + "-other"
    file.write_text(json.dumps(manifest))
    assert _invoke(["unpack"]) == 0
    assert "re-keyed" in capsys.readouterr().out
    assert not (tmp_path / ".agent6").exists()


def test_subdirectory_resolves_pack_at_project_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert _invoke(["pack"]) == 0
    assert (tmp_path / ".agent6/pack.json").is_file()
    assert not (sub / ".agent6").exists()
