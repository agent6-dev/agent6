from __future__ import annotations

import errno
import hashlib
import json
import os
import pathlib
import shutil
import socket

import pytest

from agent6 import errors, pack, paths


def _write(path: pathlib.Path, text: str = "payload", mode: int = 0o640) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    os.utime(path, (1700000000, 1700000000))
    return path


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_pack_snapshot(tmp_path: pathlib.Path) -> None:
    """A pack carries global settings, skills and all repo state except locks."""
    root = tmp_path / "project"
    root.mkdir()
    cfg = paths.global_config_dir()
    for name in ("config.toml", "ui.toml", "secrets.toml"):
        _write(cfg / name, name)
    _write(paths.data_dir() / "skills/demo/run.sh", mode=0o750)
    state = paths.state_dir(root)
    for name in ("config.toml", "sessions/runs/s1/journal.jsonl", "memory/a.md", "other"):
        _write(state / name)
    _write(state / "locks/writer", "do not carry")
    _write(paths.cache_dir() / "cache", "do not carry")
    _write(state / "pack-base.json", json.dumps({"repo/other": _digest("base")}))

    pack.pack(root)
    packed = root / ".agent6"
    manifest = json.loads((packed / "pack.json").read_text())
    assert manifest["version"] == 1
    assert manifest["packed_at"].endswith(("Z", "+00:00"))
    assert manifest["origin"] == {
        "host": socket.gethostname(),
        "home": str(paths.effective_user().home),
        "repo_id": paths.repo_id(root),
        "workspace": str(root),
    }
    assert set(manifest["files"]) == {
        "config/config.toml",
        "config/ui.toml",
        "config/secrets.toml",
        "skills/demo/run.sh",
        "repo/config.toml",
        "repo/sessions/runs/s1/journal.jsonl",
        "repo/memory/a.md",
        "repo/other",
        "repo/pack-base.json",
    }
    for name, facts in manifest["files"].items():
        assert facts["sha256"] == hashlib.sha256((packed / name).read_bytes()).hexdigest()
        assert facts["mtime"] == "2023-11-14T22:13:20Z"
        assert facts["base"] == (_digest("base") if name == "repo/other" else None)
    assert (packed / "config").stat().st_mode & 0o777 == 0o700
    assert (packed / "config/secrets.toml").stat().st_mode & 0o777 == 0o600
    assert (packed / "skills/demo/run.sh").stat().st_mode & 0o777 == 0o750
    assert (packed / "repo/other").stat().st_mode & 0o777 == 0o640
    assert (cfg / "secrets.toml").exists()
    assert (state / "other").exists()


def test_pack_replaces_own_snapshot_and_refuses_foreign(tmp_path: pathlib.Path) -> None:
    file = _write(paths.state_dir(tmp_path) / "note", "first")
    pack.pack(tmp_path)
    file.write_text("second")
    pack.pack(tmp_path)
    packed = tmp_path / ".agent6"
    assert (packed / "repo/note").read_text() == "second"
    manifest_path = packed / "pack.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["origin"]["home"] = "/foreign/home"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(errors.OperatorError, match="unpack first"):
        pack.pack(tmp_path)
    assert (packed / "repo/note").read_text() == "second"


@pytest.mark.parametrize(
    ("packed", "base", "home", "outcome"),
    [
        ("p", None, None, "move"),
        ("p", None, "p", "keep"),
        ("p", "b", "b", "move"),
        ("b", "b", "h", "keep"),
        ("p", None, "h", "conflict"),
        ("p", "b", "h", "conflict"),
    ],
)
def test_three_way_rule(packed: str, base: str | None, home: str | None, outcome: str) -> None:
    assert pack.choose_file(packed, base, home) == outcome


@pytest.mark.parametrize("cross_device", [False, True])
def test_unpack_rekeys_and_moves(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, cross_device: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    state = paths.state_dir(source)
    _write(state / "memory/note.md")
    _write(paths.secrets_path(), "secret", 0o600)
    pack.pack(source)
    dest = tmp_path / "destination"
    dest.mkdir()
    shutil.move(source / ".agent6", dest / ".agent6")
    paths.secrets_path().unlink()
    kept = _write(paths.state_dir(dest) / "unrelated", "keep")
    if cross_device:

        def exdev(*args: object, **kwargs: object) -> None:
            raise OSError(errno.EXDEV, "cross device")

        monkeypatch.setattr(os, "rename", exdev)
    result = pack.unpack(dest)
    landed = paths.state_dir(dest) / "memory/note.md"
    assert landed.read_text() == "payload"
    assert landed.stat().st_mtime == 1700000000
    assert landed.stat().st_mode & 0o777 == 0o640
    assert paths.secrets_path().stat().st_mode & 0o777 == 0o600
    assert paths.secrets_path().stat().st_uid == os.getuid()
    assert kept.read_text() == "keep"
    assert not (dest / ".agent6").exists()
    assert (state / "memory/note.md").exists()
    assert result.origin.workspace == str(source)
    assert len(result.moved) == 2
    assert json.loads((paths.state_dir(dest) / "pack-base.json").read_text()) == {
        "repo/memory/note.md": _digest("payload"),
        "config/secrets.toml": _digest("secret"),
    }


def test_unpack_conflicts_are_all_reported_before_any_move(tmp_path: pathlib.Path) -> None:
    state = paths.state_dir(tmp_path)
    _write(state / "a", "old a")
    _write(state / "b", "old b")
    _write(state / "absent", "move me")
    pack.pack(tmp_path)
    (state / "absent").unlink()
    for name in ("a", "b"):
        (state / name).write_text("new " + name)
    with pytest.raises(errors.OperatorError) as caught:
        pack.unpack(tmp_path)
    text = str(caught.value)
    for name in ("a", "b"):
        assert str(state / name) in text
        assert str(tmp_path / ".agent6/repo" / name) in text
    assert "2023-11-14T22:13:20" in text
    assert "hand-merge" in text.lower() and "delete" in text
    assert not (state / "absent").exists()
    assert (tmp_path / ".agent6/repo/absent").exists()


@pytest.mark.parametrize(
    ("packed", "base", "home", "wanted"),
    [
        ("same", None, "same", "same"),
        ("changed", "base", "base", "changed"),
        ("base", "base", "home edit", "home edit"),
    ],
)
def test_unpack_three_way_files(
    tmp_path: pathlib.Path, packed: str, base: str | None, home: str, wanted: str
) -> None:
    state = paths.state_dir(tmp_path)
    note = _write(state / "note", packed)
    if base:
        _write(state / "pack-base.json", json.dumps({"repo/note": _digest(base)}))
    pack.pack(tmp_path)
    note.write_text(home)
    inode = note.stat().st_ino
    pack.unpack(tmp_path)
    assert note.read_text() == wanted
    if wanted == home:
        assert note.stat().st_ino == inode
    assert json.loads((state / "pack-base.json").read_text())["repo/note"] == _digest(packed)


@pytest.mark.parametrize(
    ("field", "value"), [("version", 2), ("packed_at", "bad"), ("files", {"../escape": {}})]
)
def test_bad_manifest_names_field(tmp_path: pathlib.Path, field: str, value: object) -> None:
    pack.pack(tmp_path)
    manifest = tmp_path / ".agent6/pack.json"
    data = json.loads(manifest.read_text())
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(errors.OperatorError, match=field):
        pack.unpack(tmp_path)
    assert manifest.exists()


def test_hand_merged_or_deleted_pack_file_can_be_retried(tmp_path: pathlib.Path) -> None:
    state = paths.state_dir(tmp_path)
    _write(state / "note", "original")
    _write(state / "deleted", "original")
    pack.pack(tmp_path)
    (tmp_path / ".agent6/repo/note").write_text("merged")
    (state / "note").unlink()
    (tmp_path / ".agent6/repo/deleted").unlink()
    pack.unpack(tmp_path)
    assert (state / "note").read_text() == "merged"
    assert (state / "deleted").read_text() == "original"


def test_pack_streams_files_over_64_mib(tmp_path: pathlib.Path) -> None:
    large = _write(paths.state_dir(tmp_path) / "large", "")
    with large.open("wb") as stream:
        stream.truncate(65 * 1024 * 1024)
    manifest = pack.pack(tmp_path)
    with large.open("rb") as stream:
        assert (
            manifest.files["repo/large"].sha256 == hashlib.file_digest(stream, "sha256").hexdigest()
        )


@pytest.mark.parametrize("home_text", ["secret", "home edit"])
def test_kept_secrets_are_private_and_owned_by_the_importer(
    tmp_path: pathlib.Path, home_text: str
) -> None:
    """Keeping home bytes still installs secrets through a fresh private descriptor."""
    secret = _write(paths.secrets_path(), "secret", 0o600)
    _write(
        paths.state_dir(tmp_path) / "pack-base.json",
        json.dumps({"config/secrets.toml": _digest("secret")}),
    )
    pack.pack(tmp_path)
    _write(secret, home_text, 0o644)
    result = pack.unpack(tmp_path)
    assert secret in result.kept
    assert secret.read_text() == home_text
    assert secret.stat().st_mode & 0o777 == 0o600
    assert secret.stat().st_uid == os.getuid()
    assert secret.stat().st_mtime == 1700000000


def test_manifest_version_is_required(tmp_path: pathlib.Path) -> None:
    pack.pack(tmp_path)
    manifest = tmp_path / ".agent6/pack.json"
    data = json.loads(manifest.read_text())
    del data["version"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(errors.PackError, match="version"):
        pack.unpack(tmp_path)
    assert manifest.exists()


@pytest.mark.parametrize("source", ["secrets", "manifest", "base"])
def test_special_inputs_refuse_before_open(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A FIFO at a fixed input path must not block the transfer on an open."""
    if source == "manifest":
        pack.pack(tmp_path)
        file = tmp_path / ".agent6/pack.json"
        file.unlink()
    elif source == "base":
        file = paths.state_dir(tmp_path) / "pack-base.json"
    else:
        file = paths.secrets_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(file)
    original_open = pathlib.Path.open

    def guard(path: pathlib.Path, *args: object, **kwargs: object) -> object:
        assert path != file, "attempted to open a non-regular input"
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pathlib.Path, "open", guard)
    with pytest.raises(errors.PackError, match="regular file"):
        if source == "manifest":
            pack.unpack(tmp_path)
        else:
            pack.pack(tmp_path)


def test_pack_refuses_symlinks(tmp_path: pathlib.Path) -> None:
    state = paths.state_dir(tmp_path)
    state.mkdir(parents=True)
    (state / "link").symlink_to(tmp_path)
    with pytest.raises(errors.OperatorError, match="symlink"):
        pack.pack(tmp_path)
