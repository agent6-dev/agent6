# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A dotfiles-symlinked config keeps being what agent6 reads after a write."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from agent6.config.write import set_config_value
from agent6.errors import OperatorError
from agent6.paths import global_config_path


def test_a_symlinked_config_stays_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """atomic_write publishes by rename, which replaces the NAME: a
    dotfiles-managed config silently became a regular file and the repo it
    was linked from stopped being what agent6 reads."""
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    real = tmp_path / "dotfiles" / "agent6.toml"
    real.parent.mkdir()
    real.write_text('[sandbox]\nrun_commands = "ask"\n', encoding="utf-8")
    link = global_config_path()
    link.symlink_to(real)

    assert set_config_value(tmp_path, "sandbox.protect_git", "false") is None

    assert link.is_symlink(), "the dotfiles symlink was replaced by a regular file"
    assert "protect_git" in real.read_text(encoding="utf-8"), "the write missed the real file"


def test_a_symlink_to_another_owner_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under sudo, following the operator's symlink would write as root
    wherever it points. Only a target the real operator owns is followed."""
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    foreign = tmp_path / "root-owned.toml"
    foreign.write_text("[sandbox]\n", encoding="utf-8")
    link = global_config_path()
    link.symlink_to(foreign)

    # Stand in for a target the operator does not own (root-owned under sudo).
    class _Foreign:
        st_uid = 0

    real_stat = Path.stat
    # Resolve ONCE, before the patch: on Python 3.12 Path.resolve() calls
    # Path.stat internally, so resolving inside fake_stat re-enters the patched
    # method and recurses forever (3.14's resolve() does not, which hid this).
    foreign_resolved = foreign.resolve()

    def fake_stat(self: Path, **kw: object) -> object:
        return _Foreign() if self == foreign_resolved else real_stat(self, **kw)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(OperatorError, match="owned by uid 0"):
        set_config_value(tmp_path, "sandbox.protect_git", "false")
    monkeypatch.undo()
    assert link.is_symlink() and foreign.read_text(encoding="utf-8") == "[sandbox]\n"


def test_every_writer_keeps_the_link_not_just_config_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config set` was taught to resolve the link and the other writers were
    not, so `add`, `remove`, `fill` and `fix` each replaced it: the dotfiles
    file silently stopped being what agent6 reads, while the command reported
    success against a path that was no longer the operator's. One resolver
    owns this for every writer.
    """
    from agent6.ui.cli import main

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "dotfiles" / "agent6.toml"
    real.parent.mkdir()
    real.write_text('[sandbox]\nfetch_hosts = ["a.example"]\n', encoding="utf-8")
    link = global_config_path()
    link.symlink_to(real)

    assert main(["config", "add", "sandbox.fetch_hosts", "b.example"]) == 0
    assert link.is_symlink(), "`config add` replaced the dotfiles symlink"
    assert "b.example" in real.read_text(encoding="utf-8"), "the write missed the real file"

    assert main(["config", "remove", "sandbox.fetch_hosts", "b.example"]) == 0
    assert link.is_symlink(), "`config remove` replaced the dotfiles symlink"

    assert main(["config", "fill", "--force"]) == 0
    assert link.is_symlink(), "`config fill` replaced the dotfiles symlink"


def test_a_symlink_whose_target_does_not_exist_yet_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linking the config into a dotfiles repo BEFORE writing the file is the
    ordinary order (`ln -s ~/dotfiles/agent6.toml ~/.config/agent6/config.toml`,
    then configure). Resolving the link stat()ed the target and raised, so
    every write refused -- `agent6 init`, `connect` and `config set` alike --
    over a link that was perfectly valid, just not filled in yet.
    """
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    real = tmp_path / "dotfiles" / "agent6.toml"
    real.parent.mkdir()  # the dotfiles dir exists; the file does not
    link = global_config_path()
    link.symlink_to(real)

    assert set_config_value(tmp_path, "sandbox.protect_git", "false") is None

    assert link.is_symlink(), "the dangling link was replaced instead of filled"
    assert "protect_git" in real.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (("set", "review.period", "8"), {"period": 8, "seats": ["tests"]}),
        (("unset", "review.period"), {"seats": ["tests"]}),
        (("add", "review.seats", "security"), {"period": 7, "seats": ["tests", "security"]}),
        (("remove", "review.seats", "tests"), {"period": 7, "seats": []}),
    ],
)
def test_machine_overlay_writers_keep_a_symlinked_machine_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
    expected: dict[str, object],
) -> None:
    """Machine overlay edits follow an owned machine-file symlink."""
    from agent6.ui.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "machines" / "real.asm.toml"
    target.parent.mkdir()
    target.write_text('[config.review]\nperiod = 7\nseats = ["tests"]\n', encoding="utf-8")
    link = tmp_path / "linked.asm.toml"
    link.symlink_to(target)

    assert main(["config", *command, "--machine-file", str(link)]) == 0

    assert link.is_symlink(), f"`config {command[0]} --machine-file` replaced the symlink"
    assert tomllib.loads(target.read_text(encoding="utf-8"))["config"]["review"] == expected
