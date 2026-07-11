# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 skills` CLI: install (file/dir/git), update, list, state, remove."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.ui.cli.skills_cmds import (
    _cmd_skills_disable,
    _cmd_skills_enable,
    _cmd_skills_install,
    _cmd_skills_list,
    _cmd_skills_remove,
    _cmd_skills_update,
    resolved_skill_names_for_completion,
)

SKILL_MD = """---
name: {name}
description: Use when testing {name}.
---

Body of {name}.
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hermetic data/config/state homes; returns the tmp root."""
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path / "cwd" if (tmp_path / "cwd").exists() else tmp_path)
    return tmp_path


def _write_skill_file(path: Path, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SKILL_MD.format(name=name), encoding="utf-8")
    return path


def _installed(tmp_path: Path, name: str) -> Path:
    return tmp_path / "data" / "skills" / name


class TestInstall:
    def test_local_skill_md_file(self, env: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert (_installed(env, "tidy") / "SKILL.md").is_file()
        assert (_installed(env, "tidy") / ".origin.toml").is_file()
        assert "installed tidy" in capsys.readouterr().out

    def test_local_repo_with_skills_dir(self, env: Path) -> None:
        repo = env / "pack"
        _write_skill_file(repo / "skills" / "aa" / "SKILL.md", "aa")
        _write_skill_file(repo / "skills" / "bb" / "SKILL.md", "bb")
        (repo / "skills" / "aa" / "references").mkdir()
        (repo / "skills" / "aa" / "references" / "x.md").write_text("REF\n", encoding="utf-8")
        assert _cmd_skills_install(str(repo), force=False) == 0
        assert (_installed(env, "aa") / "references" / "x.md").read_text() == "REF\n"
        assert (_installed(env, "bb") / "SKILL.md").is_file()

    def test_git_repo_install(self, env: Path) -> None:
        repo = env / "gitpack"
        _write_skill_file(repo / "skills" / "gg" / "SKILL.md", "gg")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
        # a git URL that is not an existing local path exercises the clone path;
        # file:// URLs hit git's local transport, no network involved
        assert _cmd_skills_install(f"file://{repo}", force=False) == 0
        assert (_installed(env, "gg") / "SKILL.md").is_file()
        origin = (_installed(env, "gg") / ".origin.toml").read_text()
        assert 'kind = "git"' in origin
        assert "source_sha" in origin

    def test_conflict_refused_then_forced(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_install(str(src), force=False) == 2
        assert _cmd_skills_install(str(src), force=True) == 0

    def test_missing_frontmatter_rejected(self, env: Path) -> None:
        bad = env / "bad.md"
        bad.write_text("no frontmatter\n", encoding="utf-8")
        assert _cmd_skills_install(str(bad), force=False) == 2


class TestUpdate:
    def test_update_reports_changed_and_unchanged(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_update("tidy") == 0
        assert "tidy: unchanged" in capsys.readouterr().out
        src.write_text(SKILL_MD.format(name="tidy") + "\nMore.\n", encoding="utf-8")
        assert _cmd_skills_update("tidy") == 0
        assert "tidy: updated" in capsys.readouterr().out
        assert "More." in (_installed(env, "tidy") / "SKILL.md").read_text()

    def test_update_unknown_name(self, env: Path) -> None:
        assert _cmd_skills_update("ghost") == 2


class TestStateCommands:
    def _install_tidy(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0

    def test_disable_writes_global_state(self, env: Path) -> None:
        self._install_tidy(env)
        assert _cmd_skills_disable("tidy", repo=False) == 0
        cfg = (env / "config" / "config.toml").read_text()
        assert 'tidy = "disabled"' in cfg

    def test_enable_always_and_back(self, env: Path) -> None:
        self._install_tidy(env)
        assert _cmd_skills_enable("tidy", always=True, repo=False) == 0
        assert 'tidy = "always"' in (env / "config" / "config.toml").read_text()
        assert _cmd_skills_enable("tidy", always=False, repo=False) == 0
        assert "tidy" not in (env / "config" / "config.toml").read_text()

    def test_unknown_skill_refused(self, env: Path) -> None:
        assert _cmd_skills_disable("ghost", repo=False) == 2
        assert _cmd_skills_enable("ghost", always=False, repo=False) == 2


class TestRemoveListComplete:
    def test_remove_installed(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_remove("tidy") == 0
        assert not _installed(env, "tidy").exists()
        assert _cmd_skills_remove("tidy") == 2

    def test_list_shows_state_and_origin(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_disable("tidy", repo=False) == 0
        assert _cmd_skills_list() == 0
        out = capsys.readouterr().out
        assert "tidy" in out
        assert "[disabled]" in out
        assert "Use when testing tidy." in out

    def test_completion_names(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert resolved_skill_names_for_completion(Path.cwd()) == ["tidy"]


class TestSkillsTaskPrefix:
    def test_prefix_contains_skill_and_unknown_errors(self, env: Path) -> None:
        from agent6.config.layer import load_effective
        from agent6.ui.cli.run import _skills_task_prefix

        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        cfg = load_effective(Path.cwd()).config
        prefix, err = _skills_task_prefix(cfg, ("tidy",))
        assert err == ""
        assert '<skill name="tidy">' in prefix
        assert "Body of tidy." in prefix
        _, err2 = _skills_task_prefix(cfg, ("ghost",))
        assert "ghost" in err2
        assert "tidy" in err2


class TestAtomicMultiInstall:
    def test_repo_conflict_refuses_whole_install(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # aa already installed; a repo carrying aa+bb must install NOTHING
        src = _write_skill_file(env / "one" / "SKILL.md", "aa")
        assert _cmd_skills_install(str(src), force=False) == 0
        repo = env / "pack"
        _write_skill_file(repo / "skills" / "aa" / "SKILL.md", "aa")
        _write_skill_file(repo / "skills" / "bb" / "SKILL.md", "bb")
        assert _cmd_skills_install(str(repo), force=False) == 2
        assert "nothing was installed" in capsys.readouterr().err
        assert not _installed(env, "bb").exists()
        # --force replaces and installs both
        assert _cmd_skills_install(str(repo), force=True) == 0
        assert _installed(env, "bb").exists()
