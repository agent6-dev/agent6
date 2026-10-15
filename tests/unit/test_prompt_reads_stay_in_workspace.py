# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A workspace file the host reads for a prompt is read only inside the workspace."""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6 import init, verify_infer
from agent6.ui.cli import prompt_cmds  # pyright: ignore[reportPrivateUsage]

_MARKER = "OUTSIDE-MARKER-7f3a"


def _repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return repo


def _prompt_show(
    capsys: pytest.CaptureFixture[str], config_paths: tuple[pathlib.Path, ...] = ()
) -> str:
    assert prompt_cmds._cmd_prompt_show(config_paths, mode="run") == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("kind", ["regular", "hidden", "symlink"])
def test_committed_agents_md_respects_hidden_paths_and_skips_symlinks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Commit-based setup inference reads no hidden blob or committed symlink target."""
    repo = _repo(tmp_path, monkeypatch)
    agents = repo / "AGENTS.md"
    if kind == "symlink":
        (repo / "docs").mkdir()
        (repo / "docs" / "AGENTS.md").write_text(_MARKER, encoding="utf-8")
        agents.symlink_to("docs/AGENTS.md")
        subprocess.run(["git", "add", "docs/AGENTS.md"], cwd=repo, check=True)
    else:
        agents.write_text(_MARKER, encoding="utf-8")
    subprocess.run(["git", "add", "AGENTS.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "agents"], cwd=repo, check=True)
    agents.unlink()
    hide = (str(agents),) if kind == "hidden" else ()
    assert verify_infer.read_agents_md(repo, hide, at="HEAD") == (
        _MARKER if kind == "regular" else ""
    )


def test_agents_md_linked_out_of_the_repo_stays_out_of_the_prompt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A committed `AGENTS.md -> ../secret.txt` put the outside file in the prompt."""
    repo = _repo(tmp_path, monkeypatch)
    (tmp_path / "secret.txt").write_text(f"Verify: echo {_MARKER}\n", encoding="utf-8")
    (repo / "AGENTS.md").symlink_to("../secret.txt")
    assert _MARKER not in _prompt_show(capsys)
    assert verify_infer.read_agents_md(repo) == ""


def test_agents_md_linked_inside_the_repo_is_still_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Containment must not cost `AGENTS.md -> docs/AGENTS.md`."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / "docs").mkdir()
    (repo / "docs" / "AGENTS.md").write_text(f"- {_MARKER}\n", encoding="utf-8")
    (repo / "AGENTS.md").symlink_to("docs/AGENTS.md")
    assert _MARKER in _prompt_show(capsys)
    assert _MARKER in verify_infer.read_agents_md(repo)


@pytest.mark.parametrize("target", ["private/notes.md", ".git/description"])
def test_agents_md_linked_into_a_hidden_path_stays_out_of_the_prompt(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    """A link to a `sandbox.hide_paths` tree or into `.git` put that file in the prompt."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / "private").mkdir()
    (repo / target).write_text(f"- {_MARKER}\n", encoding="utf-8")
    (repo / "AGENTS.md").symlink_to(target)
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[sandbox]\nhide_paths = ["{repo / "private"}"]\n', encoding="utf-8")
    assert _MARKER not in _prompt_show(capsys, (cfg,))
    hide = (str(repo / "private"),)
    assert verify_infer.read_agents_md(repo, hide) == ""


def test_agents_md_linked_into_the_state_dir_stays_out_of_the_prompt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """agent6's private dirs are hidden with no config naming them."""
    repo = _repo(tmp_path, monkeypatch)
    state = tmp_path / "state" / "agent6"
    state.mkdir(parents=True)
    (state / "notes.md").write_text(f"- {_MARKER}\n", encoding="utf-8")
    (repo / "AGENTS.md").symlink_to(state / "notes.md")
    assert _MARKER not in _prompt_show(capsys)


def test_manifest_linked_out_of_the_repo_stays_out_of_the_inference_context(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `pyproject.toml` link's outside target stays out of the inference call to the provider."""
    repo = _repo(tmp_path, monkeypatch)
    (tmp_path / "secret.txt").write_text(f"test: # {_MARKER}\n", encoding="utf-8")
    for name in ("pyproject.toml", "Makefile", "package.json"):
        (repo / name).symlink_to("../secret.txt")
    (repo / "docs").mkdir()
    (repo / "docs" / "go.mod").write_text("module inside\n", encoding="utf-8")
    (repo / "go.mod").symlink_to("docs/go.mod")
    context = verify_infer.gather_repo_manifests(repo, "")
    assert _MARKER not in context and "module inside" in context
    # The `test:` target is the outside file's, so the Makefile signal does not match.
    assert verify_infer.repo_signals(repo)[0][1] == "pyproject"


def test_init_infers_no_verify_command_from_a_hidden_agents_md(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 init` offered a verify command read through a link into `sandbox.hide_paths`."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / "private").mkdir()
    (repo / "private" / "notes.md").write_text(f"Verify: make {_MARKER}\n", encoding="utf-8")
    (repo / "AGENTS.md").symlink_to("private/notes.md")
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[sandbox]\nhide_paths = ["{repo / "private"}"]\n', encoding="utf-8")
    asked: list[str] = []

    def ask(question: str, default: bool) -> bool:
        asked.append(question)
        return False

    init._setup_verify_command(repo, ecosystem="", ask=ask, config_paths=(cfg,))
    assert _MARKER not in "".join(asked) + capsys.readouterr().out


def test_verify_inference_in_a_subdirectory_reads_a_link_inside_the_git_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inference reads `sub/AGENTS.md -> ../real.md` as the prompt does, not as absent."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / "real.md").write_text(f"Verify: make {_MARKER}\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("Verify: make OUT\n", encoding="utf-8")
    sub, out = repo / "sub", repo / "out"
    sub.mkdir()
    out.mkdir()
    (sub / "AGENTS.md").symlink_to("../real.md")
    (out / "AGENTS.md").symlink_to("../../secret.txt")
    assert _MARKER in verify_infer.read_agents_md(sub, ())
    assert verify_infer.read_agents_md(out, ()) == ""
