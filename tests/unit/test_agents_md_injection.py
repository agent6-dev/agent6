# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""AGENTS.md is injected whole, with the repo root's file from a subdirectory.

Whole-file injection with an operator warning replaced a silent 16k clip: a
mid-file cut hid the tail of a large conventions file from the model while the
operator saw nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent6.workflows._context import AGENTS_MD_WARN_CHARS, agents_md_notices, agents_md_text


def _git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


def test_large_agents_md_is_injected_whole(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    body = "x" * 50_000
    (tmp_path / "AGENTS.md").write_text(body, encoding="utf-8")
    assert agents_md_text(tmp_path) == body  # no clip, no marker


def test_oversize_warns_the_operator(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("y" * (AGENTS_MD_WARN_CHARS + 1), encoding="utf-8")
    notices = agents_md_notices(tmp_path)
    assert any("WARNING" in n and "chars" in n for n in notices)


def test_under_the_line_stays_silent(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("small\n", encoding="utf-8")
    assert agents_md_notices(tmp_path) == ()


def test_subdirectory_start_loads_the_repo_roots_file(tmp_path: Path) -> None:
    """pi and Claude Code collect ancestor context files; a subdir start here
    must carry the repo's conventions, not silently miss them."""
    _git_repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("ROOT RULES\n", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    text = agents_md_text(sub)
    assert "ROOT RULES" in text
    notices = agents_md_notices(sub)
    assert any("repo root" in n for n in notices)


def test_subdirectory_with_its_own_file_gets_both_labeled(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("ROOT RULES\n", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("SUB RULES\n", encoding="utf-8")
    text = agents_md_text(sub)
    # Root first (broader), the subdir's under a heading naming its directory.
    assert text.index("ROOT RULES") < text.index("SUB RULES")
    assert "pkg/" in text
    assert any("plus this directory's" in n for n in agents_md_notices(sub))


def test_nested_start_loads_each_ancestor_file_in_order(tmp_path: Path) -> None:
    """A start below an intermediate package must not skip that package's
    AGENTS.md while loading the repository root and leaf files."""
    _git_repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("ROOT RULES\n", encoding="utf-8")
    package = tmp_path / "packages"
    package.mkdir()
    (package / "AGENTS.md").write_text("PACKAGE RULES\n", encoding="utf-8")
    leaf = package / "api"
    leaf.mkdir()
    (leaf / "AGENTS.md").write_text("API RULES\n", encoding="utf-8")

    text = agents_md_text(leaf)

    assert text.index("ROOT RULES") < text.index("PACKAGE RULES") < text.index("API RULES")
    assert "packages/" in text and "packages/api/" in text
    notice = " ".join(agents_md_notices(leaf))
    assert "2 ancestor" in notice and "this directory's" in notice


def test_a_start_below_a_root_without_agents_md_labels_the_file(tmp_path: Path) -> None:
    """With no root file, the outermost file found is a subdirectory's; bare,
    its rules read as the repository's."""
    _git_repo(tmp_path)
    sub = tmp_path / "a"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("A RULES\n", encoding="utf-8")

    text = agents_md_text(sub)

    assert text.startswith("# AGENTS.md in a/ (this run's working directory)")
    assert text.endswith("A RULES\n")


def test_repo_root_start_emits_no_subdir_notice(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("ROOT RULES\n", encoding="utf-8")
    assert agents_md_notices(tmp_path) == ()


def test_non_git_dir_reads_only_its_own_file(tmp_path: Path) -> None:
    """`agent6 ask` runs outside git; the loader must not crash or wander."""
    (tmp_path / "AGENTS.md").write_text("LOCAL\n", encoding="utf-8")
    assert agents_md_text(tmp_path) == "LOCAL\n"
    assert agents_md_notices(tmp_path) == ()
