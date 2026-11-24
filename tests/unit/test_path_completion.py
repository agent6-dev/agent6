# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Path completion lists matching files in the workspace."""

from __future__ import annotations

import pathlib
import subprocess
import warnings

from agent6.ui import path_completion


def _repo(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_ref_fragment_is_the_reference_the_text_ends_in() -> None:
    assert path_completion.ref_fragment("read @src/ma") == "src/ma"
    assert path_completion.ref_fragment("read @") == ""
    assert path_completion.ref_fragment("@") == ""
    assert path_completion.ref_fragment("mail user@exam") is None
    assert path_completion.ref_fragment("read @src/main.py then") is None
    assert path_completion.ref_fragment("plain words") is None


def test_suggestions_put_prefix_matches_first_and_stop_at_the_cap(tmp_path: pathlib.Path) -> None:
    """The completion source returns at most its cap and counts what it left out."""
    names = {f"src/mod{i:02d}.py": "" for i in range(12)}
    repo = _repo(tmp_path, {**names, "docs/src-notes.md": "", ".gitignore": "ignored.txt\n"})
    (repo / "ignored.txt").write_text("", encoding="utf-8")

    found = path_completion.suggest_refs(repo, (), "src")

    assert len(found.paths) == path_completion.REF_SUGGESTIONS_MAX == 8
    assert found.paths == tuple(f"src/mod{i:02d}.py" for i in range(8))
    assert found.more == 5  # four more modules, then the path that only contains the fragment
    assert found.common == "src/mod"
    assert path_completion.suggest_refs(repo, (), "notes").paths == ("docs/src-notes.md",)
    assert path_completion.suggest_refs(repo, (), "ignored").paths == ()
    assert path_completion.suggest_refs(repo, (), "", limit=100).more == 0


def test_a_completions_common_prefix_is_counted_in_characters(tmp_path: pathlib.Path) -> None:
    """`os.path.commonprefix` did the count and warns as deprecated on Python 3.15."""
    repo = _repo(tmp_path, {"src/main.py": "", "src/map.py": ""})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert path_completion.suggest_refs(repo, (), "src").common == "src/ma"
    assert path_completion.common_prefix(["/parallel", "/pin", "/plan"]) == "/p"
    assert path_completion.common_prefix(["ab", "abc"]) == "ab"
    assert path_completion.common_prefix(["x", "y"]) == path_completion.common_prefix([]) == ""


def test_suggestions_skip_hidden_paths_symlinks_and_spaces(tmp_path: pathlib.Path) -> None:
    """A hidden path, a symlink and a name outside the reference characters are never offered."""
    repo = _repo(
        tmp_path,
        {"secrets/key.pem": "k", "open/key.txt": "t", "open/key with space.txt": "s"},
    )
    (repo / "open" / "key-link").symlink_to("key.txt")

    found = path_completion.suggest_refs(repo, (str(repo / "secrets"),), "key")

    assert found.paths == ("open/key.txt",)
    assert path_completion.suggest_refs(repo, (), "key").paths == (
        "open/key.txt",
        "secrets/key.pem",
    )


def test_suggestions_are_empty_outside_a_repository(tmp_path: pathlib.Path) -> None:
    (tmp_path / "notes.md").write_text("", encoding="utf-8")
    assert path_completion.suggest_refs(tmp_path, (), "no") == path_completion.RefSuggestions(
        (), 0, "no"
    )
