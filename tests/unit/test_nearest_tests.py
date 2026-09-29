# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The scoped-verify selection.

Nearest_test_paths picks the test files most likely to judge a change, and diff_changed_paths reads
a diff's file list.
"""

from __future__ import annotations

import pathlib
from typing import Any

from agent6.harness import _nearest_tests

DIFF = """\
diff --git a/pkg/mod.py b/pkg/mod.py
index 1..2 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-x
+y
diff --git a/README.md b/README.md
index 3..4 100644
"""


def test_diff_changed_paths_reads_b_side_in_order() -> None:
    assert _nearest_tests.diff_changed_paths(DIFF) == ("pkg/mod.py", "README.md")
    assert _nearest_tests.diff_changed_paths("") == ()


def _touch(root: pathlib.Path, *rels: str) -> None:
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def test_sibling_and_mirror_and_scan_layouts_are_found(tmp_path: pathlib.Path) -> None:
    """One changed source finds its tests across the common layouts, most specific first.

    Same-dir sibling, package tests dir, repo-root mirror, then the bounded name scan for layouts
    the conventions miss.
    """
    _touch(
        tmp_path,
        "pkg/mod.py",
        "pkg/test_mod.py",
        "pkg/tests/test_mod.py",
        "tests/pkg/test_mod.py",
        "tests/unit/test_mod.py",
    )
    got = _nearest_tests.nearest_test_paths(tmp_path, ("pkg/mod.py",))
    assert got[0] == "pkg/test_mod.py"
    assert set(got) == {
        "pkg/test_mod.py",
        "pkg/tests/test_mod.py",
        "tests/pkg/test_mod.py",
        "tests/unit/test_mod.py",
    }


def test_changed_test_files_select_themselves(tmp_path: pathlib.Path) -> None:
    """Changed test files select themselves.

    A run that edited a test file runs that file; a deleted one (absent from the tree) is dropped
    rather than handed to pytest as an error.
    """
    _touch(tmp_path, "tests/test_a.py")
    got = _nearest_tests.nearest_test_paths(tmp_path, ("tests/test_a.py", "tests/test_gone.py"))
    assert got == ("tests/test_a.py",)


def test_non_python_and_testless_changes_select_nothing(tmp_path: pathlib.Path) -> None:
    _touch(tmp_path, "docs/page.md", "pkg/mod.py")
    assert _nearest_tests.nearest_test_paths(tmp_path, ("docs/page.md",)) == ()
    assert _nearest_tests.nearest_test_paths(tmp_path, ("pkg/mod.py",)) == ()


def test_cap_bounds_the_selection(tmp_path: pathlib.Path) -> None:
    _touch(tmp_path, *[f"pkg/m{i}.py" for i in range(30)])
    _touch(tmp_path, *[f"pkg/test_m{i}.py" for i in range(30)])
    got = _nearest_tests.nearest_test_paths(tmp_path, tuple(f"pkg/m{i}.py" for i in range(30)))
    assert len(got) == 20


def test_is_bare_pytest_names_the_shape_that_takes_appended_paths() -> None:
    """Only a pytest argv naming no paths selects the files appended to it."""
    assert _nearest_tests.is_bare_pytest(("pytest",))
    assert _nearest_tests.is_bare_pytest(("python", "-m", "pytest", "-q", "-x"))
    assert _nearest_tests.is_bare_pytest(("/venv/bin/pytest", "--timeout=30"))
    assert not _nearest_tests.is_bare_pytest(("sh", "-c", "uv run ruff check && uv run pytest"))
    assert not _nearest_tests.is_bare_pytest(("python", "-m", "pytest", "-q", "tests"))
    assert not _nearest_tests.is_bare_pytest(("make", "test"))
    assert not _nearest_tests.is_bare_pytest(())


def test_a_changed_helper_under_a_tests_dir_is_not_handed_to_pytest(tmp_path: pathlib.Path) -> None:
    """conftest.py and helper modules under tests/ are not test files.

    They are neither selected as themselves nor mirrored as sources.
    """
    _touch(tmp_path, "tests/conftest.py", "tests/helpers.py", "tests/test_a.py")
    changed = ("tests/conftest.py", "tests/helpers.py", "tests/test_a.py")
    assert _nearest_tests.nearest_test_paths(tmp_path, changed) == ("tests/test_a.py",)


def test_the_scan_examines_every_entry_within_its_cap(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """The cap counts entries examined.

    With three entries under tests/ and a cap of three, the last one (the nested test file) is still
    seen.
    """
    monkeypatch.setattr("agent6.harness._nearest_tests._SCAN_CAP", 3)
    _touch(tmp_path, "pkg/mod.py", "tests/test_a.py", "tests/unit/test_mod.py")
    assert _nearest_tests.nearest_test_paths(tmp_path, ("pkg/mod.py",)) == (
        "tests/unit/test_mod.py",
    )


def test_package_level_test_dirs_are_scanned(tmp_path: pathlib.Path) -> None:
    """Package-level test dirs are scanned.

    pandas-layout repos keep tests under the package (pandas/tests), often dropping path segments
    (core/): the scan covers test dirs beside every source ancestor, not only the repo root, or a
    timed-out gate finds nothing to scope to.
    """
    _touch(tmp_path, "pkg/core/indexes/base.py", "pkg/tests/indexes/test_base.py")
    got = _nearest_tests.nearest_test_paths(tmp_path, ("pkg/core/indexes/base.py",))
    assert got == ("pkg/tests/indexes/test_base.py",)
