# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for git co-change mining."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent6.git_ops import co_change_pairs


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _setup_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@test")
    _git(tmp_path, "config", "user.name", "test")
    return tmp_path


def _commit(tmp_path: Path, files: dict[str, str], msg: str) -> None:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", msg)


def test_co_change_pairs_finds_frequent_pairs(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    # 4 commits that each change a.py + b.py together. Strong co-change.
    for i in range(4):
        _commit(tmp_path, {"src/a.py": f"v{i}", "src/b.py": f"v{i}"}, f"both v{i}")
    # 1 commit that changes c.py alone.
    _commit(tmp_path, {"src/c.py": "x"}, "c alone")
    pairs = co_change_pairs(tmp_path, n_commits=10, min_pair_count=2)
    # a.py and b.py should be the top pair with count=4.
    assert pairs, "expected at least one pair"
    top = pairs[0]
    assert top[0] == "src/a.py"
    assert top[1] == "src/b.py"
    assert top[2] == 4
    # c.py should NOT appear (didn't co-change with anything 2+ times).
    for a, b, _c in pairs:
        assert "src/c.py" not in (a, b)


def test_co_change_pairs_respects_min_count(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    # 1 commit changing a + b. Below default min_pair_count=2.
    _commit(tmp_path, {"a.py": "x", "b.py": "x"}, "both once")
    pairs = co_change_pairs(tmp_path, n_commits=10, min_pair_count=2)
    assert pairs == []


def test_co_change_pairs_returns_empty_on_shallow_history(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    # Single commit; no co-change signal possible.
    _commit(tmp_path, {"only.py": "x"}, "init")
    assert co_change_pairs(tmp_path, n_commits=10, min_pair_count=2) == []


def test_co_change_pairs_caps_results(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    # 3 commits each touching 5 files - that's C(5,2)=10 pairs per commit,
    # 30 distinct pairs all with count=3.
    files = [f"f{i}.py" for i in range(5)]
    for i in range(3):
        _commit(tmp_path, {f: f"v{i}" for f in files}, f"all v{i}")
    pairs = co_change_pairs(tmp_path, n_commits=10, min_pair_count=2, max_pairs=5)
    assert len(pairs) == 5
    # All returned pairs should have count >= the threshold.
    for _a, _b, c in pairs:
        assert c >= 2


def test_co_change_pairs_skips_merge_commits(tmp_path: Path) -> None:
    """Merge commits would inflate co-change frequencies via multi-parent
    diffs; the function passes --no-merges to avoid this."""
    _setup_repo(tmp_path)
    _commit(tmp_path, {"a.py": "v1"}, "v1")
    _git(tmp_path, "branch", "feat")
    _commit(tmp_path, {"a.py": "v2", "b.py": "v2"}, "v2 on main")
    _git(tmp_path, "checkout", "feat")
    _commit(tmp_path, {"c.py": "v2", "d.py": "v2"}, "v2 on feat")
    _git(tmp_path, "checkout", "main")
    _git(tmp_path, "merge", "-q", "-m", "merge feat", "feat")
    # The merge commit "touches" a/b/c/d - we want this NOT to count.
    # Only the v2 commit's a<->b pair should be considered.
    pairs = co_change_pairs(tmp_path, n_commits=10, min_pair_count=1)
    pair_set = {(a, b) for a, b, _ in pairs}
    # a<->b should be present (from "v2 on main" commit).
    assert ("a.py", "b.py") in pair_set
    # c<->d should be present (from "v2 on feat" commit).
    assert ("c.py", "d.py") in pair_set
    # But a<->c should NOT be present (only the merge commit "linked"
    # them, and merges are excluded).
    assert ("a.py", "c.py") not in pair_set
