# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` collects the working-tree diff WITHOUT mutating the index.

The working-tree comparison uses a temporary index to include untracked files.
The shared index and checkout stay untouched.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

from agent6.ui.cli import review_cmds  # pyright: ignore[reportPrivateUsage]
from agent6.vcs import _git_command, git_read, review


def _git(root: pathlib.Path, *args: str) -> str:
    git = shutil.which("git")
    assert git is not None
    out = subprocess.run(
        [git, "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_pathspec_environment_does_not_empty_the_review(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GIT_LITERAL_PATHSPECS` in the operator's shell made every listing empty: no diff, exit 0."""
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    compared = review.comparison(repo, base="", head="HEAD", paths=(), hide=(str(repo / "h"),))
    assert "+x = 2" in compared.diff


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_review_diff_failure_escapes_repository_diagnostics(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Git failure naming a selected path must not emit terminal commands."""
    name = "unsafe\x1b]52;c;c2VjcmV0\x07"
    (repo / name).write_text("changed\n")
    _git(repo, "config", "diff.algorithm", "invalid")
    monkeypatch.chdir(repo)

    assert review_cmds._reviewed_diff("", "HEAD", (name,)) == 2

    rendered = capsys.readouterr().err
    assert "git diff failed" in rendered
    assert "\x1b" not in rendered and "\x07" not in rendered, repr(rendered)
    assert r"unsafe\u001b]52;c;c2VjcmV0\u0007" in rendered


@pytest.mark.parametrize("changed", [False, True])
def test_review_labels_escape_selected_paths(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changed: bool,
) -> None:
    """Empty-diff and review labels must show escaped names, not terminal commands."""
    name = "unsafe\x1b]52;c;c2VjcmV0\x07"
    (repo / name).write_text("old\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "file")
    if changed:
        (repo / name).write_text("new\n")
    monkeypatch.chdir(repo)

    result = review_cmds._reviewed_diff("", "HEAD", (name,))
    rendered = capsys.readouterr().err
    if changed:
        assert not isinstance(result, int)
        rendered += result[3]
    else:
        assert result == 0
    assert "\x1b" not in rendered and "\x07" not in rendered, repr(rendered)
    assert r"unsafe\u001b]52;c;c2VjcmV0\u0007" in rendered


def test_review_diff_includes_untracked_and_restores_index(repo: pathlib.Path) -> None:
    git = shutil.which("git")
    assert git is not None
    (repo / "newfile.py").write_text("y = 2\n", encoding="utf-8")  # untracked
    (repo / "tracked.py").write_text("x = 99\n", encoding="utf-8")  # modified tracked

    proc = review.comparison(repo, base="", head="HEAD", paths=())
    # Both the modification and the new untracked file appear in the review diff.
    assert "tracked.py" in proc.diff
    assert "newfile.py" in proc.diff
    assert "y = 2" in proc.diff

    # ...and the index is left as found: newfile.py is still UNTRACKED ("??"),
    # not a lingering intent-to-add entry.
    status = _git(repo, "status", "--porcelain")
    assert "?? newfile.py" in status
    assert "A  newfile.py" not in status
    assert " A newfile.py" not in status


@pytest.mark.parametrize("linked", [False, True], ids=["main", "linked"])
def test_review_includes_a_staged_addition_even_when_ignored(
    repo: pathlib.Path, linked: bool
) -> None:
    """An addition tracked only by the index remains reviewable after it is ignored."""
    root = repo
    if linked:
        root = repo / "linked"
        _git(repo, "worktree", "add", "--detach", str(root))
    added = root / "new.py"
    added.write_text("staged addition\n")
    _git(root, "add", "new.py")
    (root / ".gitignore").write_text("new.py\n")
    added.write_text("worktree addition\n")
    index_path = pathlib.Path(
        _git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").strip()
    )
    index = index_path.read_bytes()

    compared = review.comparison(root, base="", head="HEAD", paths=())

    assert "+worktree addition" in compared.diff
    assert _git(root, "show", f"{compared.target.head_tree}:new.py") == "worktree addition\n"
    assert index_path.read_bytes() == index
    assert _git(root, "show", ":new.py") == "staged addition\n"


@pytest.mark.parametrize(
    "state",
    [
        "tracked",
        "staged",
        "untracked",
        "driver",
        "nested",
        "nested-driver",
        "hidden-child",
        "clean",
        "hidden",
        "unselected",
        "excluded",
        "revisions",
    ],
)
def test_review_reports_dirty_submodules_without_snapshotting_contents(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
) -> None:
    """A dirty gitlink cannot silently become an empty immutable-tree review."""
    source = repo.parent / (repo.name + "-source")
    source.mkdir()
    _git(source, "init", "-q")
    (source / "file.txt").write_text("committed content\n")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "source")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "vendor/sub")
    _git(repo, "commit", "-qam", "submodule")
    sub = repo / "vendor/sub"
    content_root = sub
    if state in ("nested", "nested-driver", "hidden-child"):
        _git(sub, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "inner")
        _git(sub, "commit", "-qam", "nested submodule")
        _git(repo, "commit", "-qam", "updated gitlink")
        content_root = sub / "inner"
    if state != "clean":
        (content_root / ("new.txt" if state == "untracked" else "file.txt")).write_text(
            "unreviewed bytes\n"
        )
    marker = source / "driver-ran"
    if state in ("driver", "nested-driver"):
        (content_root / "file.txt").write_text("changed content!!\n")
        (content_root / ".gitattributes").write_text("file.txt filter=inside\n")
        _git(content_root, "config", "filter.inside.clean", f"touch {marker}; cat")
    if state == "staged":
        _git(sub, "add", "file.txt")
    if state == "tracked":
        assert "-dirty" in _git(repo, "diff", "HEAD")
    blob = _git(
        content_root,
        "hash-object",
        "--no-filters",
        "new.txt" if state == "untracked" else "file.txt",
    ).strip()
    submodules = {sub, content_root}
    indexes = {
        pathlib.Path(_git(path, "rev-parse", "--absolute-git-dir").strip()) / "index"
        for path in submodules
    }
    before_indexes = {path: path.read_bytes() for path in indexes}
    submodule_objects = {
        path: set(
            _git(
                path, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)"
            ).splitlines()
        )
        for path in submodules
    }
    super_index = (repo / ".git/index").read_bytes()
    objects_before = _git(repo, "count-objects", "-v")
    selection = (
        (":(top)tracked.py",)
        if state == "unselected"
        else ((":(top,exclude)vendor/sub",) if state == "excluded" else ())
    )
    hide = (
        (str(sub),)
        if state == "hidden"
        else ((str(content_root),) if state == "hidden-child" else ())
    )
    base = "HEAD" if state == "revisions" else ""
    monkeypatch.chdir(repo / "vendor")
    assert not marker.exists()

    compared = review.comparison(
        pathlib.Path.cwd(), base=base, head="HEAD", paths=selection, hide=hide
    )
    result = review_cmds._reviewed_diff(base, "HEAD", selection, hide)
    stderr = capsys.readouterr().err

    assert result == 0 and not compared.diff
    assert not marker.exists(), "submodule status ran a repository content driver"
    expected = state in ("tracked", "staged", "untracked", "driver", "nested", "nested-driver")
    assert ("submodule working-tree contents are not reviewed: vendor/sub" in stderr) is expected, (
        stderr
    )
    assert (repo / ".git/index").read_bytes() == super_index
    assert {path: path.read_bytes() for path in indexes} == before_indexes
    for path, before in submodule_objects.items():
        after = set(
            _git(
                path, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)"
            ).splitlines()
        )
        assert after == before, path
    assert _git(repo, "ls-tree", compared.target.head_tree, "vendor/sub").startswith(
        "160000 commit"
    )
    assert _git(repo, "count-objects", "-v") == objects_before
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=repo, capture_output=True, check=False
        ).returncode
        != 0
    )


def test_review_diff_with_base_is_plain_diff(repo: pathlib.Path) -> None:
    git = shutil.which("git")
    assert git is not None
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")
    proc = review.comparison(repo, base="HEAD~1", head="HEAD", paths=())
    assert "x = 2" in proc.diff


@pytest.mark.parametrize("base", ["", "base"])
def test_review_refuses_a_range_in_head(base: str, capsys: pytest.CaptureFixture[str]) -> None:
    """A range in --head cannot discard --base or select the working tree."""
    assert review_cmds._cmd_review((), base=base, head="a..b", paths=()) == 2
    assert "--head must be a ref; use --base a..b" in capsys.readouterr().err


@pytest.mark.parametrize("explicit_range", [False, True])
def test_review_branch_after_base_advanced(
    repo: pathlib.Path, explicit_range: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ref pairs exclude base-only changes; explicit two-dot ranges retain them."""
    git = shutil.which("git")
    assert git is not None
    fork = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "branch", "base")
    _git(repo, "checkout", "-qb", "topic")
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "topic change")
    _git(repo, "checkout", "base")
    for name in ("base_only_a.py", "base_only_b.py"):
        (repo / name).write_text("keep = True\n", encoding="utf-8")
        _git(repo, "add", name)
        _git(repo, "commit", "-qm", name)
    proc = review.comparison(
        repo, base="base..topic" if explicit_range else "base", head="topic", paths=()
    )
    assert "+x = 2" in proc.diff
    assert ("base_only_a.py" in proc.diff) is explicit_range
    assert ("base_only_b.py" in proc.diff) is explicit_range
    monkeypatch.chdir(repo)
    reviewed = review_cmds._reviewed_diff("base..topic" if explicit_range else "base", "topic", ())
    assert not isinstance(reviewed, int)
    start = _git(repo, "rev-parse", "base").strip() if explicit_range else fork
    assert f"(from {start})" in reviewed[3]
    assert ("base_only_a.py" in reviewed[2]) is explicit_range
    assert ("base_only_b.py" in reviewed[2]) is explicit_range


@pytest.mark.parametrize("committed", [False, True], ids=["worktree", "revisions"])
@pytest.mark.parametrize("exclude", [":(top,exclude)", ":(exclude,top)", ":!/", ":^/", ":/!"])
def test_exclusion_only_selection_is_repository_wide(
    repo: pathlib.Path, committed: bool, exclude: str
) -> None:
    """An empty cwd must not hide root edits that an exclusion-only diff selects."""
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "tracked.py").write_text("x = 99\n")
    (repo / "ignored.txt").write_text("excluded bytes\n")
    if committed:
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "edit")
    subdir = repo / "empty"
    subdir.mkdir()
    index = (repo / ".git/index").read_bytes()

    compared = review.comparison(
        subdir, base=base if committed else "", head="HEAD", paths=(exclude + "ignored.txt",)
    )

    assert "+x = 99" in compared.diff
    assert compared.target.paths == ("tracked.py",)
    assert "excluded bytes" not in compared.diff
    assert _git(repo, "show", f"{compared.target.head_tree}:tracked.py") == "x = 99\n"
    if not committed:
        assert not _git(repo, "ls-tree", compared.target.head_tree, "--", "ignored.txt")
    assert (repo / ".git/index").read_bytes() == index


@pytest.mark.parametrize("committed", [False, True], ids=["worktree", "revisions"])
@pytest.mark.parametrize("nested", [False, True], ids=["root", "subdirectory"])
def test_coverage_names_the_same_repository_paths_as_the_diff(
    repo: pathlib.Path, committed: bool, nested: bool
) -> None:
    """A subdirectory's f.py cannot be recorded as the repository root's f.py."""
    (repo / "sub").mkdir()
    for name in ("f.py", "sub/f.py"):
        (repo / name).write_text("old\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "files")
    base = _git(repo, "rev-parse", "HEAD").strip()
    for name in ("f.py", "sub/f.py"):
        (repo / name).write_text("new\n")
    if committed:
        _git(repo, "commit", "-qam", "edit")
    compared = review.comparison(
        repo / "sub" if nested else repo,
        base=base if committed else "",
        head="HEAD",
        paths=("f.py",),
    )
    expected = "sub/f.py" if nested else "f.py"
    assert compared.target.paths == (expected,)
    assert list(review.diff_hunks(compared.diff)) == [(expected, "base"), (expected, "head")]


@pytest.mark.parametrize(
    "selected",
    [
        ("gone.py",),
        ("*.py",),
        ("gone.py", "tracked.py"),
        ("gone.py", "tracked.py", ":(exclude)tracked.py"),
    ],
)
def test_review_coverage_includes_staged_deletions(
    repo: pathlib.Path, selected: tuple[str, ...]
) -> None:
    """Paths removed from the index still belong to the reviewed base and its coverage."""
    (repo / "gone.py").write_text("deleted content\n")
    _git(repo, "add", "gone.py")
    _git(repo, "commit", "-qm", "file to delete")
    _git(repo, "rm", "gone.py")
    (repo / "tracked.py").write_text("changed\n")
    index = (repo / ".git/index").read_bytes()

    compared = review.comparison(repo, base="", head="HEAD", paths=selected)

    assert "gone.py" in compared.target.paths
    assert "deleted file mode" in compared.diff and "-deleted content" in compared.diff
    tracked_selected = selected in (("*.py",), ("gone.py", "tracked.py"))
    assert ("+changed" in compared.diff) is tracked_selected
    assert _git(repo, "show", f"{compared.target.head_tree}:tracked.py") == (
        "changed\n" if tracked_selected else "x = 1\n"
    )
    assert (repo / ".git/index").read_bytes() == index


@pytest.mark.parametrize("selection", ["path", "exclusion", "hidden"])
@pytest.mark.parametrize("deleted", [False, True], ids=["base-file", "base-deletion"])
def test_scoped_review_keeps_unselected_conflicts_at_the_base(
    repo: pathlib.Path, selection: str, deleted: bool
) -> None:
    """An unselected merge conflict cannot block a review or change the real index."""
    (repo / "a.py").write_text("old a\n")
    conflict = repo / "b.py"
    conflict.write_text("base b\n")
    _git(repo, "add", "a.py", "b.py")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-qb", "other")
    conflict.write_text("other b\n")
    _git(repo, "commit", "-qam", "other")
    _git(repo, "checkout", "-qb", "ours", base)
    if deleted:
        _git(repo, "rm", "b.py")
    else:
        conflict.write_text("ours b\n")
    _git(repo, "commit", "-qam", "ours")
    merged = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "merge", "other"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert merged.returncode == 1
    unmerged = _git(repo, "ls-files", "-u")
    assert "b.py" in unmerged
    (repo / "a.py").write_text("changed a\n")
    before = (repo / ".git/index").read_bytes()
    conflict_text = conflict.read_text()
    paths = {"path": ("a.py",), "exclusion": (":/", ":(top,exclude)b.py"), "hidden": ()}

    compared = review.comparison(
        repo,
        base="",
        head="HEAD",
        paths=paths[selection],
        hide=(str(conflict),) if selection == "hidden" else (),
    )

    assert "+changed a" in compared.diff
    assert {name for name, _side in review.diff_hunks(compared.diff)} == {"a.py"}
    assert _git(repo, "ls-tree", compared.target.head_tree, "--", "b.py") == _git(
        repo, "ls-tree", compared.target.base_tree, "--", "b.py"
    )
    if not deleted:
        assert _git(repo, "show", f"{compared.target.head_tree}:b.py") == "ours b\n"
    assert (repo / ".git/index").read_bytes() == before
    assert _git(repo, "ls-files", "-u") == unmerged
    assert conflict.read_text() == conflict_text


def test_review_rechecks_a_same_timestamp_same_size_edit(repo: pathlib.Path) -> None:
    """A copied index must retain its mtime so Git rechecks racy entries."""
    _git(repo, "config", "core.trustctime", "false")
    target = repo / "tracked.py"
    index = repo / ".git/index"
    past = target.stat().st_mtime_ns - 10_000_000_000
    os.utime(target, ns=(past, past))
    _git(repo, "add", "tracked.py")
    target.write_text("x = 2\n")
    os.utime(target, ns=(past, past))
    os.utime(index, ns=(past, past))
    before = index.read_bytes()

    compared = review.comparison(repo, base="", head="HEAD", paths=())

    assert "+x = 2" in compared.diff
    assert index.read_bytes() == before
    assert index.stat().st_mtime_ns == past


@pytest.mark.parametrize("directory_first", [False, True], ids=["file-to-dir", "dir-to-file"])
def test_review_stages_file_directory_replacements(
    repo: pathlib.Path, directory_first: bool
) -> None:
    """Exact-name staging must replace conflicting file and directory index entries."""
    replaced = repo / "replaced"
    if directory_first:
        replaced.mkdir()
        (replaced / "child.py").write_text("old\n")
    else:
        replaced.write_text("old\n")
    _git(repo, "add", "replaced")
    _git(repo, "commit", "-qm", "replacement base")
    if directory_first:
        (replaced / "child.py").unlink()
        replaced.rmdir()
        replaced.write_text("new\n")
    else:
        replaced.unlink()
        replaced.mkdir()
        (replaced / "child.py").write_text("new\n")
    before = (repo / ".git/index").read_bytes()

    compared = review.comparison(repo, base="", head="HEAD", paths=("replaced",))

    assert "-old" in compared.diff and "+new" in compared.diff
    name = "replaced" if directory_first else "replaced/child.py"
    assert _git(repo, "show", f"{compared.target.head_tree}:{name}") == "new\n"
    assert (repo / ".git/index").read_bytes() == before


def test_unborn_review_diffs_the_empty_tree_without_writing_it(tmp_path: pathlib.Path) -> None:
    """Git accepts the empty-tree ID without an explicit object write."""
    _git(tmp_path, "init", "-q")
    empty = git_read.EMPTY_TREE
    object_file = tmp_path / ".git/objects" / empty[:2] / empty[2:]
    assert not object_file.exists()
    assert _git(tmp_path, "diff", empty, empty, "--") == ""
    (tmp_path / "new.py").write_text("new\n")
    compared = review.comparison(tmp_path, base="", head="HEAD", paths=())
    assert compared.target.base_tree == empty
    assert "+new" in compared.diff
    assert not object_file.exists()


@pytest.mark.parametrize("committed", [False, True], ids=["worktree", "revisions"])
@pytest.mark.parametrize("selected", [(), ("*.py",)], ids=["all", "pathspec"])
def test_review_commands_do_not_expand_paths_per_repository_file(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
    selected: tuple[str, ...],
) -> None:
    """A one-file review must not send every repository filename back to Git."""
    for number in range(400):
        (repo / f"unchanged_{number:04}.py").write_text("unchanged\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "files")
    (repo / "tracked.py").write_text("changed\n")
    if committed:
        _git(repo, "commit", "-qam", "change")
    spawn = _git_command.spawn

    def bounded_spawn(
        cwd: pathlib.Path,
        args: tuple[str, ...],
        env_extra: dict[str, str] | None,
        stdin_text: str | None,
        repo: pathlib.Path | None = None,
    ) -> tuple[tuple[str, ...], int, bytes, bytes]:
        assert len(args) < 40, f"{args[0]} received {len(args)} arguments"
        if args[0] == "update-index" and "--stdin" in args:
            assert stdin_text == "tracked.py\0", "staging must list only changed files"
        for index, arg in enumerate(args):
            if arg == "--pathspec-from-file" or arg.startswith("--pathspec-from-file="):
                name = args[index + 1] if arg == "--pathspec-from-file" else arg.split("=", 1)[1]
                data = (
                    stdin_text.encode()
                    if name == "-" and stdin_text
                    else pathlib.Path(name).read_bytes()
                )
                separator = b"\0" if "--pathspec-file-nul" in args else b"\n"
                count = len(data.rstrip(separator).split(separator))
                assert count < 40, f"{args[0]} received {count} pathspecs"
        return spawn(cwd, args, env_extra, stdin_text, repo=repo)

    monkeypatch.setattr(_git_command, "spawn", bounded_spawn)
    compared = review.comparison(
        repo, base="HEAD~1" if committed else "", head="HEAD", paths=selected
    )
    assert "+changed" in compared.diff
    assert "unchanged_" not in compared.diff


def test_an_explicit_range_names_its_right_endpoint_as_the_reviewed_head() -> None:
    """`review --base main..topic` checked the repo checkout against HEAD, not topic.

    `main..` is `main..HEAD` to git; the helper named "" as its head.
    """
    assert review_cmds._reviewed_head("main..topic", "HEAD") == "topic"  # pyright: ignore[reportPrivateUsage]
    assert review_cmds._reviewed_head("main", "HEAD") == "HEAD"  # pyright: ignore[reportPrivateUsage]
    assert review_cmds._reviewed_head("main..", "HEAD") == "HEAD"  # pyright: ignore[reportPrivateUsage]
