from __future__ import annotations

import pathlib

import pytest

from agent6 import errors, kinds, pack, path_safety, paths
from agent6.app import confine
from agent6.config import Config
from agent6.sandbox import jail
from agent6.sessions import lock
from agent6.tools import dispatch


@pytest.mark.parametrize("isolation", ["full", "partial", "none"])
def test_the_directory_is_an_ordinary_one_to_the_file_tools(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """The start check is a pack's one protection: a tree agent6 keeps there stays readable.

    A rule that hid every `.agent6` path component hid a session's own worktree placed under it.
    """
    tree = tmp_path / ".agent6/worktrees/run-1"
    tree.mkdir(parents=True)
    (tree / "module.py").write_text("def kept_symbol():\n    pass\n")
    d = dispatch.ToolDispatcher(root=tmp_path, config=Config(), isolation=isolation)
    try:
        read = d.dispatch("read_file", {"path": ".agent6/worktrees/run-1/module.py"}).to_wire()
        assert "kept_symbol" in str(read)
        assert ".agent6/" in d.dispatch("list_dir", {"path": "."}).to_wire()["entries"]
    finally:
        d.close()
    assert not path_safety.Workspace(tmp_path).is_denied(tree / "module.py")
    assert (
        str(tmp_path / ".agent6")
        not in jail.policy_spec(kinds.JailPolicy(cwd=tmp_path, argv=("/bin/true",)))["hide_paths"]
    )


def test_pack_and_unpack_leave_the_directorys_other_names(tmp_path: pathlib.Path) -> None:
    """`pack` deleted every other entry of `.agent6/` and `unpack` removed the directory."""
    kept = tmp_path / ".agent6/worktrees/run-1/file.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("a session's tree")
    note = paths.state_dir(tmp_path) / "note"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("first")

    pack.pack(tmp_path)
    assert kept.read_text() == "a session's tree"
    assert (tmp_path / ".agent6/repo/note").read_text() == "first"
    pack.pack(tmp_path)  # a second pack replaces its own names only
    assert kept.read_text() == "a session's tree"

    pack.unpack(tmp_path)
    assert kept.read_text() == "a session's tree"
    assert sorted(path.name for path in (tmp_path / ".agent6").iterdir()) == ["worktrees"]


def test_unpack_refuses_an_unlisted_file_in_its_own_areas_only(tmp_path: pathlib.Path) -> None:
    """A file beside the pack is not the pack's; one slipped into `config/` still refuses."""
    pack.pack(tmp_path)
    (tmp_path / ".agent6/plans").mkdir()
    (tmp_path / ".agent6/plans/plan.md").write_text("not a pack file")
    (tmp_path / ".agent6/config/extra.toml").write_text("slipped in")
    with pytest.raises(errors.PackError, match=r"config/extra\.toml") as refused:
        pack.unpack(tmp_path)
    assert "plans" not in str(refused.value)


@pytest.mark.parametrize("isolation", ["full", "partial", "none"])
def test_present_pack_refuses_before_exposure_warning(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    manifest = pack.pack(tmp_path)
    refusal = confine.config_refusal(Config(), isolation, tmp_path)
    assert refusal is not None
    assert manifest.origin.host in refusal and manifest.origin.home in refusal
    assert "agent6 unpack" in refusal and "delete" in refusal


@pytest.mark.parametrize("isolation", ["full", "partial", "none"])
def test_nested_packs_refuse_at_run_boundary(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """Every nested checkout pack is named before a run can expose its secrets."""
    (tmp_path / ".git").mkdir()
    for name in ("child", "other/deeper"):
        child = tmp_path / name
        child.mkdir(parents=True)
        (child / ".git").mkdir()
        pack.pack(child)
    refusal = confine.config_refusal(Config(), isolation, tmp_path)
    assert refusal is not None
    for name in ("child", "other/deeper"):
        assert str(tmp_path / name / ".agent6") in refusal
    assert "agent6 unpack" in refusal and "delete" in refusal


def test_pack_scan_prunes_git_and_does_not_follow_symlinks(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    pack.pack(root / ".git")
    outside = tmp_path / "outside"
    outside.mkdir()
    pack.pack(outside)
    (root / "link").symlink_to(outside, target_is_directory=True)
    assert confine.config_refusal(Config(), "full", root) is None


def test_pack_refuses_while_a_user_of_the_checkout_holds_the_lease(tmp_path: pathlib.Path) -> None:
    """A pack made under a live sandbox put the secrets within its commands' reach.

    An `exec --new` started before `agent6 pack` read the packed secret afterwards, under
    `full` and `partial` alike: the startup refusal of a present pack runs only when a command
    starts, and the `full` mask covers only a directory present at jail setup.
    """
    assert lock.hold_workspace(tmp_path)
    with pytest.raises(errors.PackError, match="live"):
        pack.pack(tmp_path)
    assert not (tmp_path / ".agent6").exists()
    lock.release_workspaces()
    pack.pack(tmp_path)
    assert (tmp_path / ".agent6" / "pack.json").exists()


def test_a_user_refuses_while_pack_writes_and_two_users_share(tmp_path: pathlib.Path) -> None:
    """`pack` takes the lease alone; runs, execs and machines hold it shared."""
    writing = lock.exclude_workspace(tmp_path)
    assert writing is not None
    assert not lock.hold_workspace(tmp_path)
    lock.release_single_writer(writing)
    assert lock.hold_workspace(tmp_path)
    assert lock.hold_workspace(tmp_path / "sub")  # the same checkout, the same lease
    assert lock.exclude_workspace(tmp_path) is None


def test_pack_of_a_nested_checkout_needs_the_leases_above_it(tmp_path: pathlib.Path) -> None:
    """A live parent workspace is not bypassed by packing a checkout nested inside it."""
    parent = tmp_path / "parent"
    (parent / ".git").mkdir(parents=True)
    child = parent / "child"
    (child / ".git").mkdir(parents=True)
    assert lock.hold_workspace(parent)
    with pytest.raises(errors.PackError, match=str(parent)):
        pack.pack(child)
    lock.release_workspaces()
    pack.pack(child)
    assert (child / ".agent6" / "pack.json").exists()


def test_a_user_of_a_nested_checkout_stops_a_pack_of_the_workspace_above_it(
    tmp_path: pathlib.Path,
) -> None:
    """A session in a worktree inside the project read a pack written beside it.

    A child's clone under `.agent6/worktrees` sits in the workspace a container mounts, so its
    commands reach the project's `.agent6/config/`: the pack waits for it as for a run of the
    project itself.
    """
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    child = project / ".agent6/worktrees/project-1/run-p1/child-1"
    (child / ".git").mkdir(parents=True)
    assert lock.hold_workspace(child)
    with pytest.raises(errors.PackError, match=str(project)):
        pack.pack(project)
    lock.release_workspaces()
    pack.pack(project)
