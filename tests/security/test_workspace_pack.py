from __future__ import annotations

import pathlib

import pytest

from agent6 import errors, kinds, pack, path_safety, paths
from agent6.app import confine
from agent6.config import Config
from agent6.sandbox import jail
from agent6.sessions import lock
from agent6.tools import dispatch, index, tool_errors
from agent6.vcs import git_read


def _secret(root: pathlib.Path) -> pathlib.Path:
    secret = root / ".agent6/config/secrets.toml"
    secret.parent.mkdir(parents=True)
    secret.write_text("packed-secret")
    return secret


def test_jail_hidden_set_contains_workspace_pack_even_when_absent(tmp_path: pathlib.Path) -> None:
    policy = kinds.JailPolicy(cwd=tmp_path, argv=("/bin/true",))
    assert str(tmp_path / ".agent6") in jail.policy_spec(policy)["hide_paths"]


@pytest.mark.parametrize("isolation", ["full", "partial", "none"])
@pytest.mark.parametrize("subdir", [".", "child"])
def test_tools_hide_pack_at_every_isolation_level(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel, subdir: str
) -> None:
    root = tmp_path / subdir
    _secret(root)
    d = dispatch.ToolDispatcher(root=tmp_path, config=Config(), isolation=isolation)
    try:
        with pytest.raises(tool_errors.ToolError, match="absent"):
            d.dispatch("read_file", {"path": f"{subdir}/.agent6/config/secrets.toml"})
        with pytest.raises(tool_errors.ToolError, match="absent"):
            d.dispatch("list_dir", {"path": f"{subdir}/.agent6"})
        assert ".agent6/" not in d.dispatch("list_dir", {"path": subdir}).to_wire()["entries"]
        (root / ".agent6/secret.py").write_text("def secret_symbol():\n    pass\n")
        assert (
            d.dispatch("find_definition", {"symbol": "secret_symbol"}).to_wire()["definitions"]
            == []
        )
    finally:
        d.close()


def test_bare_workspace_and_host_reads_hide_pack(tmp_path: pathlib.Path) -> None:
    secret = _secret(tmp_path)
    for ws in (path_safety.Workspace(tmp_path), path_safety.host_workspace(tmp_path)):
        assert ws.is_denied(secret)
        with pytest.raises(path_safety.ContainmentError, match="absent"):
            ws.resolve_read(".agent6/config/secrets.toml")
    assert path_safety.read_workspace_text(tmp_path, ".agent6/config/secrets.toml") == ""
    symbols = index.SymbolIndex(path_safety.Workspace(tmp_path))
    assert symbols._included_rel(secret) is None  # pyright: ignore[reportPrivateUsage]
    public = tmp_path / "public.py"
    public.write_text("def visible(): pass\n")
    assert symbols._included_rel(public) == pathlib.Path("public.py")  # pyright: ignore[reportPrivateUsage]


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


def test_file_boundary_denies_pack_names_inside_grants(tmp_path: pathlib.Path) -> None:
    """An extra grant or memory exemption cannot expose a nested pack to file tools."""
    root = tmp_path / "repo"
    root.mkdir()
    grant = paths.state_dir(root) / "memory"
    secret = _secret(grant / "child")
    ws = path_safety.Workspace(root, read_roots=(grant,), exempt=(grant,))
    assert ws.is_denied(secret)
    with pytest.raises(path_safety.ContainmentError, match="absent"):
        ws.resolve_read(str(secret))


def test_pack_is_excluded_from_git_snapshots(tmp_path: pathlib.Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _secret(tmp_path)
    assert ".agent6" in git_read.hidden_in_repo(tmp_path, ())


@pytest.mark.needs_namespaces
def test_full_jail_masks_workspace_pack(tmp_path: pathlib.Path) -> None:
    """A directory mask exposes no packed secret bytes through the workspace mount."""
    _secret(tmp_path)
    result = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=tmp_path, argv=("/bin/cat", ".agent6/config/secrets.toml"), isolation="full"
        )
    )
    assert result.stdout == ""
    assert result.returncode != 0


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
