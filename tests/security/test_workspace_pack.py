from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds
from agent6.app import confine
from agent6.config import Config
from agent6.sandbox import jail
from agent6.tools import _path_safety, dispatch, errors, index
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
def test_tools_hide_pack_at_every_isolation_level(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    _secret(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=Config(), isolation=isolation)
    try:
        with pytest.raises(errors.ToolError, match="absent"):
            d.dispatch("read_file", {"path": ".agent6/config/secrets.toml"})
        with pytest.raises(errors.ToolError, match="absent"):
            d.dispatch("list_dir", {"path": ".agent6"})
        assert ".agent6/" not in d.dispatch("list_dir", {}).to_wire()["entries"]
        (tmp_path / ".agent6/secret.py").write_text("def secret_symbol():\n    pass\n")
        assert (
            d.dispatch("find_definition", {"symbol": "secret_symbol"}).to_wire()["definitions"]
            == []
        )
    finally:
        d.close()


def test_bare_workspace_and_host_reads_hide_pack(tmp_path: pathlib.Path) -> None:
    secret = _secret(tmp_path)
    for ws in (_path_safety.Workspace(tmp_path), _path_safety.host_workspace(tmp_path)):
        assert ws.is_denied(secret)
        with pytest.raises(errors.ToolError, match="absent"):
            ws.resolve_read(".agent6/config/secrets.toml")
    assert _path_safety.read_workspace_text(tmp_path, ".agent6/config/secrets.toml") == ""
    symbols = index.SymbolIndex(_path_safety.Workspace(tmp_path))
    assert symbols._included_rel(secret) is None  # pyright: ignore[reportPrivateUsage]
    public = tmp_path / "public.py"
    public.write_text("def visible(): pass\n")
    assert symbols._included_rel(public) == pathlib.Path("public.py")  # pyright: ignore[reportPrivateUsage]


def test_present_pack_warns_under_partial_like_other_implicit_private_paths(
    tmp_path: pathlib.Path,
) -> None:
    cfg = Config()
    assert not any(
        h == tmp_path / ".agent6"
        for h, _, _ in confine.unmaskable_exposures(cfg, "partial", tmp_path)
    )
    _secret(tmp_path)
    exposures = confine.unmaskable_exposures(cfg, "partial", tmp_path)
    packed = [(h, r, s) for h, r, s in exposures if h == tmp_path / ".agent6"]
    assert len(packed) == 1
    assert "jailed commands can read" in confine.exposure_warning(*packed[0])
    assert confine.check_hide_paths_support(cfg, "partial", tmp_path) is None
    explicit = Config.model_validate({"sandbox": {"hide_paths": [str(tmp_path / ".agent6")]}})
    assert confine.check_hide_paths_support(explicit, "partial", tmp_path) is not None


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
