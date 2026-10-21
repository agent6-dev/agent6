# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the policy builder grants a jailed command beyond the workspace."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.tools import policy


def _host_cfg() -> Config:
    return Config.model_validate({"sandbox": {"network": "host"}})


@pytest.mark.parametrize("isolation", ["full", "partial"])
def test_the_host_network_grants_the_resolver_and_trust_store_files(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A jail on the host network had no resolver file and no CA store, so no name resolved."""
    reads = policy.host_network_reads()
    for name in (
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/ssl",
        "/etc/pki",
        "/etc/ca-certificates",
    ):
        assert (pathlib.Path(name) in reads) == pathlib.Path(name).exists(), name
    resolv = pathlib.Path("/etc/resolv.conf")
    if resolv.exists():
        # The launcher binds the name the resolver opens; Landlock needs the file it points to.
        assert resolv in reads and resolv.resolve() in reads
    assert reads, "no host in the test matrix lacks every one of them"
    granted = policy.jail_policy(tmp_path, _host_cfg(), isolation, ("true",)).extra_ro_paths
    assert set(reads) <= set(granted)


def test_a_session_or_absent_network_grants_no_host_network_file(tmp_path: pathlib.Path) -> None:
    """The resolver and trust-store grants follow the host network only."""
    reads = set(policy.host_network_reads())
    for network in ("session", "none"):
        built = policy.jail_policy(tmp_path, Config(), "full", ("true",), network=network)  # pyright: ignore[reportArgumentType]
        assert built.network == network
        assert not reads & set(built.extra_ro_paths)


def test_a_host_network_file_is_listed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolver file that is no symlink is its own target, granted once."""
    real = pathlib.Path("/etc/hosts")
    monkeypatch.setattr(policy, "_HOST_NETWORK_FILES", (str(real),))
    assert policy.host_network_reads() == ((real,) if real.exists() else ())


def test_a_symlinked_grant_reaches_the_launcher_under_both_names(tmp_path: pathlib.Path) -> None:
    """The launcher bound whatever a symlinked grant pointed to at each launch.

    The policy carries the path the grant resolved to, so the launcher follows no link.
    """
    real, link, plain = tmp_path / "tool-1.2", tmp_path / "tool", tmp_path / "data"
    real.mkdir()
    plain.mkdir()
    link.symlink_to(real, target_is_directory=True)
    # Pinned to `full`: pytest's tmp_path sits in /tmp, where a link counts under auto.
    grants = {
        "isolation": "full",
        "extra_read_paths": [str(link)],
        "extra_write_paths": [str(plain)],
    }
    cfg = Config.model_validate({"sandbox": grants})
    built = policy.jail_policy(tmp_path / "repo", cfg, "full", ("true",))
    assert link in built.extra_ro_paths and plain in built.extra_rw_paths
    assert built.grant_sources == ((link, real),)


def test_a_grant_that_links_through_the_workspace_is_dropped(tmp_path: pathlib.Path) -> None:
    """A grant resolved through a link inside the workspace was kept when it ended outside.

    A jailed command of one run can re-point that link, and the next run would bind wherever
    it then leads. The grant is dropped; a chain that never enters a writable region is kept.
    """
    repo, tools, real = tmp_path / "repo", tmp_path / "tools", tmp_path / "real"
    for made in (repo, tools, real):
        made.mkdir()
    (repo / "current").symlink_to(real, target_is_directory=True)
    (tools / "through").symlink_to(repo / "current", target_is_directory=True)
    (tools / "direct").symlink_to(real, target_is_directory=True)
    # Pinned to `full`: pytest's tmp_path sits in /tmp, where a link counts under auto.
    cfg = Config.model_validate(
        {
            "sandbox": {
                "isolation": "full",
                "extra_read_paths": [str(tools / "through"), str(tools / "direct")],
            }
        }
    )

    grants = policy.resolve_grants(cfg, repo)

    assert grants.read == ((tools / "direct", real),)
    (dropped,) = grants.dropped
    assert (dropped.path, dropped.real, dropped.region) == (tools / "through", real, repo)
    assert f"it sits inside {repo}" not in dropped.line(), "the grant itself is outside"
    assert f"a link on its way sits inside {repo}, which jailed commands write" in dropped.line()


def test_a_grant_that_links_through_tmp_is_dropped(tmp_path: pathlib.Path) -> None:
    """Commands under `partial` write the host's `/tmp`, so a grant there is re-pointable.

    Spelled inside it or reached through a link there, the grant is dropped; what it held stays
    reachable as part of `/tmp`. Under `full`, whose `/tmp` is private, both are kept.
    """
    import os

    repo, real = tmp_path / "repo", tmp_path / "real"
    repo.mkdir()
    real.mkdir()
    link = pathlib.Path("/tmp") / f"agent6-grant-link-{os.getpid()}"
    link.symlink_to(real, target_is_directory=True)
    spelled = pathlib.Path("/tmp") / f"agent6-grant-dir-{os.getpid()}"
    spelled.mkdir()
    try:
        paths = [str(link / "data"), str(spelled)]
        partial = Config.model_validate(
            {"sandbox": {"isolation": "partial", "extra_read_paths": paths}}
        )
        (real / "data").mkdir()
        grants = policy.resolve_grants(partial, repo)
        full = Config.model_validate({"sandbox": {"isolation": "full", "extra_read_paths": paths}})
        kept = policy.resolve_grants(full, repo)
    finally:
        link.unlink()
        spelled.rmdir()
    assert grants.read == ()
    linked, inside = grants.dropped
    tmp = pathlib.Path("/tmp").resolve()
    assert (linked.path, linked.region) == (link / "data", tmp)
    assert (inside.path, inside.region, inside.via_link) == (spelled, tmp, False)
    assert "which this run's commands write" in inside.line()
    assert "'full' to keep it" in inside.line()
    assert kept.read == ((link / "data", real / "data"), (spelled, spelled.resolve()))
    assert not kept.dropped


def test_a_grant_that_is_the_workspace_under_another_name_is_dropped(
    tmp_path: pathlib.Path,
) -> None:
    """A write grant whose link resolves to the workspace root was bound as a second copy.

    The launcher re-binds `.git` read-only under the workspace path alone, so a jailed command
    wrote `.git` through the alias.
    """
    repo, alias = tmp_path / "repo", tmp_path / "alias"
    repo.mkdir()
    alias.symlink_to(repo, target_is_directory=True)
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "full", "extra_write_paths": [str(alias)]}}
    )

    grants = policy.resolve_grants(cfg, repo)

    assert grants.write == ()
    (dropped,) = grants.dropped
    assert (dropped.real, dropped.region, dropped.via_link) == (repo, repo, False)
    assert f"it is {repo} itself" in dropped.line()


def test_a_dropped_link_that_leads_out_of_its_region_names_the_targets_own_path(
    tmp_path: pathlib.Path,
) -> None:
    """The region does not hold what the grant held, so the line says to grant the target."""
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "out").symlink_to(outside, target_is_directory=True)
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "full", "extra_read_paths": [str(repo / "out")]}}
    )
    (dropped,) = policy.resolve_grants(cfg, repo).dropped
    assert dropped.real == outside and dropped.region == repo
    assert "stays reachable" not in dropped.line()
    assert f"Grant {outside} by its own path" in dropped.line()
