# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`warn_sandbox_gaps`: the run-entry warning when the resolved isolation
confines less than its name promises (`none`, or strict without Landlock)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.confine import check_network_support, warn_sandbox_gaps
from agent6.config import Config, SandboxConfig
from agent6.sandbox.detect import Environment, KernelInfo


def _env(landlock_abi: int) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=landlock_abi,
        sandbox_available=True,
    )


def _cfg(tool_network: str = "auto") -> Config:
    return Config(
        sandbox=SandboxConfig(tool_network=tool_network)  # type: ignore[arg-type]
    )


def test_none_warns_unsandboxed(capsys: pytest.CaptureFixture[str]) -> None:
    warn_sandbox_gaps("none", _env(4), _cfg())
    assert "UNSANDBOXED" in capsys.readouterr().err


def test_strict_without_landlock_warns(capsys: pytest.CaptureFixture[str]) -> None:
    """strict on a Landlock-less kernel (ABI 0) silently lost a documented
    layer: the launcher's best-effort ruleset enforces nothing and no surface
    said so, breaking the "no silent downgrade, always loudly" contract."""
    warn_sandbox_gaps("strict", _env(0), _cfg())
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "Landlock" in err


def test_strict_with_landlock_is_silent(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent6.app.confine.unreachable_tools", tuple)
    warn_sandbox_gaps("strict", _env(2), _cfg())
    assert capsys.readouterr().err == ""


def test_unreachable_tool_is_named_once(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bin symlink whose target sits directly in $HOME cannot be mounted
    (mounting home would hand the jail every credential), so the tool dies in
    the jail with no explanation -- the preflight warning is the explanation."""
    monkeypatch.setattr(
        "agent6.app.confine.unreachable_tools",
        lambda: ("/home/op/.local/bin/x -> /home/op/x.sh",),
    )
    warn_sandbox_gaps("strict", _env(2), _cfg())
    err = capsys.readouterr().err
    assert "/home/op/.local/bin/x -> /home/op/x.sh" in err
    assert "never" in err and "mounted" in err


def test_hardened_auto_warns_tool_network_degrade(capsys: pytest.CaptureFixture[str]) -> None:
    """tool_network='auto' (the secure default) can't be offline on hardened
    (no netns), so it degrades to sharing the host network -- and must SAY so,
    never silently."""
    warn_sandbox_gaps("hardened", _env(4), _cfg("auto"))
    err = capsys.readouterr().err
    assert "WARNING" in err and "tool_network" in err and "network namespace" in err


def test_hardened_allow_says_nothing_about_the_network(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An operator who set tool_network='allow' asked for the tool to have the
    # network, so no degrade warning for it. `.git` is a separate degrade and
    # is expected here: hardened cannot protect it at all.
    warn_sandbox_gaps("hardened", _env(4), _cfg("allow"))
    err = capsys.readouterr().err
    assert "network" not in err.lower().split("cannot protect .git")[-1]
    assert "cannot protect .git" in err


def test_explicit_block_refuses_on_hardened() -> None:
    """tool_network='block' is an ENFORCE setting: it needs a netns only strict
    provides, so on hardened we refuse (name what's unsupported + the fix)
    rather than run silently under-confined. 'auto' degrades instead."""
    err = check_network_support(_cfg("block"), "hardened")
    assert err is not None
    assert "tool_network = 'block'" in err and "auto" in err and "strict" in err
    # auto is NOT refused (it degrades with a warning) -> None.
    assert check_network_support(_cfg("auto"), "hardened") is None
    # On strict, block is enforceable -> no refusal.
    assert check_network_support(_cfg("block"), "strict") is None


def test_unreachable_tools_scans_home_direct_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scanner reports a symlink resolving DIRECTLY into $HOME and stays
    quiet about one resolving into its own subdir (that parent mounts fine)."""
    from agent6.sandbox import jail as jail_mod

    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / "tools").mkdir()
    (home / "x.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "tools" / "y.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (home / ".local" / "bin" / "x").symlink_to(home / "x.sh")
    (home / ".local" / "bin" / "y").symlink_to(home / "tools" / "y.sh")
    monkeypatch.setattr(jail_mod.Path, "home", classmethod(lambda _cls: home))

    dropped = jail_mod.unreachable_tools()
    assert len(dropped) == 1
    assert dropped[0] == f"{home}/.local/bin/x -> {home}/x.sh"


def test_hardened_refuses_a_grant_containing_a_hidden_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Landlock has no deny rules: on hardened a granted region that CONTAINS
    a hidden dir would leave it readable -- silently ineffective security --
    so the run refuses, naming the pair. Strict masks it and runs."""
    from agent6.app.confine import check_hide_paths_support

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.chdir(tmp_path)
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(home),)))

    err = check_hide_paths_support(cfg, "hardened")
    assert err is not None and str(home) in err and str(cfg_dir) in err
    assert check_hide_paths_support(cfg, "strict") is None


def test_hardened_refuses_a_private_dir_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workspace is granted implicitly, so a config dir INSIDE it is
    readable on hardened with no grant listed at all -- verified live: a
    jailed `cat` printed secrets.toml. The refusal is what closes it."""
    from agent6.app.confine import check_hide_paths_support

    cfg_dir = tmp_path / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.chdir(tmp_path)

    err = check_hide_paths_support(Config(), "hardened")
    assert err is not None and str(cfg_dir) in err
    assert check_hide_paths_support(Config(), "strict") is None


def test_hardened_refuses_a_hide_entry_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.app.confine import check_hide_paths_support

    monkeypatch.chdir(tmp_path)
    hidden = tmp_path / "cred.txt"
    cfg = Config(sandbox=SandboxConfig(hide_paths=(str(hidden),)))
    err = check_hide_paths_support(cfg, "hardened")
    assert err is not None and str(hidden) in err
    assert check_hide_paths_support(cfg, "strict") is None


def test_plain_hardened_run_is_not_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No hidden path anywhere near a granted region: nothing to mask, so an
    ordinary hardened run is untouched."""
    from agent6.app.confine import check_hide_paths_support

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    assert check_hide_paths_support(Config(), "hardened") is None
