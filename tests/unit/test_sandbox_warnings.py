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
from agent6.sandbox.jail import ToolMountNotes


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
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    warn_sandbox_gaps("strict", _env(2), _cfg())
    assert capsys.readouterr().err == ""


def test_unreachable_tool_is_named_once(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bin symlink whose target sits directly in $HOME cannot be mounted
    (mounting home would hand the jail every credential), so the tool dies in
    the jail with no explanation -- the preflight warning is the explanation."""
    monkeypatch.setattr(
        "agent6.app.confine.tool_mount_notes",
        lambda: ToolMountNotes(unreachable=("/home/op/.local/bin/x -> /home/op/x.sh",)),
    )
    warn_sandbox_gaps("strict", _env(2), _cfg())
    err = capsys.readouterr().err
    assert "/home/op/.local/bin/x -> /home/op/x.sh" in err
    assert "never" in err and "mounted" in err


def test_a_tool_dragging_a_home_dir_into_the_jail_is_not_a_per_run_warning(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/bin/x -> ~/.ssh/helper` mounts ~/.ssh read-only into the jail, which
    stays ALLOWED (the operator placed the symlink, and guessing at which dirs
    hold keys would be enumerating badness). It is not warned per run either:
    on a normal machine every uv-installed tool in ~/.local/bin points into
    ~/.local/share, so this fired a dozen times a run and buried the messages
    that mattered. `agent6 check` lists it, where someone is asking."""
    monkeypatch.setattr(
        "agent6.app.confine.tool_mount_notes",
        lambda: ToolMountNotes(exposes_home_dir=("/home/op/.local/bin/x -> /home/op/.ssh/helper",)),
    )
    warn_sandbox_gaps("strict", _env(2), _cfg())
    assert capsys.readouterr().err == ""


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


def test_scanner_separates_unreachable_from_home_exposing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink resolving DIRECTLY into $HOME is unreachable (home is never
    mounted); one resolving into a home SUBDIR is reachable but drags that
    subdir in; one resolving inside its own bin dir is neither."""
    from agent6.sandbox import jail as jail_mod

    home = tmp_path / "home"
    binf = home / ".local" / "bin"
    binf.mkdir(parents=True)
    (home / "tools").mkdir()
    (home / "x.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "tools" / "y.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (binf / "z.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (binf / "x").symlink_to(home / "x.sh")
    (binf / "y").symlink_to(home / "tools" / "y.sh")
    (binf / "z").symlink_to(binf / "z.sh")
    monkeypatch.setattr(jail_mod.Path, "home", classmethod(lambda _cls: home))

    notes = jail_mod.tool_mount_notes()
    assert notes.unreachable == (f"{binf}/x -> {home}/x.sh",)
    assert notes.exposes_home_dir == (f"{binf}/y -> {home}/tools/y.sh",)


def test_hardened_warns_loudly_when_a_grant_exposes_the_private_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Granting a region containing the config dir is a choice the operator may
    mean: real protection remains on hardened (writes stay confined, seccomp
    applies), so refusing would be paternalism. It warns instead and names what
    becomes readable. Strict masks the same grant and says nothing."""
    from agent6.app.confine import check_hide_paths_support

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(home),)))

    warn_sandbox_gaps("hardened", _env(4), cfg)
    err = capsys.readouterr().err
    assert "WARNING" in err and "READ" in err
    assert str(cfg_dir) in err and str(home) in err
    assert check_hide_paths_support(cfg, "hardened") is None  # warned, not refused

    warn_sandbox_gaps("strict", _env(4), cfg)
    assert str(cfg_dir) not in capsys.readouterr().err


def test_the_workspace_itself_counts_as_a_granted_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verified live before this existed: with the config dir INSIDE the
    workspace, a jailed `cat` on hardened printed secrets.toml. The workspace
    is granted implicitly, so it has to be checked like any other region."""
    cfg_dir = tmp_path / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)

    warn_sandbox_gaps("hardened", _env(4), Config())
    assert str(cfg_dir) in capsys.readouterr().err


def test_hardened_refuses_an_explicit_hide_entry_it_cannot_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who wrote hide_paths down asked explicitly, so the rule the
    other knobs follow applies: a default degrades with a warning, an explicit
    value refuses rather than being silently ineffective."""
    from agent6.app.confine import check_hide_paths_support

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    hidden = ws / "cred.txt"
    cfg = Config(sandbox=SandboxConfig(hide_paths=(str(hidden),)))
    err = check_hide_paths_support(cfg, "hardened")
    assert err is not None and str(hidden) in err
    assert check_hide_paths_support(cfg, "strict") is None


def test_a_plain_hardened_run_neither_warns_nor_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.app.confine import check_hide_paths_support

    for var in ("CONFIG", "STATE", "DATA", "CACHE"):
        monkeypatch.setenv(f"AGENT6_{var}_HOME", str(tmp_path / var.lower()))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(tool_network="allow", protect_git=False))
    warn_sandbox_gaps("hardened", _env(4), cfg)
    assert capsys.readouterr().err == ""
    assert check_hide_paths_support(cfg, "hardened") is None
