# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`warn_sandbox_gaps`: the run-entry warning when the isolation confines less than its name.

`none`, strict without Landlock, or hardened on Landlock below ABI 3, where truncation is
unconfined.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.app import confine
from agent6.config import Config, SandboxConfig
from agent6.sandbox import detect, tool_paths


def _env(landlock_abi: int) -> detect.Environment:
    return detect.Environment(
        in_container=False,
        container_signals=(),
        kernel=detect.KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=landlock_abi,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def _cfg(tool_network: str = "auto", isolation: str = "auto") -> Config:
    return Config(
        sandbox=SandboxConfig(network=tool_network, isolation=isolation)  # type: ignore[arg-type]
    )


def test_none_warns_unsandboxed(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The list of what is absent has to be complete, and the memory cap is not on it.

    Measured with `memory_limit_mb = 64` and isolation `none`: a 400 MB allocation raises
    MemoryError through the run's jail session (the launcher applies the rlimit with confinement
    off), and succeeds only on the one-shot path, which runs a plain subprocess.
    """
    confine.warn_sandbox_gaps("none", _env(4), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "UNSANDBOXED" in err
    assert "memory_limit_mb" in err


def test_strict_without_landlock_warns(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Strict on a Landlock-less kernel (ABI 0) silently lost a documented layer.

    The launcher's best-effort ruleset enforces nothing and no surface said so, breaking the "no
    silent downgrade, always loudly" contract.
    """
    confine.warn_sandbox_gaps("strict", _env(0), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "Landlock" in err


def test_strict_with_landlock_is_silent(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    confine.warn_sandbox_gaps("strict", _env(2), _cfg(), root=tmp_path)
    assert capsys.readouterr().err == ""


def test_unreachable_tool_is_named_once(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable tool is named once.

    A bin symlink whose target sits directly in $HOME cannot be mounted (mounting home would hand
    the jail every credential), so the tool dies in the jail with no explanation; the preflight
    warning is the explanation.
    """
    monkeypatch.setattr(
        "agent6.app.confine.tool_mount_notes",
        lambda: tool_paths.ToolMountNotes(unreachable=("/home/op/.local/bin/x -> /home/op/x.sh",)),
    )
    confine.warn_sandbox_gaps("strict", _env(2), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "/home/op/.local/bin/x -> /home/op/x.sh" in err
    assert "never" in err and "mounted" in err


def test_a_tool_dragging_a_home_dir_into_the_jail_is_not_a_per_run_warning(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool dragging a home dir into the jail is not a per-run warning.

    `~/bin/x -> ~/.ssh/helper` mounts ~/.ssh read-only into the jail, which stays allowed: the
    operator placed the symlink, and guessing which dirs hold keys would be enumerating badness. On
    a normal machine every uv-installed tool in ~/.local/bin points into ~/.local/share, so a per-
    run warning fires a dozen times and buries the messages that matter; `agent6 check` lists it,
    where someone is asking.
    """
    monkeypatch.setattr(
        "agent6.app.confine.tool_mount_notes",
        lambda: tool_paths.ToolMountNotes(
            exposes_home_dir=("/home/op/.local/bin/x -> /home/op/.ssh/helper",)
        ),
    )
    confine.warn_sandbox_gaps("strict", _env(2), _cfg(), root=tmp_path)
    assert capsys.readouterr().err == ""


def test_hardened_auto_warns_tool_network_degrade(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hardened with network='auto' says it degrades to the host network.

    The secure default cannot be offline on hardened (no netns); the degrade is never silent.
    """
    confine.warn_sandbox_gaps("hardened", _env(4), _cfg("auto"), root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err and "network" in err and "network namespace" in err


@pytest.mark.parametrize("abi", [1, 2])
def test_hardened_below_abi3_warns_truncate_unconfined(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], abi: int
) -> None:
    """Hardened below Landlock ABI 3 warns that truncate is unconfined.

    ABI 1 and 2 do not confine truncate, so a jailed command can truncate files outside its write
    grants; `auto` still resolves to hardened there, so the over-promise is said once per run,
    naming ABI 3 and Linux 6.2.
    """
    confine.warn_sandbox_gaps("hardened", _env(abi), _cfg("host"), root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err and "truncat" in err
    assert "ABI 3" in err and "6.2" in err


def test_hardened_abi3_plus_is_silent_on_truncate(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """From ABI 3 up Landlock confines truncation, so no truncate warning."""
    confine.warn_sandbox_gaps("hardened", _env(3), _cfg("host"), root=tmp_path)
    assert "truncat" not in capsys.readouterr().err


def test_hardened_allow_says_nothing_about_the_network(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An operator who set network='allow' asked for the tool to have the
    # network, so no degrade warning for it. `.git` is a separate degrade and
    # is expected here: hardened cannot protect it at all.
    confine.warn_sandbox_gaps("hardened", _env(4), _cfg("host"), root=tmp_path)
    err = capsys.readouterr().err
    assert "network" not in err.lower().split("cannot protect .git")[-1]
    assert "cannot protect .git" in err


def test_hardened_warning_names_shared_tmp_and_persistent_home(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hardened warnings name the shared /tmp and the persistent HOME.

    Strict gives each run a private /tmp tmpfs with HOME (/tmp/agent6-home) inside it, gone when the
    run ends; hardened has no mount namespace, so /tmp is the host's and HOME is the persistent
    cache dir. The HOME warning stands on its own rather than riding the .git warning, which
    `protect_git = false` removes.
    """
    for cfg in (_cfg(), Config(sandbox=SandboxConfig(protect_git=False))):
        confine.warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
        err = capsys.readouterr().err
        assert str(paths.jail_cache_home()) in err
        assert "/tmp/agent6-home" not in err
        assert "persists across runs" in err and "executable" in err
        assert ("cannot protect .git" in err) == cfg.sandbox.protect_git


def test_strict_cache_home_warns_naming_the_cost(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`home = "cache"` under strict is an explicit widening.

    It runs, with a loud warning naming the persistence and the executable grant. The default strict
    HOME warns nothing.
    """
    confine.warn_sandbox_gaps(
        "strict", _env(4), Config(sandbox=SandboxConfig(home="cache")), root=tmp_path
    )
    err = capsys.readouterr().err
    assert "sandbox.home = 'cache'" in err and str(paths.jail_cache_home()) in err
    assert "persists across runs" in err and "executable" in err
    confine.warn_sandbox_gaps("strict", _env(4), _cfg(), root=tmp_path)
    assert str(paths.jail_cache_home()) not in capsys.readouterr().err


def test_a_cleartext_credential_endpoint_warns_at_run_entry(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cleartext credential endpoint warns once at run entry and never refuses.

    Sending a credential over plaintext http to a non-loopback host is explicit but discouraged
    config: the warning names the endpoint and the cost. A clean config warns nothing.
    """
    cfg = Config.model_validate(
        {
            "providers": {
                "corp": {
                    "api_format": "openai",
                    "base_url": "http://llm.corp.internal/v1",
                    "api_key_env": "K",
                }
            }
        }
    )
    confine.warn_cleartext_credential_endpoints(cfg)
    err = capsys.readouterr().err
    assert "WARNING" in err and "[providers.corp]" in err and "plaintext http" in err
    confine.warn_cleartext_credential_endpoints(Config())
    assert capsys.readouterr().err == ""


def test_explicit_block_refuses_on_hardened(tmp_path: pathlib.Path) -> None:
    """network='session' is an ENFORCE setting.

    It needs a netns only strict provides, so on hardened we refuse (name what's unsupported + the
    fix) rather than run silently under-confined. 'auto' degrades instead.
    """
    err = confine.check_network_support(_cfg("session"), "hardened")
    assert err is not None
    assert "sandbox.network = 'session'" in err and "auto" in err and "strict" in err
    # auto is NOT refused (it degrades with a warning) -> None.
    assert confine.check_network_support(_cfg("auto"), "hardened") is None
    # On strict, block is enforceable -> no refusal.
    assert confine.check_network_support(_cfg("session"), "strict") is None


def test_scanner_separates_unreachable_from_home_exposing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scanner separates an unreachable symlink from a home-exposing one.

    A symlink resolving directly into $HOME is unreachable (home is never mounted); one resolving
    into a home subdir is reachable but drags that subdir in; one resolving inside its own bin dir
    is neither.
    """
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
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda _cls: home))

    notes = tool_paths.tool_mount_notes()
    assert notes.unreachable == (f"{binf}/x -> {home}/x.sh",)
    assert notes.exposes_home_dir == (f"{binf}/y -> {home}/tools/y.sh",)


def test_hardened_warns_loudly_when_a_grant_exposes_the_private_dirs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Granting a region containing the config dir is a choice the operator may mean.

    Real protection remains on hardened (writes stay confined, seccomp applies), so refusing would
    be paternalism. It warns instead and names what becomes readable. Strict masks the same grant
    and says nothing.
    """
    home = tmp_path / "home"
    cfg_dir = home / ".config" / "agent6"
    (cfg_dir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(home),)))

    confine.warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err and "can read" in err
    assert str(cfg_dir) in err and str(home) in err
    assert (
        confine.check_hide_paths_support(cfg, "hardened", tmp_path) is None
    )  # warned, not refused

    confine.warn_sandbox_gaps("strict", _env(4), cfg, root=tmp_path)
    assert str(cfg_dir) not in capsys.readouterr().err


def test_the_workspace_itself_counts_as_a_granted_region(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verified live before this existed.

    With the config dir INSIDE the workspace, a jailed `cat` on hardened printed secrets.toml. The
    workspace is granted implicitly, so it has to be checked like any other region.
    """
    cfg_dir = tmp_path / ".config" / "agent6"
    (cfg_dir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)

    confine.warn_sandbox_gaps("hardened", _env(4), Config(), root=tmp_path)
    assert str(cfg_dir) in capsys.readouterr().err


def test_hardened_refuses_an_explicit_hide_entry_it_cannot_mask(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardened refuses an explicit hide_paths entry it cannot mask.

    An operator who wrote hide_paths down asked explicitly: a default degrades with a warning, an
    explicit value refuses rather than being silently ineffective.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    hidden = ws / "cred.txt"
    cfg = Config(sandbox=SandboxConfig(hide_paths=(str(hidden),)))
    err = confine.check_hide_paths_support(cfg, "hardened", tmp_path)
    assert err is not None and str(hidden) in err
    assert confine.check_hide_paths_support(cfg, "strict", tmp_path) is None


def test_a_plain_hardened_run_neither_warns_nor_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Private homes OUTSIDE every hardened grant region (/tmp is granted RW,
    # so a tmp-based home is genuinely exposed there -- the twin below pins
    # that as a true positive). The normal ~/.local layout is this case. The
    # cache stays where the suite put it: the jail's HOME lives there, and the
    # policy build creates it.
    for var in ("CONFIG", "STATE", "DATA"):
        monkeypatch.setenv(f"XDG_{var}_HOME", f"/nonexistent-private/{var.lower()}")
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(network="host", protect_git=False))
    confine.warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
    err = capsys.readouterr().err
    # hardened's persistent HOME is the one notice every such run carries;
    # nothing here is an exposure.
    assert err.count("WARNING") == 1 and str(paths.jail_cache_home()) in err, err
    assert "can read" not in err
    assert confine.check_hide_paths_support(cfg, "hardened", tmp_path) is None


def test_hardened_warns_when_private_state_sits_in_a_granted_region(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hardened warns when private state sits in a granted region.

    The hardened launcher grants the host's shared /tmp read-write, so private dirs under it are
    readable by every command; a preflight that sees only cwd and the extra grants misses them.
    """
    for var in ("CONFIG", "STATE", "DATA", "CACHE"):
        monkeypatch.setenv(f"AGENT6_{var}_HOME", str(tmp_path / var.lower()))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(network="host", protect_git=False))
    confine.warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
    err = capsys.readouterr().err
    assert str(tmp_path).startswith("/tmp"), "the fixture premise: pytest tmp lives under /tmp"
    assert "jailed commands can read" in err and "/tmp" in err


def test_root_on_hardened_names_what_it_costs(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root on hardened warns and names what it costs.

    Running as root is the operator's explicit widening, so it warns rather than refuses, and the
    warning names the cost, not just the choice. Verified as real uid 0: under hardened a jailed
    command reads /etc/shadow, /etc/sudoers and the host's ssh private keys, because Landlock grants
    the documented read-only system set and root stops file permissions narrowing it. The root
    banner names running as root; it does not name this.
    """
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    monkeypatch.setattr("agent6.app.confine.is_root", lambda: True)
    confine.warn_sandbox_gaps(
        "hardened", _env(4), Config(sandbox=SandboxConfig(protect_git=False)), root=tmp_path
    )
    err = capsys.readouterr().err
    assert "running as root" in err
    assert "/etc/shadow" in err and "ssh private keys" in err


def test_root_on_strict_says_nothing_about_it(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root on strict says nothing about it.

    Strict pivots into a minimal rootfs: verified as real uid 0, its /etc holds a single entry and
    none of those files exist. A warning there would name a cost the operator is not paying.
    """
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    monkeypatch.setattr("agent6.app.confine.is_root", lambda: True)
    confine.warn_sandbox_gaps("strict", _env(4), Config(), root=tmp_path)
    assert capsys.readouterr().err == ""


def test_a_normal_user_on_hardened_is_not_told_about_root(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)
    monkeypatch.setattr("agent6.app.confine.is_root", lambda: False)
    confine.warn_sandbox_gaps(
        "hardened", _env(4), Config(sandbox=SandboxConfig(protect_git=False)), root=tmp_path
    )
    assert "running as root" not in capsys.readouterr().err


def test_auto_degrade_warns_with_the_reason(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The degrade ITSELF is loud, not only its consequences.

    Auto landing on hardened printed the network/protect_git consequences but never why strict was
    skipped. One owner (detect.degrade_reason) feeds this line, check sandbox, and check config.
    """
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)

    def _why(_env: object) -> str:
        return "userns blocked (test)"

    monkeypatch.setattr("agent6.app.confine.degrade_reason", _why)
    confine.warn_sandbox_gaps("hardened", _env(4), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "'auto' selected 'hardened', not 'strict': userns blocked (test)" in err


def test_explicit_hardened_has_no_degrade_line(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who WROTE hardened chose it; nothing degraded."""
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)

    def _why(_env: object) -> str:
        return "userns blocked (test)"

    monkeypatch.setattr("agent6.app.confine.degrade_reason", _why)
    confine.warn_sandbox_gaps("hardened", _env(4), _cfg(isolation="hardened"), root=tmp_path)
    err = capsys.readouterr().err
    assert "not 'strict'" not in err


def test_unsandboxed_origin_says_auto_or_the_operator(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UNSANDBOXED banner attributes `isolation = 'none'` to `auto` or to the operator.

    `auto` resolves there on a host with no confinement mechanism; that is not the operator's
    choice.
    """
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", tool_paths.ToolMountNotes)

    def _why(_env: object) -> str:
        return "nothing here (test)"

    monkeypatch.setattr("agent6.app.confine.degrade_reason", _why)
    confine.warn_sandbox_gaps("none", _env(0), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "'auto' found no confinement mechanism" in err
    assert "sandbox.isolation = 'none'" not in err
    confine.warn_sandbox_gaps("none", _env(0), _cfg(isolation="none"), root=tmp_path)
    err = capsys.readouterr().err
    assert "sandbox.isolation = 'none'" in err
