# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`probed_environment` asks the jail launcher itself whether `full` works.

`probe_userns_supported` runs `/usr/bin/unshare`, which is wrong in both directions: an AppArmor
profile can grant the jail binary userns but not unshare, and inside Docker with a relaxed seccomp
profile unshare succeeds while AppArmor denies the jail's mount.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from agent6.config import Config
from agent6.sandbox import detect, jail
from agent6.tools import policy


def _env(userns: bool, *, sandbox: bool = True) -> detect.Environment:
    return detect.Environment(
        in_container=False,
        container_signals=(),
        kernel=detect.KernelInfo(raw="7.0.0", major=7, minor=0),
        userns_supported=userns,
        landlock_abi=4 if sandbox else 0,
        seccomp_arch_supported=True,
        sandbox_available=sandbox,
    )


def _fail_probe() -> bool:
    pytest.fail("namespaces_work should not be called here")


def test_trial_launch_failure_reaches_refusal_and_degrade_reason(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed full-jail launch keeps its error, even when the unshare probe succeeds."""
    fake = tmp_path / "failjail"
    failure = "agent6-jail: fork failed: EAGAIN"
    fake.write_text(f"#!/bin/sh\necho '{failure}' >&2\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))
    monkeypatch.delenv("AGENT6_DISABLE_SANDBOX", raising=False)
    monkeypatch.setattr(detect, "environment", lambda: _env(True))
    monkeypatch.setattr(detect, "apparmor_userns_restricted", lambda: True)
    jail.namespaces_work.cache_clear()
    try:
        env = detect.probed_environment()
        assert detect.resolve_isolation("auto", env) == "partial"
        reason = detect.degrade_reason(env) or ""
        assert "trial full jail launch" in reason
        assert repr(f"agent6-jail launcher exited 1: {failure}") in reason
        assert "AppArmor" not in reason
        with pytest.raises(detect.IsolationUnavailableError) as caught:
            detect.resolve_isolation("full", env)
        assert reason in str(caught.value)
    finally:
        jail.namespaces_work.cache_clear()


def test_probe_does_not_probe_where_there_is_no_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-Linux host has no jail to ask."""
    monkeypatch.setattr(detect, "environment", lambda: _env(False, sandbox=False))
    monkeypatch.setattr(jail, "namespaces_work", _fail_probe)
    assert detect.probed_environment().detected_isolation == "none"


def test_probe_keeps_userns_when_the_jail_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect, "environment", lambda: _env(True))
    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(True))
    assert detect.probed_environment().userns_supported is True


def test_probe_drops_to_partial_when_the_jail_cannot_do_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A jail that fails its own probe resolves `auto` down, never announces `full` and then dies.

    The Docker case: `unshare` succeeds under a relaxed seccomp profile and AppArmor then denies the
    jail's `mount`.
    """
    monkeypatch.setattr(detect, "environment", lambda: _env(True))
    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(False, "trial failed"))
    env = detect.probed_environment()
    assert env.userns_supported is False
    assert env.detected_isolation == "partial"


def test_probe_upgrades_to_full_via_jail_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The AppArmor-isolation case: unshare blocked, but the jail binary can userns.
    monkeypatch.setattr(detect, "environment", lambda: _env(False))
    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(True))
    assert detect.probed_environment().userns_supported is True


def test_probe_refuses_over_a_binary_it_cannot_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe answers only the namespace question; an unusable binary's own refusal propagates.

    A 0644 or wrong-architecture AGENT6_JAIL_BIN is not "userns supported: False".
    """
    from agent6.app import preflight, session_runtime
    from agent6.app import reporter as app_reporter
    from agent6.config import Config

    fake = tmp_path / "agent6-jail"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o644)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))
    jail.namespaces_work.cache_clear()
    try:
        with pytest.raises(jail.JailUnavailableError, match=re.escape(str(fake))):
            detect.probed_environment()
        said: list[str] = []
        reporter = app_reporter.Reporter(out=said.append, err=said.append)
        with pytest.raises(preflight.SessionRefusedError):
            session_runtime.select_isolation(
                Config(),
                cwd=tmp_path,
                confirm_unconfined=lambda _level, _cfg: True,
                reporter=reporter,
            )
        assert any(str(fake) in line and "uv sync --reinstall-package" in line for line in said)
    finally:
        jail.namespaces_work.cache_clear()


def test_probe_stays_partial_when_jail_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect, "environment", lambda: _env(False))
    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(False, "trial failed"))
    assert detect.probed_environment().userns_supported is False


def test_probe_skips_probe_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "environment", lambda: _env(False, sandbox=False))
    monkeypatch.setattr(jail, "namespaces_work", _fail_probe)  # not consulted
    assert detect.probed_environment().userns_supported is False


def test_degrade_reason_full_strength_is_none() -> None:
    assert detect.degrade_reason(_env(True)) is None


def test_degrade_reason_names_the_blocking_mechanism(monkeypatch: pytest.MonkeyPatch) -> None:
    """The degrade reason names the mechanism, since each has a different fix.

    AppArmor sysctl, max_user_namespaces or container policy; every reporting surface prints the
    line.
    """
    e = _env(False)
    monkeypatch.setattr(detect, "apparmor_userns_restricted", lambda: True)
    assert "apparmor_restrict_unprivileged_userns" in (detect.degrade_reason(e) or "")
    assert "agent6 system apparmor install" in (detect.degrade_reason(e) or "")

    monkeypatch.setattr(detect, "apparmor_userns_restricted", lambda: False)
    monkeypatch.setattr(detect, "_read_max_userns", lambda: "0")
    assert "user.max_user_namespaces = 0" in (detect.degrade_reason(e) or "")

    monkeypatch.setattr(detect, "_read_max_userns", lambda: "58135")
    import dataclasses

    assert "container" in (detect.degrade_reason(dataclasses.replace(e, in_container=True)) or "")
    assert "unshare -U -r true" in (detect.degrade_reason(e) or "")


def test_degrade_reason_covers_the_landlock_less_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "apparmor_userns_restricted", lambda: False)
    monkeypatch.setattr(detect, "_read_max_userns", lambda: None)
    import dataclasses

    no_landlock = dataclasses.replace(_env(False), landlock_abi=0)
    reason = detect.degrade_reason(no_landlock) or ""
    assert "no Landlock" in reason
    assert detect.degrade_reason(_env(False, sandbox=False)) is not None


def test_unsupported_seccomp_arch_degrades_auto_and_refuses_explicit() -> None:
    """With no seccomp filter (off x86_64 and aarch64) `auto` degrades to `none`, `full` refuses."""
    import dataclasses

    e = dataclasses.replace(_env(True), seccomp_arch_supported=False)
    assert e.detected_isolation == "none"
    assert "seccomp filter" in (detect.degrade_reason(e) or "")
    for requested in ("full", "partial"):
        with pytest.raises(detect.IsolationUnavailableError, match="seccomp filter"):
            detect.resolve_isolation(requested, e)
    # auto and the explicit opt-out still resolve.
    assert detect.resolve_isolation("auto", e) == "none"
    assert detect.resolve_isolation("none", e) == "none"


def test_a_missing_launcher_degrades_auto_and_refuses_the_real_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No launcher: `auto` and `none` run unsandboxed with the cause; the real levels refuse."""
    monkeypatch.delenv("AGENT6_DISABLE_SANDBOX", raising=False)
    monkeypatch.setattr(detect, "environment", lambda: _env(True))
    monkeypatch.setattr(jail, "locate_jail_binary", lambda: None)
    monkeypatch.setattr(jail, "namespaces_work", _fail_probe)
    env = detect.probed_environment()
    assert env.launcher_found is False
    assert detect.resolve_isolation("none", env) == "none"
    assert detect.resolve_isolation("auto", env) == "none"
    assert "agent6-jail launcher" in (detect.degrade_reason(env) or "")
    for level in ("full", "partial"):
        with pytest.raises(detect.IsolationUnavailableError, match="agent6-jail launcher"):
            detect.resolve_isolation(level, env)


def test_a_disabled_sandbox_never_launches_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AGENT6_DISABLE_SANDBOX=1` needs no launcher at all."""
    monkeypatch.setenv("AGENT6_DISABLE_SANDBOX", "1")
    monkeypatch.setattr(detect, "environment", lambda: _env(True))
    monkeypatch.setattr(jail, "locate_jail_binary", lambda: None)
    monkeypatch.setattr(jail, "namespaces_work", _fail_probe)
    assert detect.resolve_isolation("auto", detect.probed_environment()) == "none"


def test_grants_follow_the_launchers_verdict_on_namespaces(
    tmp_path: pathlib.Path, host_tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unshare` says `full`, the launcher says `partial`: a grant under the host's /tmp drops."""
    monkeypatch.delenv("AGENT6_DISABLE_SANDBOX", raising=False)
    monkeypatch.setattr(detect, "environment", lambda: _env(True))
    monkeypatch.setattr(jail, "locate_jail_binary", lambda: pathlib.Path("/usr/bin/true"))
    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(False, "trial failed"))
    root = tmp_path / "repo"
    root.mkdir()
    grant = host_tmp_path / "grant"
    grant.mkdir()
    assert not grant.is_relative_to(root.resolve())
    cfg = Config.model_validate({"sandbox": {"extra_write_paths": [str(grant)]}})
    grants = policy.resolve_grants(cfg, root)
    assert grants.write == ()
    (dropped,) = grants.dropped
    host_tmp = pathlib.Path("/tmp").resolve()
    assert (dropped.path, dropped.real, dropped.region) == (grant, grant, host_tmp)
    assert f"it sits inside {host_tmp}, which this run's commands write" in dropped.line()

    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(True))
    kept = policy.resolve_grants(cfg, root)
    assert kept.write == ((grant, grant),)
    assert not kept.dropped
