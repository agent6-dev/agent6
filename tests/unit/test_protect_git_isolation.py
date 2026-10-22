# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`protect_git = "on"` requires `full` `sandbox.isolation` to bind-remount `.git` as read-only.

Landlock has no deny rules: protecting `.git` on `partial` meant not granting
the workspace ROOT, because a grant is recursive and granting the root its own
create/remove rights grants them over `.git` too. That cost every top-level
write, which is more than the protection is worth when the operator can have
the real thing by using `full`.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6.app import confine
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.sandbox import detect


def _said() -> tuple[app_reporter.Reporter, list[str]]:
    lines: list[str] = []
    return app_reporter.Reporter(out=lines.append, err=lines.append), lines


def _env_without_namespaces() -> detect.Environment:
    return detect.Environment(
        in_container=False,
        container_signals=(),
        kernel=detect.KernelInfo(raw="7.0.0", major=7, minor=0),
        userns_supported=False,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def test_auto_degrades_with_a_warning(tmp_path: pathlib.Path) -> None:
    """`auto` is the default, and a default this host cannot honour runs loudly, never silently."""
    reporter, lines = _said()
    confine.warn_sandbox_gaps(
        "partial", _env_without_namespaces(), Config(), reporter=reporter, root=tmp_path
    )
    assert any("jailed commands can write .git" in line for line in lines)
    assert confine.check_protect_git_support(Config(), "partial") is None


@pytest.mark.parametrize("isolation", ["partial", "none"])
def test_on_refuses_where_it_cannot_be_honoured(isolation: str) -> None:
    """`on` asks for something specific; running without it would be a lie.

    The value carries the demand, so a config written by `config fill` or by hand behaves the
    same: nothing depends on which layer the word came from.
    """
    on = Config.model_validate({"sandbox": {"protect_git": "on"}})
    message = confine.check_protect_git_support(on, isolation)  # pyright: ignore[reportArgumentType]
    assert message is not None
    assert "requires sandbox.isolation = 'full'" in message
    assert "sandbox.protect_git = 'auto'" in message  # names the fix


def test_full_provides_it_for_auto_and_on() -> None:
    for word in ("auto", "on"):
        cfg = Config.model_validate({"sandbox": {"protect_git": word}})
        assert confine.check_protect_git_support(cfg, "full") is None


def test_off_never_refuses() -> None:
    off = Config.model_validate({"sandbox": {"protect_git": "off"}})
    assert confine.check_protect_git_support(off, "partial") is None


def test_auto_resolves_off_when_the_model_controls_git(tmp_path: pathlib.Path) -> None:
    """`git.control = "model"` needs a writable .git; `auto` yields to it and `on` is refused."""
    from agent6.tools import policy

    (tmp_path / ".git").mkdir()
    model = Config.model_validate({"git": {"control": "model"}})
    assert policy.jail_policy(tmp_path, model, "full", ("true",)).extra_protect_paths == ()
    with pytest.raises(ValueError, match="protect_git"):
        Config.model_validate({"git": {"control": "model"}, "sandbox": {"protect_git": "on"}})


def test_only_full_carves_git_out_of_the_jail(tmp_path: pathlib.Path) -> None:
    """Top-level writes work on `partial` because there is no carve-out.

    A carve-out at the workspace root is what would cost them; `touch newfile` there works.
    """
    from agent6.tools import policy

    (tmp_path / ".git").mkdir()
    full = policy.jail_policy(tmp_path, Config(), "full", ("true",))
    partial = policy.jail_policy(tmp_path, Config(), "partial", ("true",))
    assert (tmp_path / ".git").resolve() in full.extra_protect_paths
    assert partial.extra_protect_paths == ()


@pytest.mark.needs_namespaces
def test_a_new_top_level_entry_can_be_created_on_partial(tmp_path: pathlib.Path) -> None:
    """A new top-level entry can be created on `partial`.

    `mkfifo`, `touch` and `mkdir` at the workspace root must not fail with a misleading "File
    exists" while the carve-out is there.
    """
    from agent6.sandbox import jail
    from agent6.tools import policy

    (tmp_path / ".git").mkdir()
    res = jail.run_in_jail(
        policy.jail_policy(
            tmp_path,
            Config(),
            "partial",
            ("sh", "-c", "touch newfile && mkdir build && mkfifo pipe && echo made-them"),
        )
    )
    assert "made-them" in res.stdout, res.stdout + res.stderr
