# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI wiring of the per-invocation sandbox/approval opt-outs.

Unit tests already cover the pieces (detect.resolve_isolation env setter,
Config.with_sandbox_overrides, the confirm gate). These cover the wiring the
CLI does on top: parsing the flags into _SandboxOverrides and applying them,
and the env mechanism a `machine run` relies on to reach its agent subprocesses.
"""

from __future__ import annotations

import argparse

import pytest

from agent6.app import session_setup
from agent6.config import Config
from agent6.sandbox import detect


def _args(**kw: bool) -> argparse.Namespace:
    return argparse.Namespace(**kw)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--disable-sandbox", "task"],
        ["resume", "--disable-sandbox", "session"],
        ["fork", "--disable-sandbox", "session"],
        ["plan", "--disable-sandbox", "task"],
        ["ask", "--disable-sandbox", "question"],
        ["exec", "--new", "--disable-sandbox", "--", "ls", "/"],
        ["exec", "--disable-sandbox", "session", "--", "ls", "/"],
        ["machine", "run", "--disable-sandbox", "loop.asm.toml"],
    ],
)
def test_every_command_runner_accepts_the_sandbox_opt_out(argv: list[str]) -> None:
    """The operator's explicit sandbox opt-out has the same spelling on every runner."""
    from agent6.ui.cli import parser as cli_parser

    args = cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(argv))
    assert args.disable_sandbox is True


def test_from_args_reads_both_flags() -> None:
    o = session_setup.SandboxOverrides.from_args(_args(disable_sandbox=True, auto_approve=True))
    assert o.disable_sandbox is True
    assert o.auto_approve is True


def test_from_args_defaults_false_when_flags_absent() -> None:
    # Commands that do not offer the flags (or an older namespace) get False.
    o = session_setup.SandboxOverrides.from_args(_args())
    assert o.disable_sandbox is False
    assert o.auto_approve is False


def test_apply_flag_path_forces_none_and_auto_approve() -> None:
    cfg = Config()
    assert cfg.sandbox.isolation == "auto"
    out = session_setup.SandboxOverrides(disable_sandbox=True, auto_approve=True).apply(cfg)
    assert out.sandbox.isolation == "none"
    assert out.sandbox.run_commands == "allow"


def test_apply_noop_when_no_flags() -> None:
    cfg = Config()
    assert session_setup.SandboxOverrides().apply(cfg) is cfg


def _linux_env() -> detect.Environment:
    return detect.Environment(
        in_container=False,
        container_signals=(),
        kernel=detect.KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def test_machine_env_mechanism_forces_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # AGENT6_DISABLE_SANDBOX=1 makes `resolve_isolation` return `none` whatever the config says,
    # and `machine run` passes that level to each agent subprocess.
    monkeypatch.setenv("AGENT6_DISABLE_SANDBOX", "1")
    assert detect.resolve_isolation("full", _linux_env()) == "none"
    assert detect.resolve_isolation("auto", _linux_env()) == "none"


def test_overrides_render_as_the_flags_that_set_them() -> None:
    """Overrides render as the flags that set them.

    A continuation this invocation spawns (a detached resume) carries the overrides as flags; an
    execution that drops them runs under the config's defaults (an `--auto-approve` run detaches and
    waits on its first approval; a `--max-usd` cap falls back to the config's).
    """
    sandbox = session_setup.SandboxOverrides(disable_sandbox=True, auto_approve=True)
    budget = session_setup.BudgetOverrides(max_usd=0.25, max_tokens_fallback=1000)
    assert sandbox.argv() == ["--disable-sandbox", "--auto-approve"]
    assert budget.argv() == ["--max-usd", "0.25", "--max-tokens-fallback", "1000"]
    assert session_setup.override_flags(budget, sandbox, None) == [*budget.argv(), *sandbox.argv()]
    assert session_setup.override_flags(None, None, None) == []
    assert (
        session_setup.SandboxOverrides().argv() == []
        and session_setup.BudgetOverrides().argv() == []
    )
