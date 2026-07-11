# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`--config FILE` parses in both positions for run/plan/resume/check.

The documented `agent6 run --config FILE` (config after the subcommand) used to
error; and a subparser `default=None` would clobber the top-level
`agent6 --config FILE run` form back to None. Both must now set `args.config`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli.parser import (
    _inject_default_verb,  # pyright: ignore[reportPrivateUsage]
    build_parser,
)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--config", "c.toml", "task"],
        ["--config", "c.toml", "run", "task"],
        # `plan` carries --config/task on its implicit `run` verb (see
        # _inject_default_verb), which `main` applies before parsing.
        ["plan", "--config", "c.toml", "task"],
        ["--config", "c.toml", "plan", "task"],
        ["resume", "rid", "--config", "c.toml"],
        ["--config", "c.toml", "resume", "rid"],
        ["check", "--config", "c.toml"],
        ["--config", "c.toml", "check"],
    ],
)
def test_config_flag_parses_in_both_positions(argv: list[str]) -> None:
    args = build_parser().parse_args(_inject_default_verb(argv))
    assert args.config == Path("c.toml")


def test_config_defaults_to_none_when_absent() -> None:
    args = build_parser().parse_args(["run", "task"])
    assert args.config is None


def test_run_decompose_flag_defaults_off_and_parses() -> None:
    # --decompose is plan-first (overrides [prompt].decompose for the run); off by default.
    p = build_parser()
    assert p.parse_args(["run", "fix it"]).decompose is False
    assert p.parse_args(["run", "--decompose", "fix it"]).decompose is True
