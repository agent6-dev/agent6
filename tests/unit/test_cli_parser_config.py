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


def test_history_bare_query_defaults_to_search() -> None:
    # `history "divide"` == `history search "divide"` (search is history's one
    # obvious action), like `runs`->list and bare `ask`.
    args = build_parser().parse_args(_inject_default_verb(["history", "divide"]))
    assert args.history_command == "search" and args.query == "divide"


def test_history_explicit_search_still_works() -> None:
    args = build_parser().parse_args(_inject_default_verb(["history", "search", "divide"]))
    assert args.history_command == "search" and args.query == "divide"
    # A flag after the bare query is carried onto the injected verb too.
    a2 = build_parser().parse_args(_inject_default_verb(["history", "--regex", "d.v"]))
    assert a2.history_command == "search" and a2.query == "d.v" and a2.regex is True


def test_config_get_does_not_offer_keys_it_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completer must offer what the command accepts, and nothing else.

    `[presets.*]` tables are stripped before validation, so they are not
    effective-config leaves: `config get presets.mine.sandbox.tool_network`
    errors with "is not a config leaf". The shared completer offered exactly
    those keys, so TAB proposed an input the command refuses. They stay on the
    write verbs, where they ARE accepted.
    """
    from agent6.ui.cli.completers import (
        _complete_config_keys,  # pyright: ignore[reportPrivateUsage]
    )

    (tmp_path / "config.toml").write_text(
        '[presets.mine.sandbox]\nrun_commands = "yes"\n', encoding="utf-8"
    )
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    for_set = _complete_config_keys("presets.")
    for_get = _complete_config_keys("presets.", include_presets=False)

    assert any(k.startswith("presets.mine.") for k in for_set), "the write verbs still offer them"
    assert not any(k.startswith("presets.") for k in for_get), f"get offered: {for_get[:3]}"


def test_fork_carries_the_same_sandbox_flags_as_its_siblings() -> None:
    """`_add_sandbox_flags` says "every paid command carries both:
    run/plan/ask/resume and machine run". A fork without `--no-run` CONTINUES a
    run, so it is one -- but it registered only the budget flags, and
    `agent6 fork --auto-approve <id>` died on "unrecognized arguments".

    Loud, not silent, which is why this is a consistency gap rather than a lie.
    But an operator who forks a run they had auto-approved should not have to
    fork with --no-run and then resume just to say so again.
    """
    from agent6.app._setup import SandboxOverrides
    from agent6.ui.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["fork", "--auto-approve", "some-session-id"])
    assert SandboxOverrides.from_args(args).auto_approve is True

    args = parser.parse_args(["fork", "--no-commands", "some-session-id"])
    assert SandboxOverrides.from_args(args).no_commands is True


def test_get_completion_offers_no_key_get_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract this file already states: a completer offers what the
    command accepts, and nothing else.

    The enum keys exist so `config set` can reach a leaf no layer has set yet.
    `config get` reads EFFECTIVE leaves and rejects those, so offering them made
    TAB suggest three keys it answers "is not a config leaf" to."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    from agent6.ui.cli import main
    from agent6.ui.cli.completers import (
        _complete_config_keys,  # pyright: ignore[reportPrivateUsage]
    )

    for key in _complete_config_keys("models.", include_presets=False):
        assert main(["config", "get", key]) == 0, f"completion offered {key!r}, which get rejects"
