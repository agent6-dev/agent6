# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`--config FILE` parses in both positions for run, plan, resume and check.

A subparser `default=None` would clobber the top-level form back to None.
"""

from __future__ import annotations

import argparse
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
        # `plan` carries --config and the task on its implicit `run` verb, applied before parsing.
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
    # `history "divide"` == `history search "divide"`, like `runs` -> list and bare `ask`.
    args = build_parser().parse_args(_inject_default_verb(["history", "divide"]))
    assert args.history_command == "search" and args.query == "divide"


def test_a_bare_sessions_is_list_with_its_flags() -> None:
    """A bare `sessions` is `list` with its flags, through `_DEFAULT_VERBS` like every shorthand."""
    args = build_parser().parse_args(_inject_default_verb(["sessions"]))
    assert args.sessions_command == "list"
    args = build_parser().parse_args(_inject_default_verb(["sessions", "--json"]))
    assert args.sessions_command == "list" and args.list_json is True


def test_ask_has_one_verb() -> None:
    """`ask list` was a poorer `sessions list`: one listing, one verb."""
    from agent6.ui.cli.parser import _DEFAULT_VERBS  # pyright: ignore[reportPrivateUsage]

    assert _DEFAULT_VERBS["ask"] == ("query", frozenset({"query"}))
    args = build_parser().parse_args(_inject_default_verb(["ask", "list"]))
    assert args.ask_command == "query" and args.task == "list"


def test_a_bare_history_names_the_query_it_needs(capsys: pytest.CaptureFixture[str]) -> None:
    """A bare `history` names the query it needs, like a bare `plan` or `ask`."""
    from agent6.ui.cli import main

    try:
        rc = main(["history"])
    except SystemExit as exc:  # argparse's own refusal
        rc = int(exc.code or 0)
    assert rc == 2
    assert "'history' needs a query" in capsys.readouterr().err


def test_history_explicit_search_still_works() -> None:
    args = build_parser().parse_args(_inject_default_verb(["history", "search", "divide"]))
    assert args.history_command == "search" and args.query == "divide"
    # A flag after the bare query is carried onto the injected verb too.
    a2 = build_parser().parse_args(_inject_default_verb(["history", "--regex", "d.v"]))
    assert a2.history_command == "search" and a2.query == "d.v" and a2.regex is True


def test_config_get_does_not_offer_keys_it_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config get` completion offers no preset key, since get rejects them.

    `[presets.*]` tables are stripped before validation, so they are not effective-config leaves;
    they stay on the write verbs, where they are accepted.
    """
    from agent6.ui.cli.completers import (
        _complete_config_keys,  # pyright: ignore[reportPrivateUsage]
    )

    (tmp_path / "agent6").mkdir()
    (tmp_path / "agent6" / "config.toml").write_text(
        '[presets.mine.sandbox]\nrun_commands = "yes"\n', encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    for_set = _complete_config_keys("presets.")
    for_get = _complete_config_keys("presets.", settable=False)

    assert any(k.startswith("presets.mine.") for k in for_set), "the write verbs still offer them"
    assert not any(k.startswith("presets.") for k in for_get), f"get offered: {for_get[:3]}"


def test_fork_carries_the_same_sandbox_flags_as_its_siblings() -> None:
    """Fork carries the same sandbox flags as its siblings.

    A fork without `--no-run` continues a run, so it is a paid command and takes `--auto-approve`
    like the rest.
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
    """Get completion offers no enum key `config get` rejects.

    The enum keys exist so `config set` can reach a leaf no layer has set yet.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    from agent6.ui.cli import main
    from agent6.ui.cli.completers import (
        _complete_config_keys,  # pyright: ignore[reportPrivateUsage]
    )

    for key in _complete_config_keys("models.", settable=False):
        assert main(["config", "get", key]) == 0, f"completion offered {key!r}, which get rejects"


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            return dict(action.choices)
    return {}


def test_default_verb_sets_are_the_parsers_real_subcommands() -> None:
    """Each default-verb group lists exactly the parser's real subcommands.

    A verb added to a parser and not here would be swallowed as the default verb's first argument.
    """
    from agent6.ui.cli.parser import _DEFAULT_VERBS  # pyright: ignore[reportPrivateUsage]

    groups = _subcommands(build_parser())
    for group, (default, verbs) in _DEFAULT_VERBS.items():
        real = set(_subcommands(groups[group]))
        assert verbs == real, f"{group}: {sorted(verbs ^ real)}"
        assert default in real


def test_bare_default_groups_are_those_whose_default_verb_takes_no_positional() -> None:
    """A group is bare exactly when its default verb takes no positional."""
    from agent6.ui.cli.parser import (
        _BARE_DEFAULT_GROUPS,  # pyright: ignore[reportPrivateUsage]
        _DEFAULT_VERBS,  # pyright: ignore[reportPrivateUsage]
    )

    groups = _subcommands(build_parser())
    bare = {
        group
        for group, (default, _verbs) in _DEFAULT_VERBS.items()
        if not any(
            not a.option_strings and a.dest != "==SUPPRESS=="
            for a in _subcommands(groups[group])[default]._actions  # pyright: ignore[reportPrivateUsage]
        )
    }
    assert bare == _BARE_DEFAULT_GROUPS


def test_a_mistyped_verb_after_a_bare_group_names_the_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped verb after a bare group names the choices, not "unrecognized arguments"."""
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_inject_default_verb(["skills", "show"]))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'show'" in err and "install" in err


@pytest.mark.parametrize(
    ("argv", "dest", "verb"),
    [
        (["skills"], "skills_command", "list"),
        (["memory"], "memory_command", "list"),
        (["mcp"], "mcp_command", "list"),
        (["config"], "config_command", "show"),
        (["config", "git.dirty_tree"], "config_command", "show"),
        (["prompt"], "prompt_command", "show"),
        (["prompt", "--json"], "prompt_command", "show"),
    ],
)
def test_a_bare_group_runs_its_listing(argv: list[str], dest: str, verb: str) -> None:
    """A bare group runs its listing; a key after `config` is a `show` of that key."""
    args = build_parser().parse_args(_inject_default_verb(argv))
    assert getattr(args, dest) == verb
