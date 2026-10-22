# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`--config FILE` parses in both positions for run, plan, resume and check.

The top-level flag and a verb's flag land in separate dests; `config_files` merges them once.
"""

from __future__ import annotations

import argparse
import pathlib

import pytest

from agent6.ui.cli import _common
from agent6.ui.cli import parser as cli_parser


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
    args = cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(argv))
    assert cli_parser.config_files(args) == (pathlib.Path("c.toml"),)


def test_config_defaults_to_no_files_when_absent() -> None:
    args = cli_parser.build_parser().parse_args(["run", "task"])
    assert cli_parser.config_files(args) == ()


def test_run_decompose_flag_defaults_off_and_parses() -> None:
    # --decompose is plan-first (overrides [prompt].decompose for the run); off by default.
    p = cli_parser.build_parser()
    assert p.parse_args(["run", "fix it"]).decompose is False
    assert p.parse_args(["run", "--decompose", "fix it"]).decompose is True


def test_history_bare_query_defaults_to_search() -> None:
    # `history "divide"` == `history search "divide"`, like `runs` -> list and bare `ask`.
    args = cli_parser.build_parser().parse_args(
        cli_parser._inject_default_verb(["history", "divide"])
    )
    assert args.history_command == "search" and args.query == "divide"


def test_a_bare_sessions_is_list_with_its_flags() -> None:
    """A bare `sessions` is `list` with its flags, through `_DEFAULT_VERBS` like every shorthand."""
    args = cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(["sessions"]))
    assert args.sessions_command == "list"
    args = cli_parser.build_parser().parse_args(
        cli_parser._inject_default_verb(["sessions", "--json"])
    )
    assert args.sessions_command == "list" and args.list_json is True


def test_ask_has_one_verb() -> None:
    """`ask list` was a poorer `sessions list`: one listing, one verb."""
    assert cli_parser._DEFAULT_VERBS["ask"] == ("query", frozenset({"query"}))
    args = cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(["ask", "list"]))
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
    args = cli_parser.build_parser().parse_args(
        cli_parser._inject_default_verb(["history", "search", "divide"])
    )
    assert args.history_command == "search" and args.query == "divide"
    # A flag after the bare query is carried onto the injected verb too.
    a2 = cli_parser.build_parser().parse_args(
        cli_parser._inject_default_verb(["history", "--regex", "d.v"])
    )
    assert a2.history_command == "search" and a2.query == "d.v" and a2.regex is True


def test_config_get_does_not_offer_keys_it_rejects(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config get` completion offers no preset key, since get rejects them.

    `[presets.*]` tables are stripped before validation, so they are not effective-config leaves;
    they stay on the write verbs, where they are accepted.
    """
    from agent6.ui.cli import completers

    (tmp_path / "agent6").mkdir()
    (tmp_path / "agent6" / "config.toml").write_text(
        '[presets.mine.sandbox]\nrun_commands = "allow"\n', encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    for_set = completers._complete_config_keys("presets.")
    for_get = completers._complete_config_keys("presets.", settable=False)

    assert any(k.startswith("presets.mine.") for k in for_set), "the write verbs still offer them"
    assert not any(k.startswith("presets.") for k in for_get), f"get offered: {for_get[:3]}"


def test_fork_carries_the_same_sandbox_flags_as_its_siblings() -> None:
    """Fork carries the same sandbox flags as its siblings.

    A fork without `--no-run` continues a run, so it is a paid command and takes `--auto-approve`
    like the rest.
    """
    from agent6.app import _setup

    parser = cli_parser.build_parser()
    args = parser.parse_args(["fork", "--auto-approve", "some-session-id"])
    assert _setup.SandboxOverrides.from_args(args).auto_approve is True

    args = parser.parse_args(["fork", "--no-commands", "some-session-id"])
    assert _setup.SandboxOverrides.from_args(args).no_commands is True


def test_get_completion_offers_no_key_get_rejects(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Get completion offers no enum key `config get` rejects.

    The enum keys exist so `config set` can reach a leaf no layer has set yet.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    from agent6.ui.cli import completers, main

    for key in completers._complete_config_keys("models.", settable=False):
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
    groups = _subcommands(cli_parser.build_parser())
    for group, (default, verbs) in cli_parser._DEFAULT_VERBS.items():
        real = set(_subcommands(groups[group]))
        assert verbs == real, f"{group}: {sorted(verbs ^ real)}"
        assert default in real


def test_bare_default_groups_are_those_whose_default_verb_takes_no_positional() -> None:
    """A group is bare exactly when its default verb takes no positional."""
    groups = _subcommands(cli_parser.build_parser())
    bare = {
        group
        for group, (default, _verbs) in cli_parser._DEFAULT_VERBS.items()
        if not any(
            not a.option_strings and a.dest != "==SUPPRESS=="
            for a in _subcommands(groups[group])[default]._actions  # pyright: ignore[reportPrivateUsage]
        )
    }
    assert bare == cli_parser._BARE_DEFAULT_GROUPS


def test_a_mistyped_verb_after_a_bare_group_names_the_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped verb after a bare group names the choices, not "unrecognized arguments"."""
    with pytest.raises(SystemExit) as exc:
        cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(["skills", "show"]))
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
    args = cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(argv))
    assert getattr(args, dest) == verb


def test_a_repeated_config_keeps_every_file_in_order() -> None:
    """Two `--config` values parse to both paths in command-line order, not the last one."""
    args = cli_parser.build_parser().parse_args(
        ["--config", "a.toml", "--config", "b.toml", "config", "show"]
    )
    assert cli_parser.config_files(args) == (pathlib.Path("a.toml"), pathlib.Path("b.toml"))


@pytest.mark.parametrize(
    "command",
    [
        ["run", "task"],
        ["plan", "task"],
        # Nested groups: the leaf verb sits two levels down.
        ["plan", "run", "task"],
        ["ask", "query", "why?"],
        ["mcp", "serve"],
    ],
)
def test_config_files_on_both_sides_keep_order(command: list[str]) -> None:
    """A `--config` after the leaf verb follows the root's files; a nested group is no different."""
    argv = ["--config", "a.toml", *command, "--config", "b.toml"]
    args = cli_parser.build_parser().parse_args(cli_parser._inject_default_verb(argv))
    assert cli_parser.config_files(args) == (pathlib.Path("a.toml"), pathlib.Path("b.toml"))


def test_config_files_after_the_verb_keep_their_order() -> None:
    """`run task --config a --config b` keeps a before b: later files win, so order is the fact."""
    args = cli_parser.build_parser().parse_args(
        ["run", "task", "--config", "a.toml", "--config", "b.toml"]
    )
    assert cli_parser.config_files(args) == (pathlib.Path("a.toml"), pathlib.Path("b.toml"))


def test_a_path_given_twice_is_kept_twice() -> None:
    """A repeated path is not deduplicated: the parser reports what was typed."""
    args = cli_parser.build_parser().parse_args(
        ["--config", "a.toml", "run", "task", "--config", "a.toml"]
    )
    assert cli_parser.config_files(args) == (pathlib.Path("a.toml"), pathlib.Path("a.toml"))


def test_a_leaf_without_the_flag_keeps_what_a_group_collected() -> None:
    """A group verb's `--config` survives its leaf: the leaf dest is SUPPRESS, not None.

    argparse copies a nested verb's whole namespace upward, so a `default=None` leaf would
    write None over the group's list. No agent6 group parser carries the flag today, so the
    shape is built here through the real `_add_config_flag`.
    """
    root = argparse.ArgumentParser(prog="x")
    root.add_argument("--config", action="append", type=pathlib.Path, default=None)
    group = root.add_subparsers(dest="command", required=True).add_parser("group")
    _common._add_config_flag(group)
    leaf = group.add_subparsers(dest="verb", required=True).add_parser("leaf")
    _common._add_config_flag(leaf)
    args = root.parse_args(["--config", "a.toml", "group", "--config", "b.toml", "leaf"])
    assert cli_parser.config_files(args) == (pathlib.Path("a.toml"), pathlib.Path("b.toml"))


def test_command_index_skips_every_config_value() -> None:
    """Repeated `--config FILE` and `--config=FILE` are all skipped on the way to the verb."""
    assert (
        cli_parser._command_index(
            ["--config", "a.toml", "--config=b.toml", "--config", "c.toml", "plan"]
        )
        == 5
    )
