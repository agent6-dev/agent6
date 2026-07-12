# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser UX guards: leaf --help descriptions, subcommand metavars, terminology
(machine *id*), completers/metavars on options, and plain punctuation."""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import pytest

from agent6.ui.cli.completers import _complete_profiles  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.parser import build_parser


def _subparsers(parser: argparse.ArgumentParser) -> Iterator[tuple[str, argparse.ArgumentParser]]:
    """Yield (name, parser) for every subparser, recursively."""
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            for name, sub in action.choices.items():
                yield name, sub
                yield from _subparsers(sub)


def _find(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for n, sub in _subparsers(parser):
        if n == name:
            return sub
    raise AssertionError(f"no subparser named {name!r}")


def _option(parser: argparse.ArgumentParser, flag: str) -> argparse.Action:
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if flag in action.option_strings:
            return action
    raise AssertionError(f"no option {flag!r}")


def _positional(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if not action.option_strings and action.dest == dest:
            return action
    raise AssertionError(f"no positional {dest!r}")


def test_every_subparser_has_a_description() -> None:
    # Leaf --help used to open with no summary at all; each add_parser now
    # mirrors its help string as the description.
    parser = build_parser()
    missing = [name for name, sub in _subparsers(parser) if not sub.description]
    assert missing == []


def test_bare_parent_command_error_names_subcommand_not_dest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A parent whose subcommand is required must name "<subcommand>", not leak
    # the argparse dest ("plan_command"). (`runs` no longer errors here: bare
    # `agent6 runs` lists runs.)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan"])
    err = capsys.readouterr().err
    assert "<subcommand>" in err
    assert "plan_command" not in err


def test_bare_agent6_prints_help_not_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    from agent6.ui.cli import main

    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage: agent6" in out
    assert "<command>" in out


def test_no_em_dashes_in_parser_help() -> None:
    parser = build_parser()
    offenders: list[str] = []
    for _name, sub in [("agent6", parser), *_subparsers(parser)]:
        if "—" in (sub.description or ""):
            offenders.append(sub.prog)
        for action in sub._actions:  # pyright: ignore[reportPrivateUsage]
            if "—" in (action.help or ""):
                offenders.append(f"{sub.prog} {action.dest}")
    assert offenders == []


def test_run_tui_help_does_not_claim_an_extra() -> None:
    # textual is a base dependency; there is no `tui` extra.
    help_text = _option(_find(build_parser(), "run"), "--tui").help or ""
    assert "extra" not in help_text


def test_plan_task_help_does_not_promise_omission() -> None:
    # plan always requires a task ("Omit to execute/offer" is `run` behavior).
    plan_run = _find(_find(build_parser(), "plan"), "run")
    help_text = _positional(plan_run, "task").help or ""
    assert "Omit" not in help_text


def test_attach_and_web_say_machine_id() -> None:
    parser = build_parser()
    for cmd in ("attach", "web"):
        help_text = _positional(_find(parser, cmd), "target").help or ""
        assert "machine id" in help_text
        assert "machine name" not in help_text


def test_profile_flags_have_the_profiles_completer() -> None:
    parser = build_parser()
    carriers = (
        _find(parser, "run"),
        _find(_find(parser, "plan"), "run"),
        _find(parser, "query"),  # `query` is ask's default verb
    )
    for sub in carriers:
        action = _option(sub, "--profile")
        assert getattr(action, "completer", None) is _complete_profiles


def test_option_metavars() -> None:
    parser = build_parser()
    assert _option(_find(parser, "create"), "--max-attempts").metavar == "N"
    assert _option(_find(parser, "search"), "--run").metavar == "RUN_ID"


def test_model_header_names_reviewer_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    # config.py: planner and reviewer fall back to worker (the header said
    # "planner/worker").
    from agent6.ui.cli import main

    assert main(["model"]) == 0
    out = capsys.readouterr().out
    assert "planner/reviewer fall back to worker" in out
