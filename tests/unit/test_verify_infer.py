# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for verify_command inference (AGENTS.md -> repo signals -> LLM)."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from agent6 import verify_infer

pytestmark = pytest.mark.usefixtures("jail_bin")

_AGENTS_PIPELINE = """\
# AGENTS.md

## Verify command

Also what `verify_command` should be in this repo's agent6 config:

```bash
uv run ruff check && uv run ruff format --check && \\
  uv run pyright && uv run pytest
```
"""

_AGENTS_SIMPLE = """\
## Verify command

```bash
# EDIT: replace with your actual verify pipeline.
.venv/bin/python -m pytest -x
```
"""


def test_agents_md_pipeline_wraps_in_sh_c() -> None:
    argv = verify_infer.verify_from_agents_md(_AGENTS_PIPELINE)
    assert argv is not None
    assert argv[0] == "sh" and argv[1] == "-c"
    # backslash-continuation joined, both halves of the && chain present.
    assert "ruff check" in argv[2] and "pytest" in argv[2]
    assert "&&" in argv[2]


def test_agents_md_simple_command_tokenises_and_skips_comments() -> None:
    argv = verify_infer.verify_from_agents_md(_AGENTS_SIMPLE)
    assert argv == (".venv/bin/python", "-m", "pytest", "-x")


def test_agents_md_inline_marker() -> None:
    assert verify_infer.verify_from_agents_md("Verify: pytest -q") == ("pytest", "-q")
    assert verify_infer.verify_from_agents_md("Test: make check") == ("make", "check")


def test_agents_md_inline_marker_written_as_code() -> None:
    """A Verify line whose command is inline code keeps the backticks out of the argv.

    Kept, `sh -c` ran the tests and then executed their output as a command.
    """
    argv = verify_infer.verify_from_agents_md("Verify: `python -m pytest -q`")
    assert argv == ("python", "-m", "pytest", "-q")
    assert verify_infer.verify_from_agents_md("Test: `make check`.") == ("make", "check")
    assert verify_infer.verify_from_agents_md("Verify: `a && b`") == ("sh", "-c", "a && b")
    assert verify_infer.verify_from_agents_md("Verify: ruff check .") == ("ruff", "check", ".")


def test_agents_md_none_when_absent() -> None:
    assert verify_infer.verify_from_agents_md("# Readme\n\nNo verify here.") is None
    assert verify_infer.verify_from_agents_md("") is None


def test_repo_signal_package_json(tmp_path: pathlib.Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (
        ("npm", "test", "--silent"),
        "package.json",
    )


def test_repo_signal_package_json_ignores_the_npm_placeholder(tmp_path: pathlib.Path) -> None:
    """The test script `npm init` writes exists to fail, so it is no gate.

    Taken as one, `npm test` made every run end red and `agent6 init --yes` saved it.
    """
    pkg = tmp_path / "package.json"
    for script in (
        'echo "Error: no test specified" && exit 1',
        "  echo 'Error: no test specified'  &&  exit 1 ",
        "echo Error: no test specified && exit 1",
    ):
        pkg.write_text(json.dumps({"scripts": {"test": script}}), encoding="utf-8")
        assert verify_infer.repo_signals(tmp_path) == [], script
    (tmp_path / "Makefile").write_text("test:\n\t./run\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (("make", "test"), "Makefile:test")
    # Nothing wider: a script that only starts like the placeholder is a real one.
    script = 'echo "Error: no test specified" && exit 1 || jest'
    pkg.write_text(json.dumps({"scripts": {"test": script}}), encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (
        ("npm", "test", "--silent"),
        "package.json",
    )


def test_repo_signal_makefile_target(tmp_path: pathlib.Path) -> None:
    (tmp_path / "Makefile").write_text("build:\n\tcc x.c\ntest:\n\t./run\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (("make", "test"), "Makefile:test")


def test_repo_signal_pyproject(tmp_path: pathlib.Path) -> None:
    # No .venv present (e.g. a container or system-python checkout) -> fall back
    # to python3 on PATH, NOT the missing .venv/bin/python that would break verify.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (
        ("python3", "-m", "pytest", "-q"),
        "pyproject",
    )


def test_repo_signal_pyproject_prefers_existing_venv(tmp_path: pathlib.Path) -> None:
    # When a project .venv/bin/python exists, prefer it (jail-visible convention).
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (
        (".venv/bin/python", "-m", "pytest", "-q"),
        "pyproject",
    )


def test_repo_signal_cargo_and_go(tmp_path: pathlib.Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (
        ("cargo", "test", "--quiet"),
        "Cargo.toml",
    )
    (tmp_path / "Cargo.toml").unlink()
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (("go", "test", "./..."), "go.mod")


def test_repo_signal_verify_sh_wins(tmp_path: pathlib.Path) -> None:
    """A root verify.sh beats every manifest, run directly when executable, else through sh."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    script = tmp_path / "verify.sh"
    script.write_text("#!/bin/sh\npython3 -m pytest -q\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (("sh", "verify.sh"), "verify.sh")
    script.chmod(0o755)
    assert verify_infer.repo_signals(tmp_path)[0] == (("./verify.sh",), "verify.sh")


def test_repo_signal_loose_python_tests_come_last(tmp_path: pathlib.Path) -> None:
    """test_*.py means pytest only when no manifest says otherwise; a Go tests/ stays `go test`."""
    (tmp_path / "test_calc.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (
        ("python3", "-m", "pytest", "-q"),
        "test_*.py",
    )
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path)[0] == (("go", "test", "./..."), "go.mod")


def test_repo_signal_none(tmp_path: pathlib.Path) -> None:
    assert verify_infer.repo_signals(tmp_path) == []


@pytest.mark.parametrize(
    ("hidden", "name"),
    [
        ("verify.sh", "verify.sh"),
        ("pyproject.toml", "pyproject.toml"),
        ("Cargo.toml", "Cargo.toml"),
        ("go.mod", "go.mod"),
        ("tests", "tests/test_calc.py"),
        ("test_calc.py", "test_calc.py"),
    ],
)
def test_a_hidden_file_gives_no_repo_signal(tmp_path: pathlib.Path, hidden: str, name: str) -> None:
    """A file under `sandbox.hide_paths` gave the gate, which then ran on the jail's empty mask.

    A hidden verify.sh became `sh verify.sh`, which exits 0 on the empty file.
    """
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text("exit 1\n", encoding="utf-8")
    assert verify_infer.repo_signals(tmp_path, (str(tmp_path / hidden),)) == []


def test_a_hidden_link_target_or_parent_reads_as_absent(tmp_path: pathlib.Path) -> None:
    """A verify.sh linked into a hidden dir stayed the gate, and a hidden `.venv` its Python."""
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "verify.sh").write_text("exit 1\n", encoding="utf-8")
    (tmp_path / "verify.sh").symlink_to("ci/verify.sh")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
    hide = (str(tmp_path / "ci"), str(tmp_path / ".venv"))
    assert verify_infer.repo_signals(tmp_path, hide) == [
        (("python3", "-m", "pytest", "-q"), "pyproject")
    ]


def test_the_model_tier_reads_no_hidden_file(tmp_path: pathlib.Path) -> None:
    """The model's context listed a hidden file, and a hidden manifest still made the call."""
    (tmp_path / "run_tests.sh").write_text("exit 1\n", encoding="utf-8")
    (tmp_path / "noxfile.py").write_text("import nox\n", encoding="utf-8")
    contexts: list[str] = []

    def fake_llm(context: str) -> str:
        contexts.append(context)
        return "[]"

    hide = (str(tmp_path / "run_tests.sh"),)
    verify_infer.infer_verify_command(tmp_path, "", llm_call=fake_llm, hide=hide)
    assert len(contexts) == 1 and "run_tests.sh" not in contexts[0]
    verify_infer.infer_verify_command(
        tmp_path, "", llm_call=fake_llm, hide=(str(tmp_path / "noxfile.py"),)
    )
    assert len(contexts) == 1


def test_a_symlink_loop_at_the_root_still_reaches_the_model_tier(tmp_path: pathlib.Path) -> None:
    """A `loop -> loop` link would make a `Path.resolve()` name check raise on Python 3.12."""
    (tmp_path / "loop").symlink_to("loop")
    (tmp_path / "noxfile.py").write_text("import nox\n", encoding="utf-8")
    contexts: list[str] = []

    def fake_llm(context: str) -> str:
        contexts.append(context)
        return "[]"

    verify_infer.infer_verify_command(tmp_path, "", llm_call=fake_llm)
    assert len(contexts) == 1 and "loop" in contexts[0]


def test_parse_llm_verify() -> None:
    assert verify_infer.parse_llm_verify('["pytest","-q"]') == ("pytest", "-q")
    assert verify_infer.parse_llm_verify('here you go:\n```json\n["cargo","test"]\n```') == (
        "cargo",
        "test",
    )
    assert verify_infer.parse_llm_verify("[]") is None
    assert verify_infer.parse_llm_verify("I cannot tell") is None
    assert verify_infer.parse_llm_verify("[1, 2]") is None
    assert verify_infer.parse_llm_verify('["", "x"]') is None


def test_infer_layering_prefers_agents_md(tmp_path: pathlib.Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "Verify: pytest -q")
    assert got.source == "agents_md" and got.argv == ("pytest", "-q")


def test_infer_falls_back_to_signals_then_llm(tmp_path: pathlib.Path) -> None:
    calls: list[str] = []

    def fake_llm(ctx: str) -> str:
        calls.append(ctx)
        return '["make","verify"]'

    # A manifest with no static match (package.json without scripts.test) is
    # what the LLM tier exists to read.
    (tmp_path / "package.json").write_text('{"scripts": {"check": "x"}}', encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "", llm_call=fake_llm)
    assert got.source == "llm" and got.argv == ("make", "verify")
    assert calls and "package.json" in calls[0]

    # A repo signal short-circuits before the LLM.
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    calls.clear()
    got2 = verify_infer.infer_verify_command(tmp_path, "", llm_call=fake_llm)
    assert got2.source == "go.mod"
    assert not calls, "repo signal should short-circuit the LLM"


def test_infer_skips_the_llm_when_there_is_nothing_to_read(tmp_path: pathlib.Path) -> None:
    """With no manifests and no AGENTS.md the run starts gateless; the LLM tier is not consulted.

    Its context would be a bare filename listing whose only non-"none" outcome is an invented
    gate; the mid-run adopter owns a project that appears later.
    """
    calls: list[str] = []

    def fake_llm(ctx: str) -> str:
        calls.append(ctx)
        return '["make","verify"]'

    (tmp_path / "notes.txt").write_text("just a file\n", encoding="utf-8")
    assert verify_infer.infer_verify_command(tmp_path, "", llm_call=fake_llm).argv == ()
    assert not calls, "the LLM tier fired with nothing to read"

    # AGENTS.md prose alone (no fenced command) still qualifies: the LLM
    # reads prose the fenced-block tier cannot.
    got = verify_infer.infer_verify_command(
        tmp_path, "Run make verify before committing.", llm_call=fake_llm
    )
    assert calls and got.source == "llm"


def test_infer_llm_failure_is_safe(tmp_path: pathlib.Path) -> None:
    # A manifest so the LLM tier is genuinely reached before it fails.
    (tmp_path / "package.json").write_text('{"scripts": {"check": "x"}}', encoding="utf-8")
    reached: list[str] = []

    def boom(ctx: str) -> str:
        reached.append(ctx)
        raise RuntimeError("provider down")

    assert verify_infer.infer_verify_command(tmp_path, "", llm_call=boom).argv == ()
    assert reached, "the failure path was never exercised"


def test_package_json_is_passed_over_when_the_sandbox_has_no_npm(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """A gate whose runner the jail PATH lacks is not inferred, and the reason is kept.

    Taken unchecked, `npm test` aborted the run as unexecutable, and the resume after it.
    """
    (jail_bin / "npm").unlink()
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert (got.argv, got.source) == ((), "none")
    why = "package.json names `npm test --silent`, and the sandbox has no `npm`"
    assert verify_infer.passed_over_text(got) == why


def test_the_test_script_itself_is_the_gate_when_npm_is_absent_and_its_runner_is_not(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """A `node --test` script runs without npm; a script needing the shell is no argv.

    A glob or a `~` that the shell expands would reach the runner as literal text, a gate that
    can never pass where the runner does not expand it itself; a second line would reach it as
    arguments; a runner given as a path would count as found, and where it is missing the start
    gate could not execute it and the run would end before its first turn.
    """
    (jail_bin / "npm").unlink()
    (tmp_path / "package.json").write_text('{"scripts": {"test": "node --test"}}', encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert (got.argv, got.source, got.passed_over) == (("node", "--test"), "package.json", ())
    # With npm present the script runs under it, hooks and env included.
    (jail_bin / "npm").write_text("#!/bin/sh\n", encoding="utf-8")
    (jail_bin / "npm").chmod(0o755)
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert (got.argv, got.source) == (("npm", "test", "--silent"), "package.json")
    (jail_bin / "npm").unlink()
    # A pretest or posttest hook runs under npm alone, so the script is not the whole gate.
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"pretest": "node lint.js", "test": "node --test"}}),
        encoding="utf-8",
    )
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert got.argv == () and "no `npm`" in got.passed_over[0][1]
    for script in (
        "node --test && echo ok",
        "CI=1 node --test",
        "node --test | tee log",
        "node --test test/*.test.js",
        "node --test ~/t.js",
        "node --test (a)",
        "node --test\nnode check.js",
        "./node_modules/.bin/mocha",
    ):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": script}}), encoding="utf-8"
        )
        got = verify_infer.infer_verify_command(tmp_path, "")
        assert got.argv == (), script
        assert "names `npm test --silent`, and the sandbox has no `npm`" in got.passed_over[0][1]


def test_a_path_runner_stays_npm_test_even_when_the_repository_holds_it(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """A path runner stays `npm test`.

    Checked on the host, the path can be untracked, outside the jail, or a link out.
    """
    (jail_bin / "npm").unlink()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "scripts" / "test.sh").chmod(0o755)
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "./scripts/test.sh --fast"}}', encoding="utf-8"
    )
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert got.argv == ()
    assert "names `npm test --silent`, and the sandbox has no `npm`" in got.passed_over[0][1]


def test_an_unresolvable_runner_falls_through_to_the_next_candidate(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """Every tier is judged: an AGENTS.md gate and a manifest both give way to a runnable one."""
    (jail_bin / "npm").unlink()
    (jail_bin / "pytest").unlink()
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    (tmp_path / "Makefile").write_text("test:\n\t./run\n", encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "Verify: pytest -q")
    assert (got.argv, got.source) == (("make", "test"), "Makefile:test")
    assert got.passed_over == (
        ("AGENTS.md", "names `pytest -q`, and the sandbox has no `pytest`"),
        ("package.json", "names `npm test --silent`, and the sandbox has no `npm`"),
    )


def test_a_model_answer_with_an_unresolvable_runner_is_not_a_gate(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """The model tier's answer is judged like a manifest's."""
    (jail_bin / "npm").unlink()
    (tmp_path / "package.json").write_text('{"scripts": {"check": "x"}}', encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "", llm_call=lambda _ctx: '["npm", "test"]')
    assert (got.argv, got.source) == ((), "none")
    assert [source for source, _ in got.passed_over] == ["the reviewer model"]


def test_a_runner_given_as_a_path_is_not_judged_by_the_jail_path(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """A first word with a slash resolves against the workspace, so the PATH rule leaves it."""
    (jail_bin / "python3").unlink()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert got.argv == (".venv/bin/python", "-m", "pytest", "-q")
    assert got.passed_over == ()


def test_the_manifests_an_ecosystem_gate_leaves_out_are_named(tmp_path: pathlib.Path) -> None:
    """One ecosystem's runner is the gate, and the other ecosystems' manifests are named.

    Unnamed, a repo with three manifests ran one suite and said nothing of the other two.
    """
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert (got.argv, got.source) == (("npm", "test", "--silent"), "package.json")
    assert got.passed_over == (
        ("Cargo.toml", "names `cargo test --quiet`, which this gate does not run"),
        ("go.mod", "names `go test ./...`, which this gate does not run"),
    )


def test_one_ecosystem_or_a_repo_entry_point_names_no_other_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """Loose tests beside a Python manifest are one gate, and a verify.sh speaks for the repo."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert (got.source, got.passed_over) == ("pyproject", ())
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    (tmp_path / "verify.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    got = verify_infer.infer_verify_command(tmp_path, "")
    assert (got.source, got.passed_over) == ("verify.sh", ())


def test_gather_repo_manifests_clips_and_includes(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    ctx = verify_infer.gather_repo_manifests(tmp_path, "Verify: pytest", cap=10)
    assert "pyproject.toml" in ctx and "AGENTS.md" in ctx and "top-level" in ctx


def test_a_comment_mark_inside_an_escaped_quote_stays() -> None:
    """A backslash-escaped quote does not end the string, so a later ` #` is not a comment."""
    assert verify_infer.line_to_argv('pytest -k "a \\" # b"') == ("pytest", "-k", 'a " # b')
    assert verify_infer.line_to_argv("pytest -q  # the fast suite") == ("pytest", "-q")
    assert verify_infer.line_to_argv("pytest -k 'a # b'") == ("pytest", "-k", "a # b")


_AGENTS_SETUP = """\
# Agents

## Setup command

Builds the environment a fresh worktree lacks:

```bash
$ uv sync --offline  # the cache is warm
```

## Verify command

```bash
uv run ruff check && \\
  uv run pytest
```
"""


def test_agents_md_setup_block_is_read_as_the_verify_block_is() -> None:
    """A `## Setup command` fence nothing read left a new worktree with no environment."""
    assert verify_infer.setup_from_agents_md(_AGENTS_SETUP) == ("uv", "sync", "--offline")
    assert verify_infer.verify_from_agents_md(_AGENTS_SETUP) == (
        "sh",
        "-c",
        "uv run ruff check && uv run pytest",
    )


def test_agents_md_setup_block_wraps_a_pipeline_and_ignores_other_headings() -> None:
    """The setup fence takes the verify fence's shell handling; a verify fence is no setup."""
    chained = "## setup command\n```sh\npython3 -m venv .venv\n.venv/bin/pip install -e .\n```\n"
    assert verify_infer.setup_from_agents_md(chained) == (
        "sh",
        "-c",
        "python3 -m venv .venv && .venv/bin/pip install -e .",
    )
    assert verify_infer.setup_from_agents_md("## Verify command\n```\npytest -q\n```\n") is None
    assert verify_infer.setup_from_agents_md("## Setup\n```\nmake deps\n```\n") is None
    assert verify_infer.setup_from_agents_md("") is None


def test_a_prompt_alone_or_before_a_tab_is_dropped() -> None:
    """A `$` alone on a line or before a tab stayed in the command, and sh ran `$` as a program."""
    assert verify_infer.verify_from_agents_md(
        "## Verify command\n\n```console\n$ pytest -q\n$\n```\n"
    ) == ("pytest", "-q")
    assert verify_infer.setup_from_agents_md("## Setup command\n```\n$ \n$\tuv sync\n```\n") == (
        "uv",
        "sync",
    )
    assert verify_infer.verify_from_agents_md("## Verify command\n```\n$\n```\n") is None
    # A `$` that starts a word is no prompt.
    assert verify_infer.verify_from_agents_md("## Verify command\n```\n$HOME/bin/t\n```\n") == (
        "sh",
        "-c",
        "$HOME/bin/t",
    )


def test_a_shell_line_is_judged_by_its_first_program(
    tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """A wrapped line's `sh` always resolves; the program it runs first must too.

    `nosuchtool check && nosuchtool test` was adopted as a gate the sandbox could never run.
    """
    assert verify_infer.command_head(("sh", "-c", "FOO=1 nosuchtool test")) == "nosuchtool"
    assert verify_infer.command_head(("sh", "-c", "pytest -q && ruff check")) == "pytest"
    assert verify_infer.command_head(("pytest", "-q")) == "pytest"
    assert verify_infer.command_head(("sh", "-c")) == "sh"
    for md in ("Verify: nosuchtool check && nosuchtool test", "Verify: FOO=1 nosuchtool test"):
        got = verify_infer.infer_verify_command(tmp_path, md)
        assert got.argv == () and "no `nosuchtool`" in got.passed_over[0][1], md


def test_an_inline_verify_line_drops_its_comment_before_the_unquote() -> None:
    """A comment after the backticks kept them, and sh ran the command's output as a command."""
    assert verify_infer.verify_from_agents_md("Verify: `make test` # the gate") == (
        "make",
        "test",
    )
