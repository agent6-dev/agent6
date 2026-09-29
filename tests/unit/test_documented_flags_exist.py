# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every flag the docs name in backticks is a flag the CLI has.

Scanning the file rather than the line catches a flag named on a continuation line.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import tomllib

import pytest

from agent6.ui.cli import parser as cli_parser

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = [*sorted((ROOT / "docs").glob("*.md")), ROOT / "README.md"]

# Other tools' flags named in prose: git's, and Claude Code's on the claude_code argv.
_NOT_OURS = {
    "--no-ext-diff",
    "--no-textconv",
    "--no-ff",
    "--allowed",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--setting-sources",
    "--strict-mcp-config",
}


def _cli_flags() -> set[str]:
    """Every long option the parser knows, at every subcommand depth."""
    found: set[str] = set()

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
            found.update(o for o in action.option_strings if o.startswith("--"))
            # Subparsers hang off a dict-valued `choices`; a plain argument's is a tuple.
            if isinstance(action.choices, dict):
                for sub in action.choices.values():
                    if isinstance(sub, argparse.ArgumentParser):
                        walk(sub)

    walk(cli_parser.build_parser())
    return found


def _named_flags(text: str) -> set[str]:
    """Every flag a doc attributes to agent6: backticked anywhere, or on an agent6 line in a fence.

    Fenced blocks are line-scoped because a shell block often mixes tools (`tailscale serve` under
    `agent6 web`).
    """
    named = {m.rstrip(".,;:)") for m in re.findall(r"`(--[a-z0-9][a-z0-9-]+)", text)}
    for block in re.finditer(r"```[a-z]*\n(.*?)```", text, re.S):
        for line in block.group(1).splitlines():
            if re.search(r"(^|\s)agent6\s", line):
                named |= {m.rstrip(".,;:)`") for m in re.findall(r"--[a-z0-9][a-z0-9-]+", line)}
    return named


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_flags_exist(doc: pathlib.Path) -> None:
    missing = _named_flags(doc.read_text("utf-8")) - _cli_flags() - _NOT_OURS
    assert not missing, f"{doc.name} names flags the CLI does not have: {sorted(missing)}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_source_links_resolve(doc: pathlib.Path) -> None:
    """Every `blob/master/<path>` link a doc carries points at a real file."""
    pat = re.compile(r"https://github\.com/agent6-dev/agent6/(?:blob|tree)/master/([^)\s#]+)")
    linked = pat.findall(doc.read_text(encoding="utf-8"))
    missing = sorted({p for p in linked if not (ROOT / p).exists()})
    assert not missing, f"{doc.name} links at paths that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_toml_examples_parse(doc: pathlib.Path) -> None:
    """Every ```toml block a doc ships parses as TOML.

    Blocks using `<name>` placeholders or `...` elisions are sketches of shape and are skipped.
    """
    text = doc.read_text(encoding="utf-8")
    for block in re.finditer(r"```toml\n(.*?)```", text, re.S):
        body = block.group(1)
        if "<" in body or "\n..." in body:
            continue
        line = text[: block.start()].count("\n") + 1
        try:
            tomllib.loads(body)
        except tomllib.TOMLDecodeError as exc:  # pragma: no cover - the failure IS the message
            pytest.fail(f"{doc.name}:{line} toml block does not parse: {exc}")


def _documented_routes(page: str) -> set[str]:
    """Every `/api/...` route the page names, with `{a,b}` groups expanded."""
    routes: set[str] = set()
    for span in re.findall(r"`(/api/[^`]+)`", page):
        head, _, rest = span.partition("{")
        if rest:
            routes.update(head + alt for alt in rest.split("}")[0].split(","))
        else:
            routes.add(span)
    return routes


def test_docs_name_every_web_write_route() -> None:
    """docs/web.md enumerates the browser UI's write surface, so a reader can audit every POST."""
    server = (ROOT / "src" / "agent6" / "ui" / "web" / "server.py").read_text(encoding="utf-8")
    post = server[server.index("def _route_post") :]
    post = post[: post.index("\n    def ", 1)]
    verbs = set(re.findall(r'verb == "([a-z_]+)"', server))
    paths = set(re.findall(r'path == "(/api/[a-z0-9_/]+)"', post))
    assert verbs and paths, "no POST verbs/routes found in the web server"
    page = (ROOT / "docs" / "web.md").read_text(encoding="utf-8")
    missing = sorted([v for v in verbs if v not in page] + list(paths - _documented_routes(page)))
    assert not missing, f"docs/web.md does not name web POST route(s): {missing}"
