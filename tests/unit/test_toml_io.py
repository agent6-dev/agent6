# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config.io scalar/table serialization + leaf surgery."""

from __future__ import annotations

import pathlib
import threading
import tomllib

import pytest

from agent6.config import ConfigError, io


def test_leaf_scan_skips_the_interior_of_a_multiline_value(tmp_path: pathlib.Path) -> None:
    r"""The leaf scan skips lines inside a triple-quoted string or multi-line array.

    It matched `^\s*leaf\s*=` on every line, rewrote the operator's string and reported success.
    """
    p = tmp_path / "c.toml"
    p.write_text(
        '[harness]\nverify_command = """\nx = 5\n"""\nx = 30\n',
        encoding="utf-8",
    )

    io.upsert_toml_leaf(p, "harness.x", 60)
    parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    assert parsed["harness"]["x"] == 60, "the real leaf must be the one rewritten"
    assert "x = 5" in parsed["harness"]["verify_command"], "the string was corrupted"

    assert io.remove_toml_leaf(p, "harness.x") is True
    parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    assert "x" not in parsed["harness"]
    assert "x = 5" in parsed["harness"]["verify_command"], "the string was corrupted"


def test_table_header_lookup_tolerates_a_trailing_comment(tmp_path: pathlib.Path) -> None:
    """A header with a trailing comment is found by the surgery.

    Matched exactly, `[sandbox]  # the jail` was invisible: unset found nothing and set
    appended a second table, making the file unparseable.
    """
    p = tmp_path / "c.toml"
    p.write_text("[sandbox]  # the jail\nprotect_git = true\n", encoding="utf-8")

    io.upsert_toml_leaf(p, "sandbox.run_commands", "yes")
    text = p.read_text(encoding="utf-8")
    assert text.count("[sandbox]") == 1, "a second table was appended"
    parsed = tomllib.loads(text)
    assert parsed["sandbox"] == {"protect_git": True, "run_commands": "yes"}

    assert io.remove_toml_leaf(p, "sandbox.protect_git") is True
    assert tomllib.loads(p.read_text(encoding="utf-8"))["sandbox"] == {"run_commands": "yes"}


def test_remove_toml_leaf_deletes_whole_multiline_array(tmp_path: pathlib.Path) -> None:
    """A multi-line array value must be removed whole.

    Deleting only the opening `leaf = [` line orphaned the continuation lines, leaving unparseable
    TOML (and `config fix` then reported the file it 'repaired' as invalid).
    """
    path = tmp_path / "c.toml"
    path.write_text(
        '[sandbox]\nallow_urls = [\n  "http://x",\n  "http://y",\n]\ntool_network = "session"\n'
    )
    assert io.remove_toml_leaf(path, "sandbox.allow_urls") is True
    out = path.read_text()
    tomllib.loads(out)  # must stay valid TOML
    assert "allow_urls" not in out
    assert 'tool_network = "session"' in out  # sibling + header preserved


def test_remove_toml_leaf_deletes_whole_multiline_string(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[a]\nx = """\nmultiline\nstring\n"""\ny = 1\n')
    assert io.remove_toml_leaf(path, "a.x") is True
    out = path.read_text()
    assert tomllib.loads(out) == {"a": {"y": 1}}


def test_remove_toml_leaf_multiline_last_leaf_drops_header(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[sandbox]\nallow_urls = [\n  1,\n]\n")
    assert io.remove_toml_leaf(path, "sandbox.allow_urls") is True
    out = path.read_text()
    tomllib.loads(out)
    assert "[sandbox]" not in out  # empty section header dropped


def test_upsert_toml_leaf_top_level_key_lands_before_first_table(tmp_path: pathlib.Path) -> None:
    """A single-segment key is written into the top region, before the first table header."""
    path = tmp_path / "c.toml"
    path.write_text('# keep me\n[sandbox]\nrun_commands = "ask"\n')
    io.upsert_toml_leaf(path, "profile", "ultra")
    out = path.read_text()
    assert tomllib.loads(out) == {"profile": "ultra", "sandbox": {"run_commands": "ask"}}
    assert "# keep me" in out


def test_upsert_toml_leaf_top_level_key_replaces_existing(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('profile = "quick"\n\n[sandbox]\nrun_commands = "ask"\n')
    io.upsert_toml_leaf(path, "profile", "ultra")
    out = path.read_text()
    assert tomllib.loads(out) == {"profile": "ultra", "sandbox": {"run_commands": "ask"}}
    assert out.count("profile") == 1


def test_upsert_toml_leaf_top_level_key_into_empty_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    io.upsert_toml_leaf(path, "profile", "ultra")
    assert tomllib.loads(path.read_text()) == {"profile": "ultra"}


def test_remove_toml_leaf_top_level_key_keeps_sections(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('profile = "ultra"\n\n[sandbox]\nrun_commands = "ask"\n')
    assert io.remove_toml_leaf(path, "profile") is True
    assert tomllib.loads(path.read_text()) == {"sandbox": {"run_commands": "ask"}}
    assert io.remove_toml_leaf(path, "profile") is False  # already gone


def test_remove_toml_leaf_top_level_key_never_touches_a_table_member(
    tmp_path: pathlib.Path,
) -> None:
    # A [review] section owning `trigger` must not lose it to a bare-key remove.
    path = tmp_path / "c.toml"
    path.write_text('[review]\ntrigger = "off"\n')
    assert io.remove_toml_leaf(path, "trigger") is False
    assert tomllib.loads(path.read_text()) == {"review": {"trigger": "off"}}


def test_upsert_toml_leaf_refuses_a_leaf_under_an_array_of_tables(tmp_path: pathlib.Path) -> None:
    """A leaf under an array-of-tables refuses with the owner message, not the parser's error."""
    p = tmp_path / "c.toml"
    p.write_text('[svc]\nname = "s"\n[[svc.items]]\nk = 1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="array-of-tables"):
        io.upsert_toml_leaf(p, "svc.items.enabled", True)


def test_upsert_toml_leaf_preserves_a_trailing_comment(tmp_path: pathlib.Path) -> None:
    """Replacing a single-line leaf keeps its trailing comment; a `#` in the string is not one."""
    p = tmp_path / "c.toml"
    p.write_text('[models.worker]\nmodel = "old"  # the good one\n', encoding="utf-8")
    io.upsert_toml_leaf(p, "models.worker.model", "new")
    out = p.read_text(encoding="utf-8")
    assert 'model = "new"  # the good one' in out
    assert tomllib.loads(out)["models"]["worker"]["model"] == "new"

    p.write_text('[a]\nx = "has # hash"\n', encoding="utf-8")
    io.upsert_toml_leaf(p, "a.x", "y")
    assert p.read_text(encoding="utf-8").splitlines()[-1] == 'x = "y"'  # no phantom comment


def test_upsert_top_level_key_replaces_conflicting_table(tmp_path: pathlib.Path) -> None:
    """Writing the bare `profile` key replaces an existing `[profile]` table.

    Writing both left the file unparseable, which the lenient set path then kept.
    """
    path = tmp_path / "c.toml"
    path.write_text('[profile]\nporifle = "x"\n\n[sandbox]\nrun_commands = "ask"\n')
    io.upsert_toml_leaf(path, "profile", "ultra")
    assert tomllib.loads(path.read_text()) == {
        "profile": "ultra",
        "sandbox": {"run_commands": "ask"},
    }


def test_upsert_table_leaf_replaces_conflicting_top_level_key(tmp_path: pathlib.Path) -> None:
    """Creating a `[profile]` table drops an existing bare `profile` key, never writing both."""
    path = tmp_path / "c.toml"
    path.write_text('profile = "ultra"\n\n[sandbox]\nrun_commands = "ask"\n')
    io.upsert_toml_leaf(path, "profile.porifle", "x")
    assert tomllib.loads(path.read_text()) == {
        "profile": {"porifle": "x"},
        "sandbox": {"run_commands": "ask"},
    }


def test_upsert_table_leaf_skips_a_key_name_inside_an_earlier_multiline_value(
    tmp_path: pathlib.Path,
) -> None:
    """Dropping a conflicting bare scalar skips a look-alike line inside a triple-quoted value."""
    path = tmp_path / "c.toml"
    path.write_text(
        'doc = """\nprofile = not a real key\n"""\nprofile = "old"\n\n'
        '[sandbox]\nrun_commands = "ask"\n',
        encoding="utf-8",
    )
    io.upsert_toml_leaf(path, "profile.name", "x")
    data = tomllib.loads(path.read_text(encoding="utf-8"))  # must still parse
    assert data["profile"] == {"name": "x"}  # bare scalar dropped, table written
    assert "profile = not a real key" in data["doc"]  # the doc string is intact
    assert data["sandbox"] == {"run_commands": "ask"}


def test_leaf_surgery_still_rejects_empty_key_segments(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    for bad in ("", "a..b", ".x", "x."):
        with pytest.raises(ConfigError, match="config key"):
            io.upsert_toml_leaf(path, bad, "v")
        with pytest.raises(ConfigError, match="config key"):
            io.remove_toml_leaf(path, bad)


def test_remove_toml_table_drops_header_body_and_subtables(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[cli]\ninput = "bar"\n[cli.sub]\nx = 1\n[budget]\nmax_usd = 1.0\n', encoding="utf-8"
    )
    assert io.remove_toml_table(path, "cli") is True
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert "cli" not in data  # header, body, and [cli.sub] all gone
    assert data["budget"] == {"max_usd": 1.0}  # the sibling table is untouched


def test_remove_toml_table_absent_returns_false(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[budget]\nmax_usd = 1.0\n", encoding="utf-8")
    assert io.remove_toml_table(path, "cli") is False
    assert path.read_text(encoding="utf-8") == "[budget]\nmax_usd = 1.0\n"


def test_header_lookup_skips_a_header_shadowed_by_an_earlier_multiline_value(
    tmp_path: pathlib.Path,
) -> None:
    """The header-locate scans skip a `[table]`-looking line inside a triple-quoted value.

    Matched first, an upsert wrote into the wrong table and a remove corrupted the operator's
    string while staying valid TOML.
    """
    base = '[b]\ndoc = """\n[a]\nx = 1\n"""\nk = 0\n\n[a]\nreal = "yes"\nkeep = 1\n'
    p = tmp_path / "c.toml"

    p.write_text(base, encoding="utf-8")
    io.upsert_toml_leaf(p, "a.new", 9)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["a"] == {"real": "yes", "keep": 1, "new": 9}  # landed in the REAL [a]
    assert "new" not in data["b"]  # not the shadowed fake
    assert "[a]" in data["b"]["doc"]  # the triple-quoted string is intact

    p.write_text(base, encoding="utf-8")
    assert io.remove_toml_leaf(p, "a.real") is True  # found the real leaf, not "nothing to unset"
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["a"] == {"keep": 1}  # real removed from the REAL [a]
    assert "[a]" in data["b"]["doc"]  # b.doc's string uncorrupted (fake header untouched)


def test_remove_toml_table_survives_a_bracketed_multiline_interior(tmp_path: pathlib.Path) -> None:
    """A table whose multi-line value has an interior line starting with `[` is dropped whole.

    The per-line scan flipped `dropping` off at that line, leaking the value's tail and every
    sibling below; `config fix` calls remove_toml_table, so this bricked recovery.
    """
    path = tmp_path / "c.toml"
    path.write_text(
        '[cli]\nhelp = """\nUsage:\n[options]\n"""\nenabled = true\n[review]\ntrigger = "off"\n',
        encoding="utf-8",
    )
    assert io.remove_toml_table(path, "cli") is True
    data = tomllib.loads(path.read_text(encoding="utf-8"))  # must still parse
    assert "cli" not in data  # header, the multi-line help, and enabled all gone
    assert data == {"review": {"trigger": "off"}}


def test_format_toml_value_round_trips_through_parse_cli_value() -> None:
    # The serializer is the exact inverse of parse_cli_value: what an editor
    # prefills from it must save back unchanged (the TUI edit box relies on
    # this for list/dict fields).
    assert io.parse_cli_value(io.format_toml_value(("uv", "run", "pytest"))) == [
        "uv",
        "run",
        "pytest",
    ]
    assert io.parse_cli_value(io.format_toml_value([])) == []


def test_toml_repr_serializes_nested_dict_as_inline_table() -> None:
    # An OpenRouter routing value round-trips through the inline-table form.
    val = {"provider": {"sort": "throughput"}}
    rendered = io.format_toml_value(val)
    assert rendered == '{ provider = { sort = "throughput" } }'
    assert tomllib.loads(f"x = {rendered}")["x"] == val


def test_toml_repr_empty_dict() -> None:
    assert io.format_toml_value({}) == "{}"


def test_config_set_whole_extra_body_value_round_trips(tmp_path: pathlib.Path) -> None:
    # The natural way to set a table-valued config (extra_body) is the whole
    # value at table granularity — the deep-leaf path would collide with the
    # inline parent. Setting the whole value must produce valid, re-parseable
    # TOML even when the section already has the key.
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    value = io.parse_cli_value('{ provider = { sort = "latency" } }')  # pyright: ignore[reportPrivateUsage]
    io.upsert_toml_leaf(  # pyright: ignore[reportPrivateUsage]
        cfg, "providers.openrouter.extra_body", value
    )
    parsed = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert parsed["providers"]["openrouter"]["extra_body"] == {"provider": {"sort": "latency"}}
    # the sibling key survived the surgery
    assert parsed["providers"]["openrouter"]["api_format"] == "openai"


def test_control_chars_serialize_to_valid_toml(tmp_path: pathlib.Path) -> None:
    """The serializer escapes control chars, so a newline-bearing value writes parseable TOML.

    `config set` reported success over the unparseable file, blaming another layer.
    """
    path = tmp_path / "c.toml"
    path.write_text("", encoding="utf-8")
    io.upsert_toml_leaf(path, "git.name", "a\nb\tc\rd\x1be")  # pyright: ignore[reportPrivateUsage]
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["git"]["name"] == "a\nb\tc\rd\x1be"
    # control-char-free strings serialize byte-identically to before
    assert io.format_toml_value('quote"and\\back') == '"quote\\"and\\\\back"'


def test_concurrent_leaf_writes_lose_no_update(tmp_path: pathlib.Path) -> None:
    """Two writers serialize on portable.locked_file, so neither publish drops the other's key.

    A CLI `config set` racing the web or TUI editor lost an update; the lock is removed on
    release.
    """
    path = tmp_path / "config.toml"
    n_writers, n_keys = 8, 5
    barrier = threading.Barrier(n_writers)

    def writer(i: int) -> None:
        barrier.wait()
        for j in range(n_keys):
            io.upsert_toml_leaf(path, f"t{i}.k{j}", i * 100 + j)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert {f"t{i}.k{j}" for i in range(n_writers) for j in range(n_keys)} == {
        f"{table}.{leaf}" for table, leaves in data.items() for leaf in leaves
    }
    assert not (tmp_path / "config.toml.lock").exists()


def test_upsert_toml_leaf_replaces_a_whole_multiline_value(tmp_path: pathlib.Path) -> None:
    """A multi-line value must be replaced whole.

    Rewriting only its opening line orphans the rest, producing unparseable TOML from every config
    writer (`config set`/`add`/`remove`, `connect`, the TUI and web editors); multi-line arrays are
    the hand-written form for allow_urls, personas, verify_command.
    """
    path = tmp_path / "config.toml"
    path.write_text(
        '[sandbox]\nallow_urls = [\n  "https://a.example",\n]\nprofile = "strict"\n',
        encoding="utf-8",
    )
    io.upsert_toml_leaf(path, "sandbox.allow_urls", ["https://a.example", "https://b.example"])
    data = tomllib.loads(path.read_text(encoding="utf-8"))  # parses at all
    assert data["sandbox"]["allow_urls"] == ["https://a.example", "https://b.example"]
    assert data["sandbox"]["profile"] == "strict"  # the sibling key survived


def test_upsert_toml_leaf_replaces_a_whole_multiline_string(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[a]\nx = """\nmultiline\nstring\n"""\ny = 1\n', encoding="utf-8")
    io.upsert_toml_leaf(path, "a.x", "flat")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["a"] == {"x": "flat", "y": 1}


def test_no_writer_deletes_a_top_level_inline_table(tmp_path: pathlib.Path) -> None:
    """`sandbox = { protect_git = false, ... }` is legal TOML.

    The bare-key drop exists for the SCALAR-vs-[table] conflict (`profile` vs `[profile]`), but its
    regex matching a table-valued key too would delete the whole line (every sibling setting in it)
    on any sandbox.* write and report success. The refusal lives in the writer so `config
    add`/`remove` and the engine-level writers behind the TUI, connect and init cannot skip it.
    """
    p = tmp_path / "c.toml"
    body = 'sandbox = { protect_git = false, run_commands = "yes", memory_limit_mb = 8000 }\n'
    p.write_text(body, encoding="utf-8")

    with pytest.raises(ConfigError, match="not a plain \\[table\\]"):
        io.upsert_toml_leaf(p, "sandbox.allow_urls", ["https://example.com"])
    assert p.read_text(encoding="utf-8") == body, "the operator's settings were deleted"

    # The conflict it DOES exist for still resolves: a bare scalar gives way.
    p.write_text('profile = "ultra"\n', encoding="utf-8")
    io.upsert_toml_leaf(p, "profile.nested", 1)
    parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    assert parsed["profile"] == {"nested": 1}


def test_remove_toml_leaf_refuses_an_undeclared_table_ancestor(tmp_path: pathlib.Path) -> None:
    """Removing a leaf inside an inline table or dotted key refuses like upsert_toml_leaf.

    It read as "not found", which every removal surface translated to "nothing to unset" while
    `config get` showed the leaf set.
    """
    p = tmp_path / "c.toml"
    p.write_text('sandbox = { protect_git = false, run_commands = "yes" }\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot be unset on its own"):
        io.remove_toml_leaf(p, "sandbox.protect_git")
    p.write_text("sandbox.protect_git = false\n", encoding="utf-8")  # the dotted-key shape
    with pytest.raises(ConfigError, match="cannot be unset on its own"):
        io.remove_toml_leaf(p, "sandbox.protect_git")
    # A genuinely absent leaf under a DECLARED header keeps the quiet False.
    p.write_text('[sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    assert io.remove_toml_leaf(p, "sandbox.protect_git") is False


def test_upsert_end_scan_skips_a_multiline_value_with_a_bracket_line(
    tmp_path: pathlib.Path,
) -> None:
    """The section-end scan skips an interior line starting with `[` inside a triple-quoted value.

    The truncated region made the leaf search stop early and the insert land inside the
    operator's string, destroying a sibling and reporting success.
    """
    p = tmp_path / "c.toml"
    p.write_text(
        '[harness.metric]\npattern = """\n[0-9]+ ms\n"""\ngoal = "minimize"\n',
        encoding="utf-8",
    )
    io.upsert_toml_leaf(p, "harness.metric.goal", "maximize")
    parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    assert parsed["harness"]["metric"]["goal"] == "maximize", "the real leaf must be rewritten"
    assert parsed["harness"]["metric"]["pattern"] == "[0-9]+ ms\n", "the string was corrupted"

    # The unset twin: it must FIND (and remove) the real leaf, not report absent.
    assert io.remove_toml_leaf(p, "harness.metric.goal") is True
    parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    assert "goal" not in parsed["harness"]["metric"]
    assert parsed["harness"]["metric"]["pattern"] == "[0-9]+ ms\n"


def test_drop_top_region_key_skips_a_multiline_value_bracket_line(tmp_path: pathlib.Path) -> None:
    # profile (bare scalar) vs [profile] surgery: dropping the top-level key must
    # not stop early at a '[' line inside a preceding top-level multi-line value.
    p = tmp_path / "c.toml"
    p.write_text(
        'note = """\n[not a header]\n"""\nprofile = "old"\n',
        encoding="utf-8",
    )
    io.upsert_toml_leaf(p, "profile.review.trigger", "off")  # forces [profile] surgery
    parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    assert parsed["note"] == "[not a header]\n", "the top-level string was corrupted"
    assert parsed["profile"]["review"]["trigger"] == "off"


def test_remove_toml_table_drops_an_array_of_tables_subtable(tmp_path: pathlib.Path) -> None:
    """Removing [cli] takes its [[cli.aliases]] array-of-tables subtable with it.

    _header_name reported [[x]] as not a table, so the subtable and everything after stayed,
    leaving the config unloadable and config fix stuck.
    """
    p = tmp_path / "c.toml"
    p.write_text(
        '[cli]\nx = 1\n\n[[cli.aliases]]\nname = "a"\n\n[sandbox]\nprotect_git = true\n',
        encoding="utf-8",
    )
    assert io.remove_toml_table(p, "cli") is True
    out = p.read_text(encoding="utf-8")
    assert "cli" not in tomllib.loads(out)
    assert "aliases" not in out  # the subtable went with its parent
    assert tomllib.loads(out)["sandbox"] == {"protect_git": True}  # sibling preserved
