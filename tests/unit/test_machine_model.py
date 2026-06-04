# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.machine.model — `.asm.toml` parse + semantic validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.machine.model import MachineError, load_machine

# The worked example from STATE_MACHINES.md §10. The canonical
# happy path; error-case tests mutate a copy of this.
VALID_MACHINE = """
machine = "item-classifier"
version = 1
initial = "poll"

[budget]
max_usd         = 25.0
max_transitions = 100000

[vars.operator]
inbox_dir = { type = "str", value = "/srv/inbox" }
poll_secs = { type = "int", value = 300 }

[vars.code]
pending = { type = "list[str]", default = [] }
cursor  = { type = "str",       default = "" }

[vars.agent]
verdict = { type = "classification", default = {} }

[schemas.classification]
label      = { type = "str", enum = ["urgent", "normal", "spam"] }
confidence = "float"

[schemas.scan_result]
pending = "list[str]"
cursor  = "str"

[states.poll]
kind = "wait"
every_secs = "{{ poll_secs }}"
on = { tick = "scan", signal = "scan" }

[states.scan]
kind = "tool"
command = ["scan-inbox", "--dir", "{{ inbox_dir }}", "--since", "{{ cursor }}"]
output_schema = "scan_result"
capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }
timeout_secs = 60
on = { ok = "have_items", nonzero = "poll", timeout = "poll" }

[states.have_items]
kind = "branch"
when = [
  { if = "len(pending) == 0", goto = "poll" },
  { else = true,              goto = "classify" },
]

[states.classify]
kind  = "agent"
model = "claude-sonnet-4-5"
prompt = "Classify these pending items: {{ pending | json }}"
output_schema = "classification"
capture = { finish_json = "verdict" }
timeout_secs = 600
on = { ok = "route", failed = "poll", budget_exhausted = "halt", timeout = "poll" }

[states.route]
kind = "branch"
when = [
  { if = "verdict.label == 'urgent' and verdict.confidence >= 0.7", goto = "record" },
  { else = true, goto = "poll" },
]

[states.record]
kind = "tool"
command = ["archive-item", "--label", "{{ verdict.label }}", "{{ pending }}"]
timeout_secs = 30
on = { ok = "poll", nonzero = "poll", timeout = "poll" }

[states.halt]
kind   = "terminal"
status = "failed"
reason = "machine budget exhausted"
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "m.asm.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _problems(tmp_path: Path, body: str) -> list[str]:
    with pytest.raises(MachineError) as excinfo:
        load_machine(_write(tmp_path, body))
    return excinfo.value.problems


def test_valid_machine_loads(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, VALID_MACHINE))
    assert spec.machine == "item-classifier"
    assert spec.initial == "poll"
    assert set(spec.states) == {
        "poll",
        "scan",
        "have_items",
        "classify",
        "route",
        "record",
        "halt",
    }


def test_missing_file(tmp_path: Path) -> None:
    problems = _problems(tmp_path, "")  # empty -> not even `machine` key
    assert problems


def test_bad_toml(tmp_path: Path) -> None:
    problems = _problems(tmp_path, "machine = ")
    assert any("not valid TOML" in p for p in problems)


# -- naming rules ----------------------------------------------------------


def test_duplicate_name_across_owners(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[vars.agent]\nverdict = { type = "classification", default = {} }',
        '[vars.agent]\nverdict = { type = "classification", default = {} }\n'
        'cursor = { type = "str", default = "" }',
    )
    problems = _problems(tmp_path, body)
    assert any("declared in both" in p and "cursor" in p for p in problems)


def test_bare_top_level_var(tmp_path: Path) -> None:
    body = VALID_MACHINE + '\n[vars.stray]\nx = { type = "str", default = "" }\n'
    # `vars.stray` becomes an owner-less subtable.
    problems = _problems(tmp_path, body)
    assert any("no owner subtable" in p for p in problems)


def test_reserved_name(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        "[vars.code]\npending",
        '[vars.code]\nresult = { type = "str", default = "" }\npending',
    )
    problems = _problems(tmp_path, body)
    assert any("reserved" in p and "result" in p for p in problems)


def test_non_identifier_variable(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'cursor  = { type = "str",       default = "" }',
        'cursor  = { type = "str",       default = "" }\n'
        '"last-seen" = { type = "str", default = "" }',
    )
    problems = _problems(tmp_path, body)
    assert any("not a valid identifier" in p for p in problems)


# -- ownership wall --------------------------------------------------------


def test_tool_cannot_write_agent_var(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }',
        'capture = { set = { verdict = "{{ result.pending }}" } }',
    )
    problems = _problems(tmp_path, body)
    assert any("may only write `[vars.code]`" in p for p in problems)


def test_capture_cannot_write_operator_var(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }',
        'capture = { set = { poll_secs = "{{ result.cursor }}" } }',
    )
    problems = _problems(tmp_path, body)
    assert any("owned by `[vars.operator]`" in p for p in problems)


# -- branches --------------------------------------------------------------


def test_branch_not_total(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '  { if = "len(pending) == 0", goto = "poll" },\n'
        '  { else = true,              goto = "classify" },',
        '  { if = "len(pending) == 0", goto = "poll" },',
    )
    problems = _problems(tmp_path, body)
    assert any("not total" in p for p in problems)


def test_branch_else_must_be_last(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '  { if = "len(pending) == 0", goto = "poll" },\n'
        '  { else = true,              goto = "classify" },',
        '  { else = true, goto = "poll" },\n  { if = "len(pending) == 0", goto = "classify" },',
    )
    problems = _problems(tmp_path, body)
    assert any("must be the final" in p for p in problems)


def test_predicate_misspelled_field(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace("verdict.confidence >= 0.7", "verdict.confidense >= 0.7")
    problems = _problems(tmp_path, body)
    assert any("has no field" in p and "confidense" in p for p in problems)


def test_predicate_unknown_variable(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace("len(pending) == 0", "len(nonsense) == 0")
    problems = _problems(tmp_path, body)
    assert any("unknown variable" in p and "nonsense" in p for p in problems)


# -- type checks -----------------------------------------------------------


def test_default_type_mismatch(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'cursor  = { type = "str",       default = "" }',
        'cursor  = { type = "str",       default = 5 }',
    )
    problems = _problems(tmp_path, body)
    assert any("expected str" in p for p in problems)


def test_unknown_type(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('poll_secs = { type = "int"', 'poll_secs = { type = "integer"')
    problems = _problems(tmp_path, body)
    assert any("unknown type" in p for p in problems)


def test_dotting_json_is_error(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[vars.code]\npending = { type = "list[str]", default = [] }',
        '[vars.code]\nblob = { type = "json", default = {} }\n'
        'pending = { type = "list[str]", default = [] }',
    )
    body = body.replace("len(pending) == 0", "blob.key == 0")
    problems = _problems(tmp_path, body)
    assert any("cannot navigate into json" in p for p in problems)


def test_enum_only_on_str(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'confidence = "float"',
        'confidence = { type = "float", enum = ["a"] }',
    )
    problems = _problems(tmp_path, body)
    assert any("enum" in p and "str" in p for p in problems)


def test_schema_cycle(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[schemas.scan_result]\npending = "list[str]"\ncursor  = "str"',
        '[schemas.scan_result]\npending = "list[str]"\ncursor  = "str"\nself = "scan_result"',
    )
    problems = _problems(tmp_path, body)
    assert any("cycle" in p for p in problems)


# -- list splicing / templates --------------------------------------------


def test_bare_list_outside_argv_is_error(tmp_path: Path) -> None:
    # Reading a bare list into a prompt (not argv) must be a load error.
    body = VALID_MACHINE.replace("{{ pending | json }}", "{{ pending }}")
    problems = _problems(tmp_path, body)
    assert any("bare reference to list" in p for p in problems)


def test_list_spliced_inside_larger_string_is_error(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('"{{ pending }}"', '"--items={{ pending }}"')
    problems = _problems(tmp_path, body)
    assert any("bare reference to list" in p for p in problems)


# -- wait timing -----------------------------------------------------------


def test_wait_requires_exactly_one_timing(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"',
        'every_secs = "{{ poll_secs }}"\nuntil = "2030-01-01T00:00:00Z"',
    )
    problems = _problems(tmp_path, body)
    assert any("exactly one of `every_secs`" in p for p in problems)


# -- on-table completeness -------------------------------------------------


def test_tool_missing_outcome_label(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'on = { ok = "have_items", nonzero = "poll", timeout = "poll" }',
        'on = { ok = "have_items", nonzero = "poll" }',
    )
    problems = _problems(tmp_path, body)
    assert any("missing outcome 'timeout'" in p for p in problems)


def test_unknown_outcome_label(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'on = { tick = "scan", signal = "scan" }',
        'on = { tick = "scan", signal = "scan", boom = "scan" }',
    )
    problems = _problems(tmp_path, body)
    assert any("unknown outcome 'boom'" in p for p in problems)


# -- graph -----------------------------------------------------------------


def test_unknown_transition_target(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('signal = "scan" }', 'signal = "nowhere" }')
    problems = _problems(tmp_path, body)
    assert any("not a declared state" in p and "nowhere" in p for p in problems)


def test_unreachable_state(tmp_path: Path) -> None:
    body = VALID_MACHINE + (
        '\n[states.orphan]\nkind = "terminal"\nstatus = "ok"\nreason = "never reached"\n'
    )
    problems = _problems(tmp_path, body)
    assert any("unreachable" in p and "orphan" in p for p in problems)


def test_initial_must_exist(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('initial = "poll"', 'initial = "ghost"')
    problems = _problems(tmp_path, body)
    assert any("initial state 'ghost'" in p for p in problems)
