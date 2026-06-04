# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `{{ ... }}` rendering and list-splicing (agent6.machine.template)."""

from __future__ import annotations

import pytest

from agent6.machine.predicate import Reference
from agent6.machine.template import (
    TemplateError,
    TemplateRuntimeError,
    parse_template,
    render_command,
    render_string,
    render_value,
    resolve_reference,
)


def test_render_string_scalar() -> None:
    tmpl = parse_template("path={{ cursor }}")
    assert render_string(tmpl, {"cursor": "abc"}, where="t") == "path=abc"


def test_render_string_int_and_bool() -> None:
    assert render_string(parse_template("{{ n }}"), {"n": 7}, where="t") == "7"
    assert render_string(parse_template("{{ b }}"), {"b": True}, where="t") == "true"
    assert render_string(parse_template("{{ b }}"), {"b": False}, where="t") == "false"


def test_render_string_json_filter_sorts_keys() -> None:
    tmpl = parse_template("{{ d | json }}")
    assert render_string(tmpl, {"d": {"b": 1, "a": 2}}, where="t") == '{"a":2,"b":1}'


def test_render_string_len_filter() -> None:
    tmpl = parse_template("{{ items | len }}")
    assert render_string(tmpl, {"items": ["a", "b", "c"]}, where="t") == "3"


def test_render_value_lone_ref_keeps_native_type() -> None:
    tmpl = parse_template("{{ items }}")
    value = render_value(tmpl, {"items": ["a", "b"]}, where="t")
    assert value == ["a", "b"]


def test_render_value_non_lone_renders_string() -> None:
    tmpl = parse_template("n={{ n }}")
    assert render_value(tmpl, {"n": 3}, where="t") == "n=3"


def test_render_command_splices_list() -> None:
    argv = render_command(("rec", "{{ items }}"), {"items": ["a", "b"]}, where="cmd")
    assert argv == ["rec", "a", "b"]


def test_render_command_lone_scalar_renders_one_arg() -> None:
    argv = render_command(("rec", "--n", "{{ n }}"), {"n": 5}, where="cmd")
    assert argv == ["rec", "--n", "5"]


def test_resolve_reference_navigates_record() -> None:
    ref = Reference(root="verdict", path=("label",))
    assert resolve_reference(ref, {"verdict": {"label": "urgent"}}) == "urgent"


def test_resolve_reference_unknown_root_raises() -> None:
    with pytest.raises(TemplateRuntimeError):
        resolve_reference(Reference(root="nope", path=()), {})


def test_resolve_reference_into_non_record_raises() -> None:
    with pytest.raises(TemplateRuntimeError):
        resolve_reference(Reference(root="x", path=("k",)), {"x": 3})


def test_unbalanced_braces_is_error() -> None:
    with pytest.raises(TemplateError):
        parse_template("{{ a }} and {{ b")
