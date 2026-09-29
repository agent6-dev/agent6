# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `{{ ... }}` rendering and list-splicing (agent6.machine.template)."""

from __future__ import annotations

import pytest

from agent6.machine import predicate, template


def test_render_string_scalar() -> None:
    tmpl = template.parse_template("path={{ cursor }}")
    assert template.render_string(tmpl, {"cursor": "abc"}, where="t") == "path=abc"


def test_render_string_int_and_bool() -> None:
    assert template.render_string(template.parse_template("{{ n }}"), {"n": 7}, where="t") == "7"
    assert (
        template.render_string(template.parse_template("{{ b }}"), {"b": True}, where="t") == "true"
    )
    assert (
        template.render_string(template.parse_template("{{ b }}"), {"b": False}, where="t")
        == "false"
    )


def test_render_string_json_filter_sorts_keys() -> None:
    tmpl = template.parse_template("{{ d | json }}")
    assert template.render_string(tmpl, {"d": {"b": 1, "a": 2}}, where="t") == '{"a":2,"b":1}'


def test_render_string_len_filter() -> None:
    tmpl = template.parse_template("{{ items | len }}")
    assert template.render_string(tmpl, {"items": ["a", "b", "c"]}, where="t") == "3"


def test_render_value_lone_ref_keeps_native_type() -> None:
    tmpl = template.parse_template("{{ items }}")
    value = template.render_value(tmpl, {"items": ["a", "b"]}, where="t")
    assert value == ["a", "b"]


def test_render_value_non_lone_renders_string() -> None:
    tmpl = template.parse_template("n={{ n }}")
    assert template.render_value(tmpl, {"n": 3}, where="t") == "n=3"


def test_render_command_splices_list() -> None:
    argv = template.render_command(("rec", "{{ items }}"), {"items": ["a", "b"]}, where="cmd")
    assert argv == ["rec", "a", "b"]


def test_render_command_lone_scalar_renders_one_arg() -> None:
    argv = template.render_command(("rec", "--n", "{{ n }}"), {"n": 5}, where="cmd")
    assert argv == ["rec", "--n", "5"]


def test_resolve_reference_navigates_record() -> None:
    ref = predicate.Reference(root="verdict", path=("label",))
    assert template.resolve_reference(ref, {"verdict": {"label": "urgent"}}) == "urgent"


def test_resolve_reference_unknown_root_raises() -> None:
    with pytest.raises(template.TemplateRuntimeError):
        template.resolve_reference(predicate.Reference(root="nope", path=()), {})


def test_resolve_reference_into_non_record_raises() -> None:
    with pytest.raises(template.TemplateRuntimeError):
        template.resolve_reference(predicate.Reference(root="x", path=("k",)), {"x": 3})


def test_unbalanced_braces_is_error() -> None:
    with pytest.raises(template.TemplateError):
        template.parse_template("{{ a }} and {{ b")


def test_an_expression_in_an_interpolation_is_refused_naming_the_grammar() -> None:
    """The interpolation error states the accepted shape (one dotted name, at most one filter)."""
    with pytest.raises(template.TemplateError) as exc:
        template.parse_template("{{ not result.passed }}")
    msg = str(exc.value)
    assert "'not result.passed' is not a valid reference" in msg
    assert "one dotted name" in msg and "json, len" in msg and "no operators" in msg
