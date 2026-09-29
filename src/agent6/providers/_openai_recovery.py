# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Recover tool calls a model wrote into its text, for the OpenAI provider.

Some servers (Ollama and llama.cpp chat templates for Qwen, Hermes and other small
models, some OpenRouter backends) leave `tool_calls` empty and leak the call into
the assistant text as bare or fenced JSON, a `<tool_call>` tag, Qwen-Coder XML or
a Gemma `tool_code` fence. Without recovery the loop sees text and no tool_use and
stalls. Recovery runs only when no native call exists and a tool was offered; every
form must name an offered tool except the `<tool_call>` tag, which keeps an unknown
name for the dispatcher's error. The text is read once, left to right: a call
quoted inside a closed fence stays text, and a call restated in a second form is
that call.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_OPENER_RE = re.compile(r"(?m)^ {0,3}(?:`{3,}|~{3,})|<tool_call>|<function\s*=")
_FENCE_OPEN_RE = re.compile(
    r"(?m)^ {0,3}(?P<marker>`{3,}|~{3,})[ \t]*(?P<lang>[^\s`~]*)[^\n]*(?:\n|$)"
)
_TOOL_CALL_CLOSE = "</tool_call>"
# Qwen-Coder XML; a closer may be missing or misspelled, so the next opener (lookahead) ends a body.
_FUNCTION_CALL_RE = re.compile(
    r"<function\s*=\s*([^>\s]+?)\s*>(.*?)(?:</function>|</tool_call>|(?=<function\s*=)|\Z)",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+?)\s*>(.*?)(?:</parameter>|(?=<parameter\s*=)|\Z)",
    re.DOTALL,
)
_PARAMETER_CLOSE = "</parameter>"
# Orphan closers Qwen's template leaves after a block go with the recovered call's span.
_TRAILING_SCAFFOLD_RE = re.compile(r"(?:\s*(?:</tool_call>|</function>|</parameter>))+")

type _Call = dict[str, Any]


def lenient_json_object(raw: object) -> dict[str, Any] | None:
    """Re-parse a tool-call `arguments` string that strict JSON rejected.

    Accepts a raw control character inside a string value and trailing junk after
    a valid object (a leaked tag or prose), the two malformations weak models emit.

    Args:
        raw: The arguments string.

    Returns:
        The object; None for a scalar, an array or a still-invalid string, so the
        caller keeps the `_raw_arguments` sentinel rather than guessing.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed, _ = json.JSONDecoder(strict=False).raw_decode(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_code_call_to_dict(
    node: ast.Call, source: str, tool_names: frozenset[str]
) -> dict[str, Any] | None:
    """Turn one call node into a tool call when it targets an offered tool.

    A non-tool wrapper such as `print(tool(...))` is unwrapped one level. Keyword
    arguments are read as literals; a non-literal, positional or splatted argument
    marks the whole input malformed rather than dropping it. A tool nested as an
    argument value is not mined.

    Args:
        node: The call node.
        source: The block's source, for the raw-arguments diagnostic.
        tool_names: The tools offered.

    Returns:
        The `{"name", "input"}` call, or None for a non-tool call.
    """
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in tool_names:
        for arg in node.args:
            if isinstance(arg, ast.Call):
                inner = _tool_code_call_to_dict(arg, source, tool_names)
                if inner is not None:
                    return inner
        return None
    raw = ast.get_source_segment(source, node) or ast.unparse(node)
    if node.args or any(kw.arg is None for kw in node.keywords):
        return {"name": node.func.id, "input": {"_raw_arguments": raw}}
    args: dict[str, Any] = {}
    for kw in node.keywords:
        assert kw.arg is not None
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return {"name": node.func.id, "input": {"_raw_arguments": raw}}
    return {"name": node.func.id, "input": args}


def _tool_code_calls(code: str, tool_names: frozenset[str]) -> list[_Call]:
    """Return the offered-tool calls in one Gemma `tool_code` block, in source order.

    The block is parsed with `ast`, never executed; top-level calls and list or
    tuple elements count, and a `print(...)` wrapper is unwrapped.
    """
    code = code.strip()
    if not code:
        return []
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return []
    out: list[_Call] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            continue
        value = stmt.value
        elements = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        for el in elements:
            if isinstance(el, ast.Call):
                call = _tool_code_call_to_dict(el, code, tool_names)
                if call is not None:
                    out.append(call)
    return out


def _coerce_param_value(value: str, declared_type: str | None) -> Any:  # noqa: PLR0911, PLR0912
    """Coerce a Qwen-XML `<parameter>` string to its schema-declared type.

    The template frames each value in newlines, which are stripped; a string
    parameter is then kept byte-exact, a structured or scalar one is parsed, and
    an undeclared one is parsed only when it looks like a JSON array or object.

    Args:
        value: The parameter text as matched.
        declared_type: The JSON schema type of the parameter; None when unknown.

    Returns:
        The coerced value, or the string when the coercion fails.
    """
    v = value
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    if declared_type == "string":
        return v
    if declared_type in ("array", "object"):
        try:
            return json.loads(v.strip())
        except (json.JSONDecodeError, TypeError):
            return v  # pydantic surfaces the validation error
    if declared_type == "integer":
        try:
            return int(v.strip())
        except ValueError:
            return v
    if declared_type == "number":
        try:
            return float(v.strip())
        except ValueError:
            return v
    if declared_type == "boolean":
        normalized = v.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
        return v
    stripped = v.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def _xml(
    text: str,
    start: int,
    tool_names: frozenset[str],
    tool_schemas: dict[str, dict[str, Any]] | None,
) -> tuple[int, list[_Call]] | None:
    """Parse the Qwen-Coder XML call opening at a position.

    A block missing its closer ends at its last closed parameter, or at the end of
    the text while a parameter is still open.

    Args:
        text: The whole text.
        start: The opener's offset.
        tool_names: The tools offered.
        tool_schemas: The offered tools' input schemas, for coercing parameters.

    Returns:
        The block's end and the one call, or None when the name is not an offered tool.
    """
    fmatch = _FUNCTION_CALL_RE.match(text, start)
    if fmatch is None:
        return None
    name = fmatch.group(1).strip()
    if name not in tool_names:
        return None
    body = fmatch.group(2)
    end = fmatch.end()
    if fmatch.end(2) == end:
        last = body.rfind(_PARAMETER_CLOSE)
        if last != -1 and "<parameter" not in body[last:]:
            body = body[: last + len(_PARAMETER_CLOSE)]
            end = fmatch.start(2) + len(body)
    schema = (tool_schemas or {}).get(name) or {}
    props = schema.get("properties") or {}
    args: dict[str, Any] = {}
    for pmatch in _PARAMETER_RE.finditer(body):
        key = pmatch.group(1).strip()
        decl = props.get(key) or {}
        decl_type = decl.get("type") if isinstance(decl, dict) else None
        args[key] = _coerce_param_value(pmatch.group(2), decl_type)
    # The template's stray closers right after the block go with the call.
    trail = _TRAILING_SCAFFOLD_RE.match(text, end)
    if trail is not None:
        end = trail.end()
    return end, [{"name": name, "input": args}]


def _extract_tool_call_obj(
    candidate: str, tool_names: frozenset[str], *, allow_unknown: bool = False
) -> dict[str, Any] | None:
    """Return the one JSON tool-call object in a text, or None.

    Args:
        candidate: The text.
        tool_names: The tools offered.
        allow_unknown: Whether a name outside the offered tools is kept.

    Returns:
        The `{"name", "input"}` call, or None when the text is not one call object.
    """
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name or (not allow_unknown and name not in tool_names):
        return None
    # Accept the common spellings local templates use for the args object.
    for key in ("arguments", "parameters", "input"):
        if key in obj:
            raw_args = obj[key]
            break
    else:
        raw_args = {}
    # A few templates double-encode the args as a JSON string.
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            repaired = lenient_json_object(raw_args)
            raw_args = repaired if repaired is not None else {"_raw_arguments": raw_args}
    if not isinstance(raw_args, dict):
        raw_args = {"_raw_arguments": json.dumps(raw_args)}
    return {"name": name, "input": raw_args}


def _fence(text: str, start: int, tool_names: frozenset[str]) -> tuple[int, list[_Call]] | None:
    """Parse the Markdown fence opening at a position.

    A `tool_code` fence holds Python calls; a `json` or bare fence holding one JSON
    call is that call; any other fence quotes its content.

    Args:
        text: The whole text.
        start: The opener's offset.
        tool_names: The tools offered.

    Returns:
        The fence's end and the calls it holds, or None for an opener with no
        closer, which quotes nothing.
    """
    opener = _FENCE_OPEN_RE.match(text, start)
    assert opener is not None
    marker = opener.group("marker")
    closer = re.compile(rf"(?m){re.escape(marker[0])}{{{len(marker)},}}[ \t]*$").search(
        text, opener.end()
    )
    if closer is None:
        return None
    content = text[opener.end() : closer.start()]
    lang = opener.group("lang")
    if lang == "tool_code":
        return closer.end(), _tool_code_calls(content, tool_names)
    if lang in ("", "json"):
        call = _extract_tool_call_obj(content, tool_names)
        return closer.end(), [call] if call is not None else []
    return closer.end(), []


def _tag(
    text: str,
    start: int,
    tool_names: frozenset[str],
    tool_schemas: dict[str, dict[str, Any]] | None,
) -> tuple[int, list[_Call]] | None:
    """Parse the `<tool_call>` tag opening at a position.

    Args:
        text: The whole text.
        start: The opener's offset.
        tool_names: The tools offered.
        tool_schemas: The offered tools' input schemas, for nested forms.

    Returns:
        The tag's end and its calls (one JSON call of any name, or the forms nested
        in it), or None for a tag with no closer or no call, which stays text.
    """
    content_start = start + len("<tool_call>")
    close = text.find(_TOOL_CALL_CLOSE, content_start)
    if close == -1:
        return None
    content = text[content_start:close]
    call = _extract_tool_call_obj(content, tool_names, allow_unknown=True)
    calls = [call] if call is not None else _scan(content, tool_names, tool_schemas)[0]
    if not calls:
        return None
    return close + len(_TOOL_CALL_CLOSE), calls


def _scan(
    text: str,
    tool_names: frozenset[str],
    tool_schemas: dict[str, dict[str, Any]] | None,
) -> tuple[list[_Call], str]:
    """Read the text once, left to right, parsing each form at its opener.

    Args:
        text: The text to scan.
        tool_names: The tools offered.
        tool_schemas: The offered tools' input schemas.

    Returns:
        The calls in source order, each once, and the text they leave.
    """
    calls: list[_Call] = []
    kept: list[str] = []
    pos = 0
    while (opener := _OPENER_RE.search(text, pos)) is not None:
        opened = opener.group()
        if opened == "<tool_call>":
            form = _tag(text, opener.start(), tool_names, tool_schemas)
        elif opened.startswith("<function"):
            form = _xml(text, opener.start(), tool_names, tool_schemas)
        else:
            form = _fence(text, opener.start(), tool_names)
        if form is None:
            kept.append(text[pos : opener.end()])
            pos = opener.end()
            continue
        end, found = form
        if found:
            kept.append(text[pos : opener.start()])
            for call in found:
                if call not in calls:
                    calls.append(call)
        else:
            kept.append(text[pos:end])
        pos = end
    kept.append(text[pos:])
    return calls, "".join(kept)


def coerce_text_tool_calls(
    text: str,
    tool_names: frozenset[str],
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[_Call], str]:
    """Recover the tool calls a model wrote into its text.

    Args:
        text: The assistant text.
        tool_names: The tools offered; empty recovers nothing.
        tool_schemas: The offered tools' input schemas.

    Returns:
        The calls in source order and the unconsumed text; with no call, the text
        as given.
    """
    if not text or not tool_names:
        return [], text
    # A bare JSON call is parsed first so call-shaped text inside its string arguments stays data.
    bare = _extract_tool_call_obj(text, tool_names)
    if bare is not None:
        return [bare], ""
    calls, remaining = _scan(text, tool_names, tool_schemas)
    return (calls, remaining.strip()) if calls else ([], text)
