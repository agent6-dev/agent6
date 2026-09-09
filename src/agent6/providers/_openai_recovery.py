# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Text-embedded tool-call recovery for the OpenAI-compatible provider.

Fallback parsing for models whose server does not populate the native
`tool_calls` array and instead leaks the call into the assistant
`content` text (Qwen/Hermes tags, Qwen-Coder XML, Gemma `tool_code`
fences, bare or fenced JSON). The rationale and the guards live on the
comment block below; `providers/_openai_parse.py`'s `parse_response` is the
only production caller.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# Some OpenAI-compatible servers (notably certain Ollama / llama.cpp
# chat templates for Qwen, Hermes, and other small local models, and some
# OpenRouter upstream backends) do NOT parse the model's tool call into the
# native `tool_calls` array. Instead the call leaks into the assistant
# `content` as plain text, in one of several shapes:
#   - a bare JSON object `{"name": ..., "arguments": {...}}`,
#   - the same wrapped in a ```json fence,
#   - Hermes/Qwen `<tool_call>{json}</tool_call>` tags, or
#   - the Qwen-Coder XML form ``<function=NAME><parameter=KEY>VALUE
#     </parameter>...</function>`` (string-valued params, NOT JSON).
# Without recovery the run loop sees text + no tool_use and stalls
# ("went quiet" / "silent_finish"), which kills an entire family of
# open-weight coding models (qwen3-coder, hermes, devstral, ...). We
# recover these into real tool_uses only when no native call exists and at
# least one tool was offered. Every form must name an offered tool except the
# explicit `<tool_call>` tag, which keeps an unknown name so the dispatcher
# returns that call's error. The text is read once, left to right: a call
# quoted inside a closed Markdown fence stays text, and a call restated in a
# second form is that call.
_OPENER_RE = re.compile(r"(?m)^ {0,3}(?:`{3,}|~{3,})|<tool_call>|<function\s*=")
_FENCE_OPEN_RE = re.compile(
    r"(?m)^ {0,3}(?P<marker>`{3,}|~{3,})[ \t]*(?P<lang>[^\s`~]*)[^\n]*(?:\n|$)"
)
_TOOL_CALL_CLOSE = "</tool_call>"
# Qwen-Coder XML tool form. The closing `</function>` is sometimes
# missing (truncation) or mis-spelled `</tool_call>`; capture the name
# and a lenient body, then mine `<parameter=...>` pairs out of it.
# The next-tag terminators are LOOKAHEADS (not consuming): when a closing tag
# is missing, the body must end *before* the next block's opening tag without
# swallowing it -- otherwise finditer consumes that opener and silently drops
# the following function/parameter (corrupts e.g. apply_edit on open-weight
# models that emit unclosed Qwen-XML tool calls).
_FUNCTION_CALL_RE = re.compile(
    r"<function\s*=\s*([^>\s]+?)\s*>(.*?)(?:</function>|</tool_call>|(?=<function\s*=)|\Z)",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+?)\s*>(.*?)(?:</parameter>|(?=<parameter\s*=)|\Z)",
    re.DOTALL,
)
_PARAMETER_CLOSE = "</parameter>"
# Leftover scaffolding to scrub from the visible text once calls are mined.
# Orphan closers Qwen's template leaves right after a </function> block (a
# stray </tool_call> most commonly); swallowed into the recovered call's span.
_TRAILING_SCAFFOLD_RE = re.compile(r"(?:\s*(?:</tool_call>|</function>|</parameter>))+")

type _Call = dict[str, Any]


def lenient_json_object(raw: object) -> dict[str, Any] | None:
    """Recover a tool-call `arguments` string that strict `json.loads`
    rejected, when the fix is safe and unambiguous. Returns the object, or None.

    Two common weak/open-model malformations:
    - a raw control char (an unescaped newline/tab) inside a string value, which
      `strict=False` accepts;
    - trailing junk after a valid object (a leaked `</invoke>` tag or prose),
      which `raw_decode` ignores by parsing only the leading value.

    Only a dict result is returned; a scalar/array (or a still-invalid string,
    e.g. a bad `\\d` regex escape) yields None so the caller keeps the
    `_raw_arguments` sentinel rather than guessing."""
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
    """Turn one `ast.Call` into `{"name", "input"}` if it (or, unwrapping a
    non-tool wrapper such as `print(tool(...))`, an inner call) targets an
    offered tool. Keyword args are read with `ast.literal_eval` (already typed),
    so no coercion; a non-literal, positional or splatted argument marks the
    whole input malformed rather than silently dropping it. Returns None for a non-tool call;
    we do NOT recurse into kwarg VALUES, so a tool nested as an
    argument (`apply_edit(path=read_file(...))`) is not separately mined."""
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in tool_names:
        # One-level unwrap: a non-tool wrapper around a single tool call.
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
    """The offered-tool calls in one Gemini/Gemma ```tool_code block.

    Parses the block with `ast` (never executes it). Top-level calls and
    list/tuple elements keep source order; a `print(...)` wrapper around one
    call is unwrapped."""
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

    The Qwen-Coder template emits each parameter value as raw text framed by
    newlines, e.g. `<parameter=path>\\ninterp.py\\n</parameter>`. Strip the
    framing newlines, then coerce by the tool's declared JSON-Schema type so
    structured params (`array`/`object`) and scalars rebuild correctly
    while string params (code in `new_string`/`old_string`) are left byte-
    exact. Unknown type: parse only if it looks like JSON array/object, else
    keep the string.
    """
    # Strip the single leading/trailing newline the template adds without
    # touching interior or leading-space indentation that code params need.
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
            return v  # let pydantic surface a clear validation error
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
    # Unknown / absent schema: only auto-parse clearly-structured JSON so a
    # plain string value is never silently turned into a number or dict.
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
    """The Qwen-Coder `<function=NAME><parameter=KEY>VALUE</parameter>` call
    opening at *start*: its end and the call. None when NAME is not an offered
    tool. A block missing its closer ends at its last closed parameter, or at
    the end of the text while a parameter is still open (a truncated call)."""
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
    """Parse one JSON tool-call object, optionally accepting an unknown name."""
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
    """The Markdown fence opening at *start*: its end and the calls it holds.
    A ```tool_code block holds Python calls; a ```json or bare fence holding
    one JSON call is that call; any other fence quotes its content. None for
    an opener with no closer, which quotes nothing."""
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
    """The `<tool_call>` tag opening at *start*: its end and the calls it
    holds, one JSON call of any name or the forms nested in it. None for a
    tag with no closer or no call, which stays text."""
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
    """Read *text* once, left to right: at each form opener the form is parsed
    and its markup consumed; a call restated in a second form is that call.
    Returns the calls in source order and the text they leave."""
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
    """Recover the tool calls a model wrote into its text, in source order,
    and return the unconsumed text."""
    if not text or not tool_names:
        return [], text
    # An exact bare JSON call is already self-delimiting. Parse it before
    # looking for markup so call-shaped text inside a string argument stays data.
    bare = _extract_tool_call_obj(text, tool_names)
    if bare is not None:
        return [bare], ""
    calls, remaining = _scan(text, tool_names, tool_schemas)
    return (calls, remaining.strip()) if calls else ([], text)
