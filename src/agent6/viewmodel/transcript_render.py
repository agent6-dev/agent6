# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a run's per-call provider transcripts into a readable conversation.

agent6 writes one JSON file per LLM round-trip under ``<run>/transcripts/`` --
the full, lossless ``{request, response}`` (secrets redacted). Each request
carries the whole conversation up to that call, so the sequence is a complete,
self-contained record (no join with ``logs.jsonl`` needed). This module folds
that sequence -- across BOTH the OpenAI and Anthropic wire shapes -- into an
ordered list of conversation turns and renders them as Markdown.

``agent6 runs transcript`` is the CLI front end (``--json`` returns the raw
transcript array instead). The fold walks transcripts in seq order, emitting
only newly-introduced messages per call, so the cumulative-snapshot growth is
not double-printed and a mid-run context-compaction reset shows as a marker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    """One normalized conversation turn (provider-agnostic).

    Deliberately mutable: `fold_conversation` builds a turn in a shape helper
    that does not know the call it came from, then stamps `seq` on it. Freezing
    would force threading seq through every builder for no gain.
    """

    role: str  # "system" | "user" | "assistant" | "tool" | "marker"
    text: str = ""
    thinking: str = ""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, args_json)
    tool_name: str = ""  # for role == "tool"
    seq: int = 0


def load_transcripts(transcripts_dir: Path) -> list[dict[str, Any]]:
    """Every transcript JSON object under a run's transcripts/ dir, in seq order."""
    if not transcripts_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(transcripts_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    out.sort(key=lambda t: t.get("seq", 0))
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _request_body(t: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(t.get("request")).get("body"))


def _response_body(t: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(t.get("response")).get("body"))


def _shape(req: dict[str, Any], resp: dict[str, Any]) -> str:
    """Detect the provider wire shape of one transcript."""
    if isinstance(resp.get("choices"), list):
        return "openai"
    if isinstance(resp.get("content"), list) and resp.get("role"):
        return "anthropic"
    # Fall back on the request: Anthropic carries a top-level `system` and
    # content-block messages; OpenAI uses a system *message* + flat strings.
    return "anthropic" if "system" in req else "openai"


def _pretty_args(raw: Any) -> str:
    """Tool-call arguments -> compact one-line JSON (best effort)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw.strip()
    try:
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(raw)


def _openai_turns(m: dict[str, Any], names: dict[str, str]) -> list[Turn]:
    role = m.get("role", "")
    if role == "tool":
        name = names.get(str(m.get("tool_call_id", "")), "")
        return [Turn(role="tool", text=str(m.get("content", "")), tool_name=name)]
    if role == "assistant":
        calls: list[tuple[str, str]] = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name", ""))
            calls.append((name, _pretty_args(fn.get("arguments", ""))))
            if isinstance(tc, dict) and tc.get("id"):
                names[str(tc["id"])] = name
        return [
            Turn(
                role="assistant",
                text=str(m.get("content") or ""),
                thinking=str(m.get("reasoning_content") or ""),
                tool_calls=calls,
            )
        ]
    return [Turn(role=role or "user", text=str(m.get("content") or ""))]


def _anthropic_turns(m: dict[str, Any], names: dict[str, str]) -> list[Turn]:
    role = m.get("role", "user")
    content = m.get("content")
    if isinstance(content, str):
        return [Turn(role=role, text=content)]
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    calls: list[tuple[str, str]] = []
    tool_results: list[Turn] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        match b.get("type"):
            case "text":
                text_parts.append(str(b.get("text", "")))
            case "thinking":
                thinking_parts.append(str(b.get("thinking", "")))
            case "tool_use":
                nm = str(b.get("name", ""))
                calls.append((nm, _pretty_args(b.get("input", {}))))
                if b.get("id"):
                    names[str(b["id"])] = nm
            case "tool_result":
                raw = b.get("content")
                txt = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                nm = names.get(str(b.get("tool_use_id", "")), "")
                tool_results.append(Turn(role="tool", text=txt, tool_name=nm))
            case _:
                pass
    if role == "assistant":
        return [
            Turn(
                role="assistant",
                text="\n".join(text_parts).strip(),
                thinking="\n".join(thinking_parts).strip(),
                tool_calls=calls,
            )
        ]
    # A user message is either prose or a batch of tool_result blocks.
    if tool_results and not "".join(text_parts).strip():
        return tool_results
    return [Turn(role=role, text="\n".join(text_parts).strip())]


def _message_turns(m: dict[str, Any], shape: str, names: dict[str, str]) -> list[Turn]:
    return _openai_turns(m, names) if shape == "openai" else _anthropic_turns(m, names)


def _response_turns(resp: dict[str, Any], shape: str, names: dict[str, str]) -> list[Turn]:
    if shape == "openai":
        choices = resp.get("choices") or []
        if not choices:
            return []
        return _openai_turns(_as_dict(choices[0].get("message")), names)
    if resp.get("content") is not None:
        return _anthropic_turns({"role": "assistant", "content": resp.get("content")}, names)
    return []


def fold_conversation(transcripts: list[dict[str, Any]]) -> list[Turn]:
    """Fold per-call transcripts into one ordered conversation (no double-print)."""
    turns: list[Turn] = []
    names: dict[str, str] = {}  # tool_call/use id -> tool name (to label results)
    prev_len = 0
    for t in transcripts:
        seq = int(t.get("seq", 0))
        req = _request_body(t)
        resp = _response_body(t)
        shape = _shape(req, resp)
        msgs = req.get("messages") or []
        # Anthropic keeps the system prompt out of `messages`; surface it once.
        if shape == "anthropic" and prev_len == 0 and req.get("system"):
            sys = req["system"]
            turns.append(
                Turn(role="system", seq=seq, text=sys if isinstance(sys, str) else json.dumps(sys))
            )
        if len(msgs) < prev_len:  # a context-compaction restart shrank the history
            turns.append(Turn(role="marker", text="context summarised / restarted", seq=seq))
            prev_len = 0
        for m in msgs[prev_len:]:
            if isinstance(m, dict):
                for tt in _message_turns(m, shape, names):
                    tt.seq = seq
                    turns.append(tt)
        for rt in _response_turns(resp, shape, names):  # this call's assistant output
            rt.seq = seq
            turns.append(rt)
        prev_len = len(msgs) + 1  # the response becomes msgs[len] in the next request
    return turns


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} chars)"


def render_markdown(
    turns: list[Turn],
    *,
    run_id: str,
    show_thinking: bool = True,
    tools: str = "both",
    result_cap: int = 4000,
) -> str:
    """Render folded turns as a Markdown conversation. `tools` in both|calls|none."""
    out: list[str] = [f"# Transcript: {run_id}", ""]
    for tn in turns:
        if tn.role == "marker":
            out.append(f"\n--- {tn.text} ---\n")
            continue
        if tn.role == "tool":
            if tools != "both":
                continue
            label = f" {tn.tool_name}" if tn.tool_name else ""
            out.append(f"  <-{label}: {_clip(tn.text, result_cap)}")
            out.append("")
            continue
        header = f"## {tn.role}"
        if tn.role == "assistant":
            header += f"  (seq {tn.seq})"
        out.append(header)
        if tn.thinking and show_thinking:
            out.append(f"<thinking>\n{tn.thinking}\n</thinking>")
        if tn.text:
            out.append(tn.text)
        if tn.tool_calls and tools != "none":
            out.extend(f"-> {name}({args})" for name, args in tn.tool_calls)
        out.append("")
    return "\n".join(out).rstrip() + "\n"
