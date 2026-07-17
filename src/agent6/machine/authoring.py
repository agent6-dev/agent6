# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt scaffolding for `agent6 machine create` (Phase 5, §7.1).

`machine create` is an ordinary jailed agent6 loop whose job is to *draft*
a `.asm.toml` state machine from a natural-language task. This module holds
the prompt-assembly pieces of that flow: the per-attempt draft→check→fix
prompt (built around the grammar reference in `agent6.prompts.machine`) and
the extractor that pulls the drafted source out of the `finish_run` payload.

It deliberately imports nothing from the workflow stack, the orchestration
(running the agent loop, validating with `load_machine`, writing the draft)
lives in the CLI, which already depends on both `agent6.machine` and
`agent6.workflows`. Keeping this module pure keeps the tach graph acyclic.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from agent6.prompts.machine import MACHINE_AUTHOR_GUIDE

__all__ = [
    "MACHINE_AUTHOR_GUIDE",
    "SCRIPTS_PAYLOAD_KEY",
    "TOML_PAYLOAD_KEY",
    "build_authoring_prompt",
    "extract_scripts",
    "extract_toml",
]

# The keys the authoring agent uses to return its draft: the `.asm.toml` source
# and the helper scripts its `tool` states reference (a map of bundle-relative
# path -> file content). Both are written by `machine create`.
TOML_PAYLOAD_KEY = "toml"
SCRIPTS_PAYLOAD_KEY = "scripts"


def build_authoring_prompt(
    task: str,
    *,
    attempt: int,
    prior_toml: str | None = None,
    diagnostics: list[str] | None = None,
    prior_scripts: dict[str, str] | None = None,
    worker_unpriced: bool = False,
) -> str:
    """Assemble the user-task prompt for one draft→check→fix attempt.

    On the first attempt only the grammar guide and the operator's task are
    included. On a retry, the prior draft, its scripts, and the validation
    diagnostics are appended so the model can PATCH its own output instead of
    re-deriving everything (most retries are a one-line script fix; without the
    prior script source the model regenerates every file blind).

    When ``worker_unpriced`` is set, the operator's configured worker model (the
    one the machine's agent states inherit) has no price data, so the draft must
    use ``best_effort_usd_limit``; a ``max_usd`` cap would make `machine run`
    refuse to start, leaving the freshly-created machine unrunnable.
    """
    parts = [
        MACHINE_AUTHOR_GUIDE,
        "",
        "## Your task",
        "",
        "Author ONE complete, valid `.asm.toml` machine for this request:",
        "",
        task.strip(),
        "",
    ]
    if worker_unpriced:
        parts += [
            "## Budget",
            "",
            "The configured worker model (which this machine's agent states will"
            " inherit) has NO price data. Use `best_effort_usd_limit` in `[budget]`"
            " and on any per-state cap, NOT `max_usd` -- a hard `max_usd` would make"
            " `machine run` refuse to start. `max_transitions` remains the binding"
            " runaway guard.",
            "",
        ]
    parts += [
        "## How to return it",
        "",
        "Do NOT write any files. When the machine is complete, call `finish_run`"
        " with a `result` object containing BOTH:",
        f"  - `{TOML_PAYLOAD_KEY}`: the entire `.asm.toml` source as a single string.",
        f"  - `{SCRIPTS_PAYLOAD_KEY}`: an object mapping EACH `scripts/...` path your"
        " `tool` states reference (AND, for any script that has a seam, its"
        " `scripts/<name>_test.py` companion) to that file's COMPLETE source."
        " Every `scripts/...` command in the TOML must have an entry here, or the"
        " machine is rejected as incomplete. Omit this key only if no state runs"
        " a `scripts/...` command.",
        "",
        "Make each script PRODUCTION-READY for the real task: it reads live inputs"
        " from their real source (real HTTP via stdlib `urllib`), reads any"
        " secrets from the environment (never hard-coded), sets"
        ' `allow_network = "allow"` on its state if it touches the network, prints'
        " ONE JSON object on stdout matching its `output_schema`, and exits 0 on"
        " success. Type-annotate it and keep it lint-clean — `machine create` runs"
        " ruff + ty and rejects it otherwise. For every script with an external"
        " seam (network/clock/files), ALSO emit a `scripts/<name>_test.py` that"
        " mocks the seam and asserts the contract; these run offline in a"
        " no-network jail so the operator can simulate the machine without live"
        " services. Put a one-line rationale per state in `summary`.",
    ]
    if prior_toml is not None and diagnostics:
        joined = "\n".join(f"  - {problem}" for problem in diagnostics)
        parts.extend(
            [
                "",
                f"## Attempt {attempt}: fix the previous draft",
                "",
                "Your previous draft did not pass validation. The diagnostics were:",
                "",
                joined,
                "",
                "Here is the draft to repair:",
                "",
                "```toml",
                prior_toml.strip(),
                "```",
            ]
        )
        for rel, content in sorted((prior_scripts or {}).items()):
            fence = "```python" if rel.endswith(".py") else "```"
            parts.extend(["", f"`{rel}`:", fence, content.strip(), "```"])
        parts.extend(
            [
                "",
                "Change ONLY what the diagnostics name; keep everything else"
                " byte-identical. Return the COMPLETE corrected machine again"
                " (full toml + every script file).",
            ]
        )
    return "\n".join(parts)


def extract_toml(payload: dict[str, Any] | None) -> str | None:
    """Pull the drafted `.asm.toml` source out of a `finish_run` payload.

    Returns the source string, or ``None`` if the agent did not return a
    non-empty ``toml`` string (the caller turns that into a diagnostic and
    retries).
    """
    if not payload:
        return None
    value = payload.get(TOML_PAYLOAD_KEY)
    if isinstance(value, str) and value.strip():
        return value
    return None


def extract_scripts(payload: dict[str, Any] | None) -> dict[str, str]:
    """Pull the helper-script bundle out of a `finish_run` payload.

    Returns a {bundle-relative-path: content} map, keeping only safe entries:
    a path under `scripts/`, relative (not absolute), with no `..` segment, and
    string content. Anything else is dropped (the missing-script validation then
    catches a command that referenced it). Never raises."""
    if not payload:
        return {}
    raw = payload.get(SCRIPTS_PAYLOAD_KEY)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, content in raw.items():
        if not isinstance(key, str) or not isinstance(content, str):
            continue
        rel = key.strip()
        if rel.startswith("./"):
            rel = rel[2:]
        # Keep only paths under scripts/, no `..`, not absolute (PurePosixPath of
        # an absolute path has "/" as parts[0], which != "scripts").
        parts = PurePosixPath(rel).parts
        if not parts or parts[0] != "scripts" or ".." in parts:
            continue
        out[rel] = content
    return out
