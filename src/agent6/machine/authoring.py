# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The prompt `agent6 machine create` hands its drafting loop.

The per-attempt prompt is built around the grammar reference in `agent6.prompts.machine`.
The orchestration lives in `app/machine/create.py`; this module imports nothing from the
harness, which keeps the tach graph acyclic.
"""

from __future__ import annotations

from agent6.prompts.machine import MACHINE_AUTHOR_GUIDE  # noqa: ICN003  # re-export

__all__ = ["MACHINE_AUTHOR_GUIDE", "build_authoring_prompt"]


def build_authoring_prompt(
    task: str,
    *,
    attempt: int,
    diagnostics: list[str] | None = None,
) -> str:
    """Assemble the task prompt for one draft, check and fix attempt.

    A retry appends the diagnostics; the draft itself is in the workspace, where the agent
    patches the files it wrote.

    Args:
        task: The operator's task.
        attempt: The attempt number, named in the retry heading.
        diagnostics: The validation problems of the previous attempt, or None.

    Returns:
        The prompt text.
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
    parts += [
        "## Where to write it",
        "",
        "This workspace is yours and starts empty: write the machine file and its"
        ' scripts here with `apply_edit`, `kind="create"` for each new file,'
        " and read them back when you need to."
        " Write exactly ONE `<machine-name>.asm.toml` at the workspace root, plus"
        " every `scripts/...` file its `tool` states reference (and, for any script"
        " with a seam, its `scripts/<name>_test.py` companion). Call `finish_session`"
        " when the bundle is complete; agent6 validates what is on disk and hands"
        " you back any problems.",
        "",
        "Make each script PRODUCTION-READY for the real task: it reads live inputs"
        " from their real source (real HTTP via stdlib `urllib`), reads any"
        " secrets from the environment (never hard-coded), sets"
        ' `network = "host"` on its state if it touches the network, prints'
        " ONE JSON object on stdout matching its `output_schema`, and exits 0 on"
        " success. Type-annotate it and keep it lint-clean: `machine create` runs"
        " ruff + ty and rejects it otherwise. For every script with an external"
        " seam (network/clock/files), ALSO write a `scripts/<name>_test.py` that"
        " mocks the seam and asserts the contract; these run offline in a"
        " no-network jail so the operator can simulate the machine without live"
        " services.",
    ]
    if diagnostics:
        joined = "\n".join(f"  - {problem}" for problem in diagnostics)
        parts.extend(
            [
                "",
                f"## Attempt {attempt}: fix the draft in this workspace",
                "",
                "Your draft did not pass validation. The diagnostics were:",
                "",
                joined,
                "",
                "Read the files you wrote, change ONLY what the diagnostics name,"
                " and finish again.",
            ]
        )
    return "\n".join(parts)
