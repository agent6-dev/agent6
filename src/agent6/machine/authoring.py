# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt scaffolding for `agent6 machine create` (Phase 5, §7.1).

`machine create` is an ordinary jailed agent6 loop whose job is to *draft*
a `.asm.toml` state machine from a natural-language task. This module holds
the pure, dependency-free pieces of that flow: the grammar reference handed
to the model, the prompt assembled for each draft→check→fix attempt, and the
extractor that pulls the drafted source out of the `finish_run` payload.

It deliberately imports nothing from the workflow stack — the orchestration
(running the agent loop, validating with `load_machine`, writing the draft)
lives in the CLI, which already depends on both `agent6.machine` and
`agent6.workflows`. Keeping this module pure keeps the tach graph acyclic.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MACHINE_AUTHOR_GUIDE",
    "TOML_PAYLOAD_KEY",
    "build_authoring_prompt",
    "extract_toml",
]

# The single key the authoring agent must use to return its draft.
TOML_PAYLOAD_KEY = "toml"

MACHINE_AUTHOR_GUIDE = """\
# agent6 state-machine (.asm.toml) authoring guide

A machine is a small, deterministic program whose building blocks are
sandboxed tool calls, agent6 agent runs, timed waits, and branches. It is
plain TOML. Author one complete machine per task.

## File skeleton

    machine = "kebab-or-snake-name"   # ^[a-z][a-z0-9_-]*$
    version = 1                       # always 1
    initial = "<state name>"          # the entry state

    [budget]
    max_usd = 1.0                     # > 0
    max_transitions = 100             # > 0; hard cap on state hops

## The blackboard: three owner tables (write-authorization, one read namespace)

Variables live under exactly one owner table. The owner controls who may
WRITE; reads are a single flat namespace (refer to a variable by its BARE
name everywhere, never `vars.code.x`).

    [vars.operator]                   # constants; READ-ONLY at runtime
    threshold = { type = "int", value = 3 }

    [vars.code]                       # written only by `tool` states
    items = { type = "list[str]", default = [] }

    [vars.agent]                      # written only by `agent` states
    verdict = { type = "verdict", default = {} }

Rules:
  - Names are globally unique across the three tables. Identifiers match
    `^[a-z][a-z0-9_]*$`.
  - Reserved names (cannot be used): vars, operator, code, agent, result.
  - operator vars use `value = ...`; code/agent vars use `default = ...`.
  - A record-typed var's default is the empty table `{}`.

## Types and schemas

Field types: `str`, `int`, `float`, `bool`, `list[<scalar>]`, `json`
(opaque, not navigable), or the name of a `[schemas.*]` record (recursive).

    [schemas.verdict]
    approved = "bool"
    note = "str"
    # optional field:           reason = { type = "str", optional = true }
    # string enum:              level  = { type = "str", enum = ["low", "high"] }

To navigate `result.field` or `somevar.field` you MUST give it a record
type via a schema. Opaque `json` cannot be dotted.

## States

Each state is `[states.<name>]` with a `kind`. Names match the identifier
grammar. Terminal states end the machine.

### tool — run a sandboxed command
    [states.scan]
    kind = "tool"
    command = ["scan", "{{ threshold }}"]   # argv; see templating below
    output_schema = "scan_result"            # optional: types `result` so result.x works
    capture = { set = { items = "{{ result.items }}" } }   # writes [vars.code] only
    timeout_secs = 5
    on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

  tool labels are exactly: ok, nonzero, timeout.

### agent — run a nested agent6 loop
    [states.review]
    kind = "agent"
    # model defaults to "inherit" (the operator's effective worker model).
    # OMIT it unless you must pin a specific one — a hardcoded model the
    # operator hasn't configured passes `machine check` but dies at run time.
    prompt = "Review the change and return a verdict."
    output_schema = "verdict"                # finish_run payload validated against this
    capture = { finish_json = "verdict" }    # whole payload -> a [vars.agent] var
    # or: capture = { set = { total = "{{ result.points }}" } }  # one field
    timeout_secs = 600
    on = { ok = "route", failed = "stop_fail", budget_exhausted = "halt", timeout = "expired" }

  agent labels are exactly: ok, failed, budget_exhausted, timeout.
  An agent state may write ONLY [vars.agent] vars.

### branch — route on predicates (MUST be total)
    [states.check]
    kind = "branch"
    when = [
      { if = "len(items) == 0", goto = "stop_ok" },
      { else = true, goto = "record" },        # final `else` is REQUIRED
    ]

  Predicate allow-list: comparisons (== != < <= > >=), `and`/`or`/`not`,
  `in`, `len(x)`, record navigation `x.field`, and literals. NO arbitrary
  function calls, attribute method calls, or comprehensions.

### wait — pause until an instant or a poke
    [states.poll]
    kind = "wait"
    every_secs = "{{ interval }}"   # OR  until = "2026-01-01T00:00:00Z"
    on = { tick = "scan", signal = "scan" }

  wait labels are exactly: tick, signal. `cron` is NOT supported by the v1
  runtime — use `every_secs` or `until`.

### terminal — stop
    [states.stop_ok]
    kind = "terminal"
    status = "ok"        # "ok" or "failed"
    reason = "done"

## Templating

`{{ ref }}` interpolates a variable; `{{ ref | len }}` / `{{ ref | json }}`
are the only two filters (both zero-arg). In an argv list, an element that
is EXACTLY `"{{ listvar }}"` splices the list into N arguments. In a
`capture.set`, a lone filter-less `{{ ref }}` captures the native VALUE
(its type must match the target var); any other template renders to a
string (target must be `str`).

## Capture ownership wall
  - `tool`  states may write only `[vars.code]`.
  - `agent` states may write only `[vars.agent]`.
  - `[vars.operator]` is read-only; `branch`/`wait`/`terminal` never write.

## Accumulating records across iterations (e.g. a paper-trade log)

The machine JOURNAL (`.agent6/machines/<id>/journal.jsonl`) already records
every transition with its fact — each `tool` stdout and each `agent` payload —
so it IS your durable, replayable audit log of everything that happened. You do
not need to write your own file to "remember" past iterations.

For values you branch on or template later, capture them into the blackboard:
keep counters / latest values (the blackboard has NO `list[record]` type —
`list[<scalar>]` elements must be scalars; use a `json` var for an opaque blob).
A `tool` or `agent` state should EMIT its record as stdout/`finish_run` JSON and
let `capture` store it — do NOT have a tool script write its own data file:
a tool's jail is READ-ONLY on the `hardened` profile (file writes only work
under `strict`), so a script that appends to a file will fail there. The
journal + blackboard capture is the portable, auditable pattern.

## Worked example — poll a value, record when it crosses a threshold

A complete, valid machine. The `tool` states call small auditable scripts you
ship alongside the `.asm.toml` (here under `scripts/`); each prints a JSON
object on stdout that the schema types and `capture` reads. The scripts only
READ + print (no file writes) so they work on every sandbox profile; the buy
record itself lives in the journal + blackboard.

    machine = "price-watch"
    version = 1
    initial = "wait_tick"

    [budget]
    max_usd = 1.0
    max_transitions = 1000

    [vars.operator]
    interval_secs = { type = "int", value = 15 }
    threshold = { type = "float", value = 100.0 }

    [vars.code]
    price = { type = "float", default = 0.0 }
    recorded = { type = "int", default = 0 }

    [schemas.price_result]
    price = "float"

    [schemas.record_result]
    recorded = "int"

    [states.wait_tick]
    kind = "wait"
    every_secs = "{{ interval_secs }}"
    on = { tick = "fetch_price", signal = "fetch_price" }

    [states.fetch_price]
    kind = "tool"
    command = ["python3", "scripts/fetch_price.py"]
    output_schema = "price_result"          # types `result` so result.price works
    capture = { set = { price = "{{ result.price }}" } }
    timeout_secs = 10
    on = { ok = "decide", nonzero = "wait_tick", timeout = "wait_tick" }

    [states.decide]
    kind = "branch"
    when = [
      { if = "price >= threshold", goto = "record_buy" },
      { else = true, goto = "wait_tick" },
    ]

    [states.record_buy]
    kind = "tool"
    # record_buy.py just prints {"recorded": N} on stdout (the journal is the
    # durable log; no file write, so it works on hardened too).
    command = ["python3", "scripts/record_buy.py", "{{ price }}"]
    output_schema = "record_result"
    capture = { set = { recorded = "{{ result.recorded }}" } }
    timeout_secs = 10
    on = { ok = "wait_tick", nonzero = "wait_tick", timeout = "wait_tick" }

## Common mistakes (each fails `machine check` or silently misbehaves)
  - Hardcoding `model = "..."` on an `agent` state — omit it (defaults to
    "inherit" = the worker model) unless pinning one on purpose.
  - `list[record]` / `list[{...}]` types — unsupported. Append records to a
    file from a tool script and keep a counter in the blackboard (see above).
  - A `tool` script that prints to stderr or exits non-zero — capture fires
    ONLY on exit 0 with non-empty stdout JSON; empty stdout silently leaves the
    var at its default. Always print your JSON to STDOUT and exit 0 on success.
  - A `tool` script that writes a file — fails on the `hardened` profile (tool
    jails are read-only there). Print results to stdout and let the journal +
    blackboard persist them instead.
  - A non-total `branch` — the last clause MUST be `{ else = true, goto = ... }`.

## Validity requirements (the file must pass `machine check`)
  - Every `on`/`goto`/`initial` target names an existing state.
  - Every state is reachable from `initial`.
  - Every `branch` is total (ends with `{ else = true, goto = ... }`).
  - Every reference resolves to a declared variable of a compatible type.
  - Every `capture` writes a var owned by the writing state kind.
"""


def build_authoring_prompt(
    task: str,
    *,
    attempt: int,
    prior_toml: str | None = None,
    diagnostics: list[str] | None = None,
) -> str:
    """Assemble the user-task prompt for one draft→check→fix attempt.

    On the first attempt only the grammar guide and the operator's task are
    included. On a retry, the prior draft and the `machine check` diagnostics
    are appended so the model can repair its own output.
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
        "## How to return it",
        "",
        "Do NOT write any files. When the machine is complete, call `finish_run`"
        f" with a `result` object whose `{TOML_PAYLOAD_KEY}` field is the entire"
        " `.asm.toml` source as a single string. Put a one-line rationale per"
        " state in your `summary`.",
    ]
    if prior_toml is not None and diagnostics:
        joined = "\n".join(f"  - {problem}" for problem in diagnostics)
        parts.extend(
            [
                "",
                f"## Attempt {attempt}: fix the previous draft",
                "",
                "Your previous draft did not pass `machine check`. The diagnostics were:",
                "",
                joined,
                "",
                "Here is the draft that failed, for you to repair:",
                "",
                "```toml",
                prior_toml.strip(),
                "```",
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
