# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt scaffolding for `agent6 machine create` (Phase 5, §7.1).

`machine create` is an ordinary jailed agent6 loop whose job is to *draft*
a `.asm.toml` state machine from a natural-language task. This module holds
the pure, dependency-free pieces of that flow: the grammar reference handed
to the model, the prompt assembled for each draft→check→fix attempt, and the
extractor that pulls the drafted source out of the `finish_run` payload.

It deliberately imports nothing from the workflow stack, the orchestration
(running the agent loop, validating with `load_machine`, writing the draft)
lives in the CLI, which already depends on both `agent6.machine` and
`agent6.workflows`. Keeping this module pure keeps the tach graph acyclic.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

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
    max_usd = 1.0                     # optional hard USD cap (see below)
    max_transitions = 100             # > 0; hard cap on state hops

  The USD cap is optional, at most one of `max_usd` /
  `best_effort_usd_limit` (both > 0). `max_usd` is hard: `machine run`
  refuses to start when an agent state's model has no price data.
  `best_effort_usd_limit` binds only when spend is measurable; use it for
  unpriced or local models. Prefer `max_usd`.

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

### tool: run a sandboxed command
    [states.scan]
    kind = "tool"
    command = ["scan", "{{ threshold }}"]   # argv; see templating below
    output_schema = "scan_result"            # optional: types `result` so result.x works
    capture = { set = { items = "{{ result.items }}" } }   # writes [vars.code] only
    timeout_secs = 5
    on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

  tool labels are exactly: ok, nonzero, timeout.

### agent: run a nested agent6 loop
    [states.review]
    kind = "agent"
    # model defaults to "inherit" (the operator's effective worker model).
    # OMIT it unless you must pin a specific one, a hardcoded model the
    # operator hasn't configured passes `machine check` but dies at run time.
    prompt = "Review the change and return a verdict."
    output_schema = "verdict"                # finish_run payload validated against this
    capture = { finish_json = "verdict" }    # whole payload -> a [vars.agent] var
    # or: capture = { set = { total = "{{ result.points }}" } }  # one field
    timeout_secs = 600
    on = { ok = "route", failed = "stop_fail", budget_exhausted = "halt", timeout = "expired" }

  agent labels are exactly: ok, failed, budget_exhausted, timeout.
  An agent state may write ONLY [vars.agent] vars.
  By default an agent state is a READ-ONLY structured-output judge (classify /
  score / decide -> a finish_run result; it cannot edit the repo). For a state
  that must do real coding work, add `mode = "run"` to give it the full edit /
  verify / commit tool set.

### branch: route on predicates (MUST be total)
    [states.check]
    kind = "branch"
    when = [
      { if = "len(items) == 0", goto = "stop_ok" },
      { else = true, goto = "record" },        # final `else` is REQUIRED
    ]

  Predicate allow-list: comparisons (== != < <= > >=), `and`/`or`/`not`,
  `in`, `len(x)`, record navigation `x.field`, and literals. NO arbitrary
  function calls, attribute method calls, or comprehensions.

### wait: pause until an instant or a poke
    [states.poll]
    kind = "wait"
    every_secs = "{{ interval }}"   # OR  until = "2026-01-01T00:00:00Z"
    on = { tick = "scan", signal = "scan" }

  wait labels are exactly: tick, signal. `cron` is NOT supported by the v1
  runtime — use `every_secs` or `until`.

### terminal: stop
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

Two durable stores, both available on every sandbox profile:

  - The JOURNAL (`.agent6/machines/<id>/journal.jsonl`) already records every
    transition with its fact (each `tool` stdout, each `agent` payload), so it
    IS a replayable audit log of everything that happened — for free.
  - `$AGENT6_MACHINE_DATA_DIR` is a persistent, WRITABLE directory every `tool`
    script may write to (the one writable spot under `hardened`, where new
    top-level files in the workspace are read-only). It's a path relative to the
    workspace, so just open it from the script's cwd — works on every profile:
    `open(os.environ["AGENT6_MACHINE_DATA_DIR"] + "/trades.jsonl", "a")`.

For values you branch on or template later, capture them into the blackboard:
keep counters / latest values (the blackboard has NO `list[record]` type —
`list[<scalar>]` elements must be scalars; use a `json` var for an opaque blob).
Don't try to grow an unbounded list of records IN the blackboard — write it to
the data dir (or rely on the journal) and keep just a count/latest in a var.

## Real inputs, secrets & the network

Author scripts that do the REAL task in production — not ones that read canned
data. So:
  - Read live inputs from their real source. For HTTP the standard library is
    enough: `urllib.request` makes real API calls — you do NOT need `requests`.
  - Pass NON-secret config (an endpoint, a user id) as an operator var spliced
    into `command`, e.g. `["python3", "scripts/fetch.py", "{{ feed_url }}"]`.
  - Read SECRETS (API tokens/keys) from the ENVIRONMENT — `os.environ["X_TOKEN"]`
    — never hard-coded in a script and never written into the `.asm.toml`. The
    operator exports them when they run the machine.
  - A tool that touches the network MUST set `allow_network = "allow"` on its
    state; without it the tool is network-isolated and the call fails. (The
    operator still has to permit egress via `sandbox.tool_network`; if their
    config blocks it, `agent6 machine run` explains the one-line fix and offers
    to apply it.)
  - Persist outputs/state to `$AGENT6_MACHINE_DATA_DIR` (the workspace is
    read-only under `hardened`).

## Test every non-trivial script (offline simulation)

Anything a script does that ISN'T a pure function of its argv/stdin is a SEAM
you must be able to control in a test: a network call, the clock
(`time`/`datetime.now`), randomness, reading an external file, a subprocess.
For each script that has a seam, ALSO emit `scripts/<name>_test.py` that:
  1. imports the script,
  2. replaces each seam with a fake — mock the network function, inject a fixed
     time, point file I/O at a `tempfile` dir,
  3. asserts the script's contract (the exact JSON it prints / the file it writes),
  4. exits 0 on success, non-zero on failure, using NO network.

Structure each script so its core logic is a small function the test calls
directly and `main()` only does argv/env/stdout — that makes the seam easy to
patch. `machine create` and `machine test` LINT (ruff), TYPE-CHECK (ty), and run
these tests in a no-network jail: a script that isn't typed, isn't lint-clean,
or whose test needs the real network fails the gate. Default to `python3` + the
standard library; you may name another available interpreter in `command`
(`["bash", "scripts/x.sh"]`), but Python is validated most thoroughly — reach
for a third-party package only when stdlib genuinely can't do it, and say in the
state's comment that the operator must install it in the jail's environment.

## Fault tolerance (machines run for days; transient failures are NORMAL)

A long-running machine WILL hit a down API, a 429, a timeout. Those surface as
labels, and your wiring decides whether one bad tick kills the machine:
  - tool failures: `nonzero` (script printed an error and exited non-zero) and
    `timeout`. Route both BACK to the wait state so the machine simply retries
    next tick: `on = { ok = ..., nonzero = "wait_tick", timeout = "wait_tick" }`.
  - agent failures: `failed` (provider error, malformed output) and `timeout`
    likewise route back to the wait state. Reserve a terminal state for
    `budget_exhausted` only — that one will not heal by retrying.
  - Never route a failure label to an `status = "ok"` terminal, and never wire
    a failure straight back to the SAME fetch state (that is a hot retry loop
    with no delay; going through the wait state is the backoff).
  - Scripts should exit non-zero on failure (stderr message) rather than
    printing fabricated JSON, so a bad tick is a visible `nonzero`, not silent
    garbage captured into the blackboard.

## Worked example: watch a live price feed, record a threshold crossing

A complete, valid, PRODUCTION machine: `fetch_price` makes a real HTTP call
(hence `allow_network = "allow"`); `record_buy` appends to the data dir. Each
script ships a `*_test.py` that mocks its seam so the whole machine simulates
offline.

    machine = "price-watch"
    version = 1
    initial = "wait_tick"

    [budget]
    max_usd = 1.0
    max_transitions = 1000

    [vars.operator]
    interval_secs = { type = "int", value = 60 }
    feed_url = { type = "str", value = "https://api.example.com/v1/price/BTC" }
    threshold = { type = "float", value = 100000.0 }

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
    command = ["python3", "scripts/fetch_price.py", "{{ feed_url }}"]
    allow_network = "allow"                 # this tool reaches the real API
    output_schema = "price_result"          # types `result` so result.price works
    capture = { set = { price = "{{ result.price }}" } }
    timeout_secs = 15
    on = { ok = "decide", nonzero = "wait_tick", timeout = "wait_tick" }

    [states.decide]
    kind = "branch"
    when = [
      { if = "price >= threshold", goto = "record_buy" },
      { else = true, goto = "wait_tick" },
    ]

    [states.record_buy]
    kind = "tool"
    command = ["python3", "scripts/record_buy.py", "{{ price }}"]
    output_schema = "record_result"
    capture = { set = { recorded = "{{ result.recorded }}" } }
    timeout_secs = 10
    on = { ok = "wait_tick", nonzero = "wait_tick", timeout = "wait_tick" }

### …and the scripts it references (return ALL of these in `result.scripts`)

Real, typed scripts plus an offline test per seam (the network seam in
`fetch_price`, the filesystem seam in `record_buy`).

`scripts/fetch_price.py`:
```python
# Fetch the current price from an HTTP JSON feed. Prints {"price": <float>}.
# The feed URL is argv[1] (an operator var); an optional bearer token comes from
# the PRICE_FEED_TOKEN env var -- secrets belong in the environment, never in the
# machine file. The tool state must set allow_network = "allow".
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def fetch_price(url: str, token: str) -> float:
    # GET *url* and return the "price" field as a float (the network seam).
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read())
    return float(payload["price"])


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    if not url:
        print("usage: fetch_price.py <feed-url>", file=sys.stderr)
        return 1
    try:
        price = fetch_price(url, os.environ.get("PRICE_FEED_TOKEN", ""))
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"price": price}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/fetch_price_test.py` — mocks the network seam so it runs offline:
```python
# Offline test for fetch_price.py: mocks the network seam (urlopen) so the
# machine validates without a live feed. Run: python3 scripts/fetch_price_test.py
from __future__ import annotations

import json
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_price  # noqa: E402


class _FakeResp:
    # Stand-in for the urlopen() context manager (the seam we control).
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_parses_price() -> None:
    body = json.dumps({"price": 123.5}).encode()
    with mock.patch.object(fetch_price.urllib.request, "urlopen", return_value=_FakeResp(body)):
        assert fetch_price.fetch_price("https://feed", "") == 123.5


if __name__ == "__main__":
    test_parses_price()
    print("ok")
```

`scripts/record_buy.py`:
```python
# Append a paper trade to $AGENT6_MACHINE_DATA_DIR/trades.jsonl; print the count.
from __future__ import annotations

import json
import os
import pathlib
import sys
import time


def record(data_dir: pathlib.Path, price: float) -> int:
    # Append one trade and return the running count (the filesystem seam).
    data_dir.mkdir(parents=True, exist_ok=True)
    log = data_dir / "trades.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({"price": price, "ts": int(time.time())}) + "\\n")
    return sum(1 for _ in log.open())


def main() -> int:
    price = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    data_dir = pathlib.Path(os.environ.get("AGENT6_MACHINE_DATA_DIR", "."))
    print(json.dumps({"recorded": record(data_dir, price)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/record_buy_test.py` — a tempfile dir replaces the data-dir seam:
```python
# Offline test for record_buy.py: a temp dir replaces the data-dir seam.
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import record_buy  # noqa: E402


def test_appends_and_counts() -> None:
    with tempfile.TemporaryDirectory() as d:
        data = pathlib.Path(d)
        assert record_buy.record(data, 10.0) == 1
        assert record_buy.record(data, 20.0) == 2
        rows = (data / "trades.jsonl").read_text().splitlines()
        assert json.loads(rows[0])["price"] == 10.0


if __name__ == "__main__":
    test_appends_and_counts()
    print("ok")
```

## Common mistakes (each fails `machine check`/`create` or silently misbehaves)
  - Hardcoding `model = "..."` on an `agent` state — omit it (defaults to
    "inherit" = the worker model) unless pinning one on purpose.
  - A `tool` that calls the network but forgets `allow_network = "allow"` — it
    runs network-isolated and the call fails.
  - Hardcoding a secret/token in a script or the `.asm.toml` — read it from the
    environment (`os.environ[...]`) instead.
  - A bare `{{ listvar }}` inside an `agent` prompt or any other string —
    a list only interpolates as `{{ listvar | json }}` (or spliced as a
    standalone argv element). Bare list references fail `machine check`.
  - A network script with no `*_test.py`, or a test that hits the REAL network —
    tests run with NO network and must mock the seam, or they fail the gate.
  - A script that isn't typed or isn't lint-clean — `machine create` runs ruff +
    ty and rejects it. Annotate functions; keep imports used.
  - `list[record]` / `list[{...}]` types — unsupported. Append records to a
    file from a tool script and keep a counter in the blackboard (see above).
  - A `tool` script that prints to stderr or exits non-zero — capture fires
    ONLY on exit 0 with non-empty stdout JSON; empty stdout silently leaves the
    var at its default. Always print your JSON to STDOUT and exit 0 on success.
  - A `tool` script that writes outside `$AGENT6_MACHINE_DATA_DIR` — on
    `hardened` the rest of the workspace is read-only to tool jails (new
    top-level files are denied). Write to the data dir (or print to stdout).
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
    prior_scripts: dict[str, str] | None = None,
) -> str:
    """Assemble the user-task prompt for one draft→check→fix attempt.

    On the first attempt only the grammar guide and the operator's task are
    included. On a retry, the prior draft, its scripts, and the validation
    diagnostics are appended so the model can PATCH its own output instead of
    re-deriving everything (most retries are a one-line script fix; without the
    prior script source the model regenerates every file blind).
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
