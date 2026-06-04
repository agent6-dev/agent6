# agent6

A sandboxed coding agent for Linux. The LLM is treated as adversarial:
it runs inside a custom Rust launcher (`agent6-jail`) that combines
user namespaces, Landlock, seccomp, `pivot_root`, `capset(0)`, and
`NO_NEW_PRIVS`, so a misbehaving or compromised model cannot escape
the workspace, reach the network beyond the provider endpoint, or
corrupt the project's git history.

- **Sandbox-first.** Every child process the model can spawn (verify
  commands, metric commands, optional shell) goes through the jail.
- **Provider-agnostic.** Native HTTP clients for Anthropic and any
  OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, vLLM,
  llama.cpp, LM Studio).
- **Per-step git commits**, snapshot-resumable runs, USD/token
  budgets with hard stops, read-only code-review subcommand.
- **Small footprint.** Two runtime dependencies (`pydantic`, `httpx`).
  No telemetry, no auto-update, no plugin system.

## Status

Pre-1.0. Public shapes — config TOML, CLI flags,
on-disk run state, IPC frames — may change without backward-compatible
shims. See [AGENTS.md](AGENTS.md) for the stability policy.

## Benchmarks

Reproducible harnesses live under [bench/](bench/). All numbers below use
`claude-sonnet-4-5` as the worker, scored by independent post-hoc
verification (fresh verify + metric re-runs on the real-world side). A
perf optimization harness also lives under [bench/perf/](bench/perf/) for
local experimentation, but its single-run cycle counts are too noisy to
quote meaningfully here.

**Real-world suite** ([bench/realworld/](bench/realworld/)) — 11
SWE-bench-Lite-style tasks across real libraries (click, csv/RFC 4180,
werkzeug safe-join, URL RFC 3986, tinydb, …). Each task is scored by a
fresh sandboxed verify on hidden tests, $1/task cap. Run head-to-head
against `claude-code` on the same worker model (`claude-sonnet-4-5`):

- **Both solve all 11 tasks** (11/11 verify pass).
- Single end-to-end run of the suite: agent6 ≈ $2.60 total, claude-code
  ≈ $3.96 total.

These are **single runs (N = 1)** — not an average or median, and we have
not measured variance. The worker is stochastic and per-task cost swings
widely run-to-run, so treat the totals as rough directional guidance that
agent6 is cost-competitive at equal task outcomes, not as a precise
benchmark. Re-run both harnesses yourself under [bench/](bench/) before
quoting numbers.

## Requirements

- Linux. The sandbox relies on Linux-only kernel APIs (Landlock,
  seccomp-bpf, user/mount namespaces, `pivot_root`). macOS and Windows
  are not supported.
- Linux kernel ≥ 6.7 for full Landlock TCP-connect rules. Older kernels
  fall back to filesystem-only Landlock with a warning.
- `kernel.unprivileged_userns_clone = 1` (default on Ubuntu, Debian,
  and most cloud images). Required for the `strict` sandbox profile;
  without it the agent falls back to `hardened`.
- Python ≥ 3.12.
- An Anthropic and/or OpenAI-compatible API key.

If installing from source, a Rust toolchain (`cargo`, `rustc`) must be
on `PATH`. The hatch build hook invokes `cargo build` to compile
`agent6-jail`. PyPI wheels bundle a prebuilt `agent6-jail`.

## Install

From PyPI:

```bash
uv tool install agent6
```

From source:

```bash
git clone https://github.com/elesiuta/agent6
cd agent6
uv sync --extra tui
uv run agent6 --help
```

The `tui` extra pulls in `textual` for the live dashboard. To override
the bundled jail binary, set `AGENT6_JAIL_BIN=/path/to/agent6-jail`.

### Shell tab-completion

Tab-completion is provided via [argcomplete](https://kislyuk.github.io/argcomplete/):

```bash
# Bash / Zsh
eval "$(register-python-argcomplete agent6)"

# Fish
register-python-argcomplete --shell fish agent6 > ~/.config/fish/completions/agent6.fish
```

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Scaffold agent6.toml + AGENTS.md and add .agent6/ to .gitignore.
agent6 init

# Sanity checks.
agent6 check-config
agent6 check-sandbox

# Run the agent on a task.
agent6 run "add a --json output mode to the CLI"

# Resume an interrupted run from its last tool-call snapshot.
agent6 resume <run-id>

# Read-only code review of a diff. Never touches the worktree.
agent6 review --base origin/main --head HEAD
```

Other commands:

- `agent6 watch [<run-id>]` — attach the live TUI to an existing run
  (defaults to the most recent).
- `agent6 memory` — manage persistent agent memory under
  `.agent6/memory/`.
- `agent6 history search <query>` — ripgrep-backed search over
  persisted transcripts.
- `agent6 history graph [<run-id>]` — render the persisted task graph.
- `agent6 check-config --fix` — interactively fill in missing config
  fields after an upgrade.

## How it works

agent6 is a single-loop agent: one provider, one model, one message
history. The model drives the run by calling tools; the workflow
dispatches tools, snapshots state, and tracks budget.

```
                    +---------------------------+
                    |         agent6 CLI        |
                    +-------------+-------------+
                                  |
                +-----------------+-----------------+
                |                 |                 |
          +-----v-----+     +-----v-----+    +------v-----+
          | workflows |     |  agents/  |    |   graph/   |
          |  run      |     |code_review|    |   curator  |
          |  review   |     |           |    | (subproc)  |
          +-----+-----+     +-----+-----+    +------------+
                |                 |
          +-----v-----------------v-----+
          |        tools/dispatch       |
          | read_file list_dir grep ... |
          | apply_edit apply_patch ...  |
          | run_verify_command ...      |
          | finish_run dag_* ...        |
          +-------------+---------------+
                        |
                +-------v-------+
                | sandbox/jail  |
                +-------+-------+
                        | JSON policy
                +-------v-------+
                | agent6-jail   |  (Rust; userns + Landlock + seccomp)
                +---------------+
```

Module boundaries are enforced by [tach](https://docs.gauge.sh/):
`cli → workflows → agents → tools → sandbox`. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the state machines.

### Threat model

The worker LLM is treated as adversarial. It must not be able to:

- write outside the project's working directory;
- read files outside the project (plus any sibling read-only paths);
- reach the network except the host:port of each `[providers.*]` block
  (when `sandbox.network = "provider_only"`);
- corrupt the project's git history or its own configuration / run
  state from inside the sandbox;
- leave background processes running after the run ends.

Enforcement is layered:

- **Tool surface** ([src/agent6/tools/schema.py](src/agent6/tools/schema.py)).
  The LLM cannot directly invoke a shell or write arbitrary files. It
  has structured-edit, read-only navigation, fixed-argv verify/metric
  commands, a DAG side-store, and a terminal `finish_run`. When
  `sandbox.run_commands` is `"yes"` or `"ask"` it additionally gets
  `run_command(argv)`.
- **`agent6-jail`** wraps every child command (verify, metric,
  `run_command`, curator): fresh user/mount/pid/ipc/uts/net namespaces,
  pivots into a minimal rootfs, applies Landlock, a seccomp filter,
  drops capabilities, sets `NO_NEW_PRIVS`. In the strict profile, the
  network namespace is empty when `sandbox.network != "allow"` —
  `git push`, `curl`, `pip install`, and DNS all fail with no route,
  even from an ad-hoc script the worker writes and executes via
  `run_command`.
- **`sandbox.protect_git` + `sandbox.protect_agent6`** (default `true`)
  make `.git/`, `agent6.toml`, and `.agent6/` read-only inside the
  child's view. In `strict` they are re-bound RO on top of the
  workspace mount; in `hardened` the launcher switches its Landlock
  policy to read-only on the cwd with read-write carve-outs for each
  top-level entry except the protect set.
- **`git_ops.py`** ([src/agent6/git_ops.py](src/agent6/git_ops.py))
  constrains the workflow's own git calls: `push`, `--force`,
  `reset --hard`, `branch -D`, and history rewrites are refused
  unconditionally.
- **Landlock on the agent process itself** further restricts what the
  agent's Python code can read or write outside the jail.

If you set `sandbox.run_commands = "yes"` and `sandbox.network =
"allow"` the worker can talk to anywhere on the public internet from
inside the sandbox. The defaults exist for a reason.

See [SECURITY.md](SECURITY.md) for the per-layer breakdown.

## Configuration

Start from [agent6.example.toml](agent6.example.toml). Every field is
required at load time; `agent6 check-config` validates without running.

```toml
[sandbox]
profile = "auto"              # auto | strict | hardened
network = "provider_only"     # no | provider_only | allow
run_commands = "ask"          # yes | no | ask
protect_git = true
protect_agent6 = true

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"  # per_step | squash | stage | none
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
verify_command = ["uv", "run", "pytest", "-x"]

[budget]
max_input_tokens  = 2000000
max_output_tokens = 200000

[providers.anthropic]
kind = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-sonnet-4-5"
```

### Sandbox profiles

- **strict** — user/mount/pid/ipc/uts/net namespaces + `pivot_root`
  into a minimal rootfs + Landlock + seccomp + `capset(0)` + rlimits +
  `NO_NEW_PRIVS`. Requires unprivileged user namespaces.
- **hardened** — no namespaces, but still Landlock + seccomp +
  `capset(0)` + rlimits + `NO_NEW_PRIVS`. Works inside default-seccomp
  Docker.
- **auto** — `strict` if the kernel allows, else `hardened`. Logs the
  chosen profile on every run.

### Providers and models

Declare any number of providers as `[providers.<name>]` blocks. Each
sets `kind = "anthropic"` or `kind = "openai"` and has its own
`base_url` and `api_key_env`. Multiple providers (e.g. OpenAI plus
OpenRouter plus a local Ollama) coexist under whatever names you pick.
Per-provider `http_timeout_s` (default `600.0`) caps each HTTP call —
raise it for slow reasoning models, lower it to fail fast on a stuck
endpoint.

Two model roles are used:

| Role       | Routed by             | Used by                                                                          |
| ---------- | --------------------- | -------------------------------------------------------------------------------- |
| `worker`   | `[models.worker]`     | `agent6 run` and `agent6 resume`. Also drives the USD↔token budget conversion.   |
| `reviewer` | `[models.reviewer]`   | `agent6 review` (read-only diff review).                                         |

## Tool surface

The set of tools given to the LLM is fixed and audited in
[src/agent6/tools/schema.py](src/agent6/tools/schema.py). Adding a tool
requires a security review note in the commit message.

Read-only navigation:

- `read_file(path, start_line?, end_line?)`
- `list_dir(path)`
- `grep(pattern, path?, glob?)`
- `outline(path)` — top-level symbol outline via tree-sitter.
- `find_definition(symbol)` / `find_references(symbol)` — symbol index.

Edits:

- `apply_edit(path, edits)` — structured replace/create blocks.
- `apply_patch(diff)` — unified-diff application for multi-file changes.

Execution (operator-fixed argv only):

- `run_verify_command()` — runs `workflow.verify_command`.
- `run_metric_command()` — runs `workflow.metric.command` if configured
  and parses its metric value out of stdout.

Control:

- `finish_run(summary)` — terminal tool that ends the run.
- `dag_add_task` / `dag_update_task` / `dag_set_cursor` /
  `dag_list_tasks` — side-store notepad backed by the curator
  subprocess; the worker can plan and replan mid-flight.

Conditional:

- `run_command(argv)` — only exposed when `sandbox.run_commands ∈
  {"yes", "ask"}`. The only tool that can spawn an LLM-chosen
  subprocess. Runs inside the jail.

There is no `write_file`, no `shell`, no `web_fetch`.

## Cost accounting

Every run prints a per-model token and cost summary at the end:

```
Token + cost summary:
  claude-sonnet-4-5:  in=8884  out=1171 cache_r=0 cache_c=0 calls=4 $0.0442
  TOTAL: in=8884/2000000 out=1171/200000 cost~$0.0442
```

Pricing lives in [src/agent6/budget.py](src/agent6/budget.py) and is
updated by hand from the providers' public pricing pages. The budgets
in `agent6.toml` hard-stop the run; a stopped run is resumable.

## Live event log and TUI

Every run writes a structured JSONL event stream to
`.agent6/runs/<run-id>/logs.jsonl`. The vocabulary is small and stable:

| Event                       | Notable fields                              |
| --------------------------- | ------------------------------------------- |
| `run.start`                 | `user_task`                                 |
| `tool.call` / `.result`     | `name`, `args` (preview), `ok`, `summary`   |
| `verify.start` / `.end`     | `cmd`, `exit_code`, `duration_s`, `*_tail`  |
| `role.call` / `.result`     | `role`, `model`, `tokens_in`, `tokens_out`  |
| `budget.update`             | totals + caps for input/output tokens       |
| `approval.prompt`/`.answer` | `id`, `prompt`, `approved`, `source`        |
| `dag.*`                     | task add / update / cursor moves            |
| `run.end`                   | `summary`                                   |

This is the data contract for any external viewer. The fold from event
stream to UI state lives in [src/agent6/ui/state.py](src/agent6/ui/state.py)
as a pure function.

When installed with the `tui` extra and stdout is a TTY, `agent6 run`
spawns a separate process running `python -m agent6.ui --watch
<run-dir>` that renders the task DAG, budget bar, tool table, log tail,
and latest diff. The TUI is read-only on the log; the only thing it
writes is `<run-dir>/approvals/<id>.answer` when the user clicks Allow
or Deny on a `run_command` approval modal. Attach later with
`agent6 watch`.

## Persistence

Each run writes to `.agent6/runs/<run-id>/`:

- `graph.jsonl` — append-only journal of every task-graph mutation.
- `graph.dot` — current task graph, regenerated atomically.
- `nodes/*.md` — one markdown file per task node, rewritten atomically.
- `logs.jsonl` — per-event log (LLM turns, tool calls, costs).
- `snapshots/` — per-tool-call JSON snapshots that drive `agent6 resume`.
- `transcripts/` — full provider request/response pairs for replay.

A separate `agent6-curator` subprocess owns all writes to this
directory and runs under its own jail policy that allows writes only
to `.agent6/`. The main agent process talks to it over a Unix-domain
socket; the curator validates every IPC frame against a pydantic
schema before applying it.

## End-of-run notify hook

Optional. If `[notify]` declares `on_complete = [...]`, agent6 runs
that argv after every `agent6 run` / `agent6 resume`, with these env
vars set: `AGENT6_RUN_ID`, `AGENT6_RUN_DIR`, `AGENT6_RUN_OK`
(`"1"`/`"0"`), `AGENT6_RUN_REASON`. The hook runs OUTSIDE the jail as
your user; the argv is operator-controlled (never derived from LLM
output). Typical use: a `notify-send` desktop popup, a Slack curl, or
piping the run-dir into `agent6 review`.

## Repository layout

```
src/agent6/
  cli.py            argparse entry points
  config.py         pydantic-strict config
  budget.py         per-model pricing + per-run accounting
  events.py         structured run-event log
  git_ops.py        git wrappers; refuses push/force/rewrite
  memory.py         persistent agent memory
  detect.py         kernel + container capability detection
  init.py           `agent6 init` scaffolding
  agents/           single-turn LLM call shapes (code_review)
  workflows/        the agent loop (run) and read-only review
  machine/          declarative state-machine layer (see STATE_MACHINES.md)
  tools/            dispatcher + schemas for the LLM tool surface
  providers/        Anthropic + OpenAI HTTP clients (httpx, no SDK)
  sandbox/          jail.py (Python wrapper) + landlock.py
  graph/            curator subprocess + UDS IPC + on-disk graph store
  ui/               event fold + JSONL tailer + optional textual TUI
jail/               Rust crate for agent6-jail
tests/
  unit/             unit tests
  integration/      crash-resume, curator IPC
  sandbox/          live jail smoke tests + Landlock probes
  security/         prompt-injection corpus tests
bench/              perf + realworld benchmark harnesses
```

## Contributing

Read [AGENTS.md](AGENTS.md) before sending a PR. The repo's
`verify_command` is the single source of truth for "is this PR
landable":

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

Security-sensitive changes — anything under `sandbox/`, `tools/`,
`git_ops.py`, `providers/`, or `graph/curator` — must include a
security review note in the commit message describing what surface
changed.

## License

[Apache-2.0](LICENSE).
