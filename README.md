# agent6

A sandboxed coding agent for Anthropic Claude and OpenAI-compatible
backends (OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp).
Linux-only.

- LLM calls are wrapped in deterministic Python workflows with a fixed
  tool surface.
- Every child process runs inside `agent6-jail` (Rust + Linux user
  namespaces + Landlock + seccomp).
- Git operations refuse `push`, `--force`, and history rewrite.
- No web UI, no plugin system, no telemetry, no auto-update.

**Status**: pre-release, version `0.0.1`. 208 tests pass. The synthetic
benchmark in [bench/results.md](bench/results.md) surfaced four
reproducible workflow issues that should land before a tagged 0.1.0 PyPI
release.

## Threat model

The worker model is treated as adversarial. It must not be able to:

- write outside the project's working directory
- read files outside the project (plus any sibling read-only paths)
- reach the network except the host:port of each `[providers.*]` block
  (when `sandbox.network = "provider_only"`)
- rewrite, push, or hard-reset git history
- leave background processes running after the run ends

Enforcement is layered:

- **Landlock** is applied to the agent's own Python process to restrict
  what *it* can read/write.
- **`agent6-jail`** (a small Rust binary) wraps every child command:
  fresh user/mount/pid/ipc/uts/net namespace, pivots into a minimal
  rootfs, applies Landlock and a seccomp filter, drops capabilities, then
  execs the child.

See [SECURITY.md](SECURITY.md) for the per-layer breakdown.

## Architecture

```
                     ┌───────────────────────────────────┐
                     │           agent6 CLI              │
                     └───────────────┬───────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
        ┌───────▼───────┐  ┌─────────▼─────────┐  ┌──────▼──────┐
        │  workflows/   │  │      agents/      │  │   graph/    │
        │   implement   │  │   planner, worker │  │   curator   │
        │   plan_mode   │  │   critic, reviewer│  │   (subproc, │
        │   review      │  │   code_review,    │  │   own jail) │
        └───────┬───────┘  │   summarizer,     │  └─────────────┘
                │          │   alignment       │
                │          └───────┬───────────┘
                │                  │
        ┌───────▼──────────────────▼────────────┐
        │             tools/dispatch            │
        │  read_file list_dir grep apply_edit   │
        │  run_verify_command [run_command]     │
        └───────────────────┬───────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │  sandbox/jail.py  │
                  └─────────┬─────────┘
                            │ JSON policy
                  ┌─────────▼─────────┐
                  │  agent6-jail      │  (Rust; userns + landlock + seccomp)
                  └───────────────────┘
```

Dependency direction is enforced by [tach](https://docs.gauge.sh/):
`cli → workflows → agents → tools → sandbox`. Workflows never import each
other; agents never import workflows or the CLI.

## Requirements

- Linux. macOS and Windows are not supported and never will be — the
  sandbox uses Linux-only kernel APIs (Landlock, seccomp-bpf, user/mount
  namespaces, pivot_root).
- Linux kernel ≥ 6.7 for full Landlock TCP-connect rules. Older kernels
  fall back to filesystem-only Landlock with a loud warning.
- `kernel.unprivileged_userns_clone = 1` (default on Ubuntu, Debian, and
  most cloud images). Required for the `strict` sandbox profile; without
  it the agent falls back to `hardened` or refuses, per config.
- Python ≥ 3.12.
- Anthropic and/or OpenAI-compatible API key.

If installing from source, you also need:

- A Rust toolchain on `PATH` (`cargo`, `rustc`). The hatch build hook
  invokes `cargo build` to compile `agent6-jail`. It does **not** install
  Rust for you — if `cargo` is not on `PATH` the hook skips with a
  message and the resulting install has no jail binary
  (`agent6 check-sandbox` will tell you).
- Released PyPI wheels bundle a prebuilt `agent6-jail` and have no Rust
  toolchain requirement.

## Install

From source (development):

```bash
git clone https://github.com/<you>/agent6
cd agent6
uv sync --extra tui
uv run agent6 --help
```

`uv sync` runs `hatch_build.py`, which:

1. invokes `cargo build --release --locked --manifest-path jail/Cargo.toml`
2. copies the resulting `agent6-jail` into
   `src/agent6/sandbox/_bin/agent6-jail` (gitignored)

The `[tui]` extra pulls in `textual` for the live dashboard. Skip it
with `uv sync` if you don't want the TUI.

From PyPI (once released):

```bash
uv tool install agent6
```

PyPI wheels ship with the jail binary inside. To override the bundled
binary (custom build, alternate path), set
`AGENT6_JAIL_BIN=/path/to/agent6-jail`.

### Shell tab-completion

agent6 supports tab-completion for all subcommands and flags via
[argcomplete](https://kislyuk.github.io/argcomplete/). To enable it for
your current shell, source the completion script once:

```bash
# Bash (add to ~/.bashrc to persist).
eval "$(register-python-argcomplete agent6)"

# Zsh (add to ~/.zshrc; needs `autoload -U compinit && compinit` first).
eval "$(register-python-argcomplete agent6)"

# Fish (one-time write).
register-python-argcomplete --shell fish agent6 > ~/.config/fish/completions/agent6.fish
```

Or, for system-wide completion in all shells, run
`activate-global-python-argcomplete` once (see argcomplete docs).

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Scaffold agent6.toml + AGENTS.md and add .agent6/ to .gitignore.
agent6 init

# Sanity checks.
agent6 check-config
agent6 check-sandbox

# Plan-only: cheap pre-flight, no code changes.
agent6 plan new "add a --json output mode to the CLI"

# Inspect a previously persisted plan (defaults to most recent).
agent6 plan show

# Apply free-form feedback to the last plan, producing a new run.
agent6 plan revise "split step 2 into smaller commits"

# Hand-edit the plan JSON in $EDITOR, producing a new run.
agent6 plan edit

# Offline Q&A: if the critic raises clarifying questions, write
# them to a file, fill in the 'answer' fields, then re-run.
agent6 plan new --questions-file q.json "add a --json output mode"
$EDITOR q.json
agent6 plan new --run-id <same-id> --answers-file q.json "add a --json output mode"

# Full implement workflow. --yes auto-confirms the plan.
agent6 run --yes "add a --json output mode to the CLI"

# Resume an interrupted run (refuses if the worktree diverged).
agent6 resume <run-id>

# Read-only code review of a diff. Never touches the worktree.
agent6 review --base origin/main --head HEAD
```

Other commands:

- `agent6 watch [<run-id>]` — attach the live TUI dashboard to an
  existing run (defaults to the most recent). Same view that
  `agent6 run` auto-launches in a TTY. Attach and detach freely.
- `agent6 memory` — manage persistent agent memory under
  `.agent6/memory/`.
- `agent6 history` — search transcripts and run data under
  `.agent6/runs/`. Subcommands:
  - `agent6 history search <query>` — ripgrep-backed text search.
  - `agent6 history graph [<run-id>]` — render the persisted task
    graph for a run as a DFS-ordered tree (defaults to the most
    recent run).
- `agent6 --help` — full subcommand list.

## Configuration

Every field in `agent6.toml` is required. No implicit defaults. Start
from [agent6.example.toml](agent6.example.toml).

Highlights:

```toml
[sandbox]
profile = "auto"             # auto | strict | hardened
network = "provider_only"    # no | provider_only | allow
run_commands = "ask"         # yes | no | ask

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"  # per_step | squash | stage | none
allow_push = false           # cannot be true; ignored if set
allow_force = false
allow_history_rewrite = false

[workflow]
default = "implement"
verify_command = ["uv", "run", "pytest", "-x"]

[budget]
max_input_tokens  = 2000000
max_output_tokens = 200000
```

Sandbox profiles:

- **strict** — user/mount/pid/ipc/uts/net namespaces + pivot_root into a
  minimal rootfs + Landlock + seccomp + capset(0) + rlimits +
  `NO_NEW_PRIVS`. Requires unprivileged user namespaces.
- **hardened** — no namespaces, but still Landlock + seccomp + capset(0)
  + rlimits + `NO_NEW_PRIVS`. Works inside default-seccomp Docker.
- **auto** — pick `strict` if the kernel allows, else `hardened`. Logs
  the chosen profile on every run.

Network modes are enforced by the jail's net-namespace setup (in
`strict`) or by Landlock's TCP-connect rules (kernel ≥ 6.7) on the agent
process.

## Providers and sub-agents

Declare any number of providers as `[providers.<name>]` blocks. Each
block sets `kind = "anthropic"` or `kind = "openai"` and has its own
`base_url` and `api_key_env`. OpenAI, OpenRouter, Ollama, vLLM, llama.cpp,
LM Studio etc. coexist under whatever names you pick.

Each sub-agent has a fixed system prompt, a pydantic-typed output schema,
and only the tools its workflow gives it. Routing per role:

| Sub-agent        | Routed by             | Purpose                                                                |
| ---------------- | --------------------- | ---------------------------------------------------------------------- |
| `planner`        | `[models.planner]`    | Decomposes the refined task into ordered steps.                        |
| `worker`         | `[models.worker]`     | Executes one plan step using the tool surface.                         |
| `critic`         | `[models.critic]`     | Raises open questions; aborts on real ambiguity.                       |
| `reviewer`       | `[models.reviewer]`   | Reviews each step's diff; approves or asks for fixes.                  |
| `code_review`    | `[models.reviewer]`   | Powers `agent6 review`: read-only review of an arbitrary diff.         |
| `summarizer`     | `[models.summarizer]` | Compresses long verify output / file context to fit budgets.           |
| `alignment`      | `[models.worker]`     | Guards `agent6 resume`: refuses if the worktree diverged.              |
| `planner_revise` | `[models.planner]`    | Revises a plan in response to a critic or reviewer objection.          |

Mixing vendors across roles (e.g. Anthropic planner + OpenRouter worker
+ Anthropic reviewer) helps catch shared-failure blind spots.

## Workflows

- **implement** — plan → critic → worker (loop) → reviewer per step.
  Verify runs after every step; failures retry within the step budget.
  State is persisted to `.agent6/runs/<run-id>/`.
- **plan_mode** — produce a frozen plan only. Useful as a cheap "is the
  plan reasonable?" pre-flight.
- **review** — drives the `code_review` sub-agent on a working-tree,
  branch-vs-base, or `<base>..<head>` diff. Always read-only.

## Tool surface given to the LLM

Fixed and audited in [src/agent6/tools/schema.py](src/agent6/tools/schema.py):

- `read_file(path, start_line?, end_line?)`
- `list_dir(path)`
- `grep(pattern, path?, glob?)`
- `apply_edit(path, edits: list[{kind: "replace"|"create", old_string?, new_string}])`
- `run_verify_command()` — runs `workflow.verify_command` only
- `run_command(argv)` — only if `sandbox.run_commands ∈ {"yes", "ask"}`

There is no `write_file`, no `shell`, no `web_fetch`. Adding a tool
requires a security review note in the commit message.

## Cost accounting

Every run prints a per-model token + cost summary at the end:

```
Token + cost summary:
  claude-opus-4-5:    in=18054 out=425  cache_r=0 cache_c=0 calls=1 $0.3027
  claude-sonnet-4-5:  in=8884  out=1171 cache_r=0 cache_c=0 calls=4 $0.0442
  TOTAL: in=26938/2000000 out=1596/200000 cost~$0.3469
```

Pricing lives in [src/agent6/budget.py](src/agent6/budget.py) and is
updated by hand from the Anthropic and OpenAI public pages. Budgets in
`agent6.toml` hard-stop the run; a stopped run is resumable.

## Live event log + TUI

Every run writes a structured JSONL event stream to
`.agent6/runs/<run-id>/logs.jsonl`. The vocabulary is small and stable:

| Event                     | Emitted by         | Notable fields                              |
| ------------------------- | ------------------ | ------------------------------------------- |
| `run.start`               | implement workflow | `user_task`                                 |
| `plan.ready`              | implement workflow | `summary`, `steps[]`                        |
| `step.start` / `step.end` | implement workflow | `index`, `title`, `status`, `commit_sha`    |
| `step.diff`               | implement workflow | `index`, `commit_sha`, `patch` (truncated)  |
| `tool.call` / `.result`   | tool dispatcher    | `name`, `args` (preview), `ok`, `summary`   |
| `verify.start` / `.end`   | tool dispatcher    | `cmd`, `exit_code`, `duration_s`, `*_tail`  |
| `role.call` / `.result`   | provider wrapper   | `role`, `model`, `tokens_in`, `tokens_out`  |
| `budget.update`           | provider wrapper   | totals + caps for input/output tokens       |
| `approval.prompt`/`.answer` | tool dispatcher  | `id`, `prompt`, `approved`, `source`        |
| `run.end`                 | implement workflow | `all_passed`                                |

The shape is the data contract for any external viewer. The fold from
event stream to UI state lives in
[src/agent6/ui/state.py](src/agent6/ui/state.py) as a pure function and
is intended to be ported 1:1 to TypeScript.

When `agent6` is installed with the `tui` extra and stdout is a TTY,
`agent6 run` spawns a separate process running
`python -m agent6.ui --watch <run-dir>` that tails `logs.jsonl` and
renders a plan tree, budget bar, tool table, log tail, and the latest
step diff. The TUI is read-only on the log; the only thing it writes is
`<run-dir>/approvals/<id>.answer` when the user clicks Allow/Deny on a
`run_command` approval modal. Killing the TUI does not affect the
workflow — it falls back to a plain stdin prompt. Pass `--no-tui` to
disable, or attach later with `agent6 watch`.

## Persistence

Each run writes to `.agent6/runs/<run-id>/`:

- `graph.jsonl` — append-only journal of every mutation to the task graph.
- `graph.dot` — current task graph (regenerated atomically on topology
  change).
- `nodes/*.md` — one markdown file per node; rewritten atomically.
- `logs.jsonl` — per-event log (planner output, tool calls, costs).
- `transcripts/` — full provider request/response pairs for replay.

A separate `agent6-curator` subprocess owns all writes to this directory
and runs under its own jail policy allowing writes only to `.agent6/`.
The main agent process talks to it over a Unix domain socket; no other
process has authority to mutate the graph.

## Benchmark

See [bench/results.md](bench/results.md) for a 4-task synthetic benchmark
(bug fix, add CLI flag, refactor, type annotations). First run completed
**0/4** tasks for **~$0.08** of spend and surfaced four reproducible
workflow issues with concrete suggested fixes. Re-run with:

```bash
bash bench/run_bench.sh
cat /tmp/agent6-bench/*/result.json
```

A direct head-to-head against `claude-code` and other coding agents is
not in this repo — it would need a shared task set and a neutral runner.

## Repository layout

```
src/agent6/
  cli.py            argparse entry points
  config.py         pydantic-strict config (all fields required)
  budget.py         per-model pricing + per-run accounting
  events.py         structured run-event log
  git_ops.py        pure-function git wrappers; refuses push/force/rewrite
  memory.py         persistent agent memory (read/write/delete)
  detect.py         kernel + container capability detection
  init.py           `agent6 init` scaffolding
  agents/           typed sub-agent prompts and pydantic IO schemas
  workflows/        deterministic Python orchestrators (implement, plan, review)
  tools/            dispatcher + schemas for the fixed LLM tool surface
  providers/        Anthropic + OpenAI HTTP clients (httpx, no SDK)
  sandbox/          jail.py (Python wrapper) + landlock.py (process-side)
  graph/            curator subprocess + UDS IPC + on-disk graph store
  ui/               pure event-fold + stdlib JSONL tailer + optional textual TUI
jail/               Rust crate for the agent6-jail launcher
tests/
  unit/             ~150 unit tests
  integration/      crash-resume, curator IPC, plan-mode, alignment
  sandbox/          live jail smoke tests + landlock probes
  security/         prompt-injection corpus tests
bench/              synthetic benchmark harness + results
.github/workflows/  build, ci, pypi (Trusted Publishing)
```

## Roadmap

Open work, in priority order:

1. **Fix the four bench findings** (F1–F4 in
   [bench/results.md](bench/results.md)): per-step verify gating,
   edit-by-old-string fragility on consecutive steps, critic
   conservatism, run-log-before-dirty-check race.
2. **Network-egress test corpus** — adversarial tests that try to
   exfiltrate via DNS, ICMP, IPv6, and unix sockets and assert they are
   all blocked under `network = "provider_only"`.
3. **`agent6 init --template`** — write a starter `AGENTS.md` for common
   stacks (Python lib, Node app, …) instead of just a stub.
4. **Headless run mode** — non-interactive `--no-confirm-anything` for
   CI; today `--yes` auto-confirms the plan only, not every
   `run_commands` prompt.
5. **`agent6 review` against a remote PR** — fetch + review without
   checkout.
6. **First PyPI release** — wired in
   [.github/workflows/pypi.yml](.github/workflows/pypi.yml) via Trusted
   Publishing inside a manylinux container so the bundled `agent6-jail`
   links against an old enough glibc. Blocked on items 1–2.

## Contributing

Read [AGENTS.md](AGENTS.md) before sending a PR. The repo's
`verify_command` is the single source of truth for "is this PR
landable":

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

Security-sensitive changes (anything in `sandbox/`, `tools/`, `git_ops`,
`providers/`, `graph/curator`) must include a security review note in
the commit message.

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text. Every
source file carries an `SPDX-License-Identifier: Apache-2.0` header and
a `Copyright 2026 Eric Lesiuta` notice.
