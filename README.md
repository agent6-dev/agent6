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
- **Small footprint.** Five runtime dependencies (`pydantic`, `httpx`,
  `argcomplete`, and the `tree-sitter` pair that backs the `outline` /
  `find_definition` / `find_references` tools). No telemetry, no
  auto-update. The LLM tool surface is fixed and audited; the only
  opt-in extension point is operator-configured MCP servers
  (`[mcp]`, off by default) — there is no in-process plugin loading.

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

Install from [PyPI](https://pypi.org/project/agent6/) with [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/how-to/install-pipx/).

```bash
uv tool install "agent6[tui]"
pipx install "agent6[tui]"
```

The `tui` extra pulls in `textual` for the live dashboard; drop it
(`agent6`) for a headless install. Both tools drop the `agent6` entry
point in a user bin directory (`~/.local/bin`); if it isn't on your
`PATH` yet, run `uv tool update-shell` or `pipx ensurepath` (then restart
your shell).

From source:

```bash
git clone https://github.com/elesiuta/agent6
cd agent6
uv sync --extra tui
uv run agent6 --help
```

To override the bundled jail binary, set
`AGENT6_JAIL_BIN=/path/to/agent6-jail`.

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
# Connect a provider once (stored in ~/.config/agent6/, key in a 0600
# secrets file). Works across every repo.
agent6 connect                # interactive: pick provider, paste API key
agent6 model worker anthropic claude-sonnet-4-5

# In a project: scaffold .agent6/config.toml + AGENTS.md and gitignore .agent6/.
agent6 init

# Audit the effective config: every value + where it came from
# (default / global / repo). `*` marks values that override the default.
agent6 config show

# Pre-flight: sandbox + config + provider keys + verify_command.
agent6 check

# Run the agent on a task.
agent6 run "add a --json output mode to the CLI"

# Resume an interrupted run from its last tool-call snapshot.
agent6 resume <run-id>

# Read-only code review of a diff. Never touches the worktree.
agent6 review --base origin/main --head HEAD
```

Config is layered: built-in **secure defaults** < global
`~/.config/agent6/config.toml` < per-repo `.agent6/config.toml` < an
explicit `--config FILE`. A repo can be zero-config when the global
config supplies a provider + model; the one thing a repo always needs is
its `verify_command`.

Other commands:

- `agent6 watch [<run-id>]` — attach the live TUI to an existing run
  (defaults to the most recent).
- `agent6 plan "<task>"` — read-only planning pass (uses the `planner`
  model, falls back to `worker`); execute with `agent6 run --from-plan`.
- `agent6 memory` — manage persistent agent memory under
  `.agent6/memories/`.
- `agent6 history search <query>` — ripgrep-backed search over
  persisted transcripts.
- `agent6 history graph [<run-id>]` — render the persisted task graph.
- `agent6 diff [<run-id>]` — print the git diff a run produced
  (`manifest.base_sha` → HEAD of the run branch).
- `agent6 machine ...` — author and run agent6 state machines
  (`.asm.toml`); see [STATE_MACHINES.md](STATE_MACHINES.md).
- `agent6 mcp serve` — expose agent6's own tools over MCP (stdio).
- `agent6 config fill` — materialize every effective value into one
  explicit config file (global by default, `--repo` for the repo).
- `agent6 config get/set/unset/add/remove <key> [value]` — read or edit a
  single dotted leaf (e.g. `sandbox.agent_network`). Writes go to the global
  config by default, `--repo` for the repo, or `--machine FILE` for a
  machine's `[config]` overlay (`providers.*` is forbidden there). `add`/
  `remove` edit list fields such as `sandbox.allow_urls`. Every edit is
  re-validated and rolled back if it would produce an invalid config.

## How it works

agent6 is a single-loop agent: one provider, one model, one message
history. The model drives the run by calling tools; the workflow
dispatches them, snapshots state before every LLM call (so any run is
resumable), commits when `verify_command` passes, and tracks budget with
hard stops. Module boundaries (`cli → workflows → agents → tools →
sandbox`) are enforced by [tach](https://docs.gauge.sh/).

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the run/review loops, the
curator subprocess, on-disk run state, and where each concern lives.

## Security

The worker LLM is treated as adversarial — it cannot write or read
outside the workspace, reach the network beyond the configured provider
endpoints, corrupt git history or its own run state, or leave processes
running after the run. This is enforced *structurally*: every LLM-chosen
child command runs in the `agent6-jail` sandbox (namespaces + `pivot_root`
+ Landlock + seccomp), the agent's own egress is broker-confined to the
provider endpoints, and `git_ops.py` refuses `push` / `--force` /
history-rewrite unconditionally. Defaults are safe
(`sandbox.agent_network = "providers"`, `sandbox.tool_network = "block"`,
`sandbox.run_commands = "ask"`, `sandbox.protect_* = true`,
`git.allow_* = false`).

See **[SECURITY.md](SECURITY.md)** for the threat model, the per-layer
breakdown, and the sandbox profiles.

## Configuration

agent6 is **secure by default**: every field has a default, and
security-sensitive ones default to the safe value. The full field
reference is [CONFIG.md](CONFIG.md); the sandbox profiles and security
model are explained in [SECURITY.md](SECURITY.md).
Get started with `agent6 connect` + `agent6 model` (global) and
`agent6 init` (per-repo). `agent6 config show` audits every effective
value and where it came from; `agent6 check` validates without running.

```toml
[agent6]
# Optional, GLOBAL config only: rename the in-repo agent6 directory
# (config + run state) from ".agent6" to a name of your choosing.
# workspace_subdir = ".agent6"

[sandbox]
profile = "auto"              # auto | strict | hardened
agent_network = "providers"   # providers | local | open  (agent's LLM egress)
tool_network = "block"        # block | only_explicit_states | allow  (jailed commands)
allow_urls = []               # extra agent egress hosts under "providers"
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
# max_usd = 10.0               # optional; converted to token caps at load

[providers.anthropic]
kind = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-opus-4-5"
```

Budget ceilings can be overridden per-run from the CLI without touching
config: `agent6 run --max-usd 5 "..."`, or
`--max-input-tokens` / `--max-output-tokens` on `run`, `plan`, and
`resume`.

### Providers and models

Declare any number of providers as `[providers.<name>]` blocks. Each
sets `kind = "anthropic"` or `kind = "openai"` and has its own
`base_url` and `api_key_env`. Multiple providers (e.g. OpenAI plus
OpenRouter plus a local Ollama) coexist under whatever names you pick.
Per-provider `http_timeout_s` (default `600.0`) caps each HTTP call —
raise it for slow reasoning models, lower it to fail fast on a stuck
endpoint.

agent6 uses three model roles:

| Role       | Routed by             | Used by                                                                          |
| ---------- | --------------------- | -------------------------------------------------------------------------------- |
| `worker`   | `[models.worker]`     | `agent6 run` and `agent6 resume`. Also drives the USD↔token budget conversion.   |
| `reviewer` | `[models.reviewer]`   | `agent6 review` (read-only diff review) and the optional in-loop critic.         |
| `planner`  | `[models.planner]`    | `agent6 plan` (read-only planning pass). Falls back to `worker` when unset.      |

Each role takes an optional `thinking` level (`off`/`low`/`medium`/`high`).
OpenAI-compatible reasoning models receive a reasoning-effort knob; Anthropic
models map it onto an extended-thinking token budget (low/medium/high ≈
4k/8k/16k thinking tokens, with `max_tokens` lifted above the budget
automatically).


## Tool surface

The set of tools given to the LLM is fixed and audited in
[src/agent6/tools/schema.py](src/agent6/tools/schema.py) (see
[SECURITY.md](SECURITY.md) §4 for why this surface is the security
boundary). Adding a tool requires a security review note in the commit
message.

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
updated by hand from the providers' public pricing pages. The `[budget]`
ceilings in your config hard-stop the run; a stopped run is resumable.

## Live view

With the `tui` extra installed and stdout a TTY, `agent6 run` auto-spawns
a textual dashboard (task DAG, budget bar, tool table, log tail, latest
diff) that owns the terminal for the run and closes when the run ends;
`--no-tui` (and `-i`, the stdin REPL) opt out. A `run_command` approval
prompt appears as an Allow/Deny modal in the dashboard (it falls back to a
stdin `[y/N]` prompt when no TUI is present). Attach to a running or
finished run from another shell with `agent6 watch [<run-id>]`; `agent6
watch --plain` is a no-deps text tail for headless terminals. The
dashboard folds a structured JSONL event stream
(`.agent6/runs/<run-id>/logs.jsonl`) that is also the contract for any
external viewer — the event vocabulary is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Persistence

Each run's state lives under `.agent6/runs/<run-id>/` (append-only task
graph, per-call snapshots that drive `agent6 resume`, full transcripts,
and the event log). It is written *exclusively* by a sandboxed
`agent6-curator` subprocess over a pydantic-validated IPC channel, so a
bug in the agent can't scribble the run directory. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the on-disk layout and the curator.

## End-of-run notify hook

Optional. If `[notify]` declares `on_complete = [...]`, agent6 runs
that argv after every `agent6 run` / `agent6 resume`, with these env
vars set: `AGENT6_RUN_ID`, `AGENT6_RUN_DIR`, `AGENT6_RUN_OK`
(`"1"`/`"0"`), `AGENT6_RUN_REASON`. The hook runs OUTSIDE the jail as
your user; the argv is operator-controlled (never derived from LLM
output). Typical use: a `notify-send` desktop popup, a Slack curl, or
piping the run-dir into `agent6 review`.

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
