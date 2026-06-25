# agent6

A sandboxed coding agent for Linux. The LLM is treated as adversarial: every
command it spawns runs inside a custom Rust launcher (`agent6-jail`) built on
user namespaces, Landlock, seccomp, `pivot_root`, `capset(0)`, and
`NO_NEW_PRIVS`, so a misbehaving model cannot escape the workspace, reach the
network beyond the provider endpoint, or corrupt git history.

Features:

- Sandboxed execution for every LLM-chosen child process (verify commands,
  metric commands, optional shell)
- Works with Anthropic and any OpenAI-compatible endpoint (OpenAI,
  OpenRouter, Ollama, vLLM, llama.cpp, LM Studio), tuned to stay effective
  on cheap open-weights models
- Per-step git commits, snapshot-resumable runs, USD and token budgets with
  hard stops
- Plan, run, review, and ask modes; a live terminal dashboard; persistent
  transcripts and a searchable run history
- State machines (`agent6 machine`) for long-running automated tasks:
  LLM-drafted, operator-reviewed, journaled, and replayable
- Small, fixed LLM tool surface; the only extension point is
  operator-configured MCP servers, off by default
- Eight runtime dependencies, no telemetry, no auto-update

## Requirements

- Linux for the sandbox. It uses Linux-only kernel APIs (Landlock, seccomp,
  user namespaces). macOS and Windows run unsandboxed: the default
  `profile = "auto"` resolves to `none`, child commands run as plain
  subprocesses behind a startup warning, and an explicit `profile = "strict"`
  or `"hardened"` is refused. Run on Linux for kernel-enforced isolation.
- Kernel 6.7 or newer for Landlock TCP rules. Older kernels fall back to
  filesystem-only Landlock with a warning.
- `kernel.unprivileged_userns_clone = 1` for the `strict` profile (default
  on Ubuntu, Debian, and most cloud images); without it agent6 falls back
  to `hardened`. On Ubuntu 24.04+ with
  `kernel.apparmor_restrict_unprivileged_userns = 1`, install the bundled
  AppArmor profile (`packaging/apparmor/agent6-jail`; `agent6 check sandbox`
  prints the commands) or set that sysctl to 0.
- Python 3.12 or newer, plus an API key for at least one provider.
- Building from source needs a Rust toolchain on `PATH`; PyPI wheels bundle
  a prebuilt `agent6-jail`.

## Install

From [PyPI](https://pypi.org/project/agent6/) with
[uv](https://docs.astral.sh/uv/getting-started/installation/) or
[pipx](https://pipx.pypa.io/stable/how-to/install-pipx/):

```bash
uv tool install agent6
pipx install agent6
```

Both drop the `agent6` entry point in `~/.local/bin`; if that is not on your
`PATH`, run `uv tool update-shell` or `pipx ensurepath` and restart your
shell.

From source:

```bash
git clone https://github.com/elesiuta/agent6
cd agent6
uv sync
uv run agent6 --help
```

`AGENT6_JAIL_BIN=/path/to/agent6-jail` overrides the bundled jail binary.

## Shell tab-completion

Via [argcomplete](https://kislyuk.github.io/argcomplete/):

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

# In a project: scaffold the per-repo config + AGENTS.md.
agent6 init

# Audit the effective config: every value and where it came from.
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

Config is layered: built-in secure defaults, then the global
`~/.config/agent6/config.toml`, then the per-repo config (out of the
workspace under `$XDG_STATE_HOME/agent6/<repo-id>/config.toml`), then an
explicit `--config FILE`. The per-repo config is per-machine, not committed.
A repo can be zero-config when the global config supplies a provider and
model; the one thing a repo always needs is its `verify_command`, which
`agent6 init` scaffolds per checkout.

Other commands:

- `agent6 watch [<run-id>]`: attach the live TUI to an existing run.
- `agent6 plan "<task>"`: read-only planning pass; execute with
  `agent6 run --from-plan`.
- `agent6 ask "<question>"`: read-only Q&A over the repo, including
  questions about agent6 itself (it consults its bundled docs). Seed
  context with `@path` or `--file`; `--run <id>` asks about a prior run.
- `agent6 memory`: persistent agent memory under the per-repo state dir
  (`<state-dir>/<repo-id>/memories/`).
- `agent6 history search <query>`: search persisted transcripts.
- `agent6 history graph [<run-id>]`: render the persisted task graph.
- `agent6 diff [<run-id>]`: print the git diff a run produced.
- `agent6 machine ...`: author and run state machines (`.asm.toml`); see
  [STATE_MACHINES.md](STATE_MACHINES.md).
- `agent6 mcp serve`: expose agent6's tools over MCP (stdio).
- `agent6 config fill`: materialize every effective value into one file.
- `agent6 config get/set/unset/add/remove <key> [value]`: edit a single
  dotted leaf. Writes go to the global config by default, `--repo` for the
  repo, `--machine FILE` for a machine overlay. Every edit is re-validated
  and rolled back if invalid.

## Configuration

Every field has a default, and security-sensitive fields default to the
safe value. The full reference is [CONFIG.md](CONFIG.md); sandbox profiles
are explained in [SECURITY.md](SECURITY.md).

```toml
[sandbox]
profile = "auto"              # auto | strict | hardened
agent_network = "providers"   # providers | local | open  (agent's LLM egress)
tool_network = "block"        # block | only_explicit_states | allow  (jailed commands)
run_commands = "ask"          # yes | no | ask
protect_git = true            # strict only: re-bind .git read-only in the jail

[git]
require_clean_worktree = true
branch_per_run = true
commit_strategy = "per_step"  # per_step | squash | stage | none
allow_push = false

[workflow]
verify_command = ["uv", "run", "pytest", "-x"]

[budget]
max_input_tokens  = 2000000
max_output_tokens = 200000
# best_effort_usd_limit = 10.0  # optional; see CONFIG.md

[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"
```

Budget ceilings can be overridden per run: `agent6 run --max-usd 5 "..."`,
or `--max-input-tokens` / `--max-output-tokens` on `run`, `plan`, and
`resume`.

### Providers and models

Declare any number of `[providers.<name>]` blocks, each with
`kind = "anthropic"` or `kind = "openai"`, its own `base_url`, and
`api_key_env`. Per-provider `http_timeout_s` (default 600) caps each HTTP
call.

agent6 uses three model roles:

| Role       | Routed by           | Used by                                                  |
| ---------- | ------------------- | -------------------------------------------------------- |
| `worker`   | `[models.worker]`   | `agent6 run` / `resume`; drives USD-to-token conversion. |
| `reviewer` | `[models.reviewer]` | `agent6 review` and the optional in-loop critic.         |
| `planner`  | `[models.planner]`  | `agent6 plan`. Falls back to `worker` when unset.        |

`agent6 model all <provider> <model>` sets every role at once. Each role
takes an optional `thinking` level (`off`/`low`/`medium`/`high`).

## Tool surface

The tools given to the LLM are fixed and declared in one place,
[src/agent6/tools/schema.py](src/agent6/tools/schema.py); adding one
requires a security review note in the commit message.

- Read-only: `read_file`, `list_dir`, `grep`, `outline`,
  `find_definition`, `find_references`
- Edits: `apply_edit` (structured blocks), `apply_patch` (unified diff)
- Execution with operator-fixed argv: `run_verify_command`,
  `run_metric_command`
- Control: `finish_run`, plus `dag_*` task-notepad tools backed by the
  curator subprocess
- Conditional: `run_command(argv)`, only exposed when
  `sandbox.run_commands` allows it, and always jailed

There is no `write_file`, `shell`, or `web_fetch`.

## How it works

agent6 is a single-loop agent: one provider, one model, one message
history. The model drives the run by calling tools; the workflow dispatches
them, snapshots state before every LLM call (so any run is resumable),
commits when `verify_command` passes, and hard-stops on budget. Module
boundaries (`cli -> workflows -> agents -> tools -> sandbox`) are enforced
by [tach](https://docs.gauge.sh/). See
[ARCHITECTURE.md](ARCHITECTURE.md) for the run/review loops, the curator
subprocess, and the on-disk layout.

For security details (threat model, per-layer breakdown, sandbox
profiles), see [SECURITY.md](SECURITY.md). Defaults are safe:
`agent_network = "providers"`, `tool_network = "block"`,
`run_commands = "ask"`, `protect_git = true`, `git.allow_* = false`, and
`git_ops.py` refuses `push`, `--force`, and history rewrites
unconditionally.

## Benchmarks

Reproducible harnesses live under [bench/](bench/):

- [bench/realworld/](bench/realworld/): 11 SWE-bench-Lite-style tasks
  scored by fresh sandboxed verifies on hidden tests. Latest recorded run:
  agent6 and claude-code both solve 11/11 on the same worker model
  (`claude-sonnet-4-5`); agent6 at about $2.60 total, claude-code at about
  $3.96. Single runs, no variance measured; re-run before quoting.
- [bench/agents/](bench/agents/): head-to-head against Claude Code,
  opencode, and aider on Go and Rust tasks with cheap models.
- [bench/machine/](bench/machine/): `machine create` attempts, cost, and
  validation results.
- [bench/perf/](bench/perf/): a perf-optimization harness for local
  experimentation; single-run numbers are too noisy to quote.

## Cost accounting

Every run ends with a per-model token and cost summary. Model prices are
fetched from the provider's models endpoint and cached (OpenRouter
publishes them; Anthropic does not, so its models report an unknown
price). Where the provider reports per-call cost, that figure is used
directly. The `[budget]` ceilings hard-stop the run; a stopped run is
resumable.

## Live view

With stdout a TTY, `agent6 run` opens a terminal dashboard (task DAG,
budget bar, tool table, live reasoning pane, log tail, latest diff);
`--no-tui` and `-i` (stdin REPL) opt out. Approval and Ctrl-C steer
prompts appear as modals, with a `/dev/tty` fallback when no TUI is
present. `agent6 plan`, `agent6 ask`, and `agent6 machine create` stream
reasoning and answers to the terminal. Attach from another shell with
`agent6 watch [<run-id>]`; `agent6 watch --plain` is a plain-text tail.
The dashboard renders the JSONL event stream at
`<state-dir>/<repo-id>/runs/<run-id>/logs.jsonl`, which is also the contract
for external viewers (vocabulary in [ARCHITECTURE.md](ARCHITECTURE.md)).

## Persistence

Per-repo state lives out of the workspace under `$XDG_STATE_HOME/agent6/<repo-id>/`
(override with `[agent6].state_dir` or `AGENT6_STATE_HOME`). Each run's state
sits under `runs/<run-id>/`: the append-only task graph, per-call snapshots
that drive `agent6 resume`, full transcripts, and the event log. The
`graph-curator` subprocess owns the task graph; the main process writes the
resume snapshots, transcripts, and event log in-process. The run directory is
safe from jailed commands because it lives outside the repo cwd they run on,
not because of any single writer.

## End-of-run notify hook

If `[notify]` declares `on_complete = [...]`, agent6 runs that argv after
every `agent6 run` / `resume` with `AGENT6_RUN_ID`, `AGENT6_RUN_DIR`,
`AGENT6_RUN_OK`, and `AGENT6_RUN_REASON` set. The hook runs outside the
jail as your user; the argv is operator-controlled.

## Contributing

Read [AGENTS.md](AGENTS.md) first. The repo's verify command decides
whether a PR is landable:

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

Changes under `sandbox/`, `tools/`, `git_ops.py`, `providers/`, or
`graph/curator` must include a security review note in the commit message.

## License

[Apache-2.0](LICENSE).
