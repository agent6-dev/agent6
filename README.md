# agent6

A sandboxed coding agent for Linux, tuned to stay effective on cheap
open-weight models (Kimi, GLM, Qwen) as well as Claude. The LLM is treated as
adversarial: every command it spawns runs inside a custom Rust launcher
(`agent6-jail`) built on user namespaces, Landlock, seccomp, `pivot_root`,
`capset(0)`, and `NO_NEW_PRIVS`, so you can point a weaker or untrusted model at
a real repository and it cannot escape the workspace, reach the network beyond
the provider endpoint, or corrupt git history. Runs commit per step and are
resumable and forkable, so an interrupted or wrong turn is never a dead end.

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
agent6 model worker anthropic claude-sonnet-4-6

# Run the agent on a task -- that's it. agent6 infers a verify command for
# the repo if you haven't set one.
agent6 run "add a --json output mode to the CLI"

# Optional: a granular setup wizard (per-repo config, a pinned verify
# command, .gitignore, AGENTS.md). Safe to run anytime; never overwrites.
agent6 init

# Audit the effective config: every value and where it came from.
agent6 config show

# Pre-flight: sandbox + config + provider keys.
agent6 check

# Resume an interrupted run from its last tool-call snapshot.
agent6 resume <run-id>

# Go back and try a different direction: clone a run, rolled back to a turn,
# into a NEW run (the original stays intact) and continue it.
agent6 fork <run-id> --at-turn 7

# Read-only code review of a diff. Never touches the worktree.
agent6 review --base origin/main --head HEAD

# Adversarial review PANEL: N grounded reviewers (findings grounded against the
# diff, so only real, block-eligible problems gate). Also runs in-loop.
agent6 review --reviewers 3 --personas security,correctness,tests

# Pick a strategy preset with one knob (quick / standard / ultra / paranoid).
agent6 run "..." --profile ultra
```

The in-loop review panel lives in the `[review]` section (`trigger`,
`panel_size`, `seats`, `decision`, `personas`); the strategy preset is the
top-level `profile` key. See [CONFIG.md](CONFIG.md).

Config is layered: built-in secure defaults, then the global
`~/.config/agent6/config.toml`, then the per-repo config (out of the
workspace under `$XDG_STATE_HOME/agent6/<repo-id>/config.toml`), then an
explicit `--config FILE`. The per-repo config is per-machine, not committed.
A repo can be zero-config when the global config supplies a provider and
model. The verify command (agent6's success gate) is optional: if a repo
hasn't set `workflow.verify_command`, `agent6 run`/`plan` infer one per run
(from AGENTS.md, then repo manifests, then a cheap model call) and print what
they picked; with none inferable the run proceeds gateless (per-step commits,
no green gate). Pin one in the per-repo config — or via `agent6 init` — to make
it deterministic.

Other commands:

- `agent6 runs <verb> [<run-id>]`: inspect a specific run (run id is a
  positional everywhere, exact or unambiguous prefix; omit for the most recent):
  - `runs show`: one-shot liveness + progress (running / crashed / finished,
    current iteration, last activity, elapsed), then exit — a quick or scripted
    check (`--json`) without the live follower.
  - `runs watch`: attach the live TUI; `--plain` is a plain-text tail.
  - `runs diff`: print the git diff a run produced.
  - `runs transcript`: render a run's full LLM conversation (assistant text +
    every tool call with complete I/O) as Markdown; `--json` for the raw
    transcript. The lossless deep-dive, vs the terse `logs.jsonl`.
  - `runs graph`: render the persisted task graph.
  - `runs show` also prints fork lineage (`forked from <parent>@turn N`) when
    the run was created by `agent6 fork`.
- `agent6 fork <run-id> [--at-turn N] [--run-id NEW] [--no-run]`: clone a run,
  rolled back to checkpoint turn N (default: the latest), into a NEW run with a
  new id and the same repo, recording lineage (parent run + the turn). The
  source run is never mutated — this is "sessions as trees" done as
  clone-to-new-session, not in-place branching. It cuts `agent6/<NEW>` at the
  turn's sha (your checkout stays put) and, by default, continues the new run
  from turn N (reusing the resume path); `--no-run` just creates it. Each run
  writes a per-turn `checkpoints/<NNNN>.json` so any turn is forkable; a run
  from before this feature forks from its latest snapshot only.
- `agent6 plan "<task>"`: read-only planning pass; execute with
  `agent6 run --from-plan`. Inspect with `plan show <id>` / `plan edit <id>`.
- `agent6 ask "<question>"`: read-only Q&A over the repo, including
  questions about agent6 itself (it consults its bundled docs). Seed
  context with `@path` or `--file`; `--run <id>` (or `--seed-latest`) asks
  about a prior run. `ask list` enumerates saved asks.
- `agent6 memory`: persistent agent memory under the per-repo state dir
  (`<state-dir>/<repo-id>/memories/`).
- `agent6 history search <query>`: cross-run search over persisted transcripts.
- `agent6 machine ...`: author and run state machines (`.asm.toml`); see
  [STATE_MACHINES.md](STATE_MACHINES.md).
- `agent6 mcp serve`: expose agent6's tools over MCP (stdio).
- `agent6 config fill`: materialize every effective value into one file.
- `agent6 config get/set/unset/add/remove <key> [value]`: edit a single
  dotted leaf. Writes go to the global config by default, `--repo` for the
  repo, `--machine-file FILE` for a machine overlay. Every edit is re-validated
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
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-6"
```

Budget ceilings can be overridden per run: `agent6 run --max-usd 5 "..."`,
or `--max-input-tokens` / `--max-output-tokens` on `run`, `plan`, and
`resume`.

### Providers and models

Declare any number of `[providers.<name>]` blocks, each with
`api_format = "anthropic"` or `api_format = "openai"`, its own `base_url`,
and `api_key_env`. Per-provider `http_timeout_s` (default 600) caps each HTTP
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

### What the agent can set up

agent6 can explore a repo, read and edit it, and run an *already-installed*
toolchain (compiler, test runner, build tool) jailed via `run_command` /
`run_verify_command`. It **cannot** install system packages or use `sudo` from
inside the jail — `NO_NEW_PRIVS` neuters `sudo` (even passwordless), egress is
confined to your provider, and writes outside the workspace are blocked. So the
operator provisions the environment (install the toolchain, create the venv,
fetch deps with your own shell) and agent6 works within it; widen access through
config (`sandbox.extra_read_paths`, `sandbox.tool_network`), never sudo. See
[SECURITY.md](SECURITY.md) §2a.

## How it works

agent6 is a single-loop agent: one provider, one model, one message
history. The model drives the run by calling tools; the workflow dispatches
them, snapshots state before every LLM call (so any run is resumable),
commits each step when `verify_command` passes (or, on a gateless run with no
verify command, every editing step), and hard-stops on budget. Module
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

### Why not just run an agent in a devcontainer?

A container is one coarse blast radius for the whole agent: anything it does,
including `git push --force` or reaching the network from a build step, is
allowed inside that box. agent6 instead jails every child process the model
runs individually (Landlock + seccomp + `pivot_root`), rebinds `.git`
read-only, and confines egress to the provider endpoint, so a misbehaving model
cannot corrupt history or exfiltrate even from inside a command it chose to run.
The two compose: agent6 runs fine inside a devcontainer, where its `hardened`
profile uses the container as the filesystem boundary and still applies the
per-process network and git protections. See [SECURITY.md](SECURITY.md).

## Benchmarks

Reproducible harnesses live under [bench/](bench/):

- [bench/realworld/](bench/realworld/): 11 SWE-bench-Lite-style tasks
  scored by fresh sandboxed verifies on hidden tests. Latest recorded run:
  agent6 and claude-code both solve 11/11 on the same worker model
  (`claude-sonnet-4-5`). Cost is profile-dependent: agent6 ran $2.60 under a
  tight per-task budget cap and $8.45 when each task was free to optimize
  toward the cap, vs claude-code at $3.96; the same suite on the open-weights
  `kimi-k2.6` worker solved 11/11 for $1.20. Single runs, no variance measured;
  re-run before quoting.
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

As a rough anchor: a small, well-scoped task on an open-weight worker (Kimi or
GLM via OpenRouter) typically runs well under $1. Cap any run with
`agent6 run --max-usd N` (a hard stop on priced models); a frontier model costs
more per token but usually finishes a task in fewer turns.

## Live view

With stdout a TTY, `agent6 run` opens a terminal dashboard (task DAG,
budget bar, tool table, live reasoning pane, log tail, latest diff);
`--no-tui` and `-i` (stdin REPL) opt out. Approval and Ctrl-C steer
prompts appear as modals, with a `/dev/tty` fallback when no TUI is
present. `agent6 plan`, `agent6 ask`, and `agent6 machine create` stream
reasoning and answers to the terminal. Attach from another shell with
`agent6 runs watch [<run-id>]`; `agent6 runs watch --plain` is a plain-text tail.
The dashboard renders the JSONL event stream at
`<state-dir>/<repo-id>/runs/<run-id>/logs.jsonl`, which is also the contract
for external viewers (vocabulary in [ARCHITECTURE.md](ARCHITECTURE.md)). The
inline log pane is a live tail; press `l` (or pick a run in the hub and press
`l`) for a full-height, scrollable log of the whole run — current or finished.

## Persistence

Per-repo state lives out of the workspace under `$XDG_STATE_HOME/agent6/<repo-id>/`
(override with `[agent6].state_dir` or `AGENT6_STATE_HOME`). Each run's state
sits under `runs/<run-id>/`: the append-only task graph, the latest per-call
snapshot (`loop_state.json`) that drives `agent6 resume`, an append-only
per-turn `checkpoints/<NNNN>.json` store (each carrying that turn's workspace sha
and DAG version) that `agent6 fork` rolls back to, full transcripts, and the
event log. The `graph-curator` subprocess owns the task graph; the main process
writes the resume snapshots, checkpoints, transcripts, and event log in-process.
A per-repo `lineage.jsonl` at the state-dir root records every fork edge
(`child`, `parent`, `turn`, `sha`). The run directory is safe from jailed
commands because it lives outside the repo cwd they run on, not because of any
single writer.

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
