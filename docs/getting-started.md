# Getting started

This assumes agent6 is already installed (see [installation](installation.md)).

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
```

The key is written to `~/.config/agent6/secrets.toml` (mode `0600`) and is shared across
every repository. `agent6 connect` only stores what you paste or an OAuth token; it never
runs anything a remote returns.

agent6 routes three model roles independently:

| Role       | Set with            | Used by                                   |
| ---------- | ------------------- | ----------------------------------------- |
| `worker`   | `[models.worker]`   | `agent6 run` and `agent6 resume`          |
| `reviewer` | `[models.reviewer]` | `agent6 review` and the in-loop critic    |
| `planner`  | `[models.planner]`  | `agent6 plan` (falls back to `worker`)    |

```sh
agent6 model worker anthropic claude-sonnet-4-6
agent6 model all openrouter moonshotai/kimi-k2.6   # set every role at once
```

## Your first run

```sh
cd your-repo
agent6 run "add a --json output mode to the CLI"
```

agent6 works on a per-run branch, edits files, runs the verify command, and commits each
step that passes. It stops when the model calls `finish_run` or a budget ceiling is hit.

The verify command is the success gate. If the repo has not set
`workflow.verify_command`, agent6 infers one per run (from AGENTS.md, then the repo's
manifest files, then a cheap model call) and prints what it picked. If none can be
inferred the run still proceeds, committing every editing step without a green gate. Pin
one in the per-repo config, or with `agent6 init`, to make it deterministic.

On a TTY, `agent6 run` opens the dashboard. `--no-tui` runs headless; `-i` drives the run
from a stdin REPL.

## Inspect a run

`agent6 runs <verb> [<run-id>]` inspects or merges a run. The id is a positional argument
everywhere (an exact id or an unambiguous prefix); omit it for the most recent run.

```sh
agent6 runs show          # status, iteration, elapsed, cost; --json for scripts
agent6 runs watch         # attach the live dashboard; --plain for a text tail
agent6 runs diff          # the git diff the run produced
agent6 runs commits       # the per-step commits on the run branch
agent6 runs merge         # merge the run branch into your branch (squash/merge/ff)
agent6 runs prune         # delete safely-merged agent6/* run branches; report the rest
agent6 runs transcript    # the full conversation, every tool call with its I/O
agent6 runs graph         # the persisted task graph
```

`agent6 history search <query>` greps across the transcripts of every run.

## When a run goes wrong

```sh
agent6 resume <run-id>                 # continue from the last snapshot
agent6 fork <run-id> --at-turn 7       # branch a new run from turn 7
```

State is snapshotted before each model call and checkpointed per turn. `fork` rolls a
copy back to a turn and continues it as a new run; the original is never changed.

## Plan, review, and ask

```sh
agent6 plan "refactor the config loader"      # read-only plan; run with --from-plan
agent6 review --base origin/main --head HEAD  # read-only diff review
agent6 ask "how does the curator subprocess work?"
```

`agent6 review --reviewers 3 --personas security,correctness,tests` runs a panel of
reviewers whose findings are checked against the diff, so only real problems gate.
`agent6 run --profile ultra` selects a strategy preset (`quick`, `standard`, `ultra`,
`paranoid`).

## Configuration

Config is layered, lowest precedence first: built-in defaults, the global
`~/.config/agent6/config.toml`, the per-repo config (kept out of the workspace, not
committed), then an explicit `--config FILE`. Every field has a default and the
security-sensitive ones default to the safe value, so a repo can be zero-config when the
global config supplies a provider and model. `agent6 config show` prints every effective
value and where it came from. The [configuration reference](config.md) documents each
field.
