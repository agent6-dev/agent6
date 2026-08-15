# Getting started

This assumes agent6 is already installed (see [installation](installation.md)).

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
```

The key is written to `~/.config/agent6/secrets.toml` (mode `0600`) and is shared across
every repository. `agent6 connect` only stores what you paste or an OAuth token; it never
runs anything a remote returns. If already connected, skip this step;
`agent6 check` verifies the stored key and `agent6 model` shows the role assignments.

agent6 routes three model roles independently:

| Role       | Set with            | Used by                                   |
| ---------- | ------------------- | ----------------------------------------- |
| `worker`   | `[models.worker]`   | `agent6 run` and `agent6 resume`          |
| `reviewer` | `[models.reviewer]` | `agent6 review` and the in-loop review panel |
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

agent6 edits files in your working tree, runs the verify command, and commits
each passing step to a per-run commit chain (plus an `agent6/<id>` branch by
default); your branch, HEAD, and index are never touched. It stops when the model calls `finish_session` or a budget ceiling is hit.

At a terminal the session then asks for the next input rather than ending: type
the next instruction to continue in the same session (no `agent6 resume` to
retype), or `/exit` to finish. Finishing leaves the session resumable like any
other; without a terminal (CI, a detached run) the resume line is printed
instead.

The verify command is the success gate. If the repo has not set
`workflow.verify_command`, agent6 infers one per run (from AGENTS.md, then the repo's
manifest files, then a model call over those manifests -- skipped when there are none)
and prints what it picked. If none can be inferred the run still proceeds, committing
every editing step without a green gate. Pin one in the per-repo config, or with
`agent6 init`, to make it deterministic.

`agent6 run` is headless by default: it streams the run in your terminal. `--tui` opens
the full-screen TUI instead (the run's conversation, with the dashboard on Ctrl+D);
`-i` drives the run from a stdin REPL.

## Inspect a run

`agent6 attach [<target>]` follows live: a run renders its conversation (the same view
as `agent6 run`; `--raw` tails the plain event stream), a machine streams its state
overview and reasoning; `--tui` opens the full-screen TUI instead. `agent6 sessions <verb> [<session-id>]` inspects or merges a run. The id is a positional
argument everywhere (an exact id or an unambiguous prefix); omit it for the most recent run.

```sh
agent6 attach              # follow the conversation live; --raw, --tui, or --json instead
agent6 sessions show          # status, iteration, elapsed, cost; --json for scripts
agent6 sessions diff          # the git diff the run produced
agent6 sessions commits       # the run's per-step commits
agent6 sessions merge         # land the run's work on your branch (squash/merge/ff)
agent6 sessions prune         # delete safely-merged agent6/* run branches; report the rest
agent6 sessions dir           # where this repo's run history lives (one line, scriptable)
agent6 sessions rm            # delete one run's history; --asks clears every saved ask
agent6 sessions compare <id> <id> ...  # advisory ranked comparison across >=2 runs
agent6 sessions transcript    # the full conversation, every tool call with its I/O
agent6 sessions graph         # the persisted task graph
```

`agent6 history search <query>` greps across the transcripts of every run.

## When a run goes wrong

```sh
agent6 resume <session-id>                 # continue from the last snapshot
agent6 fork <session-id> --at-turn 7       # branch a new run from turn 7; --steer "try X" seeds the new direction
```

State is snapshotted before each model call and checkpointed per turn. `fork` rolls a
copy back to a turn and continues it as a new run; the original is never changed.

**Exit codes** (`agent6 run` / `resume`; scripts branch on these): `0` finished
with a green gate or nothing to gate on; `1` the run broke (crash, provider
error); `2` operator error (bad flag or config); `3` budget exhausted;
`4` finished over a red or never-run verify gate; `5` finished but no commit
landed and the edits sit uncommitted in the working tree (a run that changed
nothing stays `0`); `130` interrupted.

## Plan, review, and ask

```sh
agent6 plan "refactor the config loader"      # edit-free plan; run with --from-plan
agent6 plan edit <session-id>                 # answer the plan's open questions in plan.md
agent6 resume <session-id> --steer "answered" # the planner re-reads plan.md and revises
agent6 review --base origin/main --head HEAD  # read-only diff review
agent6 ask "how does the task-graph curator work?"
```

- `agent6 review --reviewers 3 --personas security,correctness,tests`: a panel
  whose findings are checked against the diff, so only real problems gate.
- `ask` works anywhere; `run` and `plan` require a git repository (branches,
  diffs, merges).
- `--preset quick|standard|ultra|paranoid` selects a strategy;
  `agent6 config presets` lists them, `agent6 config set preset <name>`
  persists one.
- `agent6 run "task" --parallel 3` (or `model-a,model-b`) fans out isolated
  lanes and prints a ranked comparison; the same fan-out spawns from the
  TUI/web composers or mid-run with the `/parallel [spec] <task>` steer
  directive. Details: [configuration](config.md#parallel).
- `agent6 run "task" --standing "hunt and fix bugs"` adds a standing goal:
  a never-finishing fallback task the run re-enters whenever the ordinary
  queue drains or the worker tries to stop. New work always outranks it
  (insert it with `add_task`; the model can also create one with
  `add_task(standing=true)`), a standing task never passes (retire it with
  skipped/obsolete), and the run still ends on its budget, an operator
  stop, or its iteration cap.

## Configuration

Config is layered, lowest precedence first: built-in defaults, the global
`~/.config/agent6/config.toml`, the per-repo config (kept out of the workspace, not
committed), then an explicit `--config FILE`. Every field has a default and the
security-sensitive ones default to the safe value, so a repo can be zero-config when the
global config supplies a provider and model. `agent6 config show` prints every effective
value and where it came from. The [configuration reference](config.md) documents each
field.
