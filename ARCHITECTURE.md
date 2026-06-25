# Architecture

This document is a map of how agent6 runs end-to-end. The diagrams
are mermaid (`mermaid` fenced blocks render natively on GitHub). For
per-file conventions and stability rules see [AGENTS.md](AGENTS.md).
For the security model (threat model, defense layers, sandbox profiles),
see [SECURITY.md](SECURITY.md).

## Layering

```
cli  ──▶  workflows  ──▶  agents  ──▶  tools  ──▶  sandbox
                              │
                              └─▶ providers (anthropic | openai)
```

Boundaries are enforced by [tach](https://docs.gauge.sh/) (see
[tach.toml](tach.toml)). Workflows never import each other; agents never
import workflows or the CLI. Crossing a boundary is almost always a
sign of the wrong design.

- **cli** ([src/agent6/cli/](src/agent6/cli/)): argument parsing,
  optional TUI spawn, top-level dispatch. Picks a workflow. Config is
  resolved by [config_layer.py](src/agent6/config_layer.py) (built-in
  secure defaults < global `~/.config/agent6/config.toml` < per-repo
  config < `--config FILE`), with paths + sudo/root
  resolution in [paths.py](src/agent6/paths.py) and API keys in
  [secrets.py](src/agent6/secrets.py). Per-repo state (config and run
  state together) lives out of the workspace under
  `$XDG_STATE_HOME/agent6/<repo-id>/`; the base is settable via the
  global-only `[agent6].state_dir` or the `AGENT6_STATE_HOME` env var.
  Roles: `worker` drives
  `run`/`resume`, `planner` drives `plan` (falls back to `worker`),
  `reviewer` drives `review` + the in-loop critic.
- **workflows** ([src/agent6/workflows/](src/agent6/workflows/)): two
  exist, `loop` (the agent loop driving `agent6 run` / `agent6 resume`)
  and `review` (the read-only review pass driving `agent6 review`).
- **agents** ([src/agent6/agents/](src/agent6/agents/)): single-turn
  LLM call shapes. The only one is `code_review`; the agent loop makes
  its own provider calls inline.
- **tools** ([src/agent6/tools/](src/agent6/tools/)): the fixed
  tool surface the LLM sees, plus dispatch.
- **sandbox** ([src/agent6/sandbox/](src/agent6/sandbox/)): Landlock
  on the agent process, `agent6-jail` for children.

## Workflow: `run`

This is the agent. One provider, one model, one message history. The
model drives by calling tools; the workflow dispatches tools, snapshots
state, and tracks budget.

```mermaid
stateDiagram-v2
    [*] --> snapshot
    snapshot --> llm_call
    llm_call --> dispatch: model emits tool calls
    llm_call --> [*]: budget exhausted
    dispatch --> snapshot: non-terminal tool
    dispatch --> commit: run_verify_command (exit 0)
    commit --> snapshot
    dispatch --> [*]: finish_run
```

Notes:

- **One LLM, one history, one loop.** No planner→worker handoff, no
  critic step, no separate reviewer agent. Multi-step work is the
  model calling the next tool in the same conversation.
- **Snapshot before every LLM call.** A `snapshots/<step>.json` is
  written to the run directory (`<state-dir>/<repo-id>/runs/<run-id>/`,
  out of the workspace) before each provider request.
  `agent6 resume <run-id>` rehydrates from the latest snapshot;
  combined with the per-tool transcripts under `transcripts/`, any
  interrupted run can be replayed deterministically up to the model
  call that comes next.
- **Per-step commits** fire when `run_verify_command` returns 0, via
  `git_ops.py` from outside the jail. Per-step is the default; the
  `git.commit_strategy` knob also allows `squash` (one commit at run
  end), `stage` (stage but never commit), and `none`.
- **DAG-as-tool.** `dag_add_task` / `dag_update_task` /
  `dag_set_cursor` / `dag_list_tasks` write to a curator-owned side
  store. They do not gate the loop; they are notes the worker keeps
  for itself and the user.
- **Context compaction.** Long runs are kept inside the model's context
  window in two tiers (thresholds in `[workflow]`): at
  `compact_drop_at_chars` the oldest tool_results are replaced by a
  short "re-call if needed" placeholder; at `compact_summarise_at_chars`
  the elided history is summarised by the `reviewer` model and the
  conversation restarts from (task + summary). The curator-owned task
  DAG survives the restart, so the worker recovers task-level state with
  `dag_list_tasks` instead of starting over.
- **`finish_run(summary)`** is the only terminal tool. Calling it
  emits a `run.end` event and returns control to the CLI.

## Workflow: `review`

A single read-only pass ([src/agent6/workflows/review.py](src/agent6/workflows/review.py))
over a diff (working tree, branch-vs-base, or arbitrary range) using
the `agents/code_review.py` agent. Produces structured findings; no
edits, no commits, no `run_command`.

```mermaid
stateDiagram-v2
    [*] --> collect_diff
    collect_diff --> code_review
    code_review --> [*]
```

## Enforcement layering

[SECURITY.md](SECURITY.md) details which guarantee each layer provides.
As a diagram:

```mermaid
flowchart TD
    LLM[LLM choice of tool] --> Tools[tools/dispatch.py]
    Tools -->|apply_edit, apply_patch, read, list, grep, outline| FS[(workspace fs)]
    Tools -->|run_verify_command, run_metric_command, run_command| Jail[agent6-jail]
    Jail --> NS[user/mount/pid/ipc/uts/net NS]
    Jail --> Pivot[pivot_root into minimal rootfs]
    Jail --> ROBinds[strict only: RO bind .git]
    Jail --> Land[Landlock V1 rules]
    Jail --> Sec[seccomp filter]
    Jail --> Caps[capset 0 + NO_NEW_PRIVS]
    Land -.-> Child[child process]
    Sec -.-> Child
    ROBinds -.-> Child
    Caps -.-> Child
    Workflow[workflow git_ops.py] -->|outside jail| Git[(.git)]
    Workflow -. blocks .-> Push[push / --force / reset --hard]
```

- `git_ops.py` runs outside the jail (the agent's own process), so
  the RO bind of `.git` does not stop the workflow from committing. It
  stops the worker.
- `protect_git` is strict-only. On strict the jail read-only
  bind-remounts `.git` on top of the workspace mount. The hardened
  profile (no mount namespace to carve with) grants blanket read-write
  on the repo cwd, so `.git` is writable by jailed commands there.
  Carving `.git` read-only on hardened would also deny new top-level
  entries and break toolchains like cargo/pytest that create `target/`
  or `.pytest_cache/`. The writable `.git` on hardened is acceptable:
  it is gated by `run_commands` (default `ask`), recoverable
  (branch-per-run, commits go through `git_ops`), and the surrounding
  container is the blast radius.
- Run state is safe from jailed commands because it lives out of the
  workspace (`<state-dir>/<repo-id>/`), unreachable from the repo cwd
  that jailed commands run on.

## Curator subprocess

The task graph is owned by a separate `graph-curator` subprocess
(`python -m agent6.graph.server`). The
main agent process writes the rest of the run state (resume snapshot,
event log, transcripts) in-process.

```mermaid
flowchart LR
    Agent[agent6 run<br/>main process] -->|UDS JSON IPC| Curator[graph-curator<br/>subprocess]
    Curator -->|task graph| Graph[(graph.jsonl, graph/*.md, graph snapshots)]
    Agent -->|in-process| Rest[(loop_state.json, logs.jsonl, transcripts)]
```

The agent talks to the curator over a Unix domain socket. The curator
validates every IPC frame against a pydantic schema before applying it,
so the on-disk graph stays consistent. What keeps the whole run
directory safe from jailed commands is its location: it lives out of the
workspace (`<state-dir>/<repo-id>/`), unreachable from the repo cwd that
jailed commands run on.

## Run state on disk

Each run's directory `<state-dir>/<repo-id>/runs/<run-id>/` holds:

- `graph.jsonl`: append-only journal of every task-graph mutation
  (curator-owned).
- `graph/*.md`: one markdown file per task node, rewritten atomically
  (curator-owned).
- `logs.jsonl`: the structured event stream (below), written by the
  main process.
- `loop_state.json`: the resume snapshot that drives `agent6 resume`,
  written by the main process.
- `transcripts/`: full provider request/response pairs for replay,
  written by the main process.

The `logs.jsonl` vocabulary is small and stable: the data contract for
any external viewer (the fold to UI state lives in
[src/agent6/ui/state.py](src/agent6/ui/state.py) as a pure function):

| Event                       | Notable fields                              |
| --------------------------- | ------------------------------------------- |
| `run.start`                 | `user_task`                                 |
| `tool.call` / `.result`     | `name`, `args` (preview), `ok`, `summary` — emitted as a pair for EVERY dispatched tool, including ones a guard rejects (`ok=false`, trusted reason), so no call is unaccounted for. Execution tools (`run_command`/`run_metric_command`) also carry capped `stdout_tail`/`stderr_tail` like `verify.end` |
| `verify.start` / `.end`     | `cmd`, `exit_code`, `duration_s`, `*_tail`  |
| `loop.verify_inferred`      | `command` (argv, `[]` if none), `source` (`agents_md`/manifest/`llm`/`none`) |
| `role.call` / `.result`     | `role`, `model`, `tokens_in`, `tokens_out`  |
| `role.text_delta`           | streamed assistant text chunk               |
| `role.thinking_delta`       | streamed reasoning chunk (TUI "thinking" pane) |
| `run.steer_requested`       | `source` (`"sigint"`): mid-run Ctrl-C       |
| `budget.update`             | totals + caps for input/output tokens       |
| `approval.prompt`/`.answer` | `id`, `prompt`, `approved`, `source` (`tui`/`stdin`) |
| `loop.*`                    | agent progress: `loop.auto_commit`, `loop.compact.*`, `loop.critic.*`, `loop.metric.*`, `loop.steer.*` |
| `loop.budget`               | per-iteration usage heartbeat: `iteration`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cost_usd` (read by `agent6 status`) |
| `loop.review.*`             | adversarial review panel: `loop.review.start` (trigger, seats), `loop.review.seat` (seat, model, verdict, findings), `loop.review.panel` (blocked, raw_blocked, decision, n_block, disarmed), `loop.review.skipped` |
| `run.end`                   | `summary`                                   |

A `run_command` approval is published as `approval.prompt`; the dashboard
TUI shows an Allow/Deny modal and writes `approvals/<id>.answer`, which the
workflow reads (falling back to a stdin prompt with no TUI), then records
`approval.answer`. The task DAG is not in this stream; it is
curator-owned and lives in `graph.jsonl` (read via `agent6 history
graph`).

## Where things live

| Concern                          | File / dir                                                            |
| -------------------------------- | --------------------------------------------------------------------- |
| Config schema                    | [src/agent6/config.py](src/agent6/config.py)                          |
| Tool surface                     | [src/agent6/tools/schema.py](src/agent6/tools/schema.py)              |
| Tool dispatch                    | [src/agent6/tools/dispatch.py](src/agent6/tools/dispatch.py)          |
| agent loop                       | [src/agent6/workflows/loop.py](src/agent6/workflows/loop.py)          |
| Review workflow                  | [src/agent6/workflows/review.py](src/agent6/workflows/review.py)      |
| Code-review agent                | [src/agent6/agents/code_review.py](src/agent6/agents/code_review.py)  |
| Jail launcher (Python wrapper)   | [src/agent6/sandbox/jail.py](src/agent6/sandbox/jail.py)              |
| Jail launcher (Rust binary)      | [src/agent6/jail/src/main.rs](src/agent6/jail/src/main.rs)            |
| Git policy                       | [src/agent6/git_ops.py](src/agent6/git_ops.py)                        |
| Provider clients                 | [src/agent6/providers/](src/agent6/providers/)                        |
| Knowledge graph (curator)        | [src/agent6/graph/](src/agent6/graph/)                                |
| Event log + UI fold              | [src/agent6/events.py](src/agent6/events.py), [src/agent6/ui/](src/agent6/ui/) |
| Run state on disk                | `<state-dir>/<repo-id>/runs/<run-id>/` (out of the workspace)         |

## Pre-1.0 stability

See [AGENTS.md](AGENTS.md). Until 1.0 every public shape (config TOML,
IPC frames, on-disk graph, CLI flags, transcript layout) is liquid;
we break cleanly rather than carry shims.
