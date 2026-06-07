# Architecture

This document is a map of how agent6 runs end-to-end. The diagrams
are mermaid (`mermaid` fenced blocks render natively on GitHub). For
per-file conventions and stability rules see [AGENTS.md](AGENTS.md).
For the security model see [SECURITY.md](SECURITY.md) and the threat
model section of [README.md](README.md).

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

- **cli** ([src/agent6/cli.py](src/agent6/cli.py)) — argument parsing,
  optional TUI spawn, top-level dispatch. Picks a workflow. Config is
  resolved by [config_layer.py](src/agent6/config_layer.py) (built-in
  secure defaults < global `~/.config/agent6/config.toml` < per-repo
  `.agent6/config.toml` < `--config FILE`), with paths + sudo/root
  resolution in [paths.py](src/agent6/paths.py) and API keys in
  [secrets.py](src/agent6/secrets.py). Roles: `worker` drives
  `run`/`resume`, `planner` drives `plan` (falls back to `worker`),
  `reviewer` drives `review` + the in-loop critic.
- **workflows** ([src/agent6/workflows/](src/agent6/workflows/)) — two
  exist: `loop` (the agent loop driving `agent6 run` / `agent6 resume`)
  and `review` (the read-only review pass driving `agent6 review`).
- **agents** ([src/agent6/agents/](src/agent6/agents/)) — single-turn
  LLM call shapes. The only one is `code_review`; the agent loop makes
  its own provider calls inline.
- **tools** ([src/agent6/tools/](src/agent6/tools/)) — the fixed,
  audited tool surface the LLM sees, plus dispatch.
- **sandbox** ([src/agent6/sandbox/](src/agent6/sandbox/)) — Landlock
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
  written to `.agent6/runs/<run-id>/` before each provider request.
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

The threat model in [README.md](README.md) summarizes which guarantee
each layer provides. As a diagram:

```mermaid
flowchart TD
    LLM[LLM choice of tool] --> Tools[tools/dispatch.py]
    Tools -->|apply_edit, apply_patch, read, list, grep, outline| FS[(workspace fs)]
    Tools -->|run_verify_command, run_metric_command, run_command| Jail[agent6-jail]
    Jail --> NS[user/mount/pid/ipc/uts/net NS]
    Jail --> Pivot[pivot_root into minimal rootfs]
    Jail --> ROBinds[RO binds: .git, .agent6/ config + run state]
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

Two things are worth calling out:

- `git_ops.py` runs **outside** the jail (the agent's own process), so
  the RO bind of `.git` does not stop the workflow from committing. It
  stops the worker.
- `protect_git` / `protect_agent6` work in both profiles. Strict uses
  a bind-remount-RO on top of the workspace mount. Hardened (no
  mount namespace) switches its Landlock setup from "RW on cwd" to
  "R on cwd + RW on each top-level entry except the protect set".
  Same end result for paths present at jail-launch time; hardened
  additionally denies writes to *new* top-level entries created at
  the cwd root (anything inside an existing top-level dir is
  unaffected by the carve-out).

## Curator subprocess

Run state (graph + transcripts + logs) is owned by a separate
`agent6-curator` subprocess, not by the main agent process.

```mermaid
flowchart LR
    Agent[agent6 run<br/>main process] -->|UDS JSON IPC| Curator[agent6-curator<br/>subprocess]
    Curator -->|sole writer| RunDir[(.agent6/runs/&lt;run-id&gt;/)]
    Curator -. own jail policy .-> JailC[agent6-jail<br/>RW only on .agent6/]
```

The agent talks to the curator over a Unix domain socket. The curator
runs under its own jail policy that allows writes only to `.agent6/`.
This means even a bug in the agent process cannot scribble over the
run directory in an unsafe way; the curator validates every IPC frame
against a pydantic schema before applying it.

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
| Run state on disk                | `.agent6/runs/<run-id>/`                                              |

## Pre-1.0 stability

See [AGENTS.md](AGENTS.md). Until 1.0 every public shape (config TOML,
IPC frames, on-disk graph, CLI flags, transcript layout) is liquid;
we break cleanly rather than carry shims.
