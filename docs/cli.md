# Command line

`agent6 --help` lists every command, and `agent6 <command> --help` its flags.
This page is the reference for the command-line surface; the patterns behind the commands are on [Usage](usage.md).

## Run options

`agent6 run` streams in your terminal; `--tui` opens the full-screen view ([terminal UI](terminal.md)) and `-i` pauses for input after each automatic commit.
`--tui` without a terminal continues in CLI mode; `-i` requires a foreground terminal and cannot be combined with `--tui`.

- `--disable-sandbox`: run commands on the host without filesystem, syscall or namespace confinement; use only on a disposable or already-isolated machine.
  Commands that can run commands share this opt-out: `run`, `resume`, `fork`, `plan`, `ask`, `exec` (including `--new`) and `machine run`.
  Place it before the session or command in `exec`; it overrides recorded isolation for that command only.
  `AGENT6_DISABLE_SANDBOX=1` applies the same opt-out across verbs.

- `--preset <name>`: a strategy preset (`no_review`, `review`, `deep_review`, or your own; `none` selects no preset), fixed for the execution; `resume --preset` switches ([presets](config.md#presets))
- `--model [provider/]model`: the model for the mode's role, over every config layer; a bare id keeps the role's provider, and a resume keeps the recorded route unless it sets its own
  The web and TUI resume composers default to that route and offer the config's routes as choices.
  A resume that changes the route prints both the chosen and recorded routes.
- `--parallel 3` (or `reviewer,provider/model-a`: a role name brings its model, effort and temperature; a `[provider/]model` runs with the worker's): isolated lanes, auto-compared into a ranked report; the fan-out is a session of its own that `attach`, `stop` and `sessions show` address, its lanes folded under it (`--lanes` expands them) ([parallel](config.md#harnessparallel))
- `--worktree`: from the repository's top level, the run works in its own git worktree at HEAD's commit, under `[git.worktree].base_dir`
    - your checkout and its uncommitted changes stay as they are and are not carried, and neither is anything git does not track ([worktrees](usage.md#worktrees)), so you can keep working there or start more runs
    - `sessions merge` lands the work and `sessions prune` then removes the worktree, as it does for a run that committed nothing and left no edits
- `--standing "<goal>"`: a fallback goal the run returns to whenever its queue drains; only the operator retires it, and ceilings still end the run (`harness.standing_patience`)
- `--pin "<text>"`: an instruction re-shown after every compaction
- `--from <id>`: seed from another session's task, outcome, diff and plan; a plan id alone runs that plan, and `sessions show` prints `seeded from`
    - a plan whose open questions have blank `**A:**` lines still runs, after one warning that names them; `agent6 plan edit <id>` answers them first
- `--decompose`: a task graph up front; `--skill NAME`: a skill in the prompt (a skill whose text waits on a person stalls an unattended run in `ask_user`; `agent6 skills disable NAME` keeps it out)
- `--session-id ID`: name the session yourself
- `agent6 prompt show [--mode run|plan|ask|read_only] [--json]`: preview the system prompt, built-in tools and first message with a task placeholder; MCP tools are discovered at run start and are not included

### Commands inside a dev container

Set `sandbox.isolation = "container"` to run setup, verification and the model's commands in your dev container.
agent6 finds the running Docker or Podman dev container of this workspace by its labels, without Node or the devcontainers CLI; start it in VS Code, with your own CLI, or with `container.start`.
`container.exec` names another container, `container.workspace` another path inside it, and `container.runtime` the runtime CLI (`["sudo", "-n", "docker"]` where docker needs sudo).
File tools, credentials and session records stay on the host; the start block shows the container, both roots, the user, HOME, who owns the container and every control the container does not enforce.
The receipt names the environment each check ran in and whether the container was left running; `container.stop` runs only for a container the run started, after every attached run ended.
Lanes, review seats, forks and `--worktree` runs get their worktrees under `container.worktree_base_dir` (`.agent6/worktrees` in the project), which the container mounts; agent6 keeps that directory out of git through `.git/info/exclude`.
With `git.worktree.kind = "linked"`, a fork or a `--worktree` run refuses when the container mounts the workspace at another path than the host's, because a linked worktree names its repository by the host path; a clone, the default, runs at any path.
`agent6 exec` refuses under container isolation (use the runtime's own exec).

```toml
[sandbox]
isolation = "container"

[container]
runtime = ["sudo", "-n", "docker"]
```

### Go build cache

Go's build cache lives under the jail's HOME, not your own.
Under `full` `sandbox.isolation`, the default `sandbox.home = "auto"` discards that HOME with the run, so a new run starts with a cold cache.
To reuse compiled packages across runs in this repository:

```sh
agent6 config set --repo sandbox.home cache
agent6 exec --new -- go test ./...
```

The first build fills the private cache; later runs reuse it.
Dependencies must already be available offline, or the operator must populate the cache with a network policy that permits fetching them.
`go test -count=1 ./...` reruns tests while still reusing compiled packages.
Under `partial` or `none` `sandbox.isolation`, `auto` already uses the persistent HOME.
Persistence also lets a poisoned cache or tool configuration reach later runs in this repository; see the [sandbox boundaries](security.md#threat-model).

## Foreground runs

`agent6 run` executes in the foreground; Ctrl-C opens its pause menu to steer it.

- the menu Tab-completes its commands; Up recalls, Ctrl-R searches past messages; a steer sent from another surface while it is open is taken as the answer
    - the [steer directives](usage.md#follow-and-steer-a-run) plus `/status`, `/tasks`, `/answer`, `/shells`, `/restate`, `/continue`, `/stop`, `/exit`, `/detach`, `/help`
- `/detach` hands the run to the background after its current turn (`agent6 attach` reattaches); Ctrl-Z prints the run's state and never suspends it
- `/exit` leaves the run resumable; `/stop`, or Ctrl-C at the pause prompt, stops it now, as `agent6 stop ID` does
- `run -i` prompts after every commit: `/continue` (bare Enter), `/cost`, `/diff`, `/watch`, `/mcp`, `/init`, `/undo`, `/help`, `/quit`, `/exit`
- hub-started runs run detached; `agent6 attach` follows them (conversation by default; `--tui`, `--raw`)

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/temps-demo.webm" type="video/webm">
</video>

## Follow and steer a run

`agent6 attach [<id>]` follows a run's conversation, or a machine's state, live.
It rings the bell and sends a desktop notification (`notify-send`, when present) as the run starts waiting on an answer and as it ends; `run` does the same in the foreground.
Ids are exact or an unambiguous prefix and default to the most recent run.
A prefix selects the single matching session whose full id prefixes every other match, provided the others have its mode or are its review seats (for example, a parallel coordinator and its lanes).
Lanes remain reachable by their own ids.

```sh
agent6 attach                 # follow live; --raw tails events, --tui full screen, --json one snapshot
agent6 steer ID "focus on X"  # an instruction at the next step; --now interrupts the call in flight
agent6 answer ID "yes"        # answer a pending question or approval (bare: list them; --question QID picks a question set)
agent6 stop ID                # stop during the current turn (the worker killed if it does not answer); --all
agent6 stop ID --after-turn   # finish the current turn, then stop
agent6 ps                     # live sessions of every repository on the machine; --lanes, --json
```

## Inspect and land a session

```sh
agent6 sessions show          # status, turn, elapsed, usage, the gate's word, where the changes are; --json
agent6 sessions diff          # the git diff the run produced; --stat, --path P
agent6 sessions commits       # the run's per-turn commits
agent6 sessions merge         # land on the run's starting branch (warns first when the gate did not certify it); --method, --into BRANCH
agent6 resume ID --merge main # merge main into the run's work first; a conflict's markers are the run's first task
agent6 sessions transcript    # the conversation as text; --no-thinking, --seq N, --tools calls|none
agent6 sessions graph         # the task graph, each line led by the id /retire takes
agent6 sessions compare <ids> # >=2 runs ranked by the reviewer model (a mechanical ranking when it is unavailable or fails), or a fan-out's saved verdict; --rejudge
agent6 sessions dir [<id>]    # the repo's run history, or that session's directory
agent6 history search <text>  # grep this repository's sessions' persisted data; --regex
```

`history search` searches the conversations, not the system prompt, the skill index or the tool declarations.
It keeps the matches it read if a file disappears or another search error occurs, prints `search incomplete: some files were not read.` and the error, and exits with code 2.
It does not report `no matches` when the search is incomplete.

`sessions show --json`, `sessions list --json` and `attach --json` return the operator's full words in `task`, the latest resume steer in `continued_task` and their headlines in `task_line`; [Formats](formats.md#cli-json) lists every field and the clipping rules.
An unsteered resume clears `continued_task`.

## Clean up sessions

```sh
agent6 sessions prune         # delete merged agent6/* branches and worktrees, report the rest
agent6 sessions prune --history --older-than 30 # delete stored transcripts, checkpoints and objects of sessions ended more than 30 days ago
agent6 sessions remove [<id>] # delete one run's history; asks [y/N] on a controlling terminal, otherwise requires an id or --yes; --yes skips the question, --asks clears saved asks
```

`sessions prune` removes merged worktrees, including squash and task merges (`tasks` is the default), but keeps worktrees holding work no commit has.
A run you landed by hand (a squash, a cherry-pick, a rebase) has no merge record: prune records it as merged when its base branch already holds its content, and says so (`<ref>: main already has its content; recorded as merged into main`).
When that base branch is gone, prune checks the checked-out branch.
A later change of the same lines on the base hides the landing: such a run stays unmerged until you delete its ref.
Only deletion of their unreachable branches and chain refs needs `--delete-squashed`.

`sessions prune --history --older-than DAYS` removes `transcripts/`, `checkpoints/` and `objects/` together from finished sessions whose last execution ended more than DAYS ago, prints each session's id and file counts, and leaves a `pruned.json` receipt; the manifest, logs, receipt, recovery state and graph store stay.
Nothing is deleted automatically, and there is no retention setting.
The [Formats reference](formats.md#provider-transcripts) describes the object store and the receipt.

At launch, a note names the state directory when its filesystem has less than 10% free; low space never refuses a session.

## Run a command in the sandbox

```sh
agent6 exec ID -- <command>   # a command inside the run's jail and network, on your terminal or pipe
agent6 exec --new [<command>] # a command, or your shell, in a fresh sandbox built from the config; no run needed
agent6 forward ID 8000        # reach a port inside the run's network; --local-port N
```

`agent6 exec` runs your own command on your own stdio.

- with stdin and stdout on a terminal the command gets a terminal of its own, with job control; otherwise stdin, stdout and stderr pass through, so pipes and redirects work
- Ctrl-Z stops a job of the shell you opened; a command you run directly keeps running
- a command a signal ended reports 128 plus the signal, at every `sandbox.isolation` level
- `agent6 exec ID -- COMMAND...` works in the run's workspace: a file your command writes there is in the run's tree, and the run's next commit holds it

## Unattended chains

Development briefs follow the [unattended-chain rule](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md#dev-environment-and-session-practice).
`AGENT6_DETACHED_AWAY=approve` grants approvals; it does not answer questions.
A headless run still waits on `ask_user(wait=true)` unless away mode is `deny`.

An external chain driver polls `agent6 attach ID --json`, as well as watching child exit.
Its parked-question stop condition is `live == true` and a member of `questions` with `answered == false` and `blocking == true`.
The driver reports the session ID and question `id`, stops scheduling subsequent work, and retains the session for an operator to answer with `agent6 answer ID --question QID TEXT` or stop with `agent6 stop ID`.
It must not fabricate an answer or automatically resume the chain.

Queued questions (`wait=false`), answered questions and questions from earlier executions do not meet this condition.
`operator_blocked` alone is insufficient: approvals can also block a run.
This is a contract for external chain drivers, not a change to interactive fan-out waits; no such external development-chain driver is included in this repository.

## Exit codes and end reasons

Exit codes for `agent6 run`, `ask` and `resume`, for scripts to branch on:

| Code | Meaning |
|---|---|
| `0` | finished with a green gate, or nothing to gate on |
| `1` | ended unfinished: out of turns, stuck, a provider error, a crash |
| `2` | operator error (bad flag or config, a refusal before anything ran) |
| `3` | budget exhausted, including a `run --parallel` fan-out whose every lane stopped at its cap |
| `4` | finished over a red or timed-out verify gate, or a never-run gate under `harness.verify.when = "finish"` or `"turn"` |
| `5` | finished, but no commit landed and the edits sit uncommitted (a run that changed nothing, or one with `[git].commit_per_turn = false`, stays `0`) |
| `130` | interrupted, or stopped by the operator: Ctrl-C, SIGTERM to a run, ask, plan or fan-out (`timeout`, CI, the kill `agent6 stop` falls back to), `agent6 stop`, abort or exit at the pause menu, stop after a commit |
| `141` | the output pipe closed (`agent6 ... \| head`) |

`passed` means the model called `finish_session` and the gate was green; it does not certify the task.
Exit `0` also covers harness-declared and gateless ends, including a run that changed nothing.
[`git.auto_merge`](config.md#git) lands a run only when the model ended it with `finish_session`; an exit `0` from any other end stays on its branch.

Every end has one reason (`stop_reason` in `agent6 sessions show --json` and `sessions list --json`), and each reason one row:

| End reason | Headline | Exit code |
|---|---|---|
| `finish_session` | `passed`, `finished`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate, unverified with the harness gate off), `4` (gate red, gate timed out, unverified under a harness gate) |
| `finish_planning` | `planned` | `0` |
| `answered` | `answered` | `0` |
| `silent_finish` | `finished · no finish call`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate, unverified with the harness gate off), `4` (gate red, gate timed out, unverified under a harness gate) |
| `went_quiet` | `failed · went quiet` | `1` |
| `budget_exhausted` | `failed · budget exhausted` | `3` |
| `provider_error` | `failed · provider error` | `1` |
| `metric_plateau` | `finished · metric plateau`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate, unverified with the harness gate off), `4` (gate red, gate timed out, unverified under a harness gate) |
| `verify_settled` | `finished · settled` | `0` |
| `settled` | `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `4` (gate red, gate timed out, unverified under a harness gate), `0` (unverified with the harness gate off, no gate) |
| `no_progress` | `failed · no progress` | `1` |
| `tool_error_stuck` | `failed · tool error stuck` | `1` |
| `verify_command_unexecutable` | `failed · verify command unexecutable` | `1` |
| `setup_command_failed` | `failed · setup command failed` | `1` |
| `loop_guard_killed` | `failed · loop guard killed` | `1` |
| `interactive_stop` | `stopped` | `130` |
| `interrupted` | `stopped` | `130`, `141` (closed output pipe) |
| `crashed` | `failed · crashed` | `1` |
| `steer_abort` | `stopped` | `130` |
| `steer_exit` | `stopped` | `130` |
| `undone` | `undone` | `0` |
| `detached` | none: no end is recorded | `0` |
| `plan_unreadable` | `failed · plan unreadable` | `1` |
| `max_turns` | `failed · max turns` | `1` |
| `ask_repl_empty` | none: no end is recorded | `0` |
| `calls_exhausted` | `failed · calls exhausted` | `1` |
| `gate_stale` | `finished · gate stale` | `4` (gate red, gate timed out, unverified under a harness gate), `0` (unverified with the harness gate off) |
| `gate_red_at_base` | `finished · gate red at base` | `4` |
| `lanes_ranked` | `finished · lanes ranked` | `0` |
| `no_lane_result` | `failed · no lane result` | `1` |
| `no_lane_passed` | `failed · no lane passed` | `4` |

- `finish_session` reads `passed` over a green gate and a bare `finished` with no gate
- `silent_finish` means the model stopped calling tools without calling `finish_session`, and the harness took its last answer as the end
- `metric_plateau` means the configured metric stopped improving and the harness ended the run
- a completed run whose edits sit uncommitted with no commit landed exits `5` in place of `0` (automatic commits under `git.control = "agent6"`)
  - `finish_session` reads `passed · not committed` over a green gate and `finished · not committed` with no gate
  - `silent_finish`, `metric_plateau`, `verify_settled` and `settled` also add `· not committed` to their status
- `verify_settled` and `settled` are the harness's own stop for a run whose turns stopped changing the tree: the worker called no `finish_session`, and `agent6 resume ID` continues it
  - `harness.settled_stop_after` sets the idle-turn limit (default 12); `-1` disables the stop but keeps the nudge

## Telemetry export

`agent6 telemetry export` reads this repository's session and machine journals and delivers what `[telemetry]` selects: spans for each execution, turn, model call, tool call, verification, review panel, operator wait and completed machine state; cumulative metrics (tokens, cost, plan points, tool calls, verification outcomes and durations, findings, approvals, waits); and log records for the significant events.
Machine worker executions are children of their state spans, and machine waits produce log records.
Model-call and tool spans follow OpenTelemetry's GenAI conventions (semantic conventions 1.45.0, status Development): they are named `chat {model}` and `execute_tool {tool}` and carry `gen_ai.*` attributes.
`gen_ai.usage.input_tokens` includes the cached tokens, as the convention defines it.
Nothing is exported until a signal is selected; a run never exports its own events.

```toml
[telemetry]
traces = "otlp"               # spans to the endpoint
metrics = "prometheus"        # or "otlp"; the textfile below for node_exporter
logs = "otlp"                 # the significant events as log records
endpoint = "https://collector.example:4318"
headers = ["Authorization"]   # values in secrets.toml under [telemetry.headers]
prometheus_file = "/var/lib/node_exporter/textfile/agent6.prom"
```

```sh
agent6 config set --secret telemetry.headers.Authorization 'Bearer <token>'
agent6 telemetry export                 # every session with a journal, once
agent6 telemetry export <id> --follow   # one session, as its journal grows
agent6 config show telemetry            # every key with its origin
```

- With `telemetry.metrics = "prometheus"` and `web.metrics_route = true` (default: `false`), `agent6 web` serves fresh hub-session metrics at `/metrics` only with `Authorization: Bearer <web-secret>` (Prometheus `authorization.credentials`), never a cookie or browser sign-in token.
- A session's meter keeps 64 distinct values per free label (model, provider, tool) and reads `other` past them; an export that merges sessions carries each session's values, so the merged set can exceed 64.
- A cursor per endpoint and session keeps a re-run from re-sending delivered spans and log records; the metrics reading is always the whole cumulative snapshot, and the Prometheus file is rewritten whole.
- The default `capture = "metadata"` exports outcomes, counts, timings and the journal's ids, never the task, arguments, outputs or diffs; `capture = "content"` adds the capped excerpts `telemetry.capture` lists ([config](config.md#telemetry)).
- `--follow` posts what it holds every five seconds while events arrive and once more on Ctrl-C; a live run's unfinished work is sent when the run finishes it, not at the export's end.
  A crashed run's unfinished work is sent once, marked incomplete.
- A request the collector answers with 429, 502, 503 or 504 is tried three times with backoff; any other refusal turns the endpoint off for that export, and the command exits 1 naming it.
- `agent6_plan_observed_points_total` counts the plan-window movement a run observed, which is the account's whole window; it is not this run's billing.

## Other commands

```sh
agent6 check [section]                               # sandbox (isolation, boundaries, probes), config (provider keys), MCP, verify
agent6 init [--yes] [--ecosystem E]                  # the setup wizard: per-repo config, verify command, .gitignore, AGENTS.md
agent6 memory add|list|show|remove                   # the index renders each fact's first non-empty line
agent6 memory list --json                           # prints a JSON array with name, first_line, written_by, written_at, edited_by, edited_at, reads and last_read_by (unknown provenance is null).
agent6 memory show NAME --json                      # prints that fact’s object from memory list --json.
agent6 memory decisions                              # the operator rulings the harness recorded
agent6 skills install|update|list                    # skills from a repo, a directory or a SKILL.md URL
agent6 skills enable NAME [--always]                 # a skill the model may use, or one it always gets; disable, remove
agent6 mcp add NAME [--url URL | -- COMMAND...]      # an MCP server; --pass-env VAR passes an environment variable to a local server; list, remove, serve
agent6 machine ...                                   # state machines over runs (see state-machines.md)
agent6 system apparmor install                       # the AppArmor profile that full sandbox.isolation needs on some hosts; status, remove
agent6 completions [bash|zsh|fish|xonsh] [--print]
agent6 config fill [--force]                         # write the defaults + global layers as one explicit global config
```

`agent6 init` detects Go (`go.mod`), Node, Python and Rust manifests for its AGENTS.md advice and build-artifact ignores.
`--ecosystem go|node|py|rust` overrides that choice, not verify inference.

See [repo memory](security.md#3-file-access) for the distinction between facts and rules.
