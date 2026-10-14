# Usage

Assumes agent6 is installed (see [installation](installation.md)).

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
agent6 check                         # every key resolves; model listings refresh
agent6 model                         # the role assignments
```

The key lands in `~/.config/agent6/secrets.toml` (mode `0600`), shared across every repository.
`connect` prompts locally and executes nothing a remote returns.

agent6 routes three model roles; `reviewer` and `planner` fall back to `worker` when unset.

| Role       | Set with            | Used by                                                                                              |
| ---------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| `worker`   | `[models.worker]`   | `agent6 run` and `agent6 resume`                                                                     |
| `reviewer` | `[models.reviewer]` | `agent6 review`, the in-loop review panel, the context summariser, the gister and the prompt reviser |
| `planner`  | `[models.planner]`  | `agent6 plan`                                                                                        |

```sh
agent6 model worker anthropic/claude-sonnet-5
agent6 model all openrouter/moonshotai/kimi-k2.6   # set every role at once
```

## Your first run

```sh
cd your-repo
agent6 run "add a --json output mode to the CLI"
```

agent6 edits your working tree, commits each step to the run's own chain (an `agent6/<id>` branch by default), and certifies the finished tree with the verify command.
Your branch, HEAD and index stay untouched unless [`git.auto_merge`](config.md#git) lands the work; `sessions merge` lands it by hand.

- commands ask for approval by default (`sandbox.run_commands = "ask"`): allow one call or the whole session
- without a terminal (CI, cron, a hub-spawned run) a prompt parks the run until a front-end answers it, or an away-mode decides it; a run whose commands are unsettled refuses to start and names the choices (`--auto-approve`, `--no-commands`, `AGENT6_DETACHED_AWAY`), see [approvals](security.md#6-approvals)
- the run ends when the model declares it finished, the operator stops it (`agent6 stop ID`; `--after-step` lets the step finish) or a ceiling (budget, iterations) stops it
- at a terminal it then asks for the next input: type to continue, `/exit` to finish
- without a terminal a finished run prints its merge and inspect commands, and an unfinished run its resume command
- a finished run takes more work with `agent6 resume <id> --steer "<what to do next>"`; a bare resume of it is refused

The verify command is the success gate.
Pin one with `harness.verify_command` or `agent6 init`; unset, each run infers one and prints it, or runs gateless when nothing is inferable.
A red gate at finish returns to the model `harness.verify_retries` times, then the run ends red.
[`[harness]`](config.md#harness) has the inference order and `verify_when`.

`agent6 run` streams in your terminal; `--tui` opens the full-screen view ([terminal UI](terminal.md)) and `-i` drives it from a stdin REPL.

## Follow and steer a run

`agent6 attach [<id>]` follows a run's conversation, or a machine's state, live.
Ids are exact or a unique prefix and default to the most recent run.

```sh
agent6 attach                 # follow live; --raw tails events, --tui full screen, --json one snapshot
agent6 steer ID "focus on X"  # an instruction at the next step; --now interrupts the call in flight
agent6 answer ID "yes"        # answer a live run's question (bare: print the question)
agent6 stop ID                # stop during the current step (the worker killed if it does not answer); --all
agent6 stop ID --after-step   # finish the current step, then stop
agent6 sessions show          # status, iteration, elapsed, cost, the gate's word, where the changes are; --json
agent6 sessions diff          # the git diff the run produced; --stat, --path P
agent6 sessions commits       # the run's per-step commits
agent6 sessions merge         # land the work on your branch (warns first when the gate did not certify it); --strategy, --into BRANCH
agent6 sessions transcript    # the conversation as text; --no-thinking, --seq N, --tools calls|none
agent6 sessions graph         # the task graph, each line led by the id /retire takes
agent6 sessions compare <ids> # >=2 runs ranked by the reviewer model (minutes), or a fan-out's verdict; --rejudge
agent6 sessions prune         # delete merged agent6/* branches and worktrees, report the rest; squash merges (the default) need --delete-squashed
agent6 sessions rm [<id>]     # delete one run's history; asks [y/N] on a controlling terminal, otherwise requires an id or --yes; --yes skips the question, --asks clears saved asks
agent6 sessions dir [<id>]    # the repo's run history, or that session's directory
agent6 exec ID -- <command>   # a command inside the run's jail and network, on your terminal or pipe
agent6 exec --new [<command>] # a command, or your shell, in a fresh sandbox built from the config; no run needed
agent6 forward ID 8000        # reach a port inside the run's network; --local-port N
agent6 history search <text>  # grep every session's persisted data; --regex
agent6 ps                     # live sessions of every repository on the machine; --lanes, --json
```

`agent6 stop ID` ends the run as `stopped`, with `operator stopped the run during step N` in the end block.
The end block adds `running command was cut short` when the stop ended a running command, or `model call was cut short` when it cut the model call.
`--after-step` leaves `running · stopping after this step` in listings until the step ends, then prints `operator stopped the run after step N`.
A starting or waiting run keeps that status and adds `stopping after this step` beside its other details.

A session's cost covers every execution, in dollars or in plan points; `sessions list`, `sessions show`, the end-of-run total and the fan-out report print the same figure.
The tokens and the points against `max_percent` on `sessions show` are the latest execution's.
`elapsed` on a resumed run is the wall time from its first start to its last event, the time between executions included.
`N questions unanswered` beside a status counts the questions of the current execution that nobody was attached to see; a resume starts the count again.

`agent6 exec` runs your own command on your own stdio.

- with stdin and stdout on a terminal the command gets a terminal of its own, with job control; otherwise stdin, stdout and stderr pass through, so pipes and redirects work
- Ctrl-Z stops a job of the shell you opened; a command you run directly keeps running, since no shell could resume it
- inside strict there is no `/dev/pts` or `/dev/tty`: `tty` prints "not a tty", and a program that opens `/dev/tty` fails
- a command a signal ended reports 128 plus the signal, on every level
- `agent6 exec ID` works in the run's workspace: a file your command writes there is in the run's tree, and the run's next commit holds it

The steer directives work from every composer, the pause menu, and `agent6 steer ID "/task ..."` from a script.

- `/task <text>` adds work to the task graph (the turn in flight never sees it); `/retire <id>` drops a task, by the number `sessions graph` prints
- `/standing <text>` sets the goal the run returns to whenever its queue drains, replacing any it had
- `/pin <text>` survives every compaction; `/compact [focus]` compacts now; `/now <text>` interrupts the call in flight
- `/btw <question>` asks a side question; `/parallel [spec] <task>` fans out lanes ([parallel](config.md#parallel))
- `/undo` takes back the last message: the tree is committed on the run's chain, every tracked path the turn changed is put back, and the run continues as a fork in the same checkout; refused while another live run drives the checkout; it acts on a view of the run, so `agent6 steer` refuses it: type it in `agent6 attach`, the TUI or the web

## Answer a parked prompt

Every front-end answers a prompt by writing one file in the session directory, and so can a script.

- a run with no terminal prints one line on stderr when it parks (`[agent6] waiting on approval-2 (run_command); answer it with: agent6 attach <id>`) and one when the prompt is answered (`[agent6] approval-2 approved (frontend)`)
- `agent6 answer <id>` prints the open question; `agent6 answer <id> TEXT...` answers it, one TEXT per question
- a prompt `agent6 attach` is showing closes once another front-end answers it: attach prints `already answered from another surface`, writes no answer and keeps following; the TUI's question modal closes the same way, with the same notice
- an approval reads `<session dir>/approvals/<id>.answer`: `yes`, `no`, `session` or `session-deny` (anything else denies; the `session` answers only where the prompt's event says `standing: true`)
- a question reads `<session dir>/questions/<id>.answer`: a JSON list of answers in the prompt's order
- the prompt's `id` is in `logs.jsonl` (`approval.prompt`, `question.prompt`); write the file atomically (a sibling, then rename), the run consumes it as soon as it exists

## When a run goes wrong

```sh
agent6 resume <session-id>           # continue from the last snapshot; --force past a diverged chain or run branch
agent6 fork <session-id> --at-turn 7 # a new run from turn 7; --steer seeds it, --no-run only creates it
```

State is snapshotted before each model call and checkpointed per turn.

- `fork` continues a copy from a turn as a new run in its own git worktree ([isolated workspaces](#isolated-workspaces)); the original run and your checkout stay as they are, and `sessions merge`, `prune` and `rm` treat it like any run ([session state](architecture.md#session-state-on-disk))
  `sessions prune` keeps a fork that has not started: its branch, chain ref and worktree stay.
- `--at-turn 0` on a run that is itself a fork picks the seed it started from, before anything it did; the fork line says so
- a `--parallel` lane continues through `agent6 fork <lane-id>`, in a worktree at the lane's last commit; `resume` refuses a lane from your checkout, whose tree is not the lane's
- a fan-out stopped only by lane caps reads `failed · budget exhausted` and exits `3`; its report says to raise the budget cap and fork each stopped lane
- a `--steer` on a fork, or on resuming a run the agent finished, becomes the new run's task and names its row and its merge subject
- a resume after the run's work landed continues from the checkout's HEAD when HEAD holds the merge: the run's next commits, `sessions diff` and its next merge then hold only the new work, so nothing lands twice
- a run whose workspace does not hold the merge (a `--worktree` run, a fork) continues on its own chain: after a resume, `sessions diff`, `sessions commits` and its next merge start from the landed tip
- a run merged `--into` a branch other than the one it started on keeps its base, and merges into that branch again from its landed tip
- `sessions merge` over a file you edited leaves your copy alone and prints two commands that merge your edit with what landed
- a commit you made on the run's `agent6/<id>` branch refuses the resume, which names it: `agent6 sessions merge <id>` lands it with the run, or `resume --force` continues without it
- files you changed between executions follow `[git].dirty_tree` on resume, as uncommitted changes do at a run's start: `ask` asks, `stash` sets them aside, `include` lets the run's next commit take them ([Config](config.md))
- when `stash` refuses over a file the run also changed, its printed `git restore --source ...` command discards only your edit and keeps the run's last tree; plain `git restore FILE` returns to HEAD, which can remove the run's work
- a new untracked file is left out of the run's commits as yours, unless the last execution died mid-turn (a kill, a crash): a command of that turn may have written it, so the run keeps it, and the resume names each file either way

Exit codes for `agent6 run`, `ask`, `resume` and `review --reviewers N`, for scripts to branch on:

| Code | Meaning |
|---|---|
| `0` | finished with a green gate, or nothing to gate on; a review panel's PASS |
| `1` | ended unfinished: out of turns, stuck, a provider error, a crash; a review panel whose every seat abstained |
| `2` | operator error (bad flag or config, a refusal before anything ran) |
| `3` | budget exhausted, including a `run --parallel` fan-out whose every lane stopped at its cap |
| `4` | finished over a red, timed-out or never-run verify gate; a review panel's BLOCK |
| `5` | finished, but no commit landed and the edits sit uncommitted (a run that changed nothing, or one with `[git].commit_per_step = false`, stays `0`) |
| `130` | interrupted, or stopped by the operator: Ctrl-C, `agent6 stop`, abort or exit at the pause menu, stop after a commit (a worker `agent6 stop` had to kill dies on the signal instead) |
| `141` | the output pipe closed (`agent6 ... \| head`) |

`passed` means the model called `finish_session` and the gate was green; it does not certify the task.
Exit `0` also covers harness-declared and gateless ends, including a run that changed nothing.
[`git.auto_merge`](config.md#git) lands a run only when the model ended it with `finish_session`; an exit `0` from any other end stays on its branch.

Every end has one reason (`reason` in `agent6 sessions show --json`), and each reason one row:

| End reason | Headline | Exit code |
|---|---|---|
| `finish_session` | `passed`, `finished`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate), `4` (gate red, gate timed out, unverified) |
| `finish_planning` | `planned` | `0` |
| `answered` | `answered` | `0` |
| `silent_finish` | `finished · no finish call`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate), `4` (gate red, gate timed out, unverified) |
| `went_quiet` | `failed · went quiet` | `1` |
| `budget_exhausted` | `failed · budget exhausted` | `3` |
| `provider_error` | `failed · provider error` | `1` |
| `metric_plateau` | `finished · metric plateau`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate), `4` (gate red, gate timed out, unverified) |
| `verify_settled` | `finished · settled` | `0` |
| `settled` | `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `4` (gate red, gate timed out, unverified), `0` (no gate) |
| `no_progress` | `failed · no progress` | `1` |
| `tool_error_stuck` | `failed · tool error stuck` | `1` |
| `verify_command_unexecutable` | `failed · verify command unexecutable` | `1` |
| `setup_command_failed` | `failed · setup command failed` | `1` |
| `loop_guard_killed` | `failed · loop guard killed` | `1` |
| `interactive_stop` | `stopped` | `130` |
| `interrupted` | `stopped` | `130` |
| `crashed` | `failed · crashed` | `1` |
| `steer_abort` | `stopped` | `130` |
| `steer_exit` | `stopped` | `130` |
| `undone` | `undone` | `0` |
| `detached` | none: no end is recorded | `0` |
| `prompt_revision_failed` | `failed · prompt revision failed` | `1` |
| `plan_unreadable` | `failed · plan unreadable` | `1` |
| `max_iterations` | `failed · max iterations` | `1` |
| `ask_repl_empty` | none: no end is recorded | `0` |
| `gate_stale` | `finished · gate stale` | `4` |
| `gate_red_at_base` | `finished · gate was already red` | `4` |
| `no_lane_result` | `failed · no lane result` | `1` |
| `no_lane_passed` | `failed · no lane passed` | `4` |

A run whose worker exited without recording an end reads `stale`, for example after SIGKILL or a machine crash.
`agent6 resume <id>` continues it from its last snapshot, and `agent6 sessions show <id>` prints that command.

- `finish_session` reads `passed` over a green gate and a bare `finished` with no gate
- `silent_finish` means the model stopped calling tools without calling `finish_session`, and the harness took its last answer as the end
- `metric_plateau` means the configured metric stopped improving and the harness ended the run
- over a red, timed-out or unverified gate, these finishes name the gate state in the headline
- a finished run whose edits sit uncommitted exits `5` in place of `0`, and reads `passed · not committed` (`finished · not committed` with no gate) in the end block and every listing
- `verify_settled` and `settled` are the harness's own stop for a run whose turns stopped changing the tree: the worker called no `finish_session`, and `agent6 resume ID` continues it

## Plan, review, and ask

```sh
agent6 plan "refactor the config loader"      # edit-free plan; run --from <id> executes it
agent6 plan show <session-id>                 # print the plan
agent6 plan edit <session-id>                 # open plan.md in $EDITOR (answer its open questions)
agent6 resume <session-id> --steer "answered" # the planner re-reads and revises
agent6 review --base origin/main --head HEAD  # read-only diff review; --path P narrows, --model M picks the reviewer
agent6 sessions review <session-id>           # read-only review of a finished run's record
agent6 ask "how does the task-graph curator work?"
```

- `review --reviewers 3 --personas security,correctness,tests`: a panel whose findings are checked against the diff, so only real problems gate; a seat pins its model in `[review].seats` (`security@openrouter/<model-id>`)
- `sessions review` reports how the run ended, what went wrong, what you corrected, and candidate memory facts and AGENTS.md lines with their evidence; it writes nothing (`agent6 memory add` lands a candidate)
- every review is saved under `<state-dir>/reviews/`; the path prints on stderr
- `ask` runs in any directory; `run` and `plan` need a git repository

## Run options

- `--preset <name>`: a strategy preset (`standard`, `quick`, `ultra`, `paranoid`, or your own), fixed for the execution; `resume --preset` switches ([presets](config.md#presets))
- `--model [provider/]model`: the model for the mode's role, over every config layer; a bare id keeps the role's provider, and a resume keeps the recorded pick unless it sets its own
- `--parallel 3` (or `provider/model-a,model-b`): isolated lanes, auto-compared into a ranked report; the fan-out is a session of its own that `attach`, `stop` and `sessions show` address, its lanes folded under it (`--lanes` expands them) ([parallel](config.md#parallel))
- `--worktree`: from the repository's top level, the run works in its own git worktree at HEAD's commit, under `[parallel].workdir`; your checkout and its uncommitted changes stay as they are and are not carried, and neither is anything git does not track ([isolated workspaces](#isolated-workspaces)), so you can keep working there or start more runs; `sessions merge` lands the work and `sessions prune` then removes the worktree, as it does for a run that committed nothing and left no edits; `agent6 sessions` marks the run `(worktree)`, and `sessions show` and `ps` print the worktree's path
- `--standing "<goal>"`: a fallback goal the run returns to whenever its queue drains; only the operator retires it, and ceilings still end the run (`harness.standing_patience`)
- `--pin "<text>"`: an instruction re-shown after every compaction
- `--from <id>`: seed from another session's task, outcome, diff and plan; a plan id runs that plan, and `sessions show` prints `seeded from`
    - a plan whose open questions have blank `**A:**` lines still runs, after one warning that names them; `agent6 plan edit <id>` answers them first
- `--decompose`: a task graph up front; `--skill NAME`: a skill in the prompt (a skill whose text waits on a person stalls an unattended run in `ask_user`; `agent6 skills disable NAME` keeps it out)
- `--session-id ID`: name the session yourself
- `agent6 prompt show [--mode run|plan|ask|agent] [--json]`: everything the model receives on the first call

## Isolated workspaces

`run --worktree`, a fork, each `--parallel` lane and a run-mode machine state work in a worktree or a clone of their own.

- It starts with the repository's tracked files at the commit it was cut from: no uncommitted change, and nothing git does not track (`.venv`, `node_modules`, a build directory).
- `[parallel].setup_command` builds what is missing; it runs once in each new workspace, before the first turn:

    ```sh
    agent6 config set --repo parallel.setup_command '["uv", "sync"]'
    ```

- Unset, a `## Setup command` fenced block in AGENTS.md supplies it; a `--worktree` run and a `--parallel` fan-out read it from the commit the workspace is cut from, not from an uncommitted edit, and print the command and that AGENTS.md gave it.
    - an AGENTS.md that is a symlink is not read there: set `parallel.setup_command`
- The command runs in the run's own sandbox, under the isolation, network and grants of every command of that run; with no route out it has to work offline ([the offline recipe](security.md#4-commands-and-environment)).
- Its command and output tail are printed when it ends; a nonzero exit ends the session `failed · setup command failed` before any model call (exit `1`).
  No turn ran, so `agent6 resume` has nothing to continue: start the run again once the command works.
- A fan-out prints each lane's setup line when it imports the lane, and a failed lane's output in the failed lanes block; `agent6 sessions show <id>` keeps the command, and a failed setup's output, after the workspace is gone.
- The files it leaves untracked stay out of the run's commits.
- It runs once per workspace: `resume` does not run it again, and a fork's new worktree runs it at the fork's first execution.
- The verify gate is settled in your checkout and carried into the workspace: a gate inferred as `.venv/bin/python -m pytest -q` stays that.
- A gate whose first word is a path git does not track, with no setup command, refuses `run --worktree`, `fork` and `run --parallel` before anything is created; the refusal names the path, the config line and the AGENTS.md block.

## Other commands

```sh
agent6 tui [target]                  # the full-screen hub, or one session's view
agent6 web [target]                  # the browser UI on 127.0.0.1:7658, behind a secret it prints (see web.md)
agent6 acp                           # speak the Agent Client Protocol on stdio (see acp.md)
agent6 check [section]               # sandbox, config (provider keys), boundaries, MCP, verify
agent6 init [--yes] [--ecosystem E]  # the setup wizard: per-repo config, verify_command, .gitignore, AGENTS.md
agent6 memory add|list|show|rm       # the repo's memory: one fact per file, restated to every run; rm asks [y/N] first on a terminal, --yes skips the question
agent6 memory decisions              # the operator rulings the harness recorded
agent6 skills install|update|list    # skills from a repo, a directory or a SKILL.md URL
agent6 skills enable [--always]      # a skill the model may use, or one it always gets; disable, remove
agent6 mcp connect [--pass-env VAR]  # an MCP server the model may call; list, remove, serve
agent6 machine ...                   # state machines over runs (see state-machines.md)
agent6 system apparmor install       # the AppArmor profile the strict jail needs on some hosts; status, remove
agent6 completions bash|zsh|fish|xonsh [--print]
agent6 config fill [--force]         # write the defaults + global layers as one explicit global config
```

## Configuration

Config is layered, lowest first: built-in defaults, `~/.config/agent6/config.toml`, the per-repo config, `--config FILE`, a machine's `[config]` overlay.
A preset is one more layer, above the config that selected it.
Every field has a default and the security-sensitive ones default safe, so a repo can be zero-config.
`agent6 config show` prints every effective value with the layer that set it; the [configuration reference](config.md) documents each field.
