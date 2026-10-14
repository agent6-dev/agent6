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
Your branch, HEAD and index are never touched; `sessions merge` lands the work.

- commands ask for approval by default (`sandbox.run_commands = "ask"`): allow one call or the whole session
- without a terminal (CI, cron, a hub-spawned run) a prompt parks the run until a front-end answers it, or an away-mode decides it; a run whose commands are unsettled refuses to start and names the choices (`--auto-approve`, `--no-commands`, `AGENT6_DETACHED_AWAY`), see [approvals](security.md#6-approvals)
- the run ends when the model declares it finished, the operator stops it (`agent6 stop ID`; `--after-step` lets the step finish) or a ceiling (budget, iterations) stops it; every end is resumable
- at a terminal it then asks for the next input: type to continue, `/exit` to finish; without one the resume line prints

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
agent6 stop ID                # stop now (the worker killed if it does not answer); --after-step, --all
agent6 sessions show          # status, iteration, elapsed, cost, the gate's word, where the changes are; --json
agent6 sessions diff          # the git diff the run produced; --stat, --path P
agent6 sessions commits       # the run's per-step commits
agent6 sessions merge         # land the work on your branch (warns first when the gate did not certify it); --strategy, --into BRANCH
agent6 sessions transcript    # the conversation as text; --no-thinking, --seq N, --tools calls|none
agent6 sessions graph         # the task graph, each line led by the id /retire takes
agent6 sessions compare <ids> # >=2 runs ranked by the reviewer model (minutes), or a fan-out's verdict; --rejudge
agent6 sessions prune         # delete merged agent6/* branches and worktrees, report the rest; squash merges (the default) need --delete-squashed
agent6 sessions rm            # delete one run's history, asking [y/N] first on a terminal; --yes skips the question, --asks clears saved asks
agent6 sessions dir [<id>]    # the repo's run history, or that session's directory
agent6 exec ID -- <command>   # a command inside the run's jail and network, on your terminal or pipe
agent6 exec --new [<command>] # a command, or your shell, in a fresh sandbox built from the config; no run needed
agent6 forward ID 8000        # reach a port inside the run's network; --local-port N
agent6 history search <text>  # grep every session's persisted data; --regex
agent6 ps                     # live sessions of every repository on the machine; --lanes, --json
```

`agent6 exec` runs your own command on your own stdio.

- with stdin and stdout on a terminal the command gets a terminal of its own, with job control (it needs util-linux `setsid` on the sandbox's PATH, and shares your terminal without it); otherwise stdin, stdout and stderr pass through, so pipes and redirects work
- inside strict there is no `/dev/pts` or `/dev/tty`: `tty` prints "not a tty", and a program that opens `/dev/tty` fails
- a command a signal ended reports 137 under strict and hardened, whatever the signal; on a terminal it reports the signal's number instead, on every level, and `setsid` prints "did not exit normally"

The steer directives work from every composer, the pause menu, and `agent6 steer ID "/task ..."` from a script.

- `/task <text>` adds work to the task graph (the turn in flight never sees it); `/retire <id>` drops a task, by the number `sessions graph` prints
- `/standing <text>` sets the goal the run returns to whenever its queue drains, replacing any it had
- `/pin <text>` survives every compaction; `/compact [focus]` compacts now; `/now <text>` interrupts the call in flight
- `/btw <question>` asks a side question; `/parallel [spec] <task>` fans out lanes ([parallel](config.md#parallel))
- `/undo` takes back the last message: the tree is committed on the run's chain, every tracked path the turn changed is put back, and the run continues as a fork in the same checkout; refused while another live run drives the checkout; it acts on a view of the run, so `agent6 steer` refuses it: type it in `agent6 attach`, the TUI or the web

## Answer a parked prompt

Every front-end answers a prompt by writing one file in the session directory, and so can a script.

- `agent6 answer <id>` prints the open question; `agent6 answer <id> TEXT...` answers it, one TEXT per question
- a prompt `agent6 attach` is showing closes once another front-end answers it: attach prints `already answered from another surface`, writes no answer and keeps following
- an approval reads `<session dir>/approvals/<id>.answer`: `yes`, `no`, `session` or `session-deny` (anything else denies; the `session` answers only where the prompt's event says `standing: true`)
- a question reads `<session dir>/questions/<id>.answer`: a JSON list of answers in the prompt's order
- the prompt's `id` is in `logs.jsonl` (`approval.prompt`, `question.prompt`); write the file atomically (a sibling, then rename), the run consumes it as soon as it exists

## When a run goes wrong

```sh
agent6 resume <session-id>           # continue from the last snapshot; --force past a diverged chain
agent6 fork <session-id> --at-turn 7 # a new run from turn 7; --steer seeds it, --no-run only creates it
```

State is snapshotted before each model call and checkpointed per turn.

- `fork` continues a copy from a turn as a new run in its own git worktree; the original run and your checkout stay as they are, and `sessions merge`, `prune` and `rm` treat it like any run ([session state](architecture.md#session-state-on-disk))
- a `--steer` on a fork, or on resuming a run the agent finished, becomes the new run's task and names its row and its merge subject
- a run that was squash-merged merges again from its landed tip, so nothing lands twice

Exit codes for `agent6 run`, `ask`, `resume` and `review --reviewers N`, for scripts to branch on:

| Code | Meaning |
|---|---|
| `0` | finished with a green gate, or nothing to gate on; a review panel's PASS |
| `1` | ended unfinished: out of turns, stuck, a provider error, a crash; a review panel whose every seat abstained |
| `2` | operator error (bad flag or config, a refusal before anything ran) |
| `3` | budget exhausted |
| `4` | finished over a red, timed-out or never-run verify gate; a review panel's BLOCK |
| `5` | finished, but no commit landed and the edits sit uncommitted (a run that changed nothing, or one with `[git].commit_per_step = false`, stays `0`) |
| `130` | interrupted, or stopped by the operator: Ctrl-C, `agent6 stop`, abort or exit at the pause menu, stop after a commit (a worker `agent6 stop` had to kill dies on the signal instead) |
| `141` | the output pipe closed (`agent6 ... \| head`) |

`passed` and exit `0` certify the finish and the gate, not the task; a gateless run that changed nothing stays `0`.

Every end has one reason (`reason` in `agent6 sessions show --json`), and each reason one row:

| End reason | Headline | Exit code |
|---|---|---|
| `finish_session` | `passed`, `finished`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate), `4` (gate red, gate timed out, unverified) |
| `finish_planning` | `planned` | `0` |
| `answered` | `answered` | `0` |
| `silent_finish` | `passed`, `finished`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate), `4` (gate red, gate timed out, unverified) |
| `went_quiet` | `failed · went quiet` | `1` |
| `budget_exhausted` | `failed · budget exhausted` | `3` |
| `provider_error` | `failed · provider error` | `1` |
| `metric_plateau` | `passed`, `finished`, `finished · gate red`, `finished · gate timed out`, `finished · unverified` | `0` (gate green, no gate), `4` (gate red, gate timed out, unverified) |
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

- where a row lists several, the gate decides: `passed` over a green gate, a bare `finished` with no gate, `finished · <detail>` over a gate that did not certify the tree
- a finished run whose edits sit uncommitted exits `5` in place of `0`
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
- `--worktree`: from the repository's top level, the run works in its own git worktree at HEAD's commit, under `[parallel].workdir`; your checkout and its uncommitted changes stay as they are and are not carried, so you can keep working there or start more runs; `sessions merge` lands the work and `sessions prune` then removes the worktree, as it does for a run that committed nothing and left no edits; `agent6 sessions` marks the run `(worktree)`, and `sessions show` and `ps` print the worktree's path
- `--standing "<goal>"`: a fallback goal the run returns to whenever its queue drains; only the operator retires it, and ceilings still end the run (`harness.standing_patience`)
- `--pin "<text>"`: an instruction re-shown after every compaction
- `--from <id>`: seed from another session's task, outcome, diff and plan; a plan id runs that plan, and `sessions show` prints `seeded from`
    - a plan whose open questions have blank `**A:**` lines still runs, after one warning that names them; `agent6 plan edit <id>` answers them first
- `--decompose`: a task graph up front; `--skill NAME`: a skill in the prompt (a skill whose text waits on a person stalls an unattended run in `ask_user`; `agent6 skills disable NAME` keeps it out)
- `--session-id ID`: name the session yourself
- `agent6 prompt show [--mode run|plan|ask|agent] [--json]`: everything the model receives on the first call

## Other commands

```sh
agent6 tui [target]                  # the full-screen hub, or one session's view
agent6 web [target]                  # the browser UI on 127.0.0.1:7658, behind a secret it prints (see web.md)
agent6 acp                           # speak the Agent Client Protocol on stdio (see acp.md)
agent6 check [section]               # sandbox, config (provider keys), boundaries, MCP, verify
agent6 init [--yes] [--ecosystem E]  # the setup wizard: per-repo config, verify_command, .gitignore, AGENTS.md
agent6 memory add|list|show|rm       # the repo's memory: one fact per file, restated to every run
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
