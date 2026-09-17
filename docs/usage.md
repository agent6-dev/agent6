# Usage

Assumes agent6 is installed (see [installation](installation.md)).

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
```

The key lands in `~/.config/agent6/secrets.toml` (mode `0600`), shared across every repository.

- `agent6 connect` prompts locally; it executes nothing a remote returns
- already connected: skip this step
- `agent6 check`: every referenced provider's key resolves; keyed providers' model listings refresh past a 10-minute cache (1.5 s each, never fatal)
- `agent6 model`: the role assignments

agent6 routes three model roles independently:

| Role       | Set with            | Used by                                                                                              |
| ---------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| `worker`   | `[models.worker]`   | `agent6 run` and `agent6 resume`                                                                     |
| `reviewer` | `[models.reviewer]` | `agent6 review`, the in-loop review panel, the context summariser, the gister and the prompt reviser |
| `planner`  | `[models.planner]`  | `agent6 plan`                                                                                        |

`reviewer` and `planner` fall back to `worker` when unset.

```sh
agent6 model worker anthropic/claude-sonnet-5
agent6 model all openrouter/moonshotai/kimi-k2.6   # set every role at once
```

## Your first run

```sh
cd your-repo
agent6 run "add a --json output mode to the CLI"
```

agent6 edits your working tree, commits each step to a per-run chain, and certifies the finished tree with the verify command.

- your branch, HEAD, and index are never touched (the chain gets an `agent6/<id>` branch by default)
- commands prompt for approval under the default `sandbox.run_commands = "ask"`: allow one call or the whole session
- a headless run refuses to start unless `AGENT6_DETACHED_AWAY` is set: `deny` (auto-deny), `wait` (park the prompt for a front-end) or `approve` (grant every scope, as the detach prompt's approve-all does); under `deny` or `approve` a question gets empty answers with a note to decide alone (`sessions show` counts them)
- a hub-spawned run parks its prompts
- with commands settled (`--auto-approve`, `--no-commands`) and no away-mode, a fetch outside `sandbox.fetch_hosts` or an MCP call parks the run at its approval until a front-end answers, and the start says so
- the run ends three ways: the model declares it finished, the operator stops it (`agent6 stop ID`, in the cheat sheet below), or a ceiling (budget, iterations) stops it
    - a stop cuts the model call, hands back a running command, ends every command the run started, and kills a worker that has not ended after 5 s
    - `--after-step` lets the step finish first; the run stays resumable
- at a terminal it then asks for the next input: type to continue the session, `/exit` to finish (still resumable)
- without a terminal (CI, detached) the resume line prints instead

The verify command is the success gate.

- unset `workflow.verify_command`: inferred per run and printed (AGENTS.md, a root `verify.sh`, manifest files, loose `test_*.py`, then a model call)
- nothing inferable: the run proceeds gateless, committing each editing step
- pin one (per-repo config or `agent6 init`) to make it deterministic
- the harness runs it when the model finishes over an uncertified tree; a red returns to the model with the output as many times as `workflow.verify_retries` allows, then the run ends red
- `workflow.verify_when` moves the harness run to every editing step (`step`) or leaves every run to the model (`never`); the model can always run it itself

`agent6 run` streams in your terminal.

- `--tui`: the full-screen conversation view, dashboard on Ctrl+D (`agent6 plan --tui` for a planning run)
- `-i`: drive the run from a stdin REPL

## Inspect a run

`agent6 attach [<target>]` follows live.

- a run renders its conversation (the `agent6 run` view); a machine streams its state overview and reasoning
- `--raw` tails a run's event stream; `--tui` opens the full-screen TUI
- the plain and raw run followers exit 0 at the run's end and non-zero if its worker dies first
- session ids are positional, exact or an unambiguous prefix; omit for the most recent run

```sh
agent6 attach                 # follow the conversation live; --raw, --tui, --json
agent6 steer ID "focus on X"  # steer a live run at its next step boundary (--now, or /now <text>, interrupts the in-flight call; it takes the composer directives too: /task, /btw, /compact; /stop is `agent6 stop`)
agent6 answer ID "yes"        # answer a live run's ask_user question (bare: print the question)
agent6 sessions show          # status, iteration, elapsed, cost, where the changes are; --json to script
agent6 sessions diff          # the git diff the run produced; --stat for the summary, --path P to narrow
agent6 sessions commits       # the run's per-step commits
agent6 sessions merge         # land the run's work on your branch; --strategy, --into BRANCH
agent6 sessions transcript    # the conversation as text, every tool call with I/O; --no-thinking, --seq N for one turn, --tools calls|none
agent6 sessions prune         # delete merged agent6/* branches; report the rest
agent6 sessions dir           # where this repo's run history lives (scriptable)
agent6 sessions dir <id>      # that session's own directory
agent6 stop ID                # stop a live run now (the worker killed if it does not answer); --after-step, --all
agent6 resume ID [--force]    # continue a stopped or parked run (--force past a diverged chain)
agent6 fork ID --at-turn 7    # a new run from a checkpoint; --no-run creates it without starting
agent6 exec ID -- <command>   # run a command inside the run's jail and network
agent6 forward ID 8000        # reach a port inside the run's session network; --local-port N
agent6 history search <text>  # search every session's persisted data; --regex
agent6 sessions rm            # delete one run's history; --asks clears saved asks
agent6 sessions compare <ids> # ranked comparison: >=2 runs (judged), or one fan-out id (its recorded verdict; --rejudge for a fresh call)
agent6 sessions graph         # the persisted task graph, each line led by the id `/retire` takes
```

`agent6 history search <query>` greps across every session's persisted transcripts and data.
`agent6 ps` lists the live sessions of every repository on the machine, with the directory to cd to and the id to attach; a fan-out's live lanes fold under it (`--lanes` lists them).

## Answer a parked prompt

A run waiting on an approval or a question takes its answer from a file in its session directory.
A parked run polls the file.
A foreground run's own terminal prompt reads it too.
Every front-end writes that file, and so can a script.

- `agent6 answer <id>` prints the open question and its options; `agent6 answer <id> TEXT...` answers it (one TEXT per question, in order) without a terminal
- `agent6 sessions dir <id>` prints the session directory; the prompt's `id` is in its `logs.jsonl` (`approval.prompt`, `question.prompt`)
- an approval: `<session dir>/approvals/<id>.answer` holding `yes`, `no`, `session` or `session-deny`; anything else denies, and the two `session` answers stand only for a prompt whose event says `standing: true`
- a question: `<session dir>/questions/<id>.answer` holding a JSON list of answers, one per question in the prompt's order (a bare string is one answer)
- write the file atomically (a sibling, then rename): the run polls every 0.2s and consumes the file as soon as it exists

## When a run goes wrong

```sh
agent6 resume <session-id>           # continue from the last snapshot
agent6 fork <session-id> --at-turn 7 # new run from turn 7 (--steer seeds it)
```

- state is snapshotted before each model call and checkpointed per turn
- `fork` rolls a copy back to a turn and continues it as a new run in its own git worktree (under `[parallel].workdir`); the original run and your checkout stay as they are
    - `sessions merge <fork>` lands it; `sessions prune` removes the worktree once it is merged; `sessions rm <fork>` removes it with the record
    - the fork's tree is the turn's committed content; the files its commits leave out are the ones untracked in its own checkout when it was created (a fresh worktree usually has none)
    - a `--steer`ed fork takes that steer as its own task: its listing row and its squashed merge subject name the work you sent it to do
    - a run the agent finished resumes only with `--steer`; it takes the steer as its task the same way, so its row and its next squash name the new work
    - a run that was squash-merged merges again from its landed tip (a resumed leg, or a fork continuing its chain), so the work the target already holds is not merged twice
- `/undo` takes back the last message in place, from the TUI and web composers, the `agent6 run` pause menu, or the `run -i` REPL; the TUI and web session views offer it on a finished run too
    - the tree as it stands is committed on the run's chain ref (and branch)
    - every tracked path that differs from the turn before is put back
    - a fork continues in the same checkout with the message back in the composer
    - the run's untracked-at-start files stay, and so do HEAD and the index; the later commits and the pre-undo commit stay on the run's ref
    - refused while another live run drives the checkout

Exit codes for `agent6 run`, `ask`, `resume` and `review --reviewers N`, for scripts to branch on:

| Code | Meaning |
|---|---|
| `0` | finished with a green gate, or nothing to gate on; a review panel's PASS |
| `1` | the run broke (crash, provider error); a review panel whose every seat abstained |
| `2` | operator error (bad flag or config, a refusal before anything ran) |
| `3` | budget exhausted |
| `4` | finished over a red or never-run verify gate; a review panel's BLOCK |
| `5` | finished, but no commit landed and the edits sit uncommitted (a run that changed nothing, or one with `[git].commit_per_step = false`, stays `0`) |
| `130` | interrupted |

## Plan, review, and ask

```sh
agent6 plan "refactor the config loader"      # edit-free plan; run --from <id> executes it
agent6 plan show <session-id>                 # print the plan
agent6 plan edit <session-id>                 # open plan.md in $EDITOR (answer its open questions)
agent6 resume <session-id> --steer "answered" # the planner re-reads and revises
agent6 review --base origin/main --head HEAD  # read-only diff review
agent6 ask "how does the task-graph curator work?"
```

- `agent6 review --reviewers 3 --personas security,correctness,tests`: a panel whose findings are checked against the diff, so only real problems gate
  - a seat can pin its model, `security@openrouter/<model-id>`, the `[review].seats` grammar
- `review --path P` narrows the diff; `--model [provider/]model` picks the reviewer for this review (a seat pinned in `[review].seats` keeps its own)
- `ask` runs in any directory; headless (no TTY) under the default `run_commands = "ask"` it needs `--auto-approve`, `--no-commands`, or an away-mode (`AGENT6_DETACHED_AWAY=wait|deny|approve`)
- `run` and `plan` need a git repository (branches, diffs, merges)

## Run options

- `--preset <name>`: a strategy preset (`standard`, `quick`, `ultra`, `paranoid`, or your own; the [presets table](config.md#presets) says what each sets)
  - `agent6 config presets` lists them; `agent6 config set preset <name>` persists one
  - a preset cannot change mid-run; `agent6 resume <id> --preset <name>` continues a stopped run under another
- `--model [provider/]model`: the run's model over every config layer, for the role the mode runs (worker for `run` and `ask`, planner for `plan`)
  - a bare id keeps the role's provider; `agent6 model <role> <provider>` lists a provider's ids
  - recorded on the run: a resume keeps it unless it sets its own `--model`, which is recorded in turn
- `--parallel 3` (or `provider/model-a,model-b`, one lane per entry; a bare id runs on the worker's provider): isolated fan-out lanes, auto-compared into a ranked report
  - the fan-out is a session of its own: `attach` follows it, `stop` ends it with its lanes, `sessions show` lists its lanes with their placement
  - its lanes nest under it in every listing, folded into a count
  - `sessions list --lanes` and `ps --lanes` list the lanes; Space in the TUI hub and the `lanes` line in the web hub expand them
  - also from the TUI and web composers, or mid-run via the `/parallel [spec] <task>` steer directive ([configuration](config.md#parallel))
- `--standing "hunt and fix bugs"`: a never-finishing fallback task the run re-enters when the queue drains
  - new work outranks it; it never passes, and only the operator retires it
  - one per run, and the operator's: `resume --standing` and `fork --standing` give one to a session that started without it, a session that has one keeps it, and the model's `add_task` has no such flag
  - `/standing <text>` sets it on a live run from any composer, retiring the goal it replaces
  - budget, stop, and the iteration cap still end the run; `workflow.standing_patience` ends it after that many re-entries in a row that ran no tool call, and by default never does
- `--pin "<text>"`: an instruction re-shown verbatim after every compaction restart, so it survives compaction (`/pin` does the same mid-run)
- `/retire <task id>` drops a task from a live run's graph, named by the number `/tasks` prints; an id the run does not hold is refused with the ones it does
- `/task <text>` adds work to a live run's task graph instead of steering it: the turn in flight never sees it, and the run works it once its open tasks drain (every composer takes it, and `agent6 steer ID "/task <text>"` from a script or another machine)
  - the first line names the task, the whole text is its spec, and `[prompt].revise_prompt` covers it as it covers the run's own task
  - the model may finish it but not retire it, and a run that ends over one names it in its receipt
- `--from <id>`: seed the run from another session (its task, outcome, diff, and its plan or ask transcript); a plan id with no task runs that plan; the run's manifest records the source and `sessions show` prints it as `seeded from`
- `--decompose`: break the task into a task graph up front (`prompt.decompose`); `--skill NAME` puts a skill in the prompt (`[skills]`)
  - an installed skill whose text gates on a person ("get your partner's approval before ...") drives an unattended run into `ask_user` on every task
  - `agent6 skills disable NAME` keeps such a skill out of the index
- `--session-id ID`: name the new session yourself (default: a generated id)
- `agent6 prompt show [--mode run|plan|ask|agent] [--json]`: everything the model receives on the first call (system prompt, tool definitions, the first user message)

## Other commands

```sh
agent6 tui [target]                  # the full-screen hub, or one session's view
agent6 web [target]                  # the browser UI on 127.0.0.1:7658 (see web.md)
agent6 acp                           # speak the Agent Client Protocol on stdio (see acp.md)
agent6 ps [--json]                   # live sessions across every repository on this machine
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

Config is layered, lowest precedence first: built-in defaults, the global `~/.config/agent6/config.toml`, the per-repo config, `--config FILE`, then a machine agent's `[config]` overlay.
A selected preset is one more layer, printed by `config show` as `preset`.
`--preset NAME` places it above the per-repo config and below `--config FILE`.
A top-level `preset` key in the global or per-repo config places it just above that config.

- every field has a default; security-sensitive fields default safe (a repo can be zero-config)
- `agent6 config show`: every effective value with the layer that set it
- the [configuration reference](config.md) documents each field
