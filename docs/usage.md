# Usage

Assumes agent6 is installed (see [installation](installation.md)).

This page holds the patterns every surface shares; the commands and flags are on [Command line](cli.md).

## Start a surface

- `agent6 tui` opens the full-screen hub in this terminal ([Terminal UI](terminal.md))
- `agent6 web` serves the browser UI on 127.0.0.1:7658, behind a secret it prints ([Web UI](web.md))
- `agent6 run "your task"` and the other commands work from any shell ([Command line](cli.md); `agent6 --help` lists them)
- `agent6 acp` speaks the Agent Client Protocol to an editor ([Editor integration](acp.md))

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
agent6 check                         # check the configuration and the provider credentials
agent6 model                         # the role assignments
```

The key lands in `~/.config/agent6/secrets.toml` (mode `0600`), shared across every repository.
`connect` prompts locally and executes nothing a remote returns.
On a terminal it ends by offering the default model when none is set, so `agent6 model default` is the second command only off a terminal or when you skip the pick.
`agent6 connect --status [NAME]` lists the configured providers' local credential state and a free live check, with the account and access-token expiry for bound sign-ins.
A ChatGPT sign-in adds the account's plan reading: the binding window's used percent and when it resets.
It refreshes and spends nothing and exits 0 even for a refusal.

```text
name        backend      authentication   state                                live                  plan
openrouter  openai       secrets.toml     ok                                   verified
chatgpt     chatgpt      ChatGPT sign-in  ok <account-id> expires in 9 d 19 h  verified              73% · resets in 2 h 10 min
claude      claude_code  none             ok                                   verified
```

A provider whose sign-in is not built in takes a hand-written [`[providers.<name>].oauth`](config.md#configured-oauth) table; `agent6 connect <name>` then signs in at that issuer and stores the grant, bound to the entry's settings.
`--flow browser|device|paste` picks the way in (default `auto`), including for `agent6 connect chatgpt`.

`--logout` attempts revocation only for a bound grant at its issuer's revocation endpoint and removes it locally either way.
Rebound grants are removed locally only; the provider configuration stays.

`agent6 check config` reports local credential refusals and exits 1; a passing local check does not verify remote authentication.

Every model call requires `[models.default]` with `model = "provider/model"`.
A model selector splits at the first slash: `openrouter/moonshotai/kimi-k2.6` selects the `openrouter` provider.
Each optional role inherits its unset `model`, `effort` and `temperature` directly from `models.default`, never from another role.
For example, `models.planner.effort = "max"` changes only the planner's effort.

| Role | Set with | Used by |
| --- | --- | --- |
| `default` | `[models.default]` | required baseline and the only fallback for every role |
| `worker` | `[models.worker]` | `agent6 run`, `agent6 ask`, their resumes, machine agent states, and model-written squash messages |
| `planner` | `[models.planner]` | `agent6 plan` and plan resumes |
| `reviewer` | `[models.reviewer]` | in-loop reviewers without a route, the default judge of `sessions compare` and `--parallel` ([`harness.parallel.judges`](config.md#harnessparallel)), and `sessions review` |
| `helper` | `[models.helper]` | helper summaries (`compaction.summarizer`), gisting (`compaction.elision_gists`), and verify-command inference |

```sh
agent6 model default anthropic/claude-sonnet-5
agent6 model helper anthropic/claude-haiku-4-5
agent6 model all openrouter/moonshotai/kimi-k2.6   # write the default and every role
```

`agent6 model` writes to the global config; `--repo` selects the repository layer.
`all` writes five tables at that layer.
`--model` on run or ask overrides the worker role above every config layer; on plan it overrides the planner role.
It does not replace the required default.
`agent6 config show` shows each inherited leaf with the default's origin.
An old worker-only config refuses without migration: rename `[models.worker]` to `[models.default]`.

## Your first run

```sh
cd your-repo
agent6 run "your task here"
```

agent6 edits your working tree and commits each editing turn to the run's own chain (an `agent6/<id>` branch by default).
Your branch, HEAD and index stay untouched until you merge the work with `sessions merge` or [`git.auto_merge`](config.md#git) merges it for you.

- commands ask for approval by default (`sandbox.run_commands = "ask"`): allow one call or the whole session
- without a terminal (CI, [cron](cli.md#scheduled-runs)) a run refuses to start under `ask` unless an away-mode is set; a hub-spawned run parks at a prompt until a front-end answers it ([approvals](security.md#6-approvals))
- the run ends when the model declares it finished, the operator stops it (`agent6 stop ID`) or a ceiling (budget, turns) stops it
- at a terminal it then asks for the next input: type to continue, `/exit` to finish
- `agent6 resume <id> --steer "<what to do next>"` continues a run that ended by `finish_session`
    - an answer saved for one of its questions (`agent6 answer`) is new work, so a bare resume delivers it

The CLI start line summarizes non-default settings, the execution target and the verify gate.
`sessions show` summarizes the latest execution's non-default settings.
`sessions show <id> --json` lists every execution's settings and their sources under `settings`.
It gives each role's recorded route and effort under `models`; an empty `effort_source` means the record did not capture effort.
The web run page also reads the recorded roles, not the live config.

## File references

An `@path` in a task or a steer you type inlines that file after the reference: `agent6 run "fix the parser in @src/parse.py"`.

- the path is relative to the session's workspace: the repository, or the run's own worktree
- at most 64 KiB of a file is inlined; a longer one ends with the count of bytes left out, which the model reads with `read_file`
- a path that is missing, a directory, outside the workspace, inside `.git` or under [`sandbox.hide_paths`](config.md#sandbox) stays as typed
- it works in the task of `agent6 run`, `plan` and `ask`, started from the command line, the TUI or the web
- it works in a steer sent to a live session with `agent6 steer`, the TUI or the web, and after `/now`
- the web and TUI composers complete a path after `@`
- sent as typed: the text of any other directive (`/pin`, `/task`, `/parallel`), an instruction that resumes an ended session (`resume --steer`), and a line typed at a foreground run's pause menu
- text a model wrote is never searched: a plan run with `--from`, a seeded session, a parent's `steer_child` message

## Images with a message

The web composer and an editor over ACP send images with a task or an instruction.
The model receives each as an image, as it does one `read_file` reads.

- on the web, paste or drop an image on a composer, or press **Image**; an editor puts `image` blocks in its prompt
- a message carries at most four images: static PNG, JPEG, GIF or WebP, each at most 2 MiB, 4 MiB together
- an image can be the whole message
- anything else is refused with the reason and nothing is sent: an SVG, a PDF, an animation, an image whose bytes are not the type it was sent as
- an image goes with the task of a run or a plan, with an instruction to a live session, and with the instruction that resumes an ended one
- an answer to a question, `/btw` and every other directive, an `ask` started from the web, and a `/parallel` task take no image and say so
- the message's text names each image, as `[image image/png, 34 KiB, sha256 1a2b3c4d5e6f]`; the journal, `sessions transcript` and every conversation view show that label
- the bytes are stored in the session directory, outside the workspace ([security](security.md#images-an-operator-sends)); a resume reloads them, a fork copies them, and a workspace pack carries them with the rest of the session
- when the model is not shown an image, its message gains a line that says why, and the conversation shows the same line: a model route without image input, a provider's rejection, a missing or changed file, an image older than the newest four of a request
- the CLI and the TUI send no image: name the file's path and the model reads it with `read_file`, from the workspace or from a folder in `sandbox.extra_read_paths`

## Follow and steer a run

Every surface follows a run and sends it the same steers; the command-line verbs are on [Command line](cli.md#follow-and-steer-a-run).
Plain and immediate steers end a `read_background` wait with the output collected so far.
The run reads the instruction at its next turn boundary, if it takes another turn.

A session's usage covers every execution, in dollars or in plan points, and in both for a run that spent both (`1pt + $0.03`); `sessions list`, `sessions show`, the end-of-run total and a ranking's table print the same figure.
A run that spent nothing reads `$0`; unpriced calls make the figure a lower bound (`~$0.03`), or `$?` with nothing priced beside them.
These points measure account-wide window movement between readings, accumulated across resets, not the run's own spend; other consumers sharing the plan contribute too.

The tokens and the points against `max_plan_points` on `sessions show` are the latest execution's.
`elapsed` on a resumed run is the wall time from its first start to its last event, the time between executions included.
`N questions unanswered` beside a status counts the questions of the current execution that nobody was attached to see; a resume starts the count again.

The steer directives work from every composer, the pause menu, and `agent6 steer ID "/task ..."` from a script.

- `/task <text>` adds work to the task graph (the turn in flight never sees it); `/retire <id>` drops a task, by the number `sessions graph` prints
- `/metric` shows, changes or retires [the run's score](#measure-a-run)
- `/standing <text>` sets the goal the run returns to whenever its queue drains, replacing any it had
- `/pin <text>` survives every compaction; `/compact [focus]` compacts now; `/now <text>` interrupts the call in flight
- `/btw <question>` asks a side question; `/parallel [SPEC --] TASK` dispatches children ([parallel](config.md#harnessparallel))
- `/review [TARGET]` reviews changes in the session ([targets and worktrees](#plan-review-and-ask))
- `/undo` is available in the TUI, web, a `run -i` prompt, or a live run's pause menu (Ctrl-C), so `agent6 steer` refuses it:
  - it takes back the last message
  - the tree is committed on the run's chain
  - every tracked path the run changed since the checkpoint before that message is put back (one changed since the run ended stays, named in the notice)
  - the run ends `undone` and an unstarted fork in the same checkout continues it (`agent6 resume <fork> --steer "<the message, edited>"`)
  - `resume` and `sessions merge` refuse the undone run and name the fork
  - it is refused while another live run drives the checkout

## Answer a parked prompt

Every front-end answers a prompt by writing one file in the session directory, and so can a script.

- `agent6 answer <id>` lists pending questions and the live approval with their prompts and options.
- `agent6 answer <id> OPTION` answers the approval with `yes`, `no`, `session` or `session-deny` (session options only when offered).
  An approval takes priority unless `--question QID` is given.
- `agent6 answer [--question QID] <id> TEXT...` answers one question set, one TEXT per question.
  A queued question's answer reaches the run at its next turn, an ended run's at its resume.
    - `finish_session` polls for queued answers before ending and defers to another model turn if one arrived.
    - An answer arriving after that poll waits for resume.
- a question is pending from its prompt until its answer; a waiting one pauses the run while the execution that asked it lives, a queued one or one an earlier execution asked pauses nothing; every surface words it the same way (waiting, queued, asked by an earlier execution, answer saved)
- an approval reads `<session dir>/approvals/<id>.answer`: `yes`, `no`, `session` or `session-deny` (anything else denies; `session` approves this call, and `session` and `session-deny` also apply to later calls only where the prompt's event says `standing: true`)
- a question reads `<session dir>/questions/<id>.answer`: a JSON list of answers in the prompt's order
- the prompt's `id` is in `logs.jsonl` (`approval.prompt`, `question.prompt`); write the file atomically (a sibling, then rename), the run consumes it as soon as it exists

## When a run goes wrong

```sh
agent6 resume <session-id>           # continue from the last snapshot; --force past a diverged chain or run branch
agent6 fork <session-id> --at-turn 7 # a new run from turn 7; --steer seeds it, --no-run only creates it
```

State is snapshotted before each model call and checkpointed per turn.

- `fork` continues a copy of a run from a turn in its own git worktree ([worktrees](#worktrees)); the original run and your checkout stay as they are, and `sessions merge`, `prune` and `remove` treat it like any run ([session state](architecture.md#session-state-on-disk))
  A plan or ask fork uses the current checkout without edit tools or automatic commits; approved commands can still modify files.
  `sessions prune` keeps a fork that has not started: its branch, chain ref and worktree stay.
- `--at-turn 0` on a run that is itself a fork picks the seed it started from, before anything it did; the fork line says so
- an unmerged child continues through its parent's `steer_child`; direct `resume` refuses a child
- a run whose manifest cannot be read (a newer agent6 wrote it, or it is broken) refuses `resume`, `fork` and `sessions merge`, naming the cause, and no surface offers them
- a `run --parallel` run whose budget ended every child reads `failed · budget exhausted` and exits `3`; `agent6 fork <child-id>` starts independent work from a child's commits
- the first `--steer` on a fork becomes the fork's task and names its row and its merge subject; later steers are continuations
- a resume with `--steer` keeps the original session label, with the latest steer beneath it as `continued: <steer>`; new questions in an ask REPL are not continuations
- under the `agent6` and `conventional` squash message styles, a resumed run's merge keeps the task as its subject (`conventional` prefixes it with the type and scope), with `continued: <steer>` in the body
- a gate withheld after denied or interrupted approval stays withheld only for that execution.
  When the snapshot records that withholding, a resume with the gate and commands allowed tells the model once that the gate is available again, without claiming it passed
- after the run's work is merged, a resume continues from the checkout's HEAD if HEAD contains that merge: the run's next commits, `sessions diff` and its next merge then hold only the new work
- a run whose workspace does not contain the merge (a `--worktree` run, a fork) continues on its own chain: after a resume, `sessions diff`, `sessions commits` and its next merge start from the run's last merged tip
- a run merged `--into` a branch other than the one it started on keeps its base; another merge into that target starts from the run's last tip merged there
- `sessions merge` over a file you edited leaves your copy alone and prints two commands that merge your edit with the target version
- a successful `sessions merge` names the other ended, unmerged runs that changed some of the same paths (`unmerged runs that changed some of the same paths: <id> (src/a.py and 2 more)`)
  The TUI's and the web's Merge confirm states that line before the merge.
- a commit you made on the run's `agent6/<id>` branch refuses the resume, which names it: `agent6 sessions merge <id>` merges that commit with the run's work, or `resume --force` continues without it
- a `sessions merge` that conflicts names `agent6 resume <id> --merge <branch>` for a run with its own worktree: the resume merges the branch into the run's work there and leaves conflict markers for the run to resolve before the next `sessions merge`
- files you changed between executions follow `[git].dirty_tree` on resume, as uncommitted changes do at a run's start: `ask` asks, `stash` sets them aside (a change under `[sandbox].hide_paths` stays in place), `include` leaves them in place, named to the model before its first call, for the run's next commit to take ([Config](config.md#git))
  Under `git.control = "model"`, resume names the uncommitted files that still match the run's last stop and leaves committing them to the model.
  It compares both working and staged contents.
  For a stop without a staged-tree record, a difference from the recorded commit remains unconfirmed.
- when `stash` refuses over a file the run also changed, its printed `git restore --source ...` command discards only your edit and keeps the run's last tree; plain `git restore FILE` returns to HEAD, which can remove the run's work
- a new untracked file is left out of the run's commits as yours, unless the last execution died mid-turn (a kill, a crash): a command of that turn may have written it, so the run keeps it, and the resume names up to four files either way

Every end has one reason and a headline every surface shows; [Exit codes and end reasons](cli.md#exit-codes-and-end-reasons) lists them with the exit codes scripts branch on.

A run whose worker exited without recording an end reads `stale`, for example after SIGKILL or a machine crash.
`agent6 resume <id>` continues it from its last snapshot, and `agent6 sessions show <id>` prints that command.
A resumed run opens with the verify fact: the last full verification's command, verdict and tree, and whether the tree changed since; every compaction restart carries the same fact.

## A row's merge note

Every listing row says what stands before a clean merge, derived from the journal on each read and never stored:

- `open findings: 2 block, 1 warn`: the run's open findings by severity; `agent6 sessions findings <id>` lists them

The status beside the note carries the rest: what a waiting run waits on, the gate's word, `unmerged`, and the questions pending or the answers saved for the resume.
Every listing orders sessions by last activity, newest first.
`sessions list --json` carries the counts as `open_findings` and the rendered note as `merge_note` ([the row](formats.md#cli-json)).

## Verification

The verify command is the success gate.

- Pin one with `harness.verify.command` or `agent6 init`; unset, each run infers one and prints it, or runs gateless when nothing is inferable.
- By default the model can run it (`run_verify_command`) and the harness never does: an end that no run of the gate covers reads `unverified` and exits 0.
  `harness.verify.when = "finish"` makes the harness run it when `finish_session` arrives over an unverified tree, and `"turn"` after every editing turn.
- With `harness.verify.at_start = true` and `harness.verify.when` set to `"finish"` or `"turn"`, a fresh run runs the gate once before its first turn, so a gate red at the base is known as that.
- Under `finish` or `turn`, a red gate at finish returns to the model `harness.verify.retries` times, then the run ends red.
- An unverified run's end guidance counts visible uncommitted changes and new untracked files as work, regardless of Git control or commit policy.
  Files untracked at start, hidden paths and ignored output stay excluded.
- The red-gate end block reads the session's first recorded baseline, including earlier executions before a resume: a green base attributes the red gate to this run's edits, a red base says the gate was red there too.
- A fork starts its own log without the parent's baseline; its base is the commit it forked from.
- [`[harness]`](config.md#harness) has the inference order and `harness.verify.when`.
- An inferred `cargo test --quiet` still needs a toolchain and dependencies inside the jail; see [Rust toolchains](security.md#rust-toolchains).

## Session findings

With `review.block_on = "none"`, a merge reports the open `block` count and proceeds.
With any other value, resolve every open `block` with a reason before merging.
Every agent6 merge path refuses unreadable review evidence.
A missing or stale review does not prevent a merge; legacy counts are not reconstructed findings.
The web [Findings card](web.md#finding-actions-api) offers the same actions and links each finding to its reviewed diff and file version.
In a run, the worker lists its own findings with the read-only `list_findings` tool; recording a response stays with the commands below.

**Pre-existing line** means the cited line bytes exist at the reviewed base.
It does not prove that the defect predates the change: a changed caller can break an unchanged callee.
The tag does not lower severity, close a finding, or exempt a block from the merge guard.
`sessions findings` prints the mapped base range and attribution commits; `--json` includes each observation's `line_origin` and the latest evidence on each finding.
File-only citations, unsupported files or base paths, ambiguous mappings, unavailable history, and blobs or patches over 4 MiB have an unknown origin with a reason.
Historical observations without this evidence remain unknown.

```bash
agent6 sessions findings <id>                    # open findings and review version
agent6 sessions findings <id> --all --json        # include resolved findings and action history
agent6 steer <id> --finding <finding-id>          # submit Fix to a live worker
agent6 steer <id> --finding <finding-id> --resume # confirm Resume to fix for a stopped run
agent6 steer <id> --finding A --finding B         # one steer carrying several findings
agent6 steer <id> --all-findings                  # every open finding without a queued Fix
agent6 sessions resolve <id> <finding-id> --as fixed --reason "Reproduced and corrected"
agent6 sessions reopen <id> <finding-id> --reason "Still reproducible"
```

- Resolve accepts `fixed`, `false_positive`, `intentional`, `wont_fix`, `duplicate`, or `pre_existing`.
  `duplicate` also requires `--duplicate-of <another-finding-id>` in the same session.
- Resolve and Reopen require a nonblank reason of at most 2,000 characters.
  `fixed` records the operator's conclusion, not a verification result.
- Fix builds its instruction from the stored finding and reviewed target, and asks the worker to reproduce the report before changing code.
  It leaves the finding open, queues behind a steer in flight, and refuses a second Fix of a finding whose Fix is still queued or of unknown delivery.
- Several findings share one steer: a block per finding under one heading, one result line per finding.
  `--all-findings` takes every open finding without a queued Fix; the web and TUI Findings card has **Fix all open** for the same.
- `--resume` confirms the background resume of the same session.
  - A successful launch reports `run resumed to fix finding <id>`.
  - `; submission acknowledgement pending` means the worker has not yet recorded taking the instruction.
    A recorded submission reports the same message without the qualifier.
  - An unavailable resume reports its failure and does not create a new run.
- Each action prints its request ID and resulting review version.
  `--request-id <id>` retries that action without sending it twice; `--expected-version <n>` refuses a stale view instead of using the current version.
- A live worker applies actions at its steer boundary.
  An action still waiting after five seconds reports `action queued`; an approval or question wait must end before the boundary is reached.
- `steer submitted` means a live instruction entered the steer bridge.
- `Fix: submitted` means the fix instruction was submitted, not that the fix completed or was verified.
  Run completion, commits, passing verification, and later clean panels do not resolve findings.
  Resolve records that conclusion with a reason.
- Fix retries read the journal, not the launch confirmation from the initiating response.
  Without a recorded submission or failure, a retry reports `delivery unknown` and never launches or sends the instruction again.

## Session receipts

`agent6 sessions receipt <id> [--json]` reads testing, review, cache use and reversal evidence from the journal and manifest.
It makes no provider call and reruns no checks.

The receipt and the run-end token summary show ordinary worker cache use across the session, including resumed executions but excluding summary attempts, helpers and children.
`Worker input cache` is cache-read tokens divided by fresh input, cache-read and cache-write tokens together.
`After compaction` counts the first logical worker call after each history rewrite and its fresh plus cache-write input tokens.
Several rewrites before one call count once; requested, refused and failed compactions do not count unless they also rewrote history.
These figures use `model.result`, which records the last attempt's usage, not all retry attempts or a causal estimate of extra cost.
Calls without input or cache-read counts are excluded, partial coverage is named, and no cache line appears when none report counts.
Historical journals that recorded an unreported count as zero cannot distinguish it from a reported zero.

Each `Summary attempt` line names its writer, cached and uncached input, outcome and any fallback reason.
An `endpoint` attempt reports the billed summary iteration.
When the endpoint drops thinking or clears tool results, the receipt shows the session counts.
If the worker or endpoint reads less than half its idle summary input from cache, the log and receipt name the limit to shorten.
This notice appears once per execution and only for reported cache counts.
`unreported` means the provider supplied no count.

- **Testing:** the receipt includes each [recorded verification attempt](formats.md#verificationevidence).
  A pass certifies only its unchanged tested tree; a scoped pass covers only the selected tests.
  An edit after a pass or a command that changes the tree leaves the final tree uncertified.
  A different commit with the same tree content does not invalidate the observation.
- **Output:** each journal stream keeps its last 2,000 characters, with truncation marked.
  Historical or incomplete captures read "capture completeness unknown".
- **Missing checks:** denial, inability to execute and interruption are not passes.
  Baseline failures stay marked separately; a run without a gate says "No verification recorded".
  Other commands are not verification evidence.
- **Review:** the receipt shows recorded panel state and names each reviewer's session.
- **How to revert:** recorded merge bounds determine the explanation.
  A squash names its commit on the target; a merge verifies its target parent; a fast-forward lists the merged commits in reverse order.
  Fast-forwards containing merges need an explicit plan.
  Missing stamps or objects do not produce guessed commands.
  "No merge recorded" does not claim that no merge occurred if recording failed.

Git commands in a receipt are explanations only.
**Ask a run to revert** in the web or TUI opens the new-work composer with a proposed task; inspect it before submitting.
You can also copy the recorded merge details into a steer request to prepare the reversal.
The reversal needs its own checks; the original receipt does not certify a merge against a moved target or the later reversal.

## Plan, review, and ask

`/review [TARGET]` works as a first message or a live steer, with no flags.
`/review` covers staged, unstaged and untracked changes against HEAD; a clean tree answers `no uncommitted changes to review` without calling the model.
`/review REF` reviews the ref's changes from its merge base with HEAD (`HEAD...REF`).
`/review A..B` compares the endpoints; `/review A...B` compares their merge base to B.
The argument always names the reviewed side: use `/review main...HEAD` for the current branch against main, or `/review SHA^..SHA` for one commit (a root commit uses the empty tree).

Start with `agent6 ask "/review main...HEAD"` for a review without edit tools, or `agent6 run "/review feature-x"` to allow fixes too.
Both modes keep the session's sandbox, approvals and git control, and can run approved checks.
A clean checkout at the range's head is reviewed in place; otherwise startup prepares a detached worktree at that head under `git.worktree.base_dir`.
A live session cannot change trees: a request for another tree names the `agent6 ask "/review TARGET"` command to start one.
A bare run review cannot stash the changes it reviews; use ask or choose `git.dirty_tree = "include"`.
The first message carries resolved revisions, a diff stat, up to 64 KiB of patch and the base tree's `AGENTS.md` rules.
With the review panel on (`[review].when` not `never`), the panel sits on the reviewed range as the session starts and records its findings like an in-loop panel: the Findings card, `sessions findings`, Resolve and Fix read them, and the worker's answer follows with the findings in hand.
Fix needs edit tools: an ask session's findings are read and resolved, and `agent6 run "/review TARGET"` fixes them.
With the panel off, the answer is the worker's markdown alone.

```sh
agent6 plan "refactor the config loader"      # no edit tools; run --from <id> executes it
agent6 plan show <session-id>                 # print the plan
agent6 plan edit <session-id>                 # open plan.md in $EDITOR (answer its open questions)
agent6 resume <session-id> --steer "answered" # the planner re-reads and revises
agent6 sessions review <session-id>           # read-only review of a finished run's record
agent6 ask "how does the task-graph curator work?"
```

- `sessions review` reports how the run ended, what went wrong, what you corrected, and candidate memory facts and AGENTS.md lines with their evidence; it touches no repo or memory file (`agent6 memory add` lands a candidate)
- every `sessions review` is saved under `<state-dir>/reviews/`; the path prints on stderr
- `ask` runs in any directory; `run` and `plan` need a git repository

In-loop panels run each reviewer as a child with its own session, and record their findings and reviewed tree IDs in the session journal.
Each batch's reviewers start under what the run has left; batches use at most `review.concurrency` and `harness.parallel.max_children` reviewers, without truncating the roster.
Results keep roster order even when reviewers finish in another order.
A reviewer opens on the task, the rules, the verification and the diff, and investigates a clone of the retained snapshot with the tools its run has: the reading tools, and `run_command` with its background commands, `run_verify_command`, `fetch`, `read_session` and `use_skill` where the run's model is offered them.
A reviewer edits nothing and dispatches nothing; `sandbox.run_commands = "deny"` withholds `run_command` and `run_verify_command` from a reviewer as it does from the run.
A reviewer takes at most `review.max_turns` turns.
Each turn tells it which reply it is (`reply 14 of 20`), and the last that it is the last: a tool call in that reply is not run, so the verdict is due then at the latest.
Each reviewer has a session of its own, listed under its parent rather than as a top-level run.
The findings card links to the reviewers on its panel.
A reviewer still running 30 minutes after its batch started is stopped and abstains; batches run one after another, each with its own 30 minutes.
Failed or unparseable reviewers remain abstentions with their diagnostics; an all-abstain panel is inconclusive, not a clean review.
Child review does not certify a later or different parent tree.
Session-owned panels retain a cloneable snapshot of the exact comparison, including uncommitted work after failed verification and with `git.commit_per_turn = false`.
Snapshotting does not checkpoint the run or move HEAD, the index or visible branches.
`sessions remove <id>` deletes the session's review snapshot refs, its children's records and its reviewers' records with its record; `sessions remove --asks --yes` does the same for the removed asks, while `sessions prune` leaves review snapshots alone.
The snapshot certifies only its recorded trees and path coverage, not later parent edits.
Under `git.control = "model"` a panel compares the base recorded at the start with the checkout's HEAD and its uncommitted edits, since the model's commits are the record.
A reviewer returns a summary and findings, with no separate vote.
Each finding carries the reviewer's category and may include a suggested fix as text or a patch.
The suggestion is evidence for the worker and the operator, not an edit.
A repeat on the same cited blob keeps its finding ID across rounds and resumes, even under a different label or suggested fix.
The worker receives each finding once unless the feedback notice cut it.
If a suggestion alone would hide the first finding, the notice marks the cut and the recorded finding keeps the full suggestion.
A later pass, a skipped panel, or the rejection cap does not erase recorded findings.
A panel asked to review trees a complete panel already reviewed with the same paths and reviewers is not sat again: the journal records `review.reused` naming that panel, whose findings stand, and the worker hears that nothing new was raised.
`agent6 attach --json` includes structured review evidence, open counts, action history and completion state.

### Code-review rules

Write review rules in `AGENTS.md`.
Code reviews read `AGENTS.md` from the reviewed base tree, from the repository root through the working directory.
A head-side edit or deletion does not change the rules supplied to that review.
Rule-file changes remain in the reviewed diff unless the selected paths exclude them.

Missing files are optional.
An unborn repository has no base rules.
A present rule file that is hidden, unreadable, non-text, or not a regular file makes the review incomplete with a named reason.
The combined rule text must fit within 64 KiB of UTF-8 bytes; an oversized set refuses the review rather than supplying a prefix.
Reduce the base-tree rule files to proceed.

The review journal records each supplied path, blob ID and byte count once at panel start.
These rules apply to the in-loop panel, not to `sessions review` or the worker's own `AGENTS.md` loading.

## Worktrees

`run --worktree`, a run-mode fork, each `--parallel` child and a run-mode machine state work in a worktree of their own.

- `[git.worktree].kind` chooses the worktree of a fork and a `--worktree` run: `clone` (the default), a clone of the repository whose commits agent6 copies into the repository as the run makes them, or `linked`, a linked git worktree; a child, a reviewer and a machine state work in a clone.
- It starts with the repository's tracked files at the commit it was cut from: no uncommitted change, and nothing git does not track (`.venv`, `node_modules`, a build directory).
- `[git.worktree].setup_command` builds what is missing; it runs once in each new worktree, before the first turn:

    ```sh
    agent6 config set --repo git.worktree.setup_command '["uv", "sync"]'
    ```

- Unset, a `## Setup command` fenced block in AGENTS.md supplies it; a `--worktree` run and a child read it from the commit the worktree is cut from, not from an uncommitted edit, and print the command and that AGENTS.md gave it.
    - an AGENTS.md that is a symlink is not read there: set `git.worktree.setup_command`
- The command runs in the run's own sandbox, under the isolation, network and grants of every command of that run; with no route out it has to work offline ([the offline recipe](security.md#4-commands-and-environment)).
- Machine clones follow the [state setup and failure rules](state-machines.md#agent) and [setup approval policy](security.md#4-commands-and-environment).
- For run, fork and child sessions, the command and output tail are printed when it ends; a nonzero exit ends the session `failed · setup command failed` before any turn (exit `1`).
    - A fork's unread steer survives the setup failure and reaches the worker on a bare resume.
  For a run or child, no turn ran, so `agent6 resume` has nothing to continue: start the run again once the command works.
- A run prints a child's failed setup command and its output when it imports the child; `agent6 sessions show <id>` keeps the command, and a failed setup's output, after the worktree is gone.
- The files it leaves untracked stay out of the run's commits.
- The verify gate is settled in your checkout and carried into the workspace: a gate inferred as `.venv/bin/python -m pytest -q` stays that.
- A gate whose first word is a path git does not track, with no setup command, refuses `run --worktree`, `fork` and `run --parallel` before anything is created; the refusal names the path, the config line and the AGENTS.md block.

## Children

### Parallel directive

`/parallel [SPEC --] TASK` runs a task in isolated children, including as the first prompt to `agent6 run`, through the same dispatch as `run --parallel SPEC TASK`.
A child belongs to the run that dispatched it: the parent hands down, the child pushes up, and the root is the one owner.
The child gets its config, task and remaining budget from its parent; its approvals, questions and spend go up to the parent while it runs; its work and memory reach the repository only when the parent imports it at its end.
Inspect the parent with `agent6 sessions list --children`, `agent6 sessions show <parent>` or `agent6 attach <parent>`.
The children list labels each work child by its attempt number within its task: two separate tasks each start with `attempt 1`.
It gives each child's task, model, status, current tool, turn, tokens, spend and elapsed time.
The child's status comes first, then the parent's outcome, such as `passed · ranked 2 of 2`.
The parent reports when a child starts, reads an operator message, stops or finishes.
`merged into parent chain` in the transcript means the child's commits are on the parent's chain, not necessarily on the checkout branch.
`sessions show <child>` reports `parent merge: merged into PARENT's chain at <sha>` and shows the checkout merge separately.
`sessions show <parent>` shows the parent's recorded checkout merge or how to merge its work.
A failed launch has a row with its reason even when there is no child session to open.
A work child shows its measured number of commits beyond start, including while it runs.
A stopped child also shows the ref that holds its work.
A reviewer shows its order and summary when it returns a verdict.
To act on one running child, use `agent6 steer <child-id> "<message>"` or `agent6 stop <child-id>`.
With `git.control = "agent6"`, every entry merges a task once: one child's work is merged onto the run's chain, and several children on one task are attempts, of which the first-ranked merges.
With `git.control = "model"`, a finished child completes its task.
Its row names the retained ref without claiming a merge.
The model can merge that ref with git.
`run --parallel` waits for its children without a model call of its own: [`harness.parallel.judges`](config.md#harnessparallel) rank several children, the first-ranked child follows this merge policy, and the run ends `finished · children ranked`.
The parent's gate still runs when its policy calls for it.
Without a parent result, a child can supply the pass if its full verify command and final tree match the parent's.
`sessions show` and `sessions receipt` name the child that verified it.
A different tree or command, or a scoped check, leaves verification to the parent's own gate policy.
A count in `judges`, such as `"2"`, starts that many judges.
A judge is a reviewer child of the run: its clone is cut at the commit the attempts started from and holds every attempt, its opening lists each attempt with its verify result, cost, finish summary and diff, and it has the tools a reviewer is handed (the reading tools, and `run_command` and `run_verify_command` where the run may run commands).
It answers with an order and its reason, within `review.max_turns` replies; the orders are summed, and the lowest sum wins.
With no judge, with one candidate, or when no judge gives a usable order, the order is verify-pass first, then the cheaper, and the merge notice says which (`Ranked by verify result, then cost: no judge gave a usable order (reviewer: no verdict within 20 replies).`).
A judge's findings are not kept as a review: only its order and reason are.
Under agent6-owned git, the other children keep their branches, so `agent6 sessions merge <child> --into <branch>` merges one by hand.
`SPEC` is a comma-separated list of positive counts, role names or `provider/model` entries, the same as `run --parallel`.
Each count starts that many children on the worker settings.
A count above one, or several selectors, means attempts at the one task; a second `/parallel` in the message is a second task.
The TUI and web composers preview the number of children and the spec for each task as you type.
At the `run -i` commit prompt, enter the line to preview it.
Repeat the same line to send it, or type another command.
A selector is an exact role name (`default`, `worker`, `planner`, `reviewer`, `helper`), which brings that role's model, effort and temperature together, or `provider/model`, which runs that model with the worker's effort and temperature; a model named like a role is written `provider/reviewer`.
The spec ends at a whitespace-delimited `--` token.
Without `--`, the whole segment is task text for one default child, even if it starts with a path, a number or a model name.
A bare `--`, a missing task, or more children than `[harness.parallel].max_children` allows to run at once, counting the children already running, is refused with the reason.

```text
/parallel src/x.py: add a test
/parallel 2 -- fix the parser
/parallel a/b,c -- compare the designs
/parallel 2 -- task A /parallel 3 -- task B
```

Repeat the `/parallel` token to batch tasks in one message.
Each segment has its own optional spec and separator.
Task text keeps its internal whitespace and newlines, including later `--` tokens.

A live run starts the children and goes on: its model keeps taking turns while they work.
When a child ends, the model reads a notice with the summary the child finished with (`summary: ...`), so a child sent to find something out answers its parent without a commit.
A task completes at the run's next turn boundary once its children have ended; a task still running holds no other.
The judges start when the last attempt ends and run in the background while the run takes its turns.
The model's roster shows the attempts as `being ranked`.
The task completes at the boundary after the judges end.
While a child's own reviewers are working, the child is `in review` and still holds the parent's finish.
Under agent6-owned git, tasks that finish before the same boundary merge in the order written, so when two touch the same lines the later one to merge gets the conflict.
Before an automatic merge, the run's uncommitted work is committed first.
One notice per task names each child's outcome: the commit on the parent chain if it merged, no changes to merge if it finished without a commit, `ranked 2 of 3` for an attempt that did not merge, `CONFLICT` with the files (the child's branch still holds its work), or `FAILED` with the reason.
A stopped child never merges by itself, whoever stopped it: its parent, the operator, the run's budget or the end of the execution that started it.
Its end notice and its line say what stopped it, that it did not merge, what holds its work, its conversation size at the last measured provider call, and how long ago it last ran; the model decides what to do with it.
Missing measurements read as unknown.
Among attempts a stopped child is no candidate; a task whose every child stopped merges nothing, says `Every child stopped: nothing merged.` and is marked failed.
Under agent6-owned git, `agent6 sessions merge <child> --into <branch>` merges a stopped child's branch by hand.

### The parallel tool

With `harness.parallel.tool = true` ([parallel](config.md#harnessparallel)), a run's model can call `parallel` with `task` text or a ready `task_id` from `list_tasks`.
`spec` has the same grammar as `--parallel`, empty for one default child.
With `{"task_id":"0002"}`, the child gets task `0002`'s title, rationale and acceptance.
The child holds that task until its result returns.
The parent's cursor moves to the next ready task or clears if none is ready.
The task passes or fails with the child's result. A refused launch leaves the task available.
A pending text dispatch under the task must return before the parent can hand off that task.
A requested merge holds its task until the merge result is recorded.
Merging a continuation's retained result records the merge commit on the task it passed.
Merging another attempt leaves that passed task alone.
Continuing a child after its task has passed or been retired leaves that task alone.
One call is one task.
The call starts its children in fresh clones of the committed tree, under the run's shared remaining budget, and returns at once with each child's session id.
With `git.control = "model"`, the model must commit its changes before calling `parallel`; agent6 does not checkpoint a dirty tree for it.
Each child is an ordinary run with the same tools as the parent and may edit its clone.
After import the clone goes, unless the child stopped or the clone holds work the parent cannot preserve.
Before cleanup or prune, the parent pins the child's recorded working and staged trees under `refs/agent6/<child>/trees/` and checks both pins.
Both operations keep a clone if its `HEAD`, index or working tree records a gitlink the child's base commit does not, or if a gitlink's path holds a nested checkout.
A gitlink names a commit, not the files in the nested checkout.
Its finished work follows the [parallel merge policy](#parallel-directive).
A child runs under the parent's own config and the run's remaining budget, with no turn cap or deadline of its own.
`read_background` with a child's id returns its id, attempt, state, task preview, `running`, `mergeable` and commit count beside its transcript tail.
`mergeable` marks stopped work or a continued child whose commits await an explicit merge.
A child whose launch was refused reads `never started`.
A merge can still be refused, for example if another attempt already merged.
With a command's `background_id` it returns the command preview, state, `returncode`, `duration_s` and `still_running` beside the output tail.
The duration runs from launch to the present while the command runs, and to its end when a `read_background` wait or a stop saw it end.
A command first found ended by a later look has a null `duration_s`.
If the launcher is lost before it reports an exit, `still_running`, `duration_s` and `returncode` are null.
With no id it lists one line per command or child; a positive `wait_s` waits for the next child to end.
A child tail shows the same bounded tool arguments and result summaries as the numbered journal entries.
A separate shell line gives a background command's exit when its result is on disk.
See [Session readers](security.md#1-trust-boundary) for what the project shares.
Every background wait uses the shorter of `wait_s` and [`harness.command_checkin_s`](config.md#harness); a check-in of 0 disables this cap.
With an id, an omitted `wait_s` uses the check-in interval; without an id, it returns the roster at once.
When the check-in ends a wait, `output` names the interval.

`steer_child` (`id`, `text`), offered with the `parallel` tool, sends an instruction to a running child or continues an unmerged child.
The same child conversation continues under the parent's remaining budget and approval policy.
The task's other attempts must have ended, with no merge pending.

- A stopped child continues in its kept clone.
- A failed or finished child continues in a fresh clone at its last commit, including an attempt ranked below the winner.
  The recorded working tree and index are restored, then the clone's setup command runs again.
- If a kill left no end tree, the notice says that uncommitted files could not be restored.
  Missing recorded trees, commits or conversation data refuse continuation.
- Under agent6-owned git, work already merged into the parent cannot continue; `parallel` starts a new child from there.
  Under model-owned git the merge status is unknown; the parent decides whether to continue the child.
- A continuation that never starts leaves the child's earlier result as it was; `steer_child` can send it again.

For a running child the instruction queues with the operator's steers on the channel `agent6 steer <child>` uses.
If the child takes another turn, it reads the message under `PARENT STEERING (an instruction from the run that started this child):`; a child can finish review without taking another turn.
A parent's steer is text for the child's model: `abort`, `/pin` or `/parallel` in it are words, not commands.
It is refused for an id that is no child of the run, or one that has not started.
`stop_background` with a child's id stops the child; its work does not merge, and its branch holds what it committed.
The roster and resumed parent state show the same stopped-child facts as its end notice, with the age updated.
The two tools' descriptions mention children when the run can start a child or already has one.
`finish_session` is refused while a child runs or its task awaits completion at a turn boundary, and the refusal names the children that hold it.
A stopped child holds no end unless its parent requests a merge or continues it.
`merge_child(id)` requests the merge of a stopped or continued child's commits at the next turn boundary and holds the end until its outcome is reported.
It is offered only while the run has stopped or continued work awaiting a merge and `git.control = "agent6"`, including children started by `/parallel` when the parallel tool is off.
A pending merge of another attempt refuses the call.
A conflict merges nothing, names the files and the ref holding the work, and permits another request.
A requested merge survives a parent resume.
The run counts its children's spend while they run and ends them once the sum reaches a cap, so the total can pass the cap by what was in flight.
The tool's description names every role's resolved model, effort and temperature and the configured providers, so the model can pick a selector such as `reviewer`.
The two tools are offered only in run mode, never inside a child and never in a front-end without a child dispatcher; a call that slips past the list is refused with the reason.
A worker child's configured harness panel may still run its reviewers; neither the worker's model nor its `/parallel` steer may dispatch work.
The knob neither authorizes nor disables `run --parallel`, the `/parallel` steer or the review panel.

A `run --parallel` run made no model call: `agent6 resume <id> --steer "..."` continues it with the worker model from the parent's tree, including any merged child work.
`agent6 stop ID` on a session with live children stops its descendants too, by the lineage their manifests record, including reviewers behind a worker that already ended.
Worker import preserves descendant records and receipts in the original repository's state directory before clone cleanup.
Their spend is included in the worker's total, not an extra charge, and their evidence certifies only the child's reviewed target, not the top-level tree.
A child that needs an approval, or waits on an `ask_user`, asks on the run that dispatched it: the prompt names the child, and a standing answer given there covers the run's children.
A question nobody answers on that run (`away mode deny`) returns `unanswered` to the child with the same note; a question the child queued (`wait: false`) stays on the child's own row.
Children end with the execution that started them: when the run stops, or ends any other way, its running children are stopped and their spend is charged.
Before its first call a resumed parent sees the ended children it can continue.
Agent6 retains their work; a continued child only merges at its parent's request.
A task whose children had ended by themselves when the run ended completes then under the same merge policy as a turn boundary; each child's spend is settled once.
`sessions list --children`, `sessions show` and the receipt list attempts and reviewers by dispatch group and number.
The receipt's `Children` section names the settled spend already included in the run's total.
The end summary splits this execution's fresh input and output tokens between the parent and its settled children, including ranking reviewers.
The receipt shows the same counts across all executions; `unreported` means an older record lacks the count.
The refusal names the child session IDs.
While a steer waits on disk, `sessions show`, the TUI header and the web session status show `steer queued, not yet delivered` with its text and the count waiting behind it.
Steers sent while one waits queue in order, one per subsequent step; `abort`, `exit`, `/undo` and `detach` take the slot at once.
The pending label clears when the worker consumes the last steer.

## Compaction

**Who compacts.** The threshold line at startup names `agent6` or `endpoint`.
With [Anthropic server compaction](anthropic.md), the endpoint clears old tool output and writes a signed summary.
The size settings, `/compact [focus]` and idle limits still apply.
The cold-start mode, summary writer, gists and history retention settings apply only when agent6 compacts by itself.

**Resume with a cold cache.** A resume is cold if the provider or model differs from the last ordinary worker request in the journal.
It is also cold if the time since that request began reaches the current provider's `idle_compact_s`.
A helper call, summary or session stop does not reset that clock.
A fork that has made no request of its own takes the route and time from its source's last request.
For a process-backed provider, streamed output or a successful attempt confirms the route and start time of its worker call.
This also applies when live text is off.
Set the limit to `-1` to disable the age rule but still detect a route change; `0` is not valid.
The wait setting `idle_compact_min_chars` does not apply at resume.

`compaction.cold_start` decides what to do before the first worker call, without a question, including when a parent continues its child:

- `drop` (default): replace every eligible old tool result, even below `drop_at_chars`.
  The newest two results, unread results, operator answers and reads of recently edited files stay whole.
  Thinking outside the newest `keep_thinking_turns` replies goes too, even when no result is replaced.
  Only `elision_gists` can call the helper during the drop.
  The usual size and manual compaction checks follow on the same request, as after `resend`.
- `resend`: keep the saved conversation for the usual size and manual compaction checks.
- `summarize`: ask the helper for a summary and restart with it and the retained history.
  The helper uses a pending `/compact` focus in that summary.
  An unusable summary leaves the history intact.
  A terminal helper error records that outcome and ends the run before the worker call.
- `truncate`: restart with the retained history and no summary call.
  The notice states what was left out and points to `read_session` when that tool is available.

`summarize` and `truncate` replace only what the retention settings do not keep.
When those settings keep the whole conversation, it is sent as it is, no helper is called, and the action reads `resend`.

A `summarize` or `truncate` restart is the only compaction on that request, and so is a failed cold summary.
The next ordinary check handles size and any pending `/compact`.
The start line and `compact.cold_start` journal entry report the route change or the gap and limit, the action, and the net context characters the cold step did not send to the worker.
This is not a token count or evidence that the endpoint evicted the cache.
At a local restart, the planner keeps one plan notice: the current `plan.md`, or its latest checkpoint copy if the fork has no file.
The planner reads `plan.md` again after local compaction on a cold start, including a `resend`.
It parks if the file becomes unreadable or disappears during compaction.
On an endpoint that writes signed summaries, the line reports that the local mode does not apply.

**What a local restart keeps.** `compaction.keep_operator_messages` reserves the newest messages the run was given, word for word: the operator's steers and answers, and the task as the oldest.
`compaction.keep_recent_turns` keeps an unbroken run of recent replies with their inputs and tool results.
Both counts default to `-1` (no count limit).
Set either to `0` to reserve no messages or keep no optional turns, respectively.
A kept turn still includes the operator text it received.
A pinned instruction is stated in the pinned block, so its `/pin` message is neither reserved nor counted.
A `/<skill>` message is reserved as the instruction that names the skill and its arguments, without the skill's text; `use_skill` loads the text again.
Older reserved messages appear in time order before the pins and any summary, including answers without their tool calls.
The restart notice leaves out a verify fact or metric notice that a kept message already states word for word, as a resume's own notices do.
The task is the whole task the worker received: the typed words with any inlined file, a plan's body, a parent's task for a child, or a machine state's prompt.
Every restart counts it as the oldest message and charges it to the cap like the others.
A restart states it once, under `TASK (the goal, verbatim)` when no kept turn holds it.
The local helper reads operator answers in the replaced history, including full answers to capped `ask_user` calls, but no other tool output.
The helper fallback for a signed endpoint summary omits those tool answers.

`compaction.keep_recent_chars` caps the two together (default `80000`); `agent6 config show` gives the effective cap for the worker.
After reserving those messages, the newest eligible turn stays whole even if it exceeds the space left.
At `0`, no optional history survives; the restart still carries pending work and any required plan or contract notice.
`compaction.keep_thinking_turns` limits how many recent replies keep their thinking when a local drop rewrites tool results, at a cold `drop`, and when a local restart keeps turns.

**Compact before the prompt cache expires.** Set `providers.<name>.idle_compact_s` below the cache lifetime of your endpoint.
The timer starts when the worker starts its model request.
It can compact during a blocking command, a background wait, an approval, a question or an interactive run parked for a message.
The whole context must exceed `providers.<name>.idle_compact_min_chars` (default `240000`).
The summary does not end the wait or add a worker turn.
Set the limit to `-1` to turn it off.
See the [provider defaults](config.md#providersname).

## Measure a run

Give one run a score to improve, independent of its verify gate and any standing goal:

```sh
agent6 run --metric 'minimize -- python3 elapsed.py' "Reduce elapsed time"
agent6 run --metric "maximize 'score: ([0-9.]+)' target 8 -- python3 bench.py" "Improve the score"
```

The metric uses UTF-8 text without carriage returns so its command and arguments survive a journal or queue read.

The flag and the live `/metric` directive take the same spec: `maximize|minimize ["capture pattern"] [target NUMBER] -- command`.
Quote the whole spec on the command line so the shell passes one argument.
The command is an argument list, not a shell script.
With a pattern, the first capture group supplies the score; without one, agent6 takes the last printed number from stdout followed by stderr.
A non-finite result or an incomplete capture with no identifiable last number gives no score.
The command uses the run's command jail and approval policy, including for automatic readings after committed edits.
The next turn sees the score and trend.

The optional target is a finite number: `maximize` reaches it at or above that score, `minimize` at or below it.
Without a target there is no maximum, even if the score line contains `8/8`.

- **Finish:** While more than a quarter of the budget remains, the metric can refuse a finish up to three times.
  It allows the finish if the best score has reached the target or the remaining budget is unknown.
- **Plateau:** After at least five parsed readings, a tie with the prior best counts as a plateau.
  If the best has reached the target, a plateau can end the run at any budget.
  Otherwise, with at most a quarter of the budget left or an unknown budget, the metric gives three plateau notices before another plateau can end the run.

`/metric` shows the applied setting; `/metric SPEC` replaces it at the next turn boundary and starts a new trend; `/metric off` retires it.
A resume, including Resume to fix, keeps the last setting; a fork takes the one at the selected checkpoint.
Children start without a metric.

## Tools

`read_file` returns UTF-8 text, a static PNG/JPEG/GIF/WebP image, or a PDF attachment.
It identifies media by content, not the filename.
A media read ignores `limit` and refuses a `start_line` other than 1.

- Images: 2 MiB each, at most four and 4 MiB total per turn and outbound request.
- PDFs: 10 MiB each, at most one per turn and outbound request, counted separately from images.
- Claude Code also refuses any encoded request line over 8 MiB, including the envelope and notices.
- Malformed or animated images are refused, not resized or truncated; crop or convert them with an otherwise-permitted jailed command.
- A provider with unknown media support gets one attempt; a recognized media rejection replaces the attachment with refusal text and retries once without rerunning tools.

Saved sessions retain metadata, not pixels or PDF payloads; resume and fork need a fresh `read_file` to see them again.
Attachments can disclose credentials in screenshots or documents.
See [Media retention](security.md#media-retention) for containment and the upstream boundary.

`read_session` without an id lists the project's sessions; `query` searches their visible tasks and entries.
With an id it returns numbered entries in journal order, such as `#41 12:31 operator: Use the blue label.`
`query` matches a literal substring regardless of case; `kinds` selects `operator`, `model` or `tool` entries without changing their numbers.
`start` is the first eligible entry number (default 1); `limit` caps matches and `max_chars` caps their combined text.
An entry too long to fit on its own ends in `…`; `next_start` gives the first match on the next page, if any.
For example, `kinds=["tool"]` with `query="pytest"` finds a recorded command with `pytest` in its arguments.
A child task names its source when recorded; `task` means the author is unknown.
The report names any skipped records over 1 MiB.
After compaction the restart notice names the session id when the model can use `read_session`.

### Web search

`providers.<name>.web_search = true` offers the provider's own hosted web search on the worker's run and ask turns.
It is off by default.

```toml
[providers.anthropic]
backend = "anthropic"
web_search = true
```

- The search runs inside the provider: agent6 adds one entry to the request's tool list and dials no new host.
- An `anthropic` or `responses` entry at the vendor's own address and a `chatgpt` entry can offer it; any other entry refuses `web_search = true` at load.
- Reviewers, judges, summaries and `machine create` drafts never get it; a child run follows its own provider entry.
- Each search shows as one line, `Search: <query>` with its URLs, and as a `model.web_search` journal event.
- The provider bills each search apart from tokens.
    - agent6 counts them: `searches=` in the cost summary and in `agent6 sessions show`, `web_searches` in `budget.update`.
    - A price listing's per-search rate prices them; a provider-reported cost or a subscription plan already covers them.
    - Without a rate the run says so at start and marks the dollar total a lower bound.
    - An Anthropic response holds at most one search, so `[budget].max_usd` is checked between searches; a Responses reply can hold several.
- `sandbox.fetch_approved_hosts` and `sandbox.network` do not bound it.
  A query is text the model chose, which can include workspace content, and it goes to the provider's search service ([Network](security.md#5-network)).

## Model call actors

A role selects model settings; `actor` tells you who made a call with them.
The actor is the calling role, optionally followed by `:` and a detail:

- `worker` runs the coding turn; `worker:squash` drafts a squash commit message.
- `helper:verify_inferer` infers a verification command; `helper:summarizer` writes helper summaries and gists.
- A panel review child calls as `reviewer:<focus>`; a review child with no actor specified calls as `reviewer`.
- `reviewer:recap` reviews a finished session's record for `agent6 sessions review <id>`.

Read `actor` on `model.*` and `provider.request` journal events and on provider transcripts.
Inside the recorded request, a message's `role` identifies its speaker, such as `user` or `assistant`.

## Carry your state with the workspace

Run `agent6 pack` before syncing a checkout to a devcontainer, remote-ssh host or new location.
It copies this home's state into `<project root>/.agent6/` and keeps it out of git through the repository's own exclude file (`.git/info/exclude`), unless git already ignores it; the project's `.gitignore` is never written.
A run adds the same line when it starts, so its commits leave `.agent6/` out.
Before copying, it asks git whether `.agent6/pack.json` is ignored and refuses if it is not (a `.gitignore` rule such as `!.agent6/` outranks the exclude file) or if any `.agent6` path is tracked.
Home keeps its copies.
Run `agent6 unpack` in the destination checkout to move the pack into that home's global directories and the state directory keyed to the new project root.
It first lists every file it would install with its destination and, under each packed config file, every setting that file sets with its value; a secret is listed by key only.
A key under `sandbox`, or one no machine file may set (a host command, a provider, a credential), is marked `[operator-only]`.
Then it asks for a yes: `--yes` (`-y`) answers ahead, and without a terminal `unpack` refuses unless `--yes` is given.
When every file already matches this home, nothing is installed and nothing is asked.
`unpack` installs the files as they are and compares none with the manifest's hashes: read the listing, since `.agent6/` is a directory any command in the workspace can write.
After a successful import it removes the pack's own entries (`pack.json`, `config/`, `repo/`, `skills/`), and `.agent6/` itself when nothing else is left in it.
A repeated `pack` replaces a snapshot from the same host and home; a pack from another origin requires `unpack` first.

Use `agent6 pack --repo-config` to carry global settings in the repository config instead of exporting global config files.
It copies every inherited leaf from global `config.toml` to the same dotted key in the repo config and prints one line per copied leaf.
Repo values win: mappings merge by key, and a repo list replaces the global list.
The fold uses the locked, validated config write path and rolls back a failed edit.
The source global files stay unchanged.
The pack omits global `config.toml` and `ui.toml`; `secrets.toml` keeps its private handling.
`unpack` leaves the destination's global settings untouched and reports that it wrote neither global file.
After unpacking, `agent6 config show` attributes the copied settings to `repo`.

File modes and mtimes are kept; `secrets.toml` is created at `0600` inside a `0700` directory.
Session activity ordering and no-id session selection use the last readable journal timestamp, falling back to the manifest's start time (filesystem metadata only for legacy state with neither recorded time).
Ordering reads backward from each journal's tail until it finds a timestamp; full session summaries still read the journal's history.
Linked worktrees are not packed: listings, session details and resume name a missing worktree rather than treating its recorded path as a local checkout.
Symlinks and special files refuse rather than following a path outside the snapshot.
**The pack contains credentials.** Use a trusted workspace sync and see [Secrets](security.md#8-secrets-and-connect).
A command already running in this checkout's sandbox can read the pack.
`agent6 pack` refuses while a run, an `agent6 exec` or a machine of this checkout is live, and a run refuses to start while `pack` writes: a pack made under a live sandbox would put the secrets within its commands' reach.
`agent6 unpack` refuses beside them too: a live command could rewrite the pack after the listing.
`agent6 pack` also refuses beside a session that died without an end, such as a killed run: resume or remove that session first.

<details markdown="1">
<summary>Pack contents and conflict rules</summary>

The pack contains:

- `pack.json`: version 1, UTC pack time, origin host, home, repo id and workspace, and each file's SHA256, UTC mtime and last shared hash.
- `config/`: global `config.toml`, `ui.toml` and `secrets.toml`, or only `secrets.toml` with `--repo-config`.
- `skills/`: installed skills from the data directory.
- `repo/`: the entire repository state directory, including repo config, sessions, asks, runs, memory, reviews and decisions, except `locks/`, `worktrees/` and `pack-base.json`.

The cache, jail homes, worktrees and completions do not travel.

`unpack` uses a three-way rule per file: P is the pack's current hash, B is its recorded base hash (possibly null), and H is the home's hash (possibly absent).

- H absent: move in.
- H equals P: identical, keep home's and drop the pack copy.
- B present and H equals B: home unchanged since the last sync, take the pack's.
- B present and P equals B: the pack is unchanged, keep home's and drop the pack copy.
- Otherwise: a conflict.

</details>

Any conflict refuses the entire import with exit 2 before moving files.
`unpack` holds the writer locks for the packed config and secrets files across preflight and import.
Hand-merge the files or delete the side to replace, then rerun `agent6 unpack`.
Pack files edited by hand are hashed again; deleting a pack file leaves the home's copy untouched.
Files at home that the pack does not carry stay untouched.
The imported hashes are saved in `<state-dir>/pack-base.json` for the trip back.

Commands that resolve the project root refuse with exit 2 if `.agent6/pack.json` is present, even for a pack this home made.
`pack` and `unpack` are exempt.
A pack beside a session of the checkout that is live, or that died without an end, is taken as that session's and moved to `.agent6/pack.invalid` instead; [Secrets](security.md#8-secrets-and-connect) has the rule and what to do with a real pack it catches.

For a devcontainer or remote-ssh round trip:

1. Run `agent6 pack` here.
2. Sync the workspace, including the hidden `.agent6/` directory, which git ignores.
3. After the sync, run `agent6 unpack` here as well, which removes your own pack.
4. Run `agent6 unpack` there and work.
5. Run `agent6 pack` there and sync the workspace back, then run `agent6 unpack` there to remove its own pack.
6. Run `agent6 unpack` here.

Recorded absolute paths in journals and session manifests remain history.
Ordinary sessions resolve against the current checkout.
A session's cached worktree does not travel; `resume` names the missing worktree and recommends `agent6 fork ID` to continue from its commits in a new worktree.
The synced checkout must carry those git objects and refs.

## Configuration

`agent6 config show` prints every effective value with the layer that set it; the [configuration reference](config.md) documents each field.
