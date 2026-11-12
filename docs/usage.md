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
The CLI also accepts a bare id on the current provider.
Each optional role inherits its unset `model`, `effort` and `temperature` directly from `models.default`, never from another role.
For example, `models.planner.effort = "max"` changes only the planner's effort.

| Role | Set with | Used by |
| --- | --- | --- |
| `default` | `[models.default]` | required baseline and the only fallback for every role |
| `worker` | `[models.worker]` | `agent6 run`, `agent6 ask`, their resumes, machine agent states, and model-written squash messages |
| `planner` | `[models.planner]` | `agent6 plan` and plan resumes |
| `reviewer` | `[models.reviewer]` | in-loop reviewers without a route, the default judge of `sessions compare` and `--parallel` ([`harness.parallel.judges`](config.md#harnessparallel)), and `sessions review` |
| `helper` | `[models.helper]` | context summarising, gisting (`compaction.elision_gists`), and verify-command inference |

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
Your branch, HEAD and index stay untouched unless [`git.auto_merge`](config.md#git) lands the work; `sessions merge` lands it by hand.

- commands ask for approval by default (`sandbox.run_commands = "ask"`): allow one call or the whole session
- without a terminal (CI, cron, a hub-spawned run) a prompt parks the run until a front-end answers it or an away-mode decides it ([approvals](security.md#6-approvals))
- the run ends when the model declares it finished, the operator stops it (`agent6 stop ID`) or a ceiling (budget, turns) stops it
- at a terminal it then asks for the next input: type to continue, `/exit` to finish
- `agent6 resume <id> --steer "<what to do next>"` continues a run that ended by `finish_session`
    - an answer saved for one of its questions (`agent6 answer`) is new work, so a bare resume delivers it

The start block names each role's provider/model route and effective effort, including the helper and each reviewer.
It names the effort's config layer and any inheritance; `effort provider default` means no effort value is sent.
For example: `worker chatgpt/gpt-6-astra effort provider default · reviewer chatgpt/gpt-6-astra effort xhigh (repo)`.
`sessions show`, the TUI dashboard header and the web run page read these recorded model facts, not the live config; a resume records its execution's values.
`sessions show --json` exposes them under `models`; an empty `effort_source` means an older record did not capture effort.

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

## Follow and steer a run

Every surface follows a run and sends it the same steers; the command-line verbs are on [Command line](cli.md#follow-and-steer-a-run).

A session's usage covers every execution, in dollars or in plan points, and in both for a run that spent both (`1pt + $0.03`); `sessions list`, `sessions show`, the end-of-run total and a ranking's table print the same figure.
These points measure account-wide window movement between readings, accumulated across resets, not the run's own spend; other consumers sharing the plan contribute too.

The tokens and the points against `max_plan_points` on `sessions show` are the latest execution's.
`elapsed` on a resumed run is the wall time from its first start to its last event, the time between executions included.
`N questions unanswered` beside a status counts the questions of the current execution that nobody was attached to see; a resume starts the count again.

The steer directives work from every composer, the pause menu, and `agent6 steer ID "/task ..."` from a script.

- `/task <text>` adds work to the task graph (the turn in flight never sees it); `/retire <id>` drops a task, by the number `sessions graph` prints
- `/standing <text>` sets the goal the run returns to whenever its queue drains, replacing any it had
- `/pin <text>` survives every compaction; `/compact [focus]` compacts now; `/now <text>` interrupts the call in flight
- `/btw <question>` asks a side question; `/parallel [SPEC --] TASK` fans out lanes ([parallel](config.md#harnessparallel))
- `/review [TARGET]` reviews changes in the session ([targets and worktrees](#plan-review-and-ask))
- `/undo` is available in the TUI, web, a `run -i` prompt, or a live run's pause menu (Ctrl-C), so `agent6 steer` refuses it:
  - it takes back the last message
  - the tree is committed on the run's chain
  - every tracked path the run changed since the checkpoint before that message is put back (one changed since the run ended stays, named in the notice)
  - the run ends `undone` and an unstarted fork in the same checkout continues it (`agent6 resume <fork> --steer "<the message, edited>"`)
  - `resume` and `sessions merge` refuse the undone run and name the fork
  - it is refused while another live run drives the checkout

### Parallel directive

`/parallel [SPEC --] TASK` runs a task in isolated lanes, including as the first prompt to `agent6 run`, through the same dispatch as `run --parallel SPEC TASK`.
A lane is a child of the run that dispatched it: the parent hands down, the child pushes up, and the root is the one owner.
The lane gets its config, task and remaining budget from its parent; its approvals, questions and spend go up to the parent while it runs; its work and memory reach the repository only when the parent imports it at its end.
Its record is in the repository's state from its first event, so `sessions list --lanes`, `attach` and the hubs read a lane while it runs.
The two land differently: a live run's steer merges each lane's work onto the run's chain in dispatch order, and `run --parallel` joins one lane.
`run --parallel` is a run that makes no model call of its own: [`harness.parallel.judges`](config.md#harnessparallel) rank the lanes, the first-ranked lane joins the run's branch, the run's own gate judges the result, and the run ends `finished · lanes ranked`.
The other lanes keep their branches, so `agent6 sessions merge <lane>` still lands one by hand.
`SPEC` is a positive lane count or a comma-separated list of selectors, the same as `run --parallel`.
A selector is an exact role name (`default`, `worker`, `planner`, `reviewer`, `helper`), which brings that role's model, effort and temperature together, or `[provider/]model`, which runs that model with the worker's effort and temperature; a model named like a role is written `provider/reviewer`.
The spec ends at a whitespace-delimited `--` token.
Without `--`, the whole segment is task text for one default lane, even if it starts with a path, a number or a model name.
A bare `--`, a missing task, or more lanes than `[harness.parallel].max_lanes` across the message's segments is refused with the reason.

```text
/parallel src/x.py: add a test
/parallel 2 -- fix the parser
/parallel a/b,c -- compare the designs
/parallel 2 -- task A /parallel 3 -- task B
```

Repeat the `/parallel` token to batch tasks in one message.
Each segment has its own optional spec and separator.
Task text keeps its internal whitespace and newlines, including later `--` tokens.

When the lanes return, each joins the run's chain in dispatch order and one notice names every outcome: `joined at <sha>`, `finished with no changes to join` (the lane ended with a result and committed nothing), `CONFLICT` (the merge is left for the operator) or `FAILED` with the reason.
Under a lane's line the notice gives the summary the lane finished with (`summary: ...`), so a lane sent to find something out answers its parent without a commit.
A lane the run ended at its deadline or at its budget says so on its line (`stopped at the run's budget with no changes to join`, `joined at <sha> (ended at its deadline)`), and a task whose lanes were all ended before changing anything is marked failed.

### The parallel tool

With `harness.parallel.tool = true` ([parallel](config.md#harnessparallel)), a run's model is offered the `parallel` tool: `task` (required) and `spec` (the same grammar as `--parallel`, empty for one default lane).
The call dispatches through the `/parallel` path: a checkpoint, fresh clones of the committed tree, the lanes joined back in dispatch order, each lane starting under the run's remaining budget, which the lanes share; the result returns as the tool's answer, naming each lane's join.
Each lane is an ordinary run with the same tools as the parent: it may edit its clone, and its commits land on the parent's chain when it joins; a lane runs under 200 turns, 30 minutes and the run's remaining budget.
The run counts its lanes' spend while they run and ends them once the sum reaches a cap, so the total can pass the cap by what was in flight.
The tool's description names every role's resolved model, effort and temperature and the configured providers, so the model can pick a selector such as `reviewer`.
The tool is offered only in run mode, never inside a child and never in a front-end without a lane dispatcher; a call that slips past the list is refused with the reason.
A worker lane's configured harness panel retains authority to create bounded read-only leaves; neither the worker's model nor its `/parallel` steer may dispatch work.
The knob neither authorizes nor disables `run --parallel`, the `/parallel` steer or the review panel.

A `run --parallel` run made no model call: `agent6 resume <id> --steer "..."` continues it from the joined work with the worker model.
`agent6 stop ID` on a session with live lanes stops its descendants too, by the lineage their manifests record, including read-only leaves behind a worker that already ended.
Worker import preserves descendant records and receipts in the original repository's state directory before clone cleanup.
Their spend is included in the worker's total, not an extra charge, and their evidence certifies only the child's reviewed target, not the top-level tree.
A lane that needs an approval, or waits on an `ask_user`, asks on the run that dispatched it: the prompt names the lane, and a standing answer given there covers the run's lanes.
A question nobody answers on that run (`away mode deny`) returns `unanswered` to the lane with the same note; a question the lane queued (`wait: false`) stays on the lane's own row.
A run interrupted while its lanes ran finishes that dispatch first when resumed: lanes still running are stopped, finished ones are imported and joined once, their spend is settled once, and the model hears the group's outcome before its first call.
Every session that dispatched lanes shows them: `sessions show`, the hubs and `attach --json` list a run's lanes under it whether it is a `run --parallel` group or a run that used `/parallel`, and the run's receipt carries a `Lanes` section (one line per lane: its join, its spend) with the spend the lanes added to the run's total named, so the total is read as inclusive.
The refusal names the lane session IDs.
While a steer waits on disk, `sessions show`, the TUI header and the web session status show `steer queued, not yet delivered` with its text and the count waiting behind it.
Steers sent while one waits queue in order and reach the model one per step; `abort`, `exit`, `/undo` and `detach` take the slot at once.
The pending label clears when the worker consumes the last steer.

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
- a `--parallel` lane continues through `agent6 fork <lane-id>`, in a worktree at the lane's last commit; `resume` refuses a lane from your checkout, whose tree is not the lane's
- a `run --parallel` run whose budget ended every lane reads `failed · budget exhausted` and exits `3`; each stopped lane continues through `agent6 fork <lane-id>`
- the first `--steer` on a fork becomes the fork's task and names its row and its merge subject; later steers are continuations
- a resume with `--steer` keeps the original session label, with the latest steer beneath it as `continued: <steer>`; new questions in an ask REPL are not continuations
- under the `agent6` and `conventional` squash message styles, a resumed run's merge keeps the task as its subject (`conventional` prefixes it with the type and scope), with `continued: <steer>` in the body
- a gate withheld after denied or interrupted approval stays withheld only for that execution.
  When the snapshot records that withholding, a resume with the gate and commands allowed tells the model once that the gate is available again, without claiming it passed
- a resume after the run's work landed continues from the checkout's HEAD when HEAD holds the merge: the run's next commits, `sessions diff` and its next merge then hold only the new work, so nothing lands twice
- a run whose workspace does not hold the merge (a `--worktree` run, a fork) continues on its own chain: after a resume, `sessions diff`, `sessions commits` and its next merge start from the landed tip
- a run merged `--into` a branch other than the one it started on keeps its base, and merges into that branch again from its landed tip
- `sessions merge` over a file you edited leaves your copy alone and prints two commands that merge your edit with what landed
- a `sessions merge` that lands names the other ended, unmerged runs that changed some of the same paths (`unmerged runs that changed some of the same paths: <id> (src/a.py and 2 more)`); each one's own merge then meets what landed
  The TUI's and the web's Merge confirm states that line before the merge.
- a commit you made on the run's `agent6/<id>` branch refuses the resume, which names it: `agent6 sessions merge <id>` lands it with the run, or `resume --force` continues without it
- a `sessions merge` that conflicts names `agent6 resume <id> --merge <branch>` for a run with its own worktree: the resume merges the branch into the run's work there, leaves each conflict's markers in the file for the run to resolve as its first task, and the next `sessions merge` lands cleanly
- files you changed between executions follow `[git].dirty_tree` on resume, as uncommitted changes do at a run's start: `ask` asks, `stash` sets them aside (a change under `[sandbox].hide_paths` stays in place), `include` leaves them in place, named to the model before its first call, for the run's next commit to take ([Config](config.md#git))
- when `stash` refuses over a file the run also changed, its printed `git restore --source ...` command discards only your edit and keeps the run's last tree; plain `git restore FILE` returns to HEAD, which can remove the run's work
- a new untracked file is left out of the run's commits as yours, unless the last execution died mid-turn (a kill, a crash): a command of that turn may have written it, so the run keeps it, and the resume names up to four files either way

Every end has one reason and a headline every surface shows; [Exit codes and end reasons](cli.md#exit-codes-and-end-reasons) lists them with the exit codes scripts branch on.

A run whose worker exited without recording an end reads `stale`, for example after SIGKILL or a machine crash.
`agent6 resume <id>` continues it from its last snapshot, and `agent6 sessions show <id>` prints that command.
A resumed run opens with the verify fact: the last full verification's command, verdict and tree, and whether the tree changed since; every compaction restart carries the same fact.

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

In-loop panels run each reviewer as a read-only lane with its own session, and record their findings and reviewed tree IDs in the session journal.
Each batch's reviewers start under what the run has left; batches use at most `review.concurrency` and `harness.parallel.max_lanes` reviewers, without truncating the roster.
Results keep roster order even when reviewers finish in another order.
A reviewer opens on the task, the rules, the verification and the diff, and investigates a clone of the retained snapshot with the tools its run has: the reading tools, and `run_command` with its background commands, `run_verify_command`, `fetch`, `read_session` and `use_skill` where the run's model is offered them.
A reviewer edits nothing and dispatches nothing; `sandbox.run_commands = "deny"` withholds `run_command` and `run_verify_command` from a reviewer as it does from the run.
A reviewer takes at most `review.max_turns` turns.
Each reviewer has a session of its own, named on the receipt and the findings card; no listing counts it among the run's lanes.
A reviewer still running 30 minutes after its batch started is stopped and abstains; batches run one after another, each with its own 30 minutes.
Failed or unparseable reviewers remain abstentions with their diagnostics; an all-abstain panel is inconclusive, not a clean review.
Child review does not certify a later or different parent tree.
Session-owned panels retain a cloneable snapshot of the exact comparison, including uncommitted work after failed verification and with `git.commit_per_turn = false`.
Snapshotting does not checkpoint the run or move HEAD, the index or visible branches.
`sessions remove <id>` deletes the session's review snapshot refs, its lanes' records and its reviewers' records with its record; `sessions remove --asks --yes` does the same for the removed asks, while `sessions prune` leaves review snapshots alone.
The snapshot certifies only its recorded trees and path coverage, not later parent edits.
Under `git.control = "model"` a panel compares the base recorded at the start with the checkout's HEAD and its uncommitted edits, since the model's commits are the record.
An exact repeat on the same cited blob keeps its finding ID across rounds and resumes.
The worker receives each finding once unless the feedback notice cut it.
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

## Session findings

Every agent6 merge path refuses open grounded `block` findings and unreadable review evidence.
Resolve each blocking finding with a reason before merging, including findings you decide not to fix.
This rule is separate from `review.block_on`, which decides whether a panel rejects the worker's finish.
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

## A row's merge note

Every listing row says what stands before a clean merge, derived from the journal on each read and never stored:

- `open findings: 2 block, 1 warn`: the run's open findings by severity; `sessions merge` refuses on an open `block`, and `agent6 sessions findings <id>` lists them

The status beside the note carries the rest: what a waiting run waits on, the gate's word, `unmerged`, and the questions pending or the answers saved for the resume.
Every listing orders sessions by last activity, newest first.
`sessions list --json` carries the counts as `open_findings` and the rendered note as `merge_note` ([the row](formats.md#cli-json)).

## Session receipts

`agent6 sessions receipt <id> [--json]` reads testing, review and reversal evidence from the journal and manifest.
It makes no provider call and reruns no checks.

- **Testing:** the receipt includes each [recorded verification attempt](formats.md#verificationevidence).
  A pass certifies only its unchanged tested tree; a scoped pass covers only the selected tests.
  An edit after a pass or a command that changes the tree leaves the final tree uncertified.
  A different commit with the same tree content does not invalidate the observation.
- **Output:** each journal stream keeps its last 2,000 characters, with truncation marked.
  Historical or incomplete captures read “capture completeness unknown”.
- **Missing checks:** denial, inability to execute and interruption are not passes.
  Baseline failures stay marked separately; a run without a gate says “No verification recorded”.
  Other commands are not verification evidence.
- **Review:** the receipt shows recorded panel state and names each reviewer's session.
- **How to revert:** recorded merge bounds determine the explanation.
  A squash names its landed commit; a merge verifies its target parent; a fast-forward lists landed commits in reverse order.
  Fast-forwards containing merges need an explicit plan.
  Missing stamps or objects do not produce guessed commands.
  “No merge recorded” does not claim that no merge occurred if recording failed.

Git commands in a receipt are explanations only.
**Ask a run to revert** in the web or TUI opens the new-work composer with a proposed task; inspect it before submitting.
You can also copy the recorded landing facts into a steer request to prepare the reversal.
The reversal needs its own checks; the original receipt does not certify a merge against a moved target or the later reversal.

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

## Worktrees

`run --worktree`, a run-mode fork, each `--parallel` lane and a run-mode machine state work in a worktree of their own.

- `[git.worktree].kind` chooses the worktree of a fork and a `--worktree` run: `clone` (the default), a clone of the repository whose commits agent6 copies into the repository as the run makes them, or `linked`, a linked git worktree; a lane, a reviewer and a machine state work in a clone.
- It starts with the repository's tracked files at the commit it was cut from: no uncommitted change, and nothing git does not track (`.venv`, `node_modules`, a build directory).
- `[git.worktree].setup_command` builds what is missing; it runs once in each new worktree, before the first turn:

    ```sh
    agent6 config set --repo git.worktree.setup_command '["uv", "sync"]'
    ```

- Unset, a `## Setup command` fenced block in AGENTS.md supplies it; a `--worktree` run and a lane read it from the commit the worktree is cut from, not from an uncommitted edit, and print the command and that AGENTS.md gave it.
    - an AGENTS.md that is a symlink is not read there: set `git.worktree.setup_command`
- The command runs in the run's own sandbox, under the isolation, network and grants of every command of that run; with no route out it has to work offline ([the offline recipe](security.md#4-commands-and-environment)).
- Machine clones follow the [state setup and failure rules](state-machines.md#agent) and [setup approval policy](security.md#4-commands-and-environment).
- For run, fork and lane sessions, the command and output tail are printed when it ends; a nonzero exit ends the session `failed · setup command failed` before any turn (exit `1`).
    - A fork's unread steer survives the setup failure and reaches the worker on a bare resume.
  For a run or lane, no turn ran, so `agent6 resume` has nothing to continue: start the run again once the command works.
- A run prints a lane's failed setup command and its output when it imports the lane; `agent6 sessions show <id>` keeps the command, and a failed setup's output, after the worktree is gone.
- The files it leaves untracked stay out of the run's commits.
- The verify gate is settled in your checkout and carried into the workspace: a gate inferred as `.venv/bin/python -m pytest -q` stays that.
- A gate whose first word is a path git does not track, with no setup command, refuses `run --worktree`, `fork` and `run --parallel` before anything is created; the refusal names the path, the config line and the AGENTS.md block.

## Carry your state with the workspace

Run `agent6 pack` before syncing a checkout to a devcontainer, remote-ssh host or new location.
It copies this home's state into `<project root>/.agent6/` and keeps it out of git through the repository's own exclude file (`.git/info/exclude`), unless git already ignores it; the project's `.gitignore` is never written.
Before copying, it asks git whether `.agent6/pack.json` is ignored and refuses if it is not (a `.gitignore` rule such as `!.agent6/` outranks the exclude file) or if any `.agent6` path is tracked.
Home keeps its copies.
Run `agent6 unpack` in the destination checkout to move the pack into that home's global directories and the state directory keyed to the new project root.
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
