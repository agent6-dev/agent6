# Architecture

How agent6 runs, end to end.
Three diagrams are built from the current source at site-build time (`docs/gen_diagrams.py`); the other three are drawn by hand, and each says so above its block.
[security.md](security.md) covers the threat model and what each isolation level enforces; [AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md) covers per-file conventions and stability rules.

## Layering

The layer stack is `ui -> app -> harness -> tools -> sandbox`.
An edge is "imports from"; a dashed edge would mark an import climbing the stack.
[tach](https://docs.gauge.sh/) records the map ([tach.toml](https://github.com/agent6-dev/agent6/blob/master/tach.toml)); the engine never imports the UI.

<!-- diagram: layering -->

Any layer may also use the shared substrate: <!-- generated: substrate-names -->.

The `vcs` package holds agent6's own git operations: `git_ops.py` writes, `git_read.py` queries, and `git_history.py` reads for display.
Only its modules import the runner, `git_command.py`.
`git_policy.py` holds those switches and the hook and driver policy; `errors.py` owns `GitError`.

- **ui** ([src/agent6/ui/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/ui)): the presentation layer and composition root, over the shared read-model fold (`viewmodel`)
    - the four front-ends: `ui/cli`, `ui/tui`, `ui/web`, `ui/acp`
    - `ui/mcp_server.py`: agent6 as an MCP server
    - the write helpers: `ui/spawn`, `ui/notify`, `ui/steer` (the steer-file seam every front-end writes) and `ui/btw` (the side question a run answers without losing its place)
- **app** ([src/agent6/app/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/app)): the pipelines composed over the engine: run/resume/fork/machine-agent lifecycles, merge and finalize, provider construction, the sandbox cross-checks (`app.confine`), the `--parallel` group
    - what it cannot do itself (own a terminal, render, spawn detached) arrives as frozen injected callables (`SessionFrontend`, `ChildRuntime`); output goes through the injected `Reporter`
- **harness** ([src/agent6/harness/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/harness)): `loop` (the agent loop behind `agent6 run` and `resume`), `review_session` (the read-only pass behind `agent6 sessions review`) and the review panel (`review_panel`, `reviewer`, `review_gate`); the agent loop makes its own provider calls through `provider_call`.
  `harness/loop_resume.py` owns snapshot carryover and persistence; `harness/loop_steer.py` applies steering at turn and execution boundaries.
- **tools** ([src/agent6/tools/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/tools)): the fixed tool surface the LLM sees, plus dispatch.
  `harness/toolset.py` picks the subset each mode exposes, and appends the MCP tools in the modes that edit.
- **sandbox** ([src/agent6/sandbox/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/sandbox)): the `agent6-jail` launcher and its policy.
  The jail bounds commands, one launcher per run; the agent process itself is never confined.

**Where the CLI resolves things.** `ui/cli` parses arguments, optionally spawns the TUI, and picks a mode.

- `cli_main` is the one error boundary: `OperatorError` (with `ConfigError`, `MemoryStoreError`) prints an `ERROR:` refusal at exit 2; a closed output pipe (`agent6 ... | head`) ends quietly at exit 141; anything else crash-reports with a saved traceback at exit 1
- config resolves through [config/layer.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/config/layer.py) (defaults, global, per-repo, `--config FILE`, then a machine's `[config]` overlay and one state's `config` table; a selected preset sits just above the layer that named it); paths and sudo/root through [paths.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/paths.py); keys through [secret_store.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/config/secret_store.py).
  `config/layer_data.py` discovers and merges layers with per-leaf provenance; `config/layer_presets.py` selects and lists presets.
- per-repo state lives out of the workspace at `$XDG_STATE_HOME/agent6/<repo-id>/`, written `<state-dir>` below, keyed on the repository (a subdirectory reaches the same runs, memory and config, and so does a linked worktree, through its `.git` file)
- `pack.py` copies and imports portable workspace state; `pack_manifest.py` validates its trust boundary without importing path resolution.
  `pack --repo-config` folds inherited global leaves into repo config through `config.write` before copying, using `config.layer_data`'s merge rules.
  `ui/cli/pack_cmds.py` owns the two command surfaces; `unpack` prints `pack.preview` and asks before it imports.
  `paths.project_root` refuses any pack until `unpack` imports it or the operator deletes it.
  A pack that a session of the checkout wrote is moved to `.agent6/pack.invalid/`: by `pack.check_start` where a command begins, by `pack.quarantine_tree` where a session, machine or `exec` ends.
- every config edit goes through [config/write.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/config/write.py): one lock-held cycle that edits the file's text, validates it and only then writes the file, so a refused edit writes nothing

Every model call requires `models.default`.
The four optional roles inherit each unset field directly from it; their jobs are listed in [Usage](usage.md#connect-a-provider).

## Units of work

| unit | what it is | bounded by |
|---|---|---|
| machine | a state-machine instance; each of its agent states hands one task to the harness | the `.asm.toml` and its journal |
| group | a `/parallel` group: N sessions on one task | the parent's session |
| session | one task's durable record: its dir, journal, task graph, chain of commits and memory use | the id; it holds several executions |
| execution | one continuous run of the harness over a session, from a `run`, `resume` or `fork` to an end | `max_turns` and the budget cap, re-armed per execution |
| turn | one harness loop iteration: a model call and the tool dispatch, advisors, gates and notices for its answer; a prose-only answer is a turn too | its number in the session: a resume continues the count, and each execution gets its own `max_turns` allowance |
| tool call | one tool the model asked for inside a turn; a finish is the last one executed | one dispatch |
| commit | one entry in the session's git history | its sha |
| checkpoint | saved harness state for resume | its checkpoint file |
| task | one node in the task graph | its task id |

A commit lands on the session's chain after a turn that changed the tree.
`git.commit_per_turn` controls whether editing turns commit; the diff pane's commit picker lists those commits.
`run` is the mode beside `plan` and `ask`, and the command that opens a session.

## The run lifecycle

`app/run.py`'s `run_task` composes the run's stages; `app/resume.py`'s `resume_task` composes a resume's.
The steps both share (the workspace lease and isolation, the session's writer lock, the headless check, the journal, transcript and operator channels, the gate pin) live in `app/session/prepare.py`, and each lifecycle calls them in its own order.
The diagram draws the run's in call order: refusals and clamps, the route preflight, isolation, git preflight, the manifest, provider construction, target preparation, gate inference, tool assembly, the loop, then auto-merge and the end report.
The stash finalize is last because it runs from `finally`, on every exit path, refusals included.

<!-- diagram: run-lifecycle -->

## A run

A run keeps one message history with one provider and one model.

- the model drives by calling tools; the harness dispatches, snapshots, tracks budget
- multi-step work is the next tool call in the same conversation: no planner-to-worker handoff, no separate reviewer by default
- the in-loop review panel is opt-in (`[review]`), layered on the same history
- `repo_context.review_rules` loads base-tree rule blobs through `vcs.review_rule_reader`; `review.start` records the paths, blob IDs, byte counts and completeness for the panel
- under `backend = "claude_code"` the provider keeps one `claude` process per execution and replays that history as labelled text and retained media whenever a call is not a continuation of its last round ([Config](config.md))

An in-loop panel records the run base and reviewed head's tree IDs, full grounded reviewer verdicts, and findings with durable IDs in `loop.review.*` events.
Panel preparation retains the exact comparison under `refs/agent6/<session>/review/<base-tree>-<head-tree>`, including uncommitted work after failed verification or with `git.commit_per_turn = false`.
The snapshot commit retains the base history as well as the head tree; an existing head commit is reused when it already retains both.
Before creating or reusing a snapshot, git checks the trees, retained history and every file object they reference.
Retention never moves the operator's index, HEAD, visible branches or run chain.
`ReviewGate.prepare` binds the panel ID and target to an immutable `ReviewInput` containing complete historical rules, prior findings and verification evidence.
The current reviewer executor consumes that context; child execution is not enabled by preparation.
Unreadable targets or failed retention skip the panel with `target_unreadable`, never substitute the chain tip for unreadable worktree content.
The refs survive chain pruning and are removed with their owning session record (`sessions remove`, including `--asks`).
`harness/review/history.py` restores those records on resume.
Only exact normalized finding content, location and the cited blob identify a repeat.
The journal keeps repeated observations even when the worker feedback omits them.
Findings cut from the feedback notice remain eligible for a later notice.

`vcs/review.py` owns diff hunk ranges and line-origin classification.
Before the panel event is written, each observation records its line origin against the immutable base and head blobs.
An exact unchanged range carries its base location and bounded blame attribution.
Unsupported files, ambiguous mappings and unavailable history carry an unknown reason.

Every execution start carries an `execution_id`, every panel has a unique `panel_id`, and every durable event has an `event_id`.
Journal append order, not timestamps, orders events.

`harness/loop.py` holds the turn: the request, the model call, the tool dispatch, what the tools did, and the ends.
What the turn leans on sits beside it, one module each:

- `run_chain` (the run's commit chain)
- `operator_bridge` (the operator's callables, the steer verbs, the pin cap, and what a steer's text means)
- `advice` (what an advisor answers with, a nudge, a stop or a refusal, and the turn's context it reads)
- `guards` (the advisors, one function per heuristic with its counters: no progress, settled, stagnation, the memory flip, the loop guard, the tool-error ladder, reachability, focus, the budget nudges)
- `metric_rules` (a metric run's plateau, ceiling and early-finish rules) and `metric_sampler` (the readings a run takes); `metric.set` in the session journal records the operator's setting or retirement, and each checkpoint names the revision its scores describe
- `quiet_turns` (a turn with no tool call: the nudges it draws, the silent finish, the went-quiet end)
- `tool_effects` (what a dispatched tool did to the turn: the verify result, the metric sample, the tree edit, the memory touch, the finish, the error)
- `finish_gates` (what a finish call declares, what it must satisfy and what an end is called)
- `standing_goal` (the standing goal's re-entry)
- `operator_tasks` (the operator's writes to the task graph: the root, the goal, `/task` and `/retire`)
- `parallel_dispatch` (a run's children: the dispatch, the merge at a turn boundary), `parallel_rank` (the judges of a task's attempts and the order they give), `parallel_panel` (a panel's reviewers, in batches through the same dispatcher), `parallel_recovery` (a resumed execution settling the dispatches left behind) and `child_work` (the children as the run's tools reach them)
- `run_checkpoint` (the per-turn commit and the final one)
- `compaction/strategy.py` (the strategy interface) and `compaction/driver.py` (the local and endpoint strategies)
- `memory_touch` (what a tool call did to the memory store)
- the settings each sibling owns (`review_gate`, `compaction_window`, `provider_call`)

The loop runs the advisors and the gates in a declared order and applies each answer; a heuristic is one function and one test file.
After stop checks, a `finish_session` turn polls for queued answers before ending.
An answer found there reaches the model on another turn before it can finish; an unanswered question does not hold the run open.

Drawn by hand against `harness/loop.py`, the turn as a state machine:

```mermaid
stateDiagram-v2
    [*] --> snapshot
    snapshot --> llm_call
    llm_call --> dispatch: model emits tool calls
    llm_call --> [*]: budget exhausted
    dispatch --> snapshot: non-terminal tool
    dispatch --> commit: verify green
    commit --> snapshot
    dispatch --> [*]: finish_session
```

The same turn with its decisions, also by hand, against `harness/loop.py`'s `_drive_loop`:

```mermaid
flowchart TD
    pre["pre-call: snapshot,<br/>nudge, compact"] --> model["provider call, streamed<br/>steer interrupts"]
    model --> tools["tool calls<br/>(commands jailed)"]
    tools --> hgate["harness gate run<br/>(harness.verify.when)"]
    hgate --> commit["auto-commit + metric"]
    commit --> review["review triggers"]
    review --> gates{"finish<br/>requested?"}
    gates -->|verify green| done(["finished"])
    gates -->|gate red, retries left| notices["notices + stop checks"]
    gates -->|gate red, retries spent| done
    gates -->|no| notices
    notices -->|budget, no progress, abort| stopped(["stopped, resumable"])
    notices -->|continue| pre
```

**Snapshot before every call.** `loop_state.json` is rewritten in full in the session directory before each provider request, with a per-turn snapshot under `checkpoints/<NNNN>.json` referencing shared message objects.
`agent6 resume` rehydrates from `loop_state.json` and `agent6 fork --at-turn N` materializes the matching checkpoint.
With the per-call transcripts, an interrupted run replays deterministically up to the next model call.
A turn's tool calls and each result that returned are recorded in `turn_in_flight.json` while they dispatch.
A resume after a death there runs none of the turn's calls again: the model gets the results that returned, and one line for each other call saying it was interrupted and not re-run.

**The harness runs the gate only when asked** (`[harness.verify].when`, default `never`: the model's own `run_verify_command` calls are the verification, and an end no run of the gate covers reads `unverified` and exits 0).

- Under `finish`, when `finish_session` arrives over a tree no verify verdict covers (green or red, nothing edited since), the loop runs `harness.verify.command` itself, through the same dispatcher path as the model's `run_verify_command`, approvals included.
- A denied approval withholds the gate for the rest of the execution, and the execution ends unverified.
- `turn` also runs it after every editing turn.
- A pytest gate naming no paths that overruns `harness.verify.timeout_s` (the harness's run or the model's own) re-runs scoped to the tests nearest the run's diff.
  It stays scoped until a full run of the gate passes.
  The notice lists the selected files, `session.stop` carries `scoped`, and a `finish_session` over a scoped green reads `passed · scoped gate` on every surface.
- A red finish certification returns to the model with the gate's output `harness.verify.retries` times (default 2), then the finish stands and the run is reported finished, never passed.
- The settled stop ends a plain run after `harness.settled_stop_after` consecutive idle turns (default 12) following a green verify of the run's own edits or an editing step no gate judged.
    - A green on the untouched base tree, from the start gate or the model's own call, is the baseline's verdict and arms nothing.
    - A verify run is neutral, as is a command start or selected-job read whose result says that command is still running.
      Completed commands, roster-only reads and reads of ended jobs count as idle even while another job runs.
    - An edit, a commit, a memory write, a tree change or an operator steer restarts the streak.
    - `-1` disables the stop, but the nudge at three idle turns still fires without announcing a stop; over a red gate the model must fix, the nudge is withheld while a finish would still go back.
    - Over a green gate of the run's own, the stop reads `finished · settled` and exits 0, since the worker called no `finish_session`.
    - A settled end whose only green was the base's carries `verification = "unverified"`: an unverified end.
    - Over a red gate it returns under the same cap, counted apart from the model's own finishes.

**Per-turn commits** fire when a gate run returns 0, through `git_ops.py` outside the jail, onto the run's detached chain (`refs/agent6/<id>/head`, temp-index staged).

- every other editing turn commits as a checkpoint, whatever the gate policy: a gateless run, a gate nobody may run (under `run_commands = "deny"`, or after a denial), a turn the model did not verify, a turn the gate judged red
- `branch_per_run` also advances a visible `agent6/<id>` branch
- HEAD never moves; when the run's own branch is the checked-out one, the index and the working tree are brought forward for the paths the commit changed, each only where it still matches the old tip
- consolidation is chosen at `sessions merge` time (`git.merge_method`: `tasks`, the default, one commit per task range; `squash`; `merge`; `ff_only`)
- `[git].control = "model"` turns the block off: no per-turn chain or run branch, the model's own commits are the record, and `sessions diff`/`commits`/`merge`, `/undo` and `fork` refuse for such a run

**The task DAG is a curator-owned side store.** `add_task` and `update_task` write it, with `depends_on` edges, cycle-checked; `list_tasks` reads it back. None of them picks the next tool.

- each turn the current task surfaces into the prompt (the cursor's subtask while it stays open, dependency-satisfied and undecomposed, else the first such subtask in tree order, else a ready standing task), advances as tasks pass, marks `in_progress`
- `finish_session` refuses while the worker's own subtasks are open, capped so an unclosable task cannot stall the run
- the surfaced banner survives tier-1 elision and re-injects after each tier-2 restart
- a focus task held without forward motion draws a split/pass/skip nudge, re-firing up to a small cap; progress resets it
- an end is final: `passed` takes only `obsolete`, and a `skipped` or `obsolete` task stays retired (needed again means a new task)

**A standing goal parks a run instead of ending it.** The operator's goal (`run --standing "<goal>"`, `/standing` on a live run) is the never-passing fallback, worked only when no ordinary subtask is ready; the model's `add_task` cannot set the flag.

- only the operator retires it: `/standing` with a new goal, `/retire`, or stopping the run
- while one exists, the soft out-of-work endings (`finish_session`, the settled family, a quiet turn) convert into re-entry
- faults, operator verbs, the turn cap, and a spent budget still end the run
- a re-entry round landing no executed tool call escalates the nudge; `[harness].standing_patience` bounds the streak (`-1` default: never self-ends; landed work resets)
- an interactive run parks the same way on a quiet turn: the conversation waits on the steer bridge; any composer or the pause menu continues it in place

**Context compaction.** The driving route reports whether the endpoint clears tool results, writes summaries or keeps its own session.
The harness selects `Endpoint` or `Tiered` once from those facts.
Both strategies use the same size, manual and idle triggers and the same growth floor.

`Endpoint` leaves earlier messages intact.
On ordinary requests over the drop threshold it asks the endpoint to clear the oldest results, in steps down to half the threshold, without removing any local results.
A route that does not write summaries reads an adopted block as a text notice.
For a summary it replays the worker request without clearing.
It adopts the signed block as the first message and keeps the worker reply after it.
The restart notice follows that reply and its tool results.
If no usable block comes back, the helper gets one attempt without changing the history first.
See [Anthropic Messages](anthropic.md) for the supported models and wire behaviour.

`Tiered` compacts locally:

- tier 1, `drop_at_chars`: above this bound, the oldest tool results become placeholders until the total reaches half the bound or no eligible results remain
- the newest two results, undelivered results and operator answers stay whole; reads of recently-edited files elide last
- dedup runs in the same tier-1 pass; `keep_thinking_turns` strips old thinking only when that pass rewrites results, avoiding separate cache breaks
- a large `read_file` decays in two stages: a model-written gist first, the bare marker under continued pressure (oldest gists first); files changing under edits are never gisted
- Tier 2 checks `summarise_at_chars` before tier 1 rewrites any messages.
- With `summariser = "worker"` (default), the driving model receives its prepared conversation and tools with a final instruction to write the summary.
  A summary tool call is never run.
- With `summariser = "helper"`, the helper reads the part being replaced without tool output.
  When that history exceeds the helper's known window, it reads the newest end with a notice that the beginning is missing.
- A rejected worker request or unusable reply gives the helper one attempt.
  The `claude_code` backend uses the helper directly because its worker process owns the session.
  Stop, budget and sign-in failures do not start another attempt.
- Only a complete text summary without a tool call or refusal replaces the history.
  The restart keeps the task, pins, decisions and verify fact.
  Size and manual restarts also keep recent whole exchanges (`keep_recent_chars`, default 80000).
- An idle restart uses the driving provider's `idle_compact_s` and `idle_compact_min_chars`.
  The limit counts from the start of the last worker request, including time spent generating the reply.
  The floor counts system text, tools and the whole conversation.
  The worker summary request omits the reply whose tool calls are still open.
  The restart keeps that reply after the summary, with its signed reasoning and tool call ids intact.
  A parked interactive run has no open call, so the summary covers the complete turn.
  A later idle restart must also meet the tier-2 growth floor.
- the DAG survives the restart: the current task re-surfaces, the summariser reports finished/new tasks (finished marked `passed`, new queued)
- `/compact [focus]` compacts on demand
- `/pin <text>` survives every restart verbatim and persists in the snapshot; over the 4000-char total cap the pinning is refused loudly and the text still lands as an ordinary steer

**Writing a compaction strategy.** Pass an object with `compact(self, context: strategy.Context) -> bool` as `Harness(compactor=...)`.
Import `strategy` from `agent6.harness.compaction`.
For local compaction the harness checks the prepared turn before saving the worker request.
For endpoint compaction it checks the successful request before dispatching the reply.
The context supplies the live conversation, settings, task, pins, restart notice, graph, journal emitter, and stop and budget checks.
`call_worker(instruction, call_id)` appends an instruction to the prepared request.
`call_helper(system, prompt, call_id)` asks the helper without tools.
`call_endpoint(instruction, call_id, count)` sends the summary instruction outside the first `count` messages.
All three calls use the run's metering and cancellation signals.

`strategy.IdleStrategy` adds an optional `idle(context, request_started)` method.
It returns one deadline for the turn's waits, or `None`.
Only a blocking wait arms the callback.
The wait joins a callback in progress before delivering its result.
Neither the deadline nor the callback changes the command check-in.

Keep the original task and complete tool exchanges.
Keep signed reasoning with its assistant message.
Use `conversation.restart(context.restart_notice() + summary, keep=tail)` with a tail that does not start with a tool result.
Return `True` after a restart so the loop restores focus notices.
Return `False` for edits in place or no change.
The strategy is supplied by Python code, not loaded from a configuration file.

**Repo memory**: one fact per markdown file under `<state-dir>/memory/`, with an index rendered from each file's first non-empty line.
A fact file is `<name>.md` under the name rule, a regular file; any other entry in the store is ignored, and `agent6 memory list` names it.

- the index injects into every run's prompt as a capped `<memory>` block; depth is a file read
- the worker writes through the ordinary edit tools under a narrow grant, in-process only; the jail never mounts it
- run mode writes; plan and ask read; machine modes see none
- models never write unprompted, so the loop sends one advisory per run, when verify first recovers green with nothing recorded; `[harness].memory_advisory = false` turns it off
- `agent6 memory add/list/show/remove` is the operator surface over the same files
- `<state-dir>/memory-use.json`, harness-written at each execution's end: per fact, the write that created it (when the record saw it), every write as a (session, when) touch in order, and its reads (count, last reader), the entry gone once the fact is deleted (by the model or `memory remove`); `memory list` prints it under each entry, so a stale or never-read fact is visible where it is pruned

Outside the store, `<state-dir>/decisions.md` holds the operator's rulings beside `memory-use.json`.
The harness appends every `ask_user` answer and every steer that answered a question, verbatim with its question, session and time; an identical ruling already on file is recorded once.
The model sees it first (a `<decisions>` block, re-shown after a compaction restart) and never opens or writes the file; `agent6 memory decisions` prints it.
A finish-time check reports any ruling missing from the file.

**Skills** resolve at run start from `<data-dir>/skills/` plus `[skills].extra_dirs`, through one resolution: the `<skills>` index and what `use_skill` serves cannot diverge.

- `always` skills inject full text; the rest get an index line and load on demand; run mode only

**The terminal tool** is `finish_session(summary)` in run mode and `finish_planning` in plan mode; ask mode has neither and ends on its final prose.
Either call, once accepted, emits a `session.stop` event and returns control to the front-end.

## OAuth credentials

`providers/oauth.py` resolves configured issuers and the fixed ChatGPT registration through `profile_for(entry)` into a frozen `OAuthProfile`.
The profile holds endpoints, encodings, callback settings, an account extractor and grant policies.
The registered ChatGPT redirect is `http://localhost:1455/auth/callback`; its listener host is `127.0.0.1`.
Configured redirect validation accepts only loopback literals.

`load_bound_grant(provider_name, entry, secrets=...)` validates a stored binding and account claim without refreshing or writing.
`OAuthCredential` pins the first account, preserves an omitted account on refresh, and refuses a different account on refresh, sibling adoption or rescue.
A stored grant without an account cannot replace a pinned identity; only a refresh of the known grant inherits its account.
A classified `OAuthError.code` of `refresh_token_reused` permits one reread after one second, accepting only a different, fresh, binding-valid, same-account grant.
The other classified codes are `refresh_token_expired`, `refresh_token_invalidated` and `invalid_grant`; unclassified refusals have `code=None`.
`oauth_exchange.py` owns the optional second exchange and its expiry parser.

Configured issuers and ChatGPT use `OAuthCredential` and bound `StoredGrant` records.
`app/role_providers.py` injects an observer that writes `provider.auth` to the run's journal.
A refresh or exchange reports success after persistence, or failure after its endpoint refuses; the observer runs outside both credential locks.
Recovery reports the answer to the one post-401 retry, not the decision to rotate.
A sibling adoption without a refresh, a cached read and an initial 403 refusal emit nothing.
Holders and transports built without an observer emit nothing.
ChatGPT model listings read through the holder with refresh disabled and fall back to the cache on failure.

`ui/cli/connect_oauth.py` selects and runs browser, device and paste flows through the resolved profile.
ChatGPT's proprietary device flow falls back to the displayed authorization URL and pasted callback on refusal or cancellation.
Entries outside the current grant format read as `missing` with the `agent6 connect <name>` remedy.
A successful connect replaces the entry atomically.
`app/session/setup.py::sign_in_refusal` shares the bound-grant diagnosis between run preflight, `check config` and `model`.
`connect --status` reads the same validator's `BoundGrantError.cause` and checks each locally usable credential without refreshing or exchanging it.
The bound store's security rules are in [Security](security.md#8-secrets-and-connect).

## Prepared execution targets

`app/session/runtime.py::prepare_local_execution` resolves a local policy and opens its command session before gate inference or workspace commands.
`tools/command_tools.py::JailAccess` carries that policy, session, resolved grants and owned local resources into dispatch; it contains no lazy session callback.
Setup, baseline verification, model commands and later verification use that same session.
A failed preparation propagates before dispatch exists.
A dispatcher without a prepared session serves the file and navigation tools; its command tools refuse.

`kinds.ExecutionTarget` holds immutable transport, workspace roots, user, HOME, environment policy, effective controls and observed protocol identity.
Unobserved runtime, container, launcher-instance, launcher-digest and lifecycle identities are `None`.
Readiness advertises the protocol version and a random launcher-instance identifier.
Container installation records the transferred binary's SHA-256 as `launcher_digest`; a local launcher's digest stays unobserved.
The record is separate from live job state and resource cleanup.

`tools/policy.py` owns permission and grant assembly; `tools/target_policy.py` resolves local control descriptions, networking and HOME selection.
The dispatcher still gates tool availability and journals approvals before execution.
Preparing a transport grants no command permission.
`JailSession.connect` accepts a process transport and performs the common supervisor handshake; command handlers need no transport-specific argv or environment branches.
Verification inference accepts an executable-availability callable: active runs use their prepared session and its environment, while passive local previews explicitly use the advisory local PATH lookup.
Mid-run gate adoption uses the same lookup.
Source discovery still reads the host workspace.

## Environment ownership

`app/environment_lease.py` keeps one lease per selected environment under `<state-dir>/environments/<key>.json`, written under a sibling lock so children in other processes see one record.
The key names the selection: the manual exec prefix, or the workspace's discovery under its runtime prefix.
A replacement of a discovered container keeps the key, so the lease sees the swap.

`acquire` discovers the container; when none runs and the operator set a start command, it runs that command (fixed argv, no shell, 120 seconds, bounded output, the curated environment) and discovers again.
It says the start before running it: a `container: starting:` line and an `environment.starting` row.
The lease the run holds is journaled as `environment.acquired`, with the role, the reason and how long the start took; the policy row and the receipt show it.
The run owns the environment only when its own start made one container appear; the lease records what was observed before and after, never an assumed creation.
A container already running, an opaque exec prefix (whose presence nothing can observe), an ambiguous result and a replacement of an owned container all leave the run a borrower, with the reason recorded.
A resumed owner keeps ownership of the same container and loses it to a replacement.

`release` removes the session's borrower entry.
The owner stops the environment through the stop command only when no borrower is live, none ended without releasing, and a stop command is set; the observation afterwards decides `stopped` or `unknown`.
Every other case looks for the container first: `left_running` with its reason when discovery sees it, `stopped` when no dev container runs, `unknown` behind `container.exec` or when the look fails.
A lease whose owner's worker died is reported as orphaned, never inherited.
`session_runtime.prepare_container_execution` composes acquisition, the checkout mapping check (`container.check_checkout`: a checkout or a `.git` pointer outside the mounted workspace, or a linked worktree where the container mounts the workspace at another path than the host's, refuses before any command; the commands' root is the session's own checkout at its path under the mount), launcher preparation and the lifecycle hook on `JailAccess`; the dispatcher reports the hook's outcome as `execution.environment` after the session closed and its jobs settled.
`sandbox.isolation = "container"` selects this path: `execution.run_execution` prepares it in place of the local session, the machine lifecycles refuse under it, and `agent6 exec` names the runtime's own exec instead.

## Container discovery

`sandbox/container.py::discover` accepts an internal frozen `ContainerOptions`, not a public config model.
It returns a `ContainerDiscovery` with an exec prefix, unprepared `ExecutionTarget` facts and unexpanded `remote_env` metadata.
`sandbox.isolation = "container"` activates it; `[container].start` is the one way a run starts a container.
Config conversion belongs in app/tools, not sandbox.

- A nonempty `exec` bypasses runtime and definition discovery; each argv element is preserved unchanged, including sudo wrappers and an operator-installed devcontainer CLI.
  The command root is the `workspace` override or the host root; runtime, container and user identities remain unknown.
- Otherwise, an explicit `runtime` prefix selects the CLI; without one, discovery searches Docker then Podman found on PATH.
  Runtime executables resolve against the caller's cwd/PATH and become absolute before queries change cwd; remaining explicit prefix arguments stay unchanged.
  Automatic discovery queries symlink and hardlink aliases of one executable once, keeping the first name.
  Distinct wrapper executables remain separate; select an explicit `runtime` when wrappers reach the same backend.
  Each query appends fixed argv to that prefix, without a shell, automatic sudo or the devcontainer CLI.
- Running queries are `ps --quiet --no-trunc --filter label=devcontainer.local_folder=<root>` for the lexical and resolved host roots, deduplicated when equal.
  IDs are deduplicated within a runtime and sorted; several matches across either runtime refuse with their IDs rather than picking one.
  A missing runtime or failed query is a refusal, not an empty result or permission to skip that runtime.
- Only when no running match exists, `ps -a` with the same flags finds labelled containers for the no-running diagnostic.
  It never selects a stopped container or starts one.
  Existence of `.devcontainer.json` or `.devcontainer/devcontainer.json` also improves that diagnostic; contents are never opened, and editor-only custom definitions may be unobservable.
- One match gets `inspect --type container <id>`; pydantic validates the JSON shape, then discovery checks identity, running state and workspace label before selection.
  `devcontainer.metadata` accepts an object or ordered array of objects; the last supplied `remoteUser` wins over the runtime's `Config.User`, whose empty value means container UID 0.
  `remoteEnv` merges in metadata order and stays unexpanded data, outside the target's execution environment.
  Preparation must resolve supported container-only substitutions and refuse host or unsupported substitutions before use.
- Bind sources resolve host aliases and parent mounts to candidate command roots.
  The deepest destination mount covering a candidate must map back to the host workspace; a shadowed path or several different mappings refuses.
  An absolute `workspace` override takes precedence and creates no mount.

The derived prefix is `<runtime> exec -i --user <selected-user> <id>`, never a tty command.
Discovery does not execute it, install a launcher, prove a live shared-tree mapping, observe HOME or acquire lifecycle ownership.
The [security boundary](security.md#container-discovery-boundary) defines query bounds and runtime authority.

## Container transport

`sandbox/container.py::prepare` installs a validated static Linux launcher through the selected exec prefix, then hands its raw stdin/stdout channel to `JailSession.connect`.
`sandbox.isolation = "container"` activates this transport; lifecycle authority is the lease's ([environment ownership](#environment-ownership)).
The application remains responsible for selecting transports and resolving effective controls.
`sandbox/container_install.py` owns ELF validation, fixed bootstrap code and the host half of the mount proof; `jail_session` imports neither container module.

The bootstrap requires `/bin/sh`, `uname`, `mktemp`, `head -c` and `chmod` inside the target.
Its only positional inputs are the validated binary's byte count and architecture; model commands, workspace paths and metadata are not shell text.
It transfers the bytes once into a fresh selected-user-owned mode-0700 `/tmp/a6.XXXXXXXX` directory, then replaces itself with the launcher without a tty.
The host withholds the policy until the bootstrap acknowledges installation, so buffered reads cannot consume policy bytes.
Transfer and readiness share a 10-second deadline; stderr drains independently throughout.
The launcher checks its installed file identity and unlinks the executable before accepting commands.
The directory supplies target-local `TMPDIR` and anonymous capture storage.
Clean shutdown removes the directory only if empty; command-created files or abrupt transport loss may leave it behind.
Cleanup never recursively deletes target files.

A protocol probe observes effective UID/GID, architecture, the inherited environment and `/proc/self/mountinfo`.
Preparation resolves `${containerEnv:NAME}` against that environment, supports null removals, and refuses other substitutions by name.
Host HOME, PATH and TMPDIR are not copied into target commands; TMPDIR always belongs to the installation.
Environment values remain in memory; existing target stamps persist policy and variable names, not values.
Executable lookup uses this same prepared command environment.

The host creates and opens a random probe directory under a private staging directory, publishes it in the workspace, and writes a random nonce through its retained descriptor.
The selected user reads it through the command root and writes an independent nonce for the host to check.
Both sides use no-follow directory descriptors, exclusive file creation and bounded reads; cleanup unlinks only the probe entries through retained descriptors.
The probe contains no secrets and permits differing host and target UIDs.
Its writable directory proves shared visibility and rejects read-only mounts, not permission to write elsewhere in the workspace.
A mismatched, read-only or symlinked mapping refuses before model commands run.
The proof creates no mount, translates no argv strings and does not establish linked-worktree git-dir visibility.

## Jail supervisor protocol

`sandbox/jail_session.py` owns one serving launcher per run over local and prepared container process transports.
`jail/src/session.rs` owns its newline-delimited JSON protocol on stdin and stdout.
The initial policy selects `mode = "serve"`; after confinement setup, the launcher sends `{"ready":true,"protocol":4,"instance":"<32 hex digits>"}`.
Python refuses malformed readiness or a different protocol version before sending any command.

- Each `run`, `background`, `status`, `read`, `stop`, `probe` or `mapping` request carries a positive, increasing `id`; its single response carries the same `id` and the readiness `instance`.
- Only one request may be active; Python is the command-channel writer, and the Rust serve loop is the response writer.
- `{"kind":"cancel","id":N}` has no response of its own and applies only to the active request with that ID.
  Cancellation requests early hand-back when `checkin_s > 0`; otherwise it stops the active job and returns its exit result.
- Cancellation for any other ID is ignored, including one that races a completed request.
- One nonblocking stdin reader belongs to the serve loop and is polled while it waits for the job supervisor.
  It forwards cancellation and EOF over the private job channel without waiting for the command deadline; a partial frame never blocks command polling.
- EOF stops the foreground command instead of handing it back, then ends the session and its background jobs.
  Malformed input also closes the session after command cleanup.

Policy and request frames are capped at 1 MiB, including the newline.
Python caps readiness at 4 KiB and gives startup writes and readiness a shared 10-second deadline.
Responses are capped at 128 MiB, enough for JSON escaping of the retained command output (4 MiB head and 4 MiB tail per stream).
Python rejects a response with the wrong request ID or launcher instance and closes that session rather than using it for another request.
`sandbox/jail_protocol.py` frames requests and responses and drains launcher stderr throughout setup and execution, retaining its last 64 KiB with a truncation marker.
Startup diagnostics are surfaced once; the retained tail accompanies later protocol failures.
Command stdout and stderr are captured separately from these protocol streams.
Closing the session shuts stdin and discards pending response bytes while stderr keeps draining.
Python waits up to 10 seconds for launcher exit before killing and reaping it.

## Jail job ownership

`jail/src/session.rs` resolves opaque handles in its own job table; handles combine a random launcher-instance identifier with the starting request's ID.
Neither `status`, `read` nor `stop` accepts a command pid.
`jail/src/jobs.rs` owns each job's private subreaper, control channel and retained stop response.
The kernel reparents detached descendants to that job owner, so jobs and launchers sharing a PID namespace cannot claim each other's children.
The [cleanup boundary and failure limits](security.md#owned-process-cleanup) apply under `full`, `partial` and `none`.

`sandbox/jail_jobs.py` keeps the last terminal response and persists it on the host.
An exit code received in status or stop remains authoritative after cleanup failure or transport loss.
No completion response means an unknown outcome, even if the transport exits successfully.
Closing a session discards unread command responses and reports unconfirmed cleanup if the supervisor does not exit cleanly.
Host process snapshots are not part of the command-session lifecycle.

## Command capture and host records

`jail/src/capture.rs` drains stdout and stderr into an anonymous launcher-owned file, retaining bounded foreground excerpts separately.
Explicit background starts run the original argv without a redirection wrapper; hand-backs carry a handle and excerpts, never a target path.
A `read` request names that handle, a byte `offset` and a `limit` from 1 to 65536.
The response carries raw bytes as a JSON integer array, the next offset, the size observed before reading and any retained `capture_error`.
The capture remains readable after stop and disappears when the session closes.

`sandbox/jail_jobs.py` copies a size snapshot into the host log through its retained descriptor, advancing the offset only for successfully written bytes.
A failed host write can retry without skipping or duplicating bytes.
`tools/background.py` retains these host descriptors, renders scrubbed tails with explicit truncation, and stops jobs and copies their final bytes before closing their sessions.
Dispatcher teardown retains capture owners, host log descriptors, temporary storage and owned networks when host persistence fails.
A later successful close releases them.
The host log directory is not a command grant.
Terminal outcomes and channel-loss reasons are host records, separate from output; replay never infers completion from log text or a command-writable status file.
An unavailable capture leaves already copied bytes intact and reports incomplete output.
Read, status and stop responses carry capture failures separately from the exit code and cleanup outcome; the host persists and displays both.

## Tool dispatch

Every LLM tool call passes the same gates: audit events wrap it, the mode backstop refuses an out-of-surface name, MCP calls take an approval, and the handler table routes the rest by name.
Commands and verify run jailed; file tools resolve through the workspace boundary.
`tools/command_tools.py` holds jailed execution, command check-ins, and background read and stop handlers.
`tools/dispatch.py` owns routing, approvals, verify and metric events, and the jail-session lifecycle.
Under `partial` and `none`, `prepare_local_execution` gives commands a shared `TMPDIR` in a private `agent6-<session>-*` (the id's first 12 characters) directory in the system temp dir, falling back to `/tmp` when that root exceeds 32 bytes to leave room for Unix socket names.
Under `full` the jail's own tmpfs `/tmp` is the run's.
The start gate, verify tool, foreground commands and background commands use this directory; the dispatcher removes it after stopping commands at teardown.
`agent6 exec SESSION -- CMD` and `agent6 exec --new` use the same rule, with a fresh directory per exec, removed when the command exits and never shared with a live run.
Programs that ignore `TMPDIR` are not redirected by this environment setting.
A killed agent can leave this directory behind in the system temp dir.
CLI crash reports go beside the active session's log, or to the system temporary directory when no session has been attached.

Command streams lose terminal escape sequences before their 20,000-character tail cap and the model's result-byte cap.
Background reads scrub the decoded tail before selecting lines; the 1 MiB file-read cap still counts raw bytes.
The shared patterns live in `terminal.py`; command shaping preserves non-escape characters, including carriage returns, while rendering drops stray controls.
Metric scores parse raw, unclipped output, including escape bytes.

<!-- diagram: tool-dispatch -->

The table routes <!-- generated: tool-names -->.

## A review

The in-loop review panel returns structured findings over the run's diff.

[Review records](data-contracts.md#review-records) owns the shared findings, reviewer replies, verdicts, immutable target identifiers, observations and operator request models in `review_records.py`.

`viewmodel/review.py` folds panel evidence and finding history for both live `SessionState` updates and `scan_session_log`.
`prompts.review.panel_prompt` renders both reviewer scopes.
An observation opens a new finding but does not reopen a resolved one.
Only resolution and reopening events change that state.
In the journal fold, a Fix request leaves the finding open and reads as delivery unknown until a submission or failure event arrives.
The fold retains action reasons in append order and never treats a fixed disposition as proof that verification passed.
Freshness compares both tree IDs and path coverage with an explicit requested comparison.
Without that comparison, freshness is unknown.
Legacy counts carry an unknown-evidence notice; an unreadable journal or structured record retains its error in `MergeReviewGuard`.

`app.merge_plan.run_merge()` is the merge the CLI, the TUI and the web each run in their own process: it validates, then calls `app.merge.execute_merge()`.
`app.merge.execute_merge()` enforces the guard for CLI, web, TUI and auto-merge, and runs its git under the hook and driver policy of the config it was given (`git_policy.use`), so a merge inside the web server leaves the server's policy as it was.
It holds the existing session writer lock while re-reading the review fold, merging a pinned source commit, and recording that tip.
Auto-merge uses the lock its lifecycle already holds; live dispositions wait in the IPC queue.
An open grounded block or unreadable evidence refuses the merge with its reason.
Missing, stale and legacy count-only reviews do not block a merge.

`app/review/actions.py` owns Fix, Resolve and Reopen for every front-end.
Live requests cross `sessions.ipc` as typed envelopes; `app/execution.py` drains them before polling the steer bridge.
The worker checks the expected version and journals the action in append order.
Offline actions use the existing session writer lock and recheck that no worker is alive.
A repeated request ID returns its journal result or retained worker reply before checking the supplied version.
A worker refusal is final for that request ID; a new attempt needs a new ID.
Fix records its request before submitting the server-built instruction, so a crash between those writes leaves delivery unknown rather than a claimed success.
An explicit resume releases the writer lock before the injected launcher runs; the resumed worker records submission when it takes that instruction.
After a successful launcher handoff, an otherwise unknown-delivery response says `run resumed to fix finding <id>; submission acknowledgement pending`.
It uses the existing `queued` status and retains the request's recorded version.
Recorded failures and submissions take precedence.
An acknowledged resume uses `submitted` and the resume-specific message without the pending qualifier.
A live submission keeps `steer submitted`.
Launch confirmation creates no journal event: a retry before acknowledgement still reads as delivery unknown and never launches again.

The web Findings card reads the same review fold as the merge guard.
`ui/web/review_routes.py` serves recorded comparisons and files from immutable tree IDs, without a checkout or private review refs.
`client_git_diff.js` owns each diff request's lifetime; `client_review.js` attaches findings through the source renderer's line attributes only when both comparison trees match.
Reviewed file links keep the panel and side, so Files reads the cited version rather than the working tree.

`agent6 sessions review <id>` ([harness/review/session.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/harness/review/session.py)) reviews a finished session's record: the journal folds into a digest (the task, the steers and rulings, the verify runs, the tool errors, the harness notices, the memory facts the session wrote, the conversation's tail, every list capped and the cap named), the reviewer role reads it beside the repo's memory index, and the markdown names how the run ended, what went wrong and why, what the operator corrected, candidate memory facts and AGENTS.md lines with their evidence, and the memory entries the record contradicts.
It writes nothing to the repo or the memory: the operator turns a candidate into `agent6 memory add` or an AGENTS.md line, or leaves it.

## Session review directive

`directive.parse_review` parses `/review [TARGET]` without importing the engine.
`app/review/directive.py` resolves its reviewed side through `vcs.review.comparison`, loads base rules through `harness.repo_context.review_rules`, and builds the bounded message from `prompts/review_directive.py`.
`app.run` prepares a detached worktree through `app.fork_worktrees` when the range needs another checkout, for both ask and run.
`app.execution` injects the live review preparer through `OperatorBridge`; a live request for another tree returns a new-session command without changing the checkout.
`steer.feedback` carries refusals and no-op answers to the shared transcript fold without a model call.
The directive uses the session's tools and permissions and returns markdown, without writing findings or a review journal.

## Parallel runs

The primitive is a task run as a child in its own worktree, whose branch joins back.
One rule holds for every child (a child, a reviewer): the parent hands down, the child pushes up, and the root is the one owner.

- handed down: the config snapshot, the task and pins, the turn cap and deadline, and what the parent's budget has left
- pushed up: the child's approvals and waiting questions, its spend while it runs, and at its end its branch, its record and its memory, which the parent imports
- owned by the root alone: the operator's answers, the one budget ledger, and every write to the repository's branches and memory

Three consumers drive it: `run --parallel`, the web and TUI composers' `/parallel` new-work directive, and a live run's `/parallel` steer.

All three share `parse_spec` in [directive.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/directive.py), a pure-stdlib leaf both `harness` and `ui` import.
It maps a child count or model list to one model per child.
The composers and steering use `parse_directive` to split `/parallel [SPEC --] TASK` segments ([grammar and examples](usage.md#parallel-directive)).

Before any clone, a spec's models are checked against what a child can run.

- a child keeps the worker's provider unless the spec names another configured one; the model is checked against what the roles name on that provider plus its cached listing ([models/validate.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/models/validate.py))
- unknown model: a miss against an existing cache re-checks that provider's live listing, and refuses with a did-you-mean only when the fresh listing confirms it; no cache, or a re-fetch that fails, proceeds with a warning (offline machines never block on a regenerable cache)
- all three consumers validate through the one helper, keeping `harness` free of a models dependency

The primitive is git plumbing in [child_repo.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/harness/child_repo.py), with no LLM, no UI, and no process spawning:

- `clone_worktree(origin, dest)`: a plain `git clone` for a child's own worktree.
- `import_run(origin, child_worktree, branch, child_session_dir)`: fetches the child's branch and its retained session refs into the origin, refusing to overwrite an existing branch; the child's record is in the origin's state already.
  With model git control it fetches HEAD into the child's chain ref and creates its checked-out branch only when that name is free.
  Cleanup keeps a clone whose HEAD or ref tips the origin cannot reach, unless a merge stamp covers them.
- `ChildSpawner` / `GroupChildSpawner`: the Protocols for running one child to its end, or starting a sibling group in the background (a `StartedChild` per child: its session id, a future for its result, a stop hook).

[app/children.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/children.py) implements those Protocols: it holds the front-end's `ChildRuntime`, plans the children, clones, spawns, awaits and imports one child on a watch thread of its own, and builds the parent's `/parallel` dispatcher.
[app/parallel_results.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/parallel_results.py) imports a finished child, carries its memory and decisions back, and stamps a ranking on the run and its children.
[app/worktrees.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/worktrees.py) places clones and linked worktrees (`[git.worktree].base_dir`; `[container].worktree_base_dir` under a container), keeps a base inside the repository out of its git, writes child configs, recovers orphaned children and removes imported or safely swept clones.
The detached spawn `app/children.py` drives (`ui.spawn`'s `spawn_child`: the fixed `python -P -E -m agent6.ui.cli.child_agent` argv, the child's pid kept) is injected as a `ChildRuntime` by the CLI adapter `ui/cli/parallel.py`; liveness probes use `sessions.process` and stop requests use the session-dir bridge (`sessions.ipc`).
A child's task, pins, caps, away mode and lineage travel in its launch file (`app/child_launch.py`, `0600` under `<origin_state>/launch/<group>/`), beside its config snapshot; the bootstrap validates both and drives `app.run.run_task` with the CLI's session seam.
A launch file that carries a `kinds.ConversationContract` starts a `reviewer` session instead of a run: the same bootstrap, `run_task` and `Harness`, with the contract's system prompt and opening in place of the task block, the turn limit `ConversationContract.max_turns`, the tools its parent hands down (`ConversationContract.allowed_tools`, within the `reviewer` mode's surface), and no baseline, MCP server, advisor, review trigger or finish gate; `git.worktree.setup_command` runs only where the session may run commands.
Each turn opens with the reply count (`prompts.review.reply_notice`: `reply 14 of 20`, and on the last that a tool call in it is not run), said after any compaction so a restart keeps it.
A response with no tool call ends it `answered`, on the last allowed turn too; a last allowed turn that still asks for tools ends it `max_turns` with that response's text and those tools not run; either way `completion.json` (`sessions/completion.py`) holds the text, the stop reason, the provider calls made and what the limit cut.
Its session dir is `sessions/reviewers/<id>` under the state of the repository that dispatched it (`child_wait.child_dir` reads the bucket from the child's mode); `resume` refuses it.

**`agent6 run --parallel N|selector,...`** resolves each selector (a role name, or `provider/model` on the worker's effort and temperature) to one `kinds.ModelSettings` through `Config.child_settings`, the resolver every entry point shares, and plans one `ChildSpec` per child, each started through the child bootstrap as an ordinary detached run with its own jail and `run_commands` policy.

`app.worktrees.write_child_config` uses `config.child_snapshot` to materialize the parent's resolved model settings, with the child's selection applied to its driving role.
A work attempt journals `worker` calls even when its selector was `reviewer`; a ranking or panel reviewer journals `reviewer` calls with the selected route and effort.
A reviewer snapshot keeps the parent's resolved panel routes for quorum validation.
The snapshot keeps only the route in `models.default`; it omits sampling keys only for resolved `None` values.
On the internal child path, `app.session.setup.load_session_config` replaces only `[models]` after normal layer discovery and preset selection, discarding lower-layer model values and their provenance.
Other sections retain normal merging and explicit-default resets, including sandbox values and budgets; ordinary operator `--config` remains a merge.

TOML cannot preserve a selected `None` temperature with the unchanged sampling vocabulary: the omitted key reloads as the existing `0.0` default.
The snapshot records each affected field in `parallel.config_notes`, written with the child's first manifest and printed in the child's end report and when the run imports the child.
These notes survive manifest rewrites, import and temporary config cleanup; they document the loss and do not override the child config.

- `run --parallel` is a run: `ExecutionInputs.opening` carries its spec and task to the harness, and the loop's opening step (`loop_steer.ranked_opening`) dispatches the task in place of the first model call and waits for its children (`ParallelDispatcher.wait`); several children join under the policy `rank`, one under `merge`
    - under `rank` the dispatcher ranks once the children settle (`harness.parallel_rank.rank`), writes the order on the dispatch record before any join, joins the first child onto the run's chain and records each other child as `ranked` with its place; the opening waits for the judges as it waited for the children
    - the journal: the dispatch, the judges' own dispatch and return (`source = "panel"`), `parallel.compared` with the ranking and rationale, `parallel.joined`, the end (`children_ranked`, `no_child_result` or `budget_exhausted`)
    - a manifest comparison record: the group id, ranking, method, rationale and judge cost; child manifests keep only their placement and group id
    - `VerifyGate.after_rank_join` applies the parent's gate policy to the joined tree
    - without a parent result, `session.stop.child_verification` can cite a child's full pass on the same tree by the same command (child, attempt id, argv and tree)
    - `git.auto_merge` requires the parent's own gate pass or no gate (`finalize.auto_merge_eligible`)
    - under `git.control = "model"` the run has no chain: every child is placed and none is joined
    - a resume with a steer continues from the joined work with the worker model
- every child's manifest names the session that dispatched it (`parallel.parent`, stamped from the private launch), so each listing nests children under their parent, folded into a count
- a child's session dir is in its mode's bucket under the state of the root repository from its first event (its launch file names that state dir, `ChildLaunch.record_state`), and so is every reviewer a child dispatches: a child is visible in every hub while it runs, and nothing is moved when it ends
- The parent polls a work child's commit line while it waits: its hidden chain under agent6-owned git, or `HEAD` under model-owned git.
  The last measured count beyond the starting commit lives in `commits_beyond_start` in the child's session directory, outside the clone; a missing measurement is unknown.
- a child's memory and rulings stay in its clone's own state: seeded from its dispatcher's when it starts, merged back when it is imported (`parallel_results.carry_back`), and left untouched by a child that fails
- on completion a child's branch is fetched into the origin; a failed-to-start child keeps its clone, and a child whose import is refused keeps its clone and says where it is (never the only copy lost)
- the child watch waits for this execution's `session.start` or `session.resume`; until then the roster says `starting` and the parent can take turns or stop the child
  The operator can inspect or stop a live preflight worker before its report.
  Reviewers in a panel batch not yet submitted read `not started`.
  A process that exits without its report fails with its stderr or exit code.
- the kernel ties the child to the watch thread that spawned it with PDEATHSIG (SIGTERM).
  On a stop before the start report, or if a running child ignores its stop request, the watch sends TERM to its process group, escalates to KILL after the grace and reaps it.
  Its clone is kept.
- the watch relays the child's open approval or blocking `ask_user` through the parent's `OperatorPrompts`, including during startup
  It writes the answer, or the note that the operator gave none, to the child's answer slot.
  A child under a relay uses away mode `wait`, so prompts from nested children reach the root.
  The parent takes its own turns; one lock covers the relay's unanswered check through its answer.
  A stop, deadline or spent budget interrupts the relay without waiting for an operator answer.
- a dispatch reserves nothing: each child starts under what the parent has left in each finite ledger (`BudgetTracker.available`, `-1` and `0` passing through per currency) as its own caps, through its launch file, and a panel's later batch under what is left then; while a child runs, the parent's watch reads its journal on each poll and counts its spend so far against the parent's caps (`BudgetTracker.note_running`), the child's own running children included (`usd_running_children` and `tokens_running_children` on its `budget.update` row, which the watch writes as that spend moves, ten seconds apart at most); a running child's spend is inside `available`, `fraction_remaining` and the breaker and outside `usd_total`; the children together reaching a cap ends them at once, so the total can pass the cap by what was in flight and one poll; a child's settlement (`settle_child`) charges its final usage once and folds its plan readings through the parent's high-water rule so movement two children both saw counts once; a child whose usage is unknown stays unsettled, its last reading counted for that execution; a resumed execution takes the children its journal already names as settled (`mark_settled`) and charges none of them again; `run --parallel` hands down nothing (its children keep the config's caps and `--max-usd`)
- `budget.update` keeps the tracker's own input and output separate from `input_settled_children` and `output_settled_children` charged through settlement.
  Each child contributes its inclusive counts once, not its per-call rows as well.
  A continuation contributes only its increase over `usage_before`.
- every live dispatch is recorded before any child launches (`sessions/child_dispatch.py`, `<session_dir>/dispatch/<group>.json`, version 13, written atomically): the identity, source, join policy, owner execution, base, the task's spec, the DAG node it was dispatched under, each child's limits and whether its end was told; a work record is one task; an unwritable record refuses the dispatch; the record advances `prepared`, `running` (across the parent's turns), `settling` (each child's usage charged once, or marked outstanding), `ranking` (a task's attempts have ended and their judges run), `joining` (each join written as it merges with the chain tip before and after; a stopped child's join is `stopped` with its ref, commit count and measured context) and `complete` with the outcome notice, then `delivered`
- `viewmodel.children.ChildView` describes one child in the parent's `children` list.
    - The dispatch gives the child its number, task, group and outcome.
    - A join names the parent and the commit on its chain, including on `sessions show <child>`.
      Only a covering merge stamp names a checkout merge destination.
    - A child that never started keeps its row and the reason it failed.
    - A stopped child's row names its ref and the number of commits it holds.
    - The child's own journal supplies its running tool, turn, tokens, spend and elapsed time.
    - A reviewer row quotes its own verdict, not the combined ranking.
    - A session without dispatch records finds its children through their recorded lineage.
- the model's `parallel` tool (`[harness.parallel].tool`, off by default) runs through the same dispatcher with `source = "model"`: the tool dispatcher's `children` is the loop's `harness.child_work.ChildWork` (the Protocol `tools.child_tools.Children`), which starts one task for the turn in progress and returns the children's session ids as the tool's answer; `read_background` and `stop_background` reach a child by that id through the same object (its roster line, a wait for its end, a stop), beside a command's `bg<N>`; `steer_child` sends a running child an instruction (`child_work.steer`): `ipc.submit_steer` on the child's session dir, the text under `operator_bridge.PARENT_STEERING`, which the child's `Steering.handle` injects as one notice and journals as `steer.parent`, apart from the operator's verbs and directives; `parallel` and `steer_child` are listed only when `ToolDispatcher.parallel_available()` holds (the knob, run mode, not a child, the loop's children installed), the same gates refuse a call that slips past the list, and the two background tools' descriptions mention children only then
- a child's end comes with the execution's at the latest: `Harness.finish` closes the dispatcher (`ParallelDispatcher.close` stops each running child, waits for its watch and charges its usage), and `GroupSpawner.close` at the execution's teardown ends by signal a child an escape left running; the records stay incomplete, and the end's summary counts the children stopped and the ones that had ended
- a resumed execution reconciles every record that is not complete or whose panel charges lack journal confirmation before its first provider call: the grant is restored as recorded (settled children released, never a second split), the recoverer finds the children by their session dirs (a live one is stopped as `agent6 stop` stops a run, one whose record is only in its clone is imported, one already imported is read as it stands), usage is settled exactly once, and a task is re-opened with its children's results (`parallel_recovery.reopened`) and settled as a turn boundary settles it: each child's end not yet told is told, a stopped child joins as `stopped` and never merges by itself, a child whose work the chain already reaches is recorded joined at its merge commit and never merged twice, a task whose children ended by themselves completes under its git policy, and the notice is delivered once; a record with no child found completes as `dispatch interrupted before launch; nothing ran`; a complete record never delivered is delivered; a session older than the records has none to reconcile
- a child's `ContinuationRecord` keeps its instruction, sequence and already charged usage; its settlement key is `<child>:<sequence>`, so live readings and recovery count only new spend and an earlier execution's settlement cannot suppress the continuation's charge
- a panel gives each reviewer `ChildLimits` (a turn cap and a wall-clock deadline) in its launch file: the bootstrap clamps `harness.max_turns` (a configured `-1` cannot lift the cap) and arms its own backstop, the stop bridges at the deadline and SIGTERM to itself a grace later; the parent's await ends at the same deadline, and a child that reaches it reads `ended at its deadline`, never a success; a work child has no limits of its own and runs under its parent's config
- the ranking is [parallel_rank.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/harness/parallel_rank.py): a judge is a reviewer child whose verdict is an order
    - `rank` runs at the turn boundary after a task's attempts have all ended: it charges the attempts, briefs each candidate (`candidates`: its diff from the base, its gate's word, its cost, its finish summary; a stopped attempt is none) and, with two candidates or more and a judge configured, starts one reviewer per `[harness.parallel].judges` entry in the background (`start`: the rank prompt, an opening that lists the attempts, `review.max_turns`, the tools the run's model is offered cut to the reviewer surface, a clone cut at the attempts' base)
    - the judges have a dispatch record of their own, `<run>-r<group>`, a `panel` record written delivered; the task's record reads `ranking` while they run, its attempts read `being ranked` in the roster and hold the end, and `ParallelDispatcher.close` stops and charges them with the execution
    - at the boundary after the judges end, `replies` reads each judge's completion record (`reply_of`: an `order` that names every attempt once, with its `summary`), `combined` sums the places and settles a tie by the mechanical order, and `winner` crowns a verified attempt or none
    - with no judge, with one candidate, or with no usable order, the mechanical order ranks (verify-pass first, then the cheaper) and `RankRecord.fallback` says why; the merge notice carries the judges' reason or that line (`parallel_join.ranked_by`)
    - a resume never starts a task's judges twice: a judges' record an earlier execution left is settled and read for what its judges answered
    - `app.compare.rank` hosts the same reviewers for `sessions compare`, which has no run: it starts them through a spawner filed under `compare-<hex>` and waits
- the app records each ranking (`app_children.build_rank_recorder`, `OperatorBridge.rank_recorder`): it stamps each child's manifest (`placement`, one writer), so every view shows the rank and why and listings star the winner, journals `parallel.compared` and prints the ranked table
- under agent6-owned git, only the first-ranked child joins; the other children keep their refs
- under model-owned git, every child keeps its ref for the model to merge with git
- `--max-usd` is the run's cap: the judges are the run's children, settled on its tracker as the attempts are

**A `/parallel` steer and the `parallel` tool start tasks in the background** through `OperatorBridge.child_spawner`, on `Harness.bridge` (the injection point keeping `harness` from importing `ui`; `app/execution.py` wires the front-end's spawner, run mode only).

- `ParallelDispatcher.dispatch` returns once the children have started, in this order:
    - expand each segment into its children: a segment is one task, one group and one dispatch record
    - refuse when the children asked for and the children still running pass `harness.parallel.max_children`
    - under agent6-owned git, chain-commit the worktree; record the chain tip or the checkout's HEAD as the children's base
    - write each record, then start each task's children (`GroupChildSpawner`), each on a watch thread that clones, spawns, awaits and imports it
- the loop takes its turns meanwhile; at each turn boundary `loop_steer.operator_boundary` calls `ParallelDispatcher.settle`, and saves the resume snapshot when it told the model anything:
    - each child that ended is told to the model once (`parallel_join.end_notice`, with its finish summary)
    - each task whose children have all ended completes, in the order written; a task with a child still running completes at a later boundary and holds no other
    - under agent6-owned git, merging chain-commits the parent's uncommitted work, then merges one child (`chain_merge` syncs merged files into the worktree) or ranks several and merges the first; where `git.commit_per_turn` is off and the tree holds uncommitted work, nothing merges and the branch holds the child's work
    - a stopped child (its result carries `stopped_by`: the operator's stop, its parent's, the budget's, the end of its execution) never merges by itself: it joins as `stopped`, is no candidate among attempts, and its notice names the ref, commit count, last measured conversation size and time since it last ran (`parallel_join.stopped_words`)
- under agent6-owned git, a child that ended with a result and no commit beyond its base joins as `no_changes` and merges nothing
- a task gets its DAG node when its dispatch completes, under the node it was dispatched from: `passed` when a child joined, finished without changes, or finished with a retained ref under model-owned git; only a join records a sha on the task; a conflict aborts that merge and names its files and the branch; the run continues either way
- an end of the run is held while a task has pending work (`finish_gates.children_finish`, in every gate list): the refusal names each child that runs or awaits task completion; a stopped child holds no end unless its parent requests a merge or continues it
- `child_merge` records a stopped child's explicit merge in `ChildRecord.merge_request` (`MergeRequest.result`, `delivered`), completes it at the next boundary through the same chain operation as a finished child, stamps the task and journals `parallel.child_merged`; the request holds finish and survives resume, a success replaces the child's join, and a conflict keeps the stopped join retryable
- `child_continue` reopens a stopped child's dispatch before its private bootstrap resumes the same session and clone (`app.child_resume` validates ownership); `ContinuationRecord` records its instruction, sequence and cumulative usage baseline, `parallel.child_continued` journals the new execution, and the parent charges only its new usage
- stopped children keep their clone and launch file; a finished continuation fast-forwards its retained ref before cleanup, and direct operator child resume remains refused
- `merge_child` is offered only to a parent with stopped unmerged work under `git.control = "agent6"`; it does not require the parallel tool knob, since an operator directive can start children
- `parallel.dispatched` and `parallel.joined` name the same group as its dispatch record.
- The parent's transcript names a steer, a refused finish, a continuation and the result of a requested merge.
- The child's transcript shows a steer from its parent as a parent message.

**A child cannot dispatch work.** `app.child_launch.dispatch_authority` separates work dispatch from configured harness-panel scheduling on `OperatorBridge.child_authority`.
Every child bootstrap sets `AGENT6_CHILD=1`; a matching private worker launch retains panel authority under `[review]`, but its model and steer cannot dispatch work even with `harness.parallel.tool = true`.
An unproven child, including unrelated machine-state execution, gets neither authority.
A conversation contract grants neither authority, so the structure ends at the reviewer.

`parallel_panel.dispatch_panel` accepts reviewers' contracts and a retained target through an internal record-only policy, independent of `harness.parallel.tool`.
It shares the work dispatch's executor, reservation, import, settlement and recovery paths; `parallel_join` owns merge and DAG projections.
`review_gate.run_panel` prepares ordered requests using provider-free reviewers resolved by `app.role_providers.resolve_reviewers`.
The dispatcher reserves the entire roster before launching batches of at most `min(review.concurrency, harness.parallel.max_children)` reviewers; no reviewer is dropped.
Every reviewer is one contract: the panel prompt, `review.max_turns` turns and the tools the parent's own model is offered at that moment (`ReviewGate.tools`), cut to the `reviewer` mode's surface (`kinds.SESSION_KINDS`, `tools.schema.mode_tools`): every base tool but the two that edit, and `use_skill`; never the task list, `parallel`, `ask_user` or an MCP tool.
A reviewer's reply has the output limit every session's has, and its conversation compacts as a run's does; a restart keeps the opening.
A reviewer runs on the common bounded Harness path, including media recovery and final-turn verdict handling.
`review_verdict` owns message preparation, finding coercion, verdict parsing and abstention diagnostics, not execution.
The parent validates each completion's session, panel, reviewer and retained target before parsing and grounding; an ordinary worker answer or successful exit is not evidence.
Events, replay and receipts keep each reviewer's session ID.
Configured top-level ask and plan reviews can schedule panels, but cannot dispatch work.
Panel shares come from the worker's remaining allowance, and their deadlines take the minimum of the requested bound, the 1,800-second panel ceiling and the owning child's deadline.

Worker import preserves descendant session directories, dispatch records, objects and lineage in the original state directory, and fetches hidden session-owned target refs before removing the clone.
Stop and cleanup discover descendants through live symlinks, across buckets and through dead intermediaries.
Cleanup retains live descendants and unimported records.
The worker's usage is inclusive of settled descendants; importing their records does not charge them again.
The dispatcher journals the cumulative budget with `settled_children` before recording settlement; ancestor settlement retains its hold until those rows confirm every panel child's inclusion.
Recovery preserves journaled charges across executions and charges only children missing from that durable accounting.
Provider calls, child settlement and finalization share `sessions.budget_events` so every cumulative row carries the same accounting fields.
Plan-window readings keep their recorded providers and replay in timestamp order across worker and descendant journals.
Child panel evidence binds only its retained target: importing or merging the worker does not certify the top-level tree.

## Enforcement layering

- `git_ops.py` runs outside the jail, in the agent's own process, so the read-only bind of `.git` stops the worker without stopping the harness's commits.

[security.md](security.md) states which guarantee each layer provides.

## The curator and its locks

An in-process `GraphCurator` (`graph/curator.py`) owns the task graph; the agent constructs one per run, and the worker, the planner and the operator's own steering all mutate through it.
The same process writes the rest of the run state.
Every mutation validates against a pydantic schema before it applies.

Drawn by hand, the one writer and what it writes:

```mermaid
flowchart LR
    Agent["agent6 run<br/>main process"] -->|in-process GraphCurator| Graph["graph.jsonl<br/>graph/**/*.md<br/>cursor.json"]
    Agent -->|in-process| Rest["loop_state.json<br/>logs.jsonl<br/>transcripts"]
```

One curator per run is an invariant (two live curators cache independently; the second write drops the first's parent-child links).

- `run`/`resume`/`fork` take a single-writer flock on `<session-dir>/worker.lock`; a second process refuses
- a crashed writer releases on death: resume-after-crash never blocks
- a per-mutation flock on the session dir guards a concurrent `fork`'s read of the graph
- a write-path fault after the in-memory update reloads from disk before surfacing: no read ever observes an unpersisted node

One live run-mode worker per checkout is the level above (`sessions/lock.py`, a flock on `<state-dir>/locks/<checkout-id>.lock`).
Run-mode workers share one working tree, so a second would interleave two runs' edits into each other's chain commits.
A second `agent6 run` parks.
The submitted task is saved verbatim in the new run's manifest (`parked_task`, with `parked_reason`, shown as "parked · checkout busy" in listings), and `agent6 resume <id>` starts it once the checkout is free.
Plan and ask expose no edit tools and spawn freely; a run's children work in isolated clones.

Files that are untracked at start are the operator's: the run records them (`untracked-at-start`) and neither commits them nor counts them as dirt.
A resume adds the files that appeared between executions and the run cannot show it wrote, and its note names them.
An edit to one of them, or to a path git ignores, stays out of the run's commits: the edit's tool result says so (`checkpoint.left_out`), and the end summary lists the paths still on disk and absent from the commits.
After an execution that died mid-turn the rule turns over: a call of that turn may have written what appeared, so those files stay the run's, and the note names them.

Uncommitted changes to tracked files are asked about over the `ask_user` channel: stash for the run, include them in its commits, or cancel, which parks the run with `parked_reason` "uncommitted changes".
`[git].dirty_tree = "stash"` and `"include"` answer without asking, and a run nobody can answer refuses before writing a manifest, discarding the empty dir.
An included tree that is exactly an unmerged run's tip is recorded (`includes` in the manifest): when the including run merges and that run is still at the tip, it is stamped merged too (`merged.via`), so its own merge has nothing left to land.

Every end the loop reaches records the tree the run left (`session.stop.tree`), and a resume applies the same policy to the paths that differ from it: those changed outside the run.
Under `git.control = "model"` the checkout's HEAD seeds the comparison with that tree, so the model's commits decide which files are tracked even when an ignore rule matches them.
`stash` sets aside those tracked changes, including staged renames and deletions.
`include` leaves them in place and names them to the model before its first call.
It refuses when the run also changed one of those paths, since stashing it would take the run's work with the edit.
A resume after a death, a detach or an older session has no recorded tree: the changes no commit has and no tool of the run wrote follow the same policy, and its note names what the next commit takes.

A resume whose merge stamp covers everything the run committed, in a checkout whose HEAD holds the landing commit, continues from HEAD (`app/resume_chain.py`).
The chain ref and run branch only move forward: to HEAD after a `merge` or `ff_only` landing, else to a commit that joins the landed tip to HEAD's tree, so the earlier commits stay reachable for `fork --at-turn`.
The manifest's `base_sha` and the merge stamp's tip move to that commit, so `sessions diff`, `sessions commits`, the squash message, the review diff and the next merge read only the new work.
`base_branch` moves to the branch the merge landed on when the checkout has it out.
Where the workspace's HEAD does not hold the landing commit (a worktree run, a fork of a squash-merged run), the chain stays as it is and a resume moves `base_sha` alone, to the commit the next merge into `base_branch` starts from.
`app/merge_message.py` builds the squash message from the commits after the base its own merge starts from, so it names what that merge lands.
A run branch that is not at the chain tip (a commit made on `agent6/<id>`) refuses the resume, naming the commit; `--force` continues on the chain and leaves the branch.

## Session state on disk

The temporary workspace `.agent6/` layout and the transfer rules are documented under [Carry your state with the workspace](usage.md#carry-your-state-with-the-workspace).

Each session directory is `<state-dir>/sessions/<bucket>/<session-id>/`, where the bucket is named for the mode (`kinds.SessionKind.bucket`): `runs/`, `plans/`, `asks/`, `machines/` for `machine create` authoring, and `reviewers/` for the bounded conversations children run under a contract.
Ids are one namespace across every bucket, since every surface addresses a session by bare id.
Minting picks an id no bucket holds; `run --session-id` refuses one another bucket holds (reusing a same-bucket id is the resume or parked-start path), and `fork --session-id` refuses one any bucket holds.

| File | Holds |
|---|---|
| `manifest.json` | the [session record and later stamps](formats.md#sessionmanifest) |
| `graph.jsonl` | append-only journal of every task-graph mutation (curator-owned) |
| `graph/**/*.md` | one markdown file per task node, nested under its parent, rewritten atomically (curator-owned) |
| `cursor.json` | the curator's current task pointer |
| `logs.jsonl` | the structured event stream |
| `loop_state.json` | the latest resume snapshot, written before each LLM call and at turn end |
| `checkpoints/<NNNN>.json` | pre-call snapshots with message references, the run's chain tip (`head_sha`) and the curator `graph_version`; kept until explicit pruning |
| `turn_in_flight.json` | the turn whose tool calls are dispatching, with each result that returned; gone once the snapshot holds the turn or the execution ends itself |
| `plan.md` | the plan itself, in plan sessions |
| `transcripts/<timestamp>-<seq>.json` | provider attempt records with redacted request references, outcomes and response evidence |
| `objects/<sha256 hex>.json` | distinct sanitized JSON values shared by transcripts and numbered checkpoints |
| `pruned.json` | explicit artifact pruning time, file counts and last transcript sequence |
| `worker.pid` | the live worker's pid and start time; every surface reads liveness here |
| `approvals/`, `questions/` | one `<id>.answer` file per prompt, whichever front-end answers; `approvals/` also holds the session-wide allow and deny marker per scope |
| `frontends/` | one file per front-end registered to answer this run, named for its pid |
| `review-actions/` | one request and one reply file per finding action (Fix, Resolve, Reopen) a surface put to the run |
| `queue/` | the operator's queued tasks, standing goal and retirements (`/task`, `/standing`, `/retire`), read at the turn boundary |
| `shells/` | the roster of background commands the run started |
| `steer.request`, `stop.request`, `compact.request` | the operator's pending asks, which the loop reads at its turn boundary |
| `steers/` | the instructions queued behind the steer in flight, one file each in the order sent; the worker moves the oldest into the slot when it clears |
| `untracked-at-start` | the files the run treats as the operator's (repo-root-relative, NUL-separated): those untracked when it started, plus those that appeared between executions and it cannot show it wrote; left out of every chain commit and dirty check; a fork records its own checkout's set, and an `/undo` fork the set of the checkout it keeps |

Transcript callers supply the object directory explicitly: sessions use the layout's `objects/`, and `sessions review` uses `<state-dir>/reviews/objects/`.
The [Formats reference](formats.md#provider-transcripts) describes provider attempt records, their pre-send write and finalization, and their call and request ids.
Review calls carry a frozen `CallContext` through `RequestContext` to the transport and transcript writer.
Panel reviewers have distinct session directories and provider transcripts, including duplicate-focus reviewers.
Their completion artifacts carry the parent-owned `(panel_id, reviewer_index, target)` binding; the parent links verdicts to those sessions.

`finish_planning` is `plan.md`'s only writer and `agent6 plan edit` its only editor.
The planner re-reads it before every turn and is shown it whenever it differs from what it last saw.
`agent6 run --from` feeds it as a new run's task.

**A fork** clones a source run's state as of a checkpoint into a new session dir with a new id, and gives a run fork a linked git worktree of its own:

- adds the worktree detached at the turn's sha (`<[git.worktree].base_dir>/<repo-id>/<new>`)
- materializes the checkpoint into the new `loop_state.json` and writes seed `checkpoints/0000.json` with references to the child's own objects
- rebuilds the curator DAG at the checkpoint's `graph_version`
- records the fork lineage and worktree in the [manifest](formats.md#sessionmanifest), including the repository git directory the jail grants
- cuts the chain ref `refs/agent6/<new>/head` at the turn's sha, and the visible `agent6/<new>` branch there too under `[git].branch_per_run`; a plan or ask fork commits nothing and cuts neither

The source run and the operator's checkout are never mutated (except by `/undo`), and one fork edge per line lands in a per-repo `lineage.jsonl`.
`agent6 resume <new>` runs the execution in that worktree; the repository's state dir and config apply there.
The jail's rules for the recorded `worktree_git_dir`, including sessions started in user-created worktrees, are in [Security](security.md#2-sandbox).
The worktree's checkout lock is removed with the worktree.
The worktree shares the repository's refs, so `sessions diff|commits|merge <new>` work from the repo like any run's; `sessions prune` removes the worktree once the fork is merged, and `sessions remove <new>` removes it with the record.
An unstarted fork keeps its branch, chain ref and worktree even when its tip is reachable from the checkout's HEAD.
The manifest is the only thing that names a worktree as agent6's.
A worktree two sessions name (an `/undo` fork and its source) stays while either still needs it.

The clone sweep deletes a group dir of `child-*` clones once the origin reaches every one of their tips, and leaves every other directory under `[git.worktree].base_dir` alone.
A clone with no child session record stays because a spawn timeout can leave a child alive before its first write, even after the parent exits.

The rebuild (`graph/replay.py`) undoes every journal-stamped mutation newer than that version, so a fork's tasks, statuses, cursor, and journal match the turn its conversation came from.
Node content the journal never records (title, rationale, acceptance, paths) is immutable after creation and comes from the current nodes; `notes` and `updated_at` cannot be unwound and stay current.
A checkpoint whose `graph_version` is 0 or less has no version to rebuild at, so its fork copies the DAG verbatim.

On a gated run, an edit not yet committed at the forked turn is absent from the fork's tree, even though the copied transcript mentions it.
A fork is a commit plus the conversation up to that turn; no checkpoint holds uncommitted bytes.
An `/undo` fork adds no worktree: it keeps the undone session's checkout.

## Machines

`agent6 machine` runs a journaled reducer ([machine/engine.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/machine/engine.py)): each impure step's result is validated and appended before the new blackboard replaces the old, so a restart after a crash replays the journal to rebuild state and continues live from the last completed step, and `machine replay` reproduces the same path offline with no world at all.
[machine/journal.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/machine/journal.py) owns one instance directory, `<state-dir>/machines/<id>/`: the source the run started from, `journal.jsonl`, the snapshots, and the single-writer lock.
That is a different place from the `sessions/machines/` bucket, where `machine create` authoring sessions live.
`app/machine/check.py` owns the bundle, config and jail-PATH checks shared by `machine check`, `test` and the create gate.
[State machines](state-machines.md) has the spec and the state kinds.

## Events

Four front-ends share one headless core: the CLI, the Textual TUI, the browser UI (`agent6 web`), and the ACP agent an editor drives.
All four fold the same event stream and render their own way.
One listing row shape (`viewmodel.summary_row`) serves `sessions list --json` and `/api/hub`, and `sessions show --json` names the state with the same `status` word.
Two shared layers sit under them.

- the read side, [viewmodel/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/viewmodel): the `SessionState` and `MachineState` fold plus its wire form, exactly what `agent6 attach --json` and the web endpoints emit
    - the lifecycle and usage both folds share: `viewmodel/session/core.py`; tolerant session journal scan: `viewmodel/session/log.py`; shared labels and status decisions: `viewmodel/session/status.py`; directory discovery and listing rows: `viewmodel/listing.py`; the telemetry projection: `viewmodel/telemetry.py` folds runs and delegates machine transitions to `viewmodel/telemetry_machine.py`, `viewmodel/telemetry_records.py` holds the spans, log records and open work, `viewmodel/telemetry_metrics.py` the metric table and meter
    - session view records: `viewmodel/session/records.py`; status, context usage and wire rendering: `viewmodel/session/view.py`; event handling: `viewmodel/session/state.py`
    - machine execution and spend reads: `viewmodel/machine/logs.py`; watcher cursors: `viewmodel/machine/watch.py`; journal folding, probes and verb refusals: `viewmodel/machine/state.py`
    - transcript event ordering: `viewmodel/transcript.py`; pending tools and verification attempts: `viewmodel/transcript_items.py`; tool previews and marker text: `viewmodel/transcript_text.py`
- the write side: [ui/spawn.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/ui/spawn.py) for detached spawns, and [sessions/ipc.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sessions/ipc.py) for the approval, question, finding-action, task-queue, steer, stop-request and compact-request file contract the harness polls
- process identities, worker and network pid records, and front-end claims: `sessions/process.py`, shared by the bridge and its callers

A run started from a hub, or detached mid-run with `/detach`, keeps going with no terminal on it (`ui/spawn.py`); `agent6 attach <id>` opens it again, tailing the same journal every other surface folds.

**Stopping a run** is one function ([app/stop.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/stop.py)) behind `agent6 stop`, the TUI's Run menu, `/stop` in every composer, the web stop route and an ACP cancel.
[Exit codes and end reasons](cli.md#exit-codes-and-end-reasons) lists what the end block and listings say.

- It writes an abort steer, which cuts a model call in flight, and the `stop.request` marker with its `now` urgency, which ends a running foreground command at once.
  Unless it is request-only, it waits for the run to end.
- A worker still alive after that wait (5 s for a run, 30 s for one draining its children) gets SIGTERM and, 3 s later, SIGKILL, along with the background commands it started on the host.
- A run's worker takes the SIGTERM as a Ctrl-C stop (an `interrupted` end at exit 130).
- Every target is named by process identity (pid plus start time), so a pid recycled before the identity check is never signalled.
- `--after-turn` writes the plain marker alone, so the turn's running command finishes (or reaches `[harness].command_checkin_s`) and its tool results and auto-commit land first.
- A session's live children, found by the lineage their manifests record, stop with it at the same urgency: a plain marker lets each child finish its turn, a `now` marker ends their running commands at once; a parent that already ended still stops them (`children_stopped`).
- An ACP cancel is request-only: it writes both requests without waiting for or signalling the worker, which is the ACP server itself.
- An immediate stop ends a foreground `run_command` the launcher hands back.
  Its tool result carries `stopped_by_operator: true` and the output so far, without `still_running` or `background_id`, so the conversation a resume reads says the stop ended the command.
- A check-in with no immediate stop still returns a running command and its background id.

`evidence.VerificationEvidence` records each configured verification attempt, including harness-started attempts, with its ID, execution, target reference, command, scope, before and after trees, outcome and capture limits.
`VerifyVerdict` and `SessionSnapshot` carry the final verification ID and tested tree across resume.
A pass covers the final tree only when both observations and the end tree match; commit identity alone does not invalidate unchanged content.
`viewmodel/receipt.py` derives `ReceiptView` from the journal and manifest, with no second receipt file or duplicate event.
`viewmodel/cache_usage.py` matches each summary writer to its `model.result` by `model_call_id` for the receipt and run-end token summary.
It reports ordinary worker cache use and the first input after a history rewrite separately.
The CLI `sessions receipt`, web receipt route, TUI screen and ACP `/receipt` read that view without a provider call, verification run or environment probe.
`VerifyRun` in `harness/review/session.py` retains IDs, trees and capture facts in the bounded digest supplied to `sessions review`.

`execution.target` owns the immutable environment evidence for an execution; `app/stamps.py` mirrors its persistence-safe `kinds.ExecutionTargetStamp` into the manifest as history, not a current-target pointer.
The stamp retains both roots, user, HOME, effective controls, environment inheritance policy and variable names, never environment values.
Its host-issued `target_id` distinguishes observations; the target also carries the observed launcher instance and any installation build identity.
The dispatcher assigns `execution_id` before setup; setup, the baseline and later verification retain their own target references across resumes.
`evidence.ExecutionObservations` folds target, setup, job-settlement and environment-lifecycle records for `scan_session_log` and the receipt.
Its `ExecutionTargets` fold also supplies the current environment to live views and run and resume start renderers, independent of manifest mirror writes.
An execution boundary without matching target evidence clears that binding; invalid target records retain history but clear the displayed environment.
Missing historical references stay unknown, even beside a newer manifest stamp.
Start blocks and session inspection use these recorded facts, not the current config or runtime.

The manifest's optional `MergeStamp.method` and `target_before` own reversal bounds.
`RevertPlan` uses those facts and `vcs/commit_graph.py` to check parents and enumerate landed commits without depending on source refs.
See [Session receipts](usage.md#session-receipts) for output caps and reversal limits.

The journal is durable.
An append failure on anything but the streaming deltas stops the run loudly (`EventWriteError`), and in-process listeners see an event only after its write landed.

[Formats](formats.md#journal) owns the journal envelope, event names and payload fields.

An approval covered by a standing answer journals only `approval.answer`, with source `session`.
Every other approval publishes as `approval.prompt` before its answer.

- one gate (`tools/operator_prompts.py`, held by the dispatcher) mints the ids and journals every prompt/answer pair, whichever front-end answers; a front-end's approver and questioner only answer and name their source
- the TUI's Allow/Deny writes the literal choice to `approvals/<id>.answer`; the gate clears that slot before journaling the prompt, and the approver reads it before the gate records `approval.answer`
- the transcript fold marks the call a prompt's `call_id` names as awaiting (`sessions show`, the web and TUI conversation, ACP's `pending`)
- the asking side decides what a choice grants: each prompt names its "allow all" scope (`command`, or one MCP server), and standing answers record per scope
- a no-standing gate sets `standing: false`, and no front-end shows the button
- after the front-end stays dead 30 consecutive seconds, the answer poll follows the [away mode](security.md#6-approvals) or falls back to stdin in an interactive terminal run
    - a page reload or a locked phone never converts a pending approval into a deny
- a watching browser registers as the run's answer front-end; prompts bridge to the page
- the task DAG rides as `graph.update` once per turn that mutated the DAG, plus the root seed, a surfaced task, an operator request (`/task`, `/standing`, `/retire`), the finish auto-pass, the compaction check-off and each parallel stamp; every mutation is curator-owned in `graph.jsonl`, read via `sessions graph`

## Where things live

| Concern | File or directory |
| --- | --- |
| Config schema | [config/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/config) (`model.py` holds `Config` and `[agent6]`; `model_roles.py` owns role inheritance; the other sections live in `sandbox_config.py`, `harness_config.py`, `git_config.py`, `surface_config.py`, `provider_config.py`) |
| Tool surface | [tools/schema.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/tools/schema.py) |
| Tool dispatch | [tools/dispatch.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/tools/dispatch.py) |
| Agent loop | [harness/loop.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/harness/loop.py) |
| Prompt text | [prompts/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/prompts) (templates and reviewer renderers the loop, review, ranking, and machine assemble; `compaction.py` holds the summariser, gist and restart-notice prompts) |
| Review panel | `harness/review/panel.py`, `harness/review/verdict.py` and `harness/review/gate.py` |
| Jail sessions and detached jobs | [sandbox/jail_session.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sandbox/jail_session.py), [sandbox/jail_jobs.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sandbox/jail_jobs.py) |
| Jail launcher | [sandbox/jail.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sandbox/jail.py) (Python), [jail/src/main.rs](https://github.com/agent6-dev/agent6/blob/master/src/agent6/jail/src/main.rs) (Rust) |
| Git policy | [git_policy.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/vcs/git/policy.py) |
| Child repository primitive | [harness/child_repo.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/harness/child_repo.py) |
| Session-dir single-writer lock | [sessions/lock.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sessions/lock.py) |
| Ranking attempts | [harness/parallel_rank.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/harness/parallel_rank.py) |
| Child orchestrator | [app/children.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/children.py) (children), [app/parallel_results.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/parallel_results.py) (import and ranking stamps), [app/worktrees.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/app/worktrees.py) (working directories), [ui/cli/parallel.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/ui/cli/parallel.py) (CLI adapter) |
| Provider clients | [providers/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/providers) (`oauth.py` holds bound grants; `oauth_profile.py` holds resolved settings and identity and expiry policies; `oauth_exchange.py` exchanges issuer bearers for model tokens; `claude_code.py` orchestrates rounds; `claude_code_protocol.py` handles control messages and tool continuations; `claude_code_session.py` owns session state and pipe I/O; `claude_code_stream.py` folds stream events and scrubs email; `claude_code_wire.py` holds wire shapes) |
| Task graph | [graph/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/graph) |
| Event log and fold | [event_log.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sessions/event_log.py) (writer), [viewmodel/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/viewmodel) (fold) |
| Front-end write bridge | [ui/spawn.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/ui/spawn.py), [ui/notify.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/ui/notify.py), [sessions/ipc.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/sessions/ipc.py) |
| Telemetry export | [ui/telemetry/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/ui/telemetry): the journal reader that encodes `viewmodel/telemetry.py` as OTLP JSON and Prometheus text, started by `agent6 telemetry export` |
| Web UI | [ui/web/](https://github.com/agent6-dev/agent6/tree/master/src/agent6/ui/web) (stdlib HTTP server; `page.py` assembles the `client_*.js` subjects in declaration order as one classic inline script, without a build step) |
| Repo memory | [memory.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/context/memory.py) (store), `<state-dir>/memory/` (data) |
| Machine engine | [machine/engine.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/machine/engine.py) (replay and live loop), [machine/execution.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/machine/execution.py) (state execution and reducer), [machine/contracts.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/machine/contracts.py) (world protocol and execution records), [machine/journal.py](https://github.com/agent6-dev/agent6/blob/master/src/agent6/machine/journal.py) (journal and instance dir) |

## Bench and development switches

These env vars are switches for benchmark arms, harness experiments and debugging, listed here so no behaviour keys off undocumented state:

- `AGENT6_SYMBOL_TOOLS`: selects a symbol-tool arm, hiding part of the navigation surface; a call to a hidden tool says so.
- `AGENT6_FORCE_STREAM=1`: streams the run's reasoning to stderr with no TTY, for a bench or CI log.
- `AGENT6_DEBUG=1`: re-raises an unexpected error with its traceback instead of writing a crash-report file, and prints the notices a run would keep to its log.

agent6 also sets markers for its own children:

- `AGENT6_CHILD=1`: a child of a session (a child, a machine state, a `/btw` question), whose model and operator work entry points cannot dispatch work ([Parallel runs](#parallel-runs)).
- `AGENT6_STREAM_TO_LOG=1`: a detached spawn or child emits its stream deltas as events with no console echo.
- `AGENT6_LAUNCH=<path>`: the child's launch file (`app/child_launch.py`): the task and pins, the caps and away mode, the config snapshot the child's sole `--config` must name, its serialization loss notes, the lineage the child's manifest records, the origin of the verify gate the parent resolved, and the operator's task; the bootstrap refuses a missing, malformed, foreign-owned or exposed file, and the config load refuses another `--config`.

## Pre-1.0 stability

See [Support levels](data-contracts.md#support-levels) for format compatibility before and from 1.0.

## Media attachments

`kinds.MediaAttachment` holds static image or PDF bytes in memory; `ToolResult.attachment` and `ToolResultItem.attachment` carry it beside unchanged text.
`ReadMediaResult.to_wire()` is one sentence naming the file and saying the attachment follows; the journal keeps the metadata, and no surface shows payload bytes.
`Conversation.to_wire()` encodes retained attachments as text-plus-media arrays under their tool-result IDs.
`read_file` reads at most the document cap plus one byte through the contained reader before inspecting media.

Anthropic keeps native image and document blocks under the tool result.
Chat Completions and Responses (including ChatGPT) keep tool replies as strings and place labelled media in a following user message after all parallel replies.
That message encodes tool output and does not change the initiator.
Claude Code receives MCP image or resource items, replays retained attachments as native blocks on respawn, and restarts when a media result changes.

- [Tools](usage.md#tools) lists the fixed image and PDF limits.
- `AttachmentBudget` counts original bytes; Claude Code checks its complete encoded stdin line against the 8 MiB line cap.
- `media_files.inspect_media` checks bounded container bytes without decoding pixels.
- Local compaction elides delivered attachments without gisting their payloads; `MEDIA_CONTEXT_CHARS` reserves context independently of base64 length and needs live calibration, especially for PDFs.
- `models.cache.cached_input_capabilities` reads explicit image and document facts from fresh listings, bound to the endpoint, provider entry and exact model.
- Missing modality data means unknown; names never imply support.
- `provider_call.call_with_media_recovery` applies the calling role's facts to canonical tool results and retries one recognized `MediaInputError` without rerunning tools.
- An endpoint that writes signed summaries keeps delivered media intact and refuses excess new attachments.
- Delivery refusals emit `media.refused` with each affected tool use's `provider_tool_call_id` and update the worker's latest snapshot before retrying.
- Confirmed unsupported modalities stay unsupported for that route during the execution; malformed media and size rejections do not change capability facts.

See [Media retention](security.md#media-retention) for persistence and visual-secret disclosure.
