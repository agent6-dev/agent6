# Editor integration

Two ways in, both spawned by the editor over stdio: ACP drives whole runs, and `agent6 mcp serve` exposes a few tools to another agent (see [As an MCP server](#as-an-mcp-server)).

## Agent Client Protocol

`agent6 acp` runs agent6 as an [Agent Client Protocol](https://agentclientprotocol.com/) agent: an editor spawns it, sends prompts, and renders the run as it happens.
It uses the same engine, config, and jail as `agent6 run`.

```jsonc
// Zed: settings.json
{
  "agent_servers": {
    "agent6": { "type": "custom", "command": "agent6", "args": ["acp"] }
  }
}
```

## What the editor sees

Every run writes one event journal; the CLI, TUI, web UI, and ACP all render it the same way.
An editor sees what `agent6 attach` shows: reasoning, each tool call and its outcome, auto-commits, and how the run ended.

ACP carries a tool call as two messages.

- `tool_call` (`in_progress`) when the run dispatches it, `tool_call_update` (`completed` or `failed`, with the output) when its result lands
- a call names its tool (`name`) beside the call's title; built-in calls carry their ACP kind, and an edit result carries each written or deleted path as an absolute follow-along location, without a line
- a call waiting on an approval or an `ask_user` answer is updated to `pending` while its prompt is open, and back to `in_progress` once answered
- a long verify shows as in progress while it runs; a call the run never returned from settles as `failed` when the run's `session.stop` is written, or when the tail ends without one (a worker killed mid-call)
- `toolCallId` is `<run id>:<execution id>:<call>`, unique for the life of the session; the dispatcher numbers each execution's calls from 1

An edit shows as a diff.

- the `tool_call_update` of a successful `apply_edit` or `apply_patch` carries its text summary, then one `diff` entry per file: the absolute `path`, `oldText` and `newText`
- a created file has `oldText: null`; a deleted file has `newText: ""`, and the summary says `deleted`
- both texts are recorded when the tool writes, so two edits to one file in a turn each show their own diff, and `session/load` replays the diffs the live run showed
- the budget is 64 KiB of UTF-8 per call, both sides of all its files together.
  A call over it carries no diff, and its summary ends `(diff omitted for size)`; a side is never cut to fit.
- a preview, a failed call, a command and an MCP tool carry no diff
- after `sessions prune` removed a run's stored texts, a replayed edit carries its summary and `diff unavailable: its stored text is gone`

Worker text and thinking arrive as they stream; output from side calls stays out of the conversation.

Inspect persisted [line-origin evidence](usage.md#session-findings) with `/findings`, `agent6 sessions findings <id>` or the web Findings card.

Four more updates ride the journal.

- `session_info_update`: the task's headline as the session title, the line `agent6 sessions list` shows
- `available_commands_update`: the [slash commands](slash-commands.md) the editor can offer for this run's mode.
  The front end handles [finding actions](#review-findings) without sending their text to the model.
  `abort`, `exit` and `detach` are task text over ACP, not terminal commands.
  The editor cancels through `session/cancel`.
- `usage_update` after each model call: the last call's whole prompt over the model's context window (the `ctx N%` rule), and the run's cumulative USD when it spent any and every call was priced (a plan-metered run carries none); nothing while the window is unknown
- `plan`: the run's task list, in the depth-first order every other surface shows it.
  Each change of the task graph sends the complete list, an empty one included; so does the run's end, which clears the focus, and `session/load` replays them.
    - the content is the task's title, and every entry is `medium` priority
    - a pending task reads `pending`; a task in progress, or the one the worker is on, reads `in_progress`; a passed task reads `completed`
    - a failed task reads `pending` with ` (failed)` after its title, since ACP has no failed status
    - skipped and obsolete tasks are left out

## Images in a prompt

`initialize` answers `promptCapabilities.image`: a `session/prompt` may carry `image` blocks.

- the block's inline `data` is read; its `uri` is never opened
- the images go with the task a first prompt starts, or with the instruction a later prompt resumes the run with, after the prompt's text
- the limits and the refusals are the ones [every image has](usage.md#images-with-a-message); a refused image answers the request with a JSON-RPC error that names it
- a prompt that is a directive (`/review`, `/parallel`) takes no image
- a request line may be 4 MiB plus the base64 of 4 MiB

## Parallel work

`/parallel [SPEC --] TASK` starts child attempts on the [same terms as a run](usage.md#parallel-directive).
Repeat `/parallel` in the prompt to start another task.
A first prompt can dispatch these tasks without a parent model call.
The model can dispatch work only when `[harness.parallel].tool` is on.

Child steps appear as labelled messages in the parent conversation while the children run.
The parent relays a child's approval request with the child id in the title.
The children run in their own clones under the run's budget and sandbox settings.
They get no editor filesystem, terminal or connection.

## Review findings

Findings and dispositions appear as `[agent6]` transcript notices, not native review threads.
Commands act on the conversation's recorded run.
An answer to a queued question reaches the run at its next turn.
An ended run receives the answer on its next prompt, which resumes it.

Use `/fix` with `--resume` to confirm **Resume to fix** when the session is idle.
`--all` selects every open finding without a queued Fix.
Fix uses the stored reports and leaves the findings open.
It refuses a missing resume point rather than starting a new run.

Live disposition actions use the worker's IPC queue; their notices arrive when recorded.
Fix does not support mid-run steering through ACP.

## Receipts

The advertised `/receipt` command reads [testing, review and reversal evidence](usage.md#session-receipts) without calling the worker or rerunning checks.
Ask for a reversal in a subsequent prompt or a new isolated run; displayed git commands are explanations, not executable actions.

## Code-review rule sources

See [Code-review rules](usage.md#code-review-rules) for discovery and limits.

## Approvals

`session/request_permission` carries every approval the CLI would prompt for.

- a command under `run_commands = "ask"`
- an MCP tool call the server's `approve` does not cover
- a `fetch` to a host outside the allow-list
- an unsandboxed autorun
- a new run on another run's branch

The editor renders the buttons; the request names the tool call it gates and carries the prompt as that call's title.
A prompt that gates no call (a pre-run question) announces a tool call of its own and closes it with the answer.
The prompt and its answer go through the same gate every front-end answers through, so `agent6 attach` and the web show the run as waiting; the answer journals `source: "acp"`, `"frontend"` when another surface answered it, or `"headless"` when the client cannot answer it.

- an unanswered request denies after five minutes, and the run continues without it
- a command or an MCP tool call offers the four answers every front-end shares: "allow" (`allow_once`) and "deny" (`reject_once`) answer the one call, "allow all" (`allow_always`) grants that scope's later calls for the session, "deny all" (`reject_always`) withdraws the scope's tools for the session
- an off-list `fetch` host is offered as `allow_once` only, so an editor's "always allow" cannot cover a different host later
- a standing "allow all", from the editor or an earlier front-end, answers that scope's later prompts without asking the editor (`source: "session"`)

## Sessions

A session is one conversation in one directory.

- `session/new` carries an absolute `cwd`, a git repository, whose own layered config applies
  Each execution uses that directory without changing the connection process's cwd.
- the first prompt starts an `agent6 run`; every later prompt resumes it with the text as its steering instruction (an execution that died before its first checkpoint leaves nothing to resume: the next prompt starts a new run, and the editor is told)
  After an undo on another surface, the next prompt follows the run's continuation.
- a busy session refuses a prompt rather than queueing it; one connection runs one prompt at a time across its sessions, and a prompt on another session queues and says for which session it waits
- `session/list` (`cwd` optional, absolute) names the repository's resumable runs, newest first: agent6's run id as `sessionId`, the task as `title`, the last activity as `updatedAt`; one page, no cursor.
  A run `agent6 resume` would refuse (a child, a run whose worktree is gone, a run with no checkpoint) is left out.
  Without `cwd` it covers every repository the connection has a session open in.
- `session/resume` opens a conversation on a recorded run by that id, so an editor reopens its thread after a restart.
  `session/resume` refuses a plan, an ask, a run whose worker is alive, or a run `agent6 resume` would refuse.
  It also refuses a run this connection has open for another directory or with a prompt in flight.
- `session/load` opens the recorded run like `session/resume`, replays its journal as ordered `session/update` notifications, then returns.
- `session/prompt` answers `end_turn` when the run finishes, a red gate included, and `max_turn_requests` at the turn cap.
  A run that fails or never starts answers with a JSON-RPC error (`-32603`, or `-32602` for a prompt agent6 refuses); the reason arrives first as a message.
- `session/close` ends a conversation; a prompt in flight is cancelled first and answers `cancelled`.
- `session/cancel` cuts a model call in flight and ends a running foreground command at once, without waiting for or signalling the worker (the ACP server itself).
  The prompt answers `cancelled` (ACP's word) and the run's record reads `stopped`, as every operator stop does; what it had not yet committed stays in the tree for the next prompt, which resumes it.
  A prompt still queued answers `cancelled` at once.
  Cancellation releases an open permission wait without waiting for the client's reply.
  A late permission reply is ignored and cannot grant permission to a later prompt.

## Model and effort

`session/new`, `session/load` and `session/resume` answer with `configOptions`: two selects an editor shows as pickers.

- `model` (category `model`): the routes the configured providers offer, the list every other front-end's picker offers.
  Its current value is the route the next prompt runs: the picked one, else the one the run recorded, else the config's.
- `effort` (category `thought_level`): `inherit`, which takes the config's effort, then `off`, `low`, `medium`, `high`, `xhigh` and `max`
- `session/set_config_option` picks a value for that session alone and answers with the complete list.
  The pick applies from the next prompt, a prompt that resumes the run included.
  It changes no running turn and writes no config file.
- a set is refused while a prompt runs on the session, and for an id or a value the session does not offer; the error says which
- `config_option_update` carries the complete list when agent6 changes it: a run cached a provider's listing, or recorded another route
- there is no `mode` option and no `modes`: an ACP session only opens runs
- a session with no route to run offers no option: its config does not load or names no model, and no run is recorded.
  The first prompt says why it cannot run.
- a run another front-end detaches to the background keeps the picked model and takes the config's effort

## Not implemented

- `session/delete`: the operator removes a run's record (`agent6 sessions remove`).
- Mid-run steering: ACP has no message for a prompt while the previous prompt is running.
- `fs/*` and `terminal/*`: ACP lets the client own the filesystem and the terminal, and agent6 keeps both behind the jail the operator configured.
- Free-text answers to a waiting `ask_user` question: the editor gets option buttons only, so a waiting question without options is unanswered and not queued (see the [frontend cases](#approvals)); `/answer` takes any text.
- Embedded resources in a prompt: text, `resource_link` and `image` blocks are read (a link rides in as its uri; the workspace boundary still decides what it reaches)
    - embedded resources are dropped; `promptCapabilities.embeddedContext` says so

## Troubleshooting

- stdout is the protocol stream: nothing but JSON-RPC
- everything agent6 would print goes to stderr (the editor's agent logs)
- a wrapper echoing to stdout before exec'ing `agent6` breaks the connection irrecoverably; write to stderr

## As an MCP server

`agent6 mcp serve` speaks MCP over stdio, so another agent (an editor's own, or a second agent6 with `[mcp.servers]`) can use agent6's jail and run state.
The cwd's config decides everything: the sandbox the commands run in, and which tools exist at all.

Five tools, of which a default config publishes two:

| Tool | Withheld when |
| --- | --- |
| `get_task_graph` (a run's task graph) | never |
| `list_sessions` (this repository's sessions) | never |
| `run_verify_command` (the configured gate, jailed) | no `[harness.verify] command`, or `[sandbox] run_commands` is not `allow` |
| `run_in_sandbox` (an argv, jailed) | `[sandbox] run_commands` is not `allow` |
| `apply_patch_in_sandbox` (a patch, then the gate) | either of the two above |

`run_commands = "ask"` withholds the command tools: the MCP boundary has no operator to prompt.

```jsonc
// Zed: settings.json -- agent6's jail as another agent's tool surface
{
  "context_servers": {
    "agent6": { "command": "agent6", "args": ["mcp", "serve"] }
  }
}
```

The tools reach the repository the server was started in, under that repository's sandbox policy; `list_sessions` and `get_task_graph` read its state dir and nothing else.
