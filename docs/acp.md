# Editor integration

Two ways in, both spawned by the editor over stdio: ACP drives whole runs, and `agent6 mcp serve` exposes a few tools to another agent (see [As an MCP server](#as-an-mcp-server)).

## Agent Client Protocol

`agent6 acp` runs agent6 as an [Agent Client Protocol](https://agentclientprotocol.com/) agent: an editor spawns it, sends prompts, and renders the run as it happens.
It uses the same engine, config, and jail as `agent6 run`.

```jsonc
// Zed: settings.json
{
  "agent_servers": {
    "agent6": { "command": "agent6", "args": ["acp"] }
  }
}
```

Any ACP client works the same way, and the command above is the whole configuration.

## What the editor sees

Every run writes one event journal; the CLI, TUI, web UI, and ACP all render it the same way.
An editor sees what `agent6 attach` shows: reasoning, each tool call and its outcome, auto-commits, and how the run ended.

ACP carries a tool call as two messages.

- `tool_call` (`in_progress`) when the run dispatches it, `tool_call_update` (`completed` or `failed`, with the output) when its result lands
- a call names its tool (`name`) beside the call's title; built-in calls carry their ACP kind, and an edit result carries each journaled path as an absolute follow-along location
- a call waiting on an approval or an `ask_user` answer is updated to `pending` while its prompt is open, and back to `in_progress` once answered
- a long verify shows as in progress while it runs; a call the run never returned from settles as `failed` when the run's `session.end` is written, or when the tail ends without one (a worker killed mid-call)
- `toolCallId` is `<run id>:<execution>:<call>`, unique for the life of the session; each prompt's execution numbers its calls from 1

Worker text and thinking arrive as they stream; side-role output stays out of the conversation.

Inspect persisted [line-origin evidence](usage.md#session-findings) with `agent6 sessions findings <id>` or the web Findings card.
ACP notices do not include line-origin tags.

Three more updates ride the journal.

- `session_info_update`: the task's headline as the session title, the line `agent6 sessions list` shows
- `available_commands_update`: `/pin <text>` and the [review commands](#review-findings).
  `/pin` reaches the run as steering text; review commands are handled by the front end, not the model.
  The composer's other directives (`/compact`, `/btw`, `/task`, ...) reach the model as words.
- `usage_update` after each model call: the last call's whole prompt over the model's context window (the `ctx N%` rule), and the run's cumulative USD when it spent any and every call was priced (a plan-metered run carries none); nothing while the window is unknown
Everything agent6 says carries `[agent6]`, the model's final summary arrives bare, and the cost receipt goes to stderr.

## Review findings

Findings and dispositions appear as `[agent6]` transcript notices, not native review threads.
The advertised commands act on the conversation's recorded run:

- `/findings [--all]` lists open findings, or also resolved findings with `--all`.
- `/review-queue` lists the repository's sessions in [review queue](usage.md#the-review-queue) order under group headings, and needs no run in the thread.
- `/answer <question-id> <text>...` answers a pending question, with options or without; a queued question's answer reaches the run at its next turn, and an ended run's is delivered by the next prompt, which resumes it.
  A question the model queued shows in the thread as a line naming it and its options.
  A needs-answer, blocked, unverified or awaiting-merge row names the CLI command that acts on it.
- `/fix ID --resume` confirms **Resume to fix** as a follow-up prompt when the session is idle.
  It uses the stored report, leaves the finding open, and refuses a missing resume point rather than starting a new run.
- `/resolve ID --as DISPOSITION --reason "TEXT" [--duplicate-of ID]` records the operator's conclusion.
  Dispositions are `fixed`, `false_positive`, `intentional`, `wont_fix`, `duplicate`, and `pre_existing`.
- `/reopen ID --reason "TEXT"` reopens a finding.

Reasons must be nonblank and at most 2,000 characters.
Live disposition actions use the worker's IPC queue; their notices arrive when recorded.
Fix does not support mid-run steering through ACP.

## Receipts

The end message includes a short evidence summary and points to `/receipt`.
The advertised `/receipt` command reads [testing, review, risk and reversal evidence](usage.md#session-receipts) without calling the worker or rerunning checks.
Ask for a reversal in a subsequent prompt or a new isolated run; displayed git commands are explanations, not executable actions.

## Code-review rule sources

Code-review starts appear as session messages with the panel ID, base tree, rule completeness, source paths, blob IDs and byte counts.
An incomplete rule set names its refusal reason; an empty complete set says that no base rules exist.
See [Code-review rules](usage.md#code-review-rules) for discovery and limits.

## Approvals

`session/request_permission` carries every approval the CLI would prompt for.

- a command under `run_commands = "ask"`
- an MCP tool call the server's `approve` does not cover
- a `fetch` to a host outside the allow-list
- an unsandboxed autorun

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
- the first prompt starts an `agent6 run`; every later prompt resumes it with the text as its steering instruction (an execution that died before its first checkpoint leaves nothing to resume: the next prompt starts a new run, and the editor is told)
- a busy session refuses a prompt rather than queueing it; one connection runs one prompt at a time across its sessions, and a prompt on another session queues and says for which session it waits
- `session/list` (`cwd` optional, absolute) names the repository's resumable runs, newest first: agent6's run id as `sessionId`, the task as `title`, the last activity as `updatedAt`; one page, no cursor.
  A run `agent6 resume` would refuse (a fan-out coordinator, a lane, a run whose worktree is gone, a run with no checkpoint) is left out.
  Without `cwd` it covers every repository the connection opened a session in.
- `session/resume` opens a conversation on a recorded run by that id, so an editor reopens its thread after a restart.
  The next prompt resumes the run with its text as the steering instruction, as every follow-up prompt does; a plan, an ask, a run whose worker is alive, or a run `agent6 resume` would refuse is refused with the same words.
- `session/close` ends a conversation; a prompt in flight is cancelled first and answers `cancelled`.
- `session/cancel` cuts a model call in flight and ends a running foreground command at once, without waiting for or signalling the worker (the ACP server itself).
  The prompt answers `cancelled` (ACP's word) and the run's record reads `stopped`, as every operator stop does; what it had not yet committed stays in the tree for the next prompt, which resumes it.
  A prompt still queued answers `cancelled` at once.
  Cancellation releases an open permission wait without waiting for the client's reply.
  A late permission reply is ignored and cannot grant permission to a later prompt.

## Not implemented

- `session/load`, the history replay: `session/resume` reopens a recorded run without replaying it, and `initialize` reports `loadSession` absent.
- `session/delete`: a run's record is the operator's to remove (`agent6 sessions rm`).
- Mid-run steering: ACP has no message for a prompt while the previous prompt is running.
  A session's follow-up is the next prompt, which resumes the run with that text as its first steering instruction.
- `fs/*` and `terminal/*`: ACP lets the client own the filesystem and the terminal, and agent6 keeps both behind the jail the operator configured.
- Free-text answers to a waiting `ask_user` question: the editor gets option buttons only, so a waiting question without options is unanswered and not queued (see the [frontend cases](#approvals)); `/answer` takes any text.
- Embedded resources in a prompt: text and `resource_link` blocks are read (a link rides in as its uri; the workspace boundary still decides what it reaches)
    - images and embedded resources are dropped; `promptCapabilities.embeddedContext` says so

## Troubleshooting

- stdout is the protocol stream: nothing but JSON-RPC
- everything agent6 would print goes to stderr (the editor's agent logs)
- a wrapper echoing to stdout before exec'ing `agent6` breaks the connection irrecoverably; write to stderr

## As an MCP server

`agent6 mcp serve` speaks MCP over stdio, so another agent (an editor's own, or a second agent6 with `[mcp.servers]`) can use agent6's jail and run state.
It is the inverse of `[mcp]` in the config, which is agent6 as an MCP client.
The cwd's config decides everything: the sandbox the commands run in, and which tools exist at all.

Five tools, of which a default config publishes two:

| Tool | Withheld when |
| --- | --- |
| `query_dag` (a run's task graph) | never |
| `list_sessions` (this repository's sessions) | never |
| `run_verify` (the configured gate, jailed) | no `[harness] verify_command`, or `[sandbox] run_commands` is not `allow` |
| `run_in_sandbox` (an argv, jailed) | `[sandbox] run_commands` is not `allow` |
| `apply_patch_in_sandbox` (a patch, then the gate) | either of the two above |

`run_commands = "ask"` withholds the command tools: the MCP boundary has no operator to prompt.
A client that calls a withheld tool by name is told which setting withheld it.

```jsonc
// Zed: settings.json -- agent6's jail as another agent's tool surface
{
  "context_servers": {
    "agent6": { "command": { "path": "agent6", "args": ["mcp", "serve"] } }
  }
}
```

The tools reach the repository the server was started in, under that repository's sandbox policy; `list_sessions` and `query_dag` read its state dir and nothing else.
