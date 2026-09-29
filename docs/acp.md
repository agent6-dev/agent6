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
- built-in calls carry their ACP kind, and an edit result carries each journaled path as an absolute follow-along location
- a call waiting on an approval or an `ask_user` answer is updated to `pending` while its prompt is open, and back to `in_progress` once answered
- a long verify shows as in progress while it runs; a call the run never returned from settles as `failed` when the run's `session.end` is written, or when the tail ends without one (a worker killed mid-call)
- `toolCallId` is `<run id>:<turn>:<call>`, unique for the life of the session; a turn's call numbers start at 1

Worker text and thinking arrive as they stream; side-role output stays out of the conversation.
What the CLI would print around the run (where the changes are, a stash notice, a refusal's reason) arrives as an `[agent6]` agent message; the cost receipt goes to stderr.

## Approvals

`session/request_permission` carries every approval the CLI would prompt for.

- a command under `run_commands = "ask"`
- an MCP tool call the server's `approve` does not cover
- a `fetch` to a host outside the allow-list
- an unsandboxed autorun

The editor renders the buttons; the request names the tool call it gates and carries the prompt as that call's title.
A prompt that gates no call (a pre-run question) announces a tool call of its own and closes it with the answer.
The prompt and its answer go through the same gate every front-end answers through, so `agent6 attach` and the web show the run as waiting; the answer journals `source: "acp"`, or `"headless"` when the client declared it cannot be asked.

- an unanswered request denies after five minutes, and the run continues without it
- an off-list `fetch` host is offered as `allow_once` only, so an editor's "always allow" cannot cover a different host later
- a standing "allow all" an earlier front-end recorded on the run answers that scope's later prompts without asking the editor (`source: "session"`)

## Sessions

A session is one conversation in one directory.

- `session/new` carries an absolute `cwd`, a git repository, whose own layered config applies
- the first prompt starts an `agent6 run`; every later prompt resumes it with the text as its steering instruction (a turn that died before its first checkpoint starts a new run, and the editor is told)
- a busy session refuses a prompt rather than queueing it; one connection runs one prompt at a time across its sessions, and a prompt on another session waits its turn and says for which session
- `session/cancel` is `agent6 stop --after-step`: the step in flight finishes and commits first (a prompt still waiting its turn answers `cancelled` at once)

## Not implemented

- `session/load`: ACP v2 reorganises it, and resume carries agent6's own semantics (`agent6 resume`, `agent6 fork`), so `initialize` reports the capability as absent.
- Mid-run steering: ACP has no message for a prompt while a turn is running.
  A session's follow-up is the next prompt, which resumes the run with that text as its first steering instruction.
- `fs/*` and `terminal/*`: ACP lets the client own the filesystem and the terminal, and agent6 keeps both behind the jail the operator configured.
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
| `run_verify` (the configured gate, jailed) | no `[harness] verify_command`, or `[sandbox] run_commands` is not `yes` |
| `run_in_sandbox` (an argv, jailed) | `[sandbox] run_commands` is not `yes` |
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
