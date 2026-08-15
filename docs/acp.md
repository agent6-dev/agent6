# Editor integration (ACP)

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

Every run writes one event journal, and the CLI, TUI, and web UI render it through the same fold.
ACP is a fourth projection of that fold, so an editor sees what `agent6 attach` shows: reasoning, each tool call and its outcome, auto-commits, and how the run ended.

A tool call arrives twice, as ACP models it: `pending`, then `completed` or `failed`.
Both land when the call finishes, because the fold emits an item only once the result is in, so a long verify stays invisible while it runs.
The pair marks the call's lifecycle and carries no progress.

## Approvals

`session/request_permission` carries every approval the CLI would prompt for: `run_commands = "ask"`, a `fetch` to a host outside the allow-list, an unsandboxed autorun.
The editor renders the buttons.

Two rules hold whoever is driving:

- An unanswered request denies: after five minutes with no reply the approval is refused and the run continues without it.
- An off-list `fetch` host is offered as `allow_once` only, so an editor's "always allow" cannot cover a different host later.

## Sessions

One session is one directory (`session/new` carries an absolute `cwd`), and its config is that directory's own layered config: global, then repo, then any preset.
The directory has to be a git repository, since it becomes what the jail mounts writable and a run needs git to branch and commit each step.

A session runs one turn at a time.
Prompting a busy session is refused rather than queued, so the editor can offer the prompt again.

A session is one conversation.
The first prompt starts an `agent6 run`, and every later prompt resumes that run with the new text as its first steering instruction (the `agent6 resume --steer` semantics), so the model keeps its history, its run branch, and its per-turn budget circuit-breaker.
A prompt whose prior turn died before the first snapshot starts fresh instead.

Runs are serialised across the connection: a second prompt waits for the first to reach a boundary.
`session/cancel` drops the same stop marker `agent6 sessions stop` does, so the step in flight finishes and commits before the run ends.

## Not implemented

- `session/load`: ACP v2 reorganises it, and resume carries agent6's own semantics (`agent6 resume`, `agent6 fork`), so `initialize` reports the capability as absent.
- Mid-run steering: a steer arrives through agent6's pause menu, which needs a terminal, so an ACP session's follow-up is the next prompt.
- `fs/*` and `terminal/*`: ACP lets the client own the filesystem and the terminal, and agent6 keeps both behind the jail the operator configured.
- Embedded resources in a prompt: a resource block's uri is client-controlled, so only text blocks are read and `promptCapabilities.embeddedContext` says so.

## Troubleshooting

stdout is the protocol stream and carries nothing but JSON-RPC.
Everything agent6 would have printed goes to stderr, where the editor shows it as agent logs.
A wrapper script that echoes to stdout before exec'ing `agent6` breaks the connection irrecoverably; write to stderr instead.
