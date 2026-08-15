# Editor integration (ACP)

`agent6 acp` runs agent6 as an [Agent Client
Protocol](https://agentclientprotocol.com/) agent: an editor spawns it, sends
prompts, and renders the run as it happens. Same engine as `agent6 run`, same
config, same jail.

```jsonc
// Zed: settings.json
{
  "agent_servers": {
    "agent6": { "command": "agent6", "args": ["acp"] }
  }
}
```

Any ACP client works the same way; the command is the whole configuration.

## What the editor sees

Every run already writes one event journal, and the CLI, the TUI and the web UI
all render it through the same fold. ACP is a fourth projection of that fold, so
an editor sees exactly what `agent6 attach` would show: reasoning, each tool
call and its outcome, auto-commits, and how the run ended.

A tool call arrives twice, as ACP models it: `pending`, then `completed` or
`failed`. Both land when the call finishes -- the shared fold does not emit
an item until the result is in, so a long verify is not yet visible while it
runs. The pair is the shape an editor keys its lifecycle on, not a progress
signal.

## Approvals

`session/request_permission` carries every approval the CLI would prompt for:
`run_commands = "ask"`, a `fetch` to a host outside the allow-list, an
unsandboxed autorun. The editor renders the buttons.

Two things do not change because an editor is driving:

- **An unanswered request is a no.** After five minutes with no reply the
  approval is denied and the run continues without it. An operator who walked
  away does not silently grant anything.
- **"Allow once" means once.** An off-list `fetch` host is offered as
  `allow_once` precisely so an editor's "always allow" cannot cover a
  *different* host later.

## Sessions

One session is one directory (`session/new` carries an absolute `cwd`), and its
config is that directory's own layered config -- global, then repo, then any
preset. That directory has to be a git repository: it becomes what the jail
mounts writable, and a run needs git to branch and commit each step.

A session runs one turn at a time. Prompting one that is busy is refused
rather than queued, so the editor can offer the prompt again; mid-run
steering is `agent6`'s pause menu, which an editor has no terminal for.

A session is one conversation: the first prompt starts an `agent6 run`, and
every later prompt resumes that same run with the new text as its first
steering instruction (the `agent6 resume --steer` semantics), so the model
keeps its history, its run branch, and its budget circuit-breaker per turn. A
prompt whose prior turn died before the first snapshot starts fresh instead.

Runs are serialised across the connection: a second prompt waits for the first
to reach a boundary. `session/cancel` drops the same stop marker `agent6 sessions stop`
does -- a marker, not a kill, so the step in flight finishes and commits before
the run ends.

## What is deliberately not implemented

- **`session/load`.** ACP v2 reorganises it, and resume is where agent6 has the
  most of its own semantics (`agent6 resume`, `agent6 fork`). `initialize`
  reports the capability as absent rather than half-answering it.
- **Mid-run steering.** A steer arrives through agent6's own pause menu,
  which needs a terminal. An ACP session's follow-up is the next prompt.
- **`fs/*` and `terminal/*`.** ACP lets the CLIENT own the filesystem and the
  terminal. agent6 inverts that on purpose: the agent owns a jail the operator
  configured, precisely so an editor cannot be talked into doing the model's
  filesystem work. The tools stay jailed.
- **Embedded resources in a prompt.** A resource block's uri is
  client-controlled, so passing one through would be path injection. Only text
  blocks are read, and `promptCapabilities.embeddedContext` says so.

## Troubleshooting

stdout is the protocol stream and carries nothing but JSON-RPC; everything
agent6 would have printed goes to stderr, where the editor shows it as agent
logs. A wrapper script that echoes to stdout before exec'ing `agent6` breaks
the connection irrecoverably -- write to stderr instead.
