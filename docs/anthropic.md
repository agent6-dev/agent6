# Anthropic Messages

`providers.<name>.server_compaction` selects who manages the message history on a supported model.
The default is `true` for `deployment = "direct"` without an explicit `base_url`.
Every other entry on this backend defaults to `false`.
`agent6 config show` prints the effective value and its origin.

The models with signed compaction are Claude Fable 5.1 and 5.2, Mythos 5.1 and 5.2, Opus 5.5, Sonnet 5.5 and Haiku 5.5.
An unsupported model or `server_compaction = false` uses [agent6's own compaction](architecture.md#a-run).

**Clearing.** Above `compaction.drop_at_chars`, agent6 asks the endpoint to clear old tool results.
It keeps sending the full history.
The edit protects operator answers from `ask_user` and keeps the newest tool use.
Clearing can cause a cache miss from the first cleared result onward.
With `prompt_caching = true`, the system prompt and last tool stay cache marked.
The harness does not move cache marks through a signed history.

**Summary.** At `compaction.summarise_at_chars`, on `/compact [focus]` or during an eligible wait, agent6 asks the endpoint to summarise the request it already received.
The summary request has the same messages, system text and tools.
It has no clearing edit, so the summary writer reads the uncleared history.
Later requests start with the returned block, including its signature.
The worker's reply and any tool result follow the block without a rewrite.
A notice then repeats the task and pins.
If the endpoint returns no block, the journal records the reason and the helper gets one attempt at a local restart.

**Thinking.** On these models with thinking enabled, direct and Vertex requests ask the API to drop a thinking block whose prefix fails verification instead of refusing the request.
This also applies when agent6 compacts by itself.
`agent6 sessions receipt ID` reports how many thinking blocks the endpoint dropped and how many tool results it cleared.
