# Anthropic Messages

`providers.<name>.server_compaction` selects who manages the message history on a supported model.
The default is `true` for `deployment = "direct"` without an explicit `base_url`.
Every other entry on this backend defaults to `false`.
`agent6 config show` prints the effective value and its origin.

The models with signed compaction are Claude Fable 5.1 and 5.2, Mythos 5.1 and 5.2, Opus 5.5, Sonnet 5.5 and Haiku 5.5.
An unsupported model or `server_compaction = false` uses [agent6's own compaction](architecture.md#compaction).

**Clearing.** Above `compaction.drop_at_chars`, agent6 asks the endpoint to clear old tool results.
It keeps sending the full history, and each request names how many of the newest tool uses stay whole.
The oldest results are cleared until the rest fit in half of `compaction.drop_at_chars`.
The same results are cleared on every later request, until the rest pass the threshold again.
The newest two results, every result the model has yet to read, and operator answers from `ask_user` are never cleared.
A step causes one cache miss from the first newly cleared result onward.
With `prompt_caching = true`, the cache marks move through the messages as they do under agent6's own compaction: the API does not count a moved mark as a change to the history.

**Summary.** At `compaction.summarize_at_chars`, on `/compact [focus]` or during an eligible wait, agent6 asks the endpoint to summarize the request it already received.
The summary request has the same messages, system text and tools.
It has no clearing edit, so the summary writer reads the uncleared history.
Later requests start with the returned block, including its signature.
The worker's reply and any tool result follow the block without a rewrite.
A notice then repeats the task and pins.
If the endpoint returns no block, the journal records the reason and the helper gets one attempt at a local restart.

**Thinking.** On these models with thinking enabled, direct and Vertex requests ask the API to drop a thinking block whose prefix fails verification instead of refusing the request.
This also applies when agent6 compacts by itself.
An operator's own `extra_body.thinking.block_binding` is sent as written.
`agent6 sessions receipt ID` reports the most thinking blocks the endpoint dropped and the most tool results it cleared on one request.
The endpoint repeats both on every request, since each request carries the full history.

**Another route.** A session resumed on a model or backend without signed compaction carries the summary as a text notice in place of the block.
