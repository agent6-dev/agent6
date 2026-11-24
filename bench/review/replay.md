# Recorded review replay

Measure what a reviewer says on its **first reply** under the recorded system prompt and the prompt in this checkout.
The replay does not run a panel or change its blocking rule.

## Run

From the checkout to measure, set `STATE` to a repository state directory containing `sessions/reviewers/`.
Use finished reviews from one route and effort per run.
To select a subset, copy the chosen reviewer directories, including `objects/`, into the same layout in a fresh state directory.
Copy the recording files themselves: replay refuses symlinks on recording paths below `STATE` before making any calls.

The call records include private finding text.
Keep the agent6 state base (`${XDG_STATE_HOME:-$HOME/.local/state}/agent6`) outside the worktree for the output.

```sh
(
  set -e
  umask 077
  base="${XDG_STATE_HOME:-$HOME/.local/state}/agent6"
  mkdir -p "$base"
  out=$(mktemp -d "$base/review-replay.XXXXXXXX")
  printf 'Replay output: %s\n' "$out"
  uv run python -m bench.review.replay "$STATE" \
    --model chatgpt/gpt-6-astra --effort high --repeats 3 \
    > "$out/calls.jsonl" 2> "$out/comparison.txt"
)
```

Use the recorded provider/model and effort in place of the example.
Use `--effort default` for an unset effort.
The command uses the checkout's effective agent6 config and budget; `--config FILE` adds an overlay.
It sends the recorded task, rules and diff to the selected provider.
Set reasoning effort with `--effort` only.
The replay refuses `extra_body.reasoning`, `reasoning_effort`, `thinking` and `output_config` overrides before spending because they can replace the adapter's effort controls.
Keep other provider settings the same as in the recording.

## Calls

- **Source:** A completion bound to a panel identifies a reviewer child; judges and children without completions do not enter the sample.
  The replay reads the first `reviewer:<focus>` transcript by sequence and resolves its object references.
  A bare `reviewer` actor does not identify the focus needed for the tree prompt and is refused.
  It does not reconstruct the opening from the current repository or the final verdict.
- **Arms:** `recorded` sends the system text from the request.
  `tree` calls `agent6.prompts.review.panel_prompt` with the recorded focus, reply limit and whether the reviewer had `run_command`.
  Both arms use the same recorded user messages, tool definitions, temperature and output cap where the wire supplies them.
- **Order:** Each repetition makes one independent call per arm.
  The recorded arm goes first on odd repetitions and second on even ones.
  The model may ask for a tool, but nothing answers that request.
  An accompanying text verdict does not count.
- **Coverage:** The reader accepts agent6 text openings on Chat Completions, Anthropic Messages and Responses (including ChatGPT).
  It reads the reply limit from the first reply notice or the limit in a single-message opening's system prompt.
  An unreadable selected opening stops the run before any model call.
  Claude Code process transcripts do not record the offered tool schemas.
  The replay does not support them, media openings or arbitrary provider extensions.
  It does not replay transport headers, cache keys or streaming settings.

## Read the results

Stdout has one JSON line per logical call: child session, source transcript, repetition, arm, model, requested effort, system text hash and outcome.
`requested_effort` records the argument, not an observation of the model's reasoning or the adapter's wire representation.
A complete run has `2 × reviews × repeats` lines.
On a provider or budget error, a `call failed` line identifies the interrupted pair and the command exits nonzero.
The input state stays unchanged.

The first table on stderr counts outcomes, block findings by category and blocks naming a reader for each review and arm.
`pass` and `block` come from the panel's reply parser **before** diff grounding or voting.
`asked for a tool`, `invalid reply` and `truncated` have no verdict.

- **Reader:** `names_reader` is a text measure on each block's title and detail.
  It counts a `reader:`, `attacker:` or `consumer:` label with a value, a qualified role such as “unauthenticated client”, or a role such as “operator”.
  The labels `unknown`, `unspecified` and `none` do not name a reader, even in backticks.
  Each JSON block includes the matching `reader_evidence`, or `null`, and the finding text for inspection.
  A match does not establish that the reader gained access; the rule can miss paraphrases or count an incidental mention.
- **Changed verdict:** Compare the two replies in the same repetition only if both have verdicts.
  The second table names each review and counts its comparable pairs and verdict changes.
  The summary counts reviews with at least one change among reviews with at least one comparable pair.
- **Kept and lost blocks:** Match recorded-arm block occurrences to tree-arm blocks by category and exact location (path, side and inclusive lines), once per occurrence per comparable pair.
  Count matched occurrences as kept and unmatched recorded occurrences as lost.
  A tool request or invalid reply cannot establish that a block was lost.
  A changed citation can count as a loss, and distinct defects at the same citation can match.

These are sample counts, not judgments of correctness or evidence that the prompt caused a change.
Read the arm distributions and the number of comparable pairs alongside the totals.
