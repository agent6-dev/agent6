# Retention counts from a journal

```sh
uv run python -m bench.compaction_offline /path/to/logs.jsonl --turn 15
```

Count the text in a recorded journal without opening its checkout, running a tool or calling a model.
Any profile or checkout can supply the journal.
The optional cut stops before the unique `budget.turn` with that number.
Without a cut the counter uses the whole file.

Each JSON line measures one keep setting with the local retention selector.
The baseline reserves all operator inputs and up to 80,000 characters of whole recent turns.
The other arms reserve no extra operator text, cap characters at 20,000, cap turns at one, keep only operator text, or replace baseline results with pointers.
`kept_chars` and `dropped_chars` partition `source_chars`.
`quote_chars` is the separately reserved part of the retained text.
`pointer_chars` measures replacement text, not retained original text.
Add it to `kept_chars` for the resulting size.

These are journal-preview characters, **not** reconstructed wire sizes or tokens.
Each `budget.turn` groups settled worker text, call arguments and result previews.
Task, steer and answer text stand for operator inputs.
A result contributes its summary and any journalled read or command tails, even if those repeat a detail.
The journal need not contain the original results, reasoning, media, notices or cache state.
The counter does not infer those, replay earlier compactions or predict correctness, re-reads or spend.
