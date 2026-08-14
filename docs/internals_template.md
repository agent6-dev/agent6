# Internals

Generated maps of the code as it is today: each diagram is built from the
current source at site-build time (`docs/gen_diagrams.py`), so it cannot
drift. [architecture.md](architecture.md) explains the design these shapes
serve.

## The module layering

Top-level packages, collapsed from the `tach` module graph. The engine stack
is `ui -> app -> workflows -> tools -> sandbox` (an edge here is "imports
from"); the shared substrate holds the leaf packages any layer may use.

<!-- diagram: layering -->

## The agent loop's turn pipeline

The drive tier of `workflows/loop.py`: `run`/`resume` enter `_drive_loop`,
which runs the phases once per model turn — snapshot, compact when needed,
call the model, dispatch its tool calls, auto-commit, then the gates,
notices, and stop checks decide continue-or-end. Extracted from the direct
calls between the phase methods; deeper tiers regenerate from source with
the same extractor.

<!-- diagram: turn-pipeline -->
