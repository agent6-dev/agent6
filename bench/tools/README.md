# Tool measurements

Each comparison uses the same committed tree, route (`chatgpt/gpt-6.2-sol`), effort (`medium`), tasks and caps on both sides.
`control` offers the shipped tools and text.

| Comparison with `control` | Development | Evaluation | Question |
|---|---|---|---|
| `no-patch` | `textkit` | `bugs`, `eventflow` | Does offering `apply_patch` change failed edit calls, turns, tokens or passes? |
| `no-docs` | `machine-test` | `plan-cap` | Does offering `agent6_docs` in `ask` change correct answers, turns or tokens? |
| `short` | `textkit` | `bugs`, `eventflow` | Does shortening the definitions of `apply_edit`, `ask_user` and `parallel` together change the editing results? |

**Fixed split.** Use development for the single-run smoke and any tuning.
Freeze the patches before opening evaluation journals.
Run three replicates per evaluation task and arm, sequentially on an otherwise idle account.
Use the same control samples for both editing comparisons.

**Editing tasks.** [Coreagent](../coreagent/README.md) supplies the prompts, starting trees and hidden graders.
`textkit` exercises implementation from a stub.
`bugs` needs several edits in one file; `eventflow` needs fixes across modules.
A pass requires every case in the original `grade.py`, not a green starter suite or a model's claim.

**Ask tasks.** The [machine bench](../machine/README.md) tests generated bundles offline.
Ask for the command that runs a bundle's mock tests (`agent6 machine test`).
The [coreagent bench](../coreagent/README.md) distinguishes a dollar budget from a ChatGPT plan cap.
Ask for the per-run flag that limits plan usage (`--max-plan-points`).
Ask in an empty repository and request only the command or flag.
Match the answer after removing surrounding whitespace and inline code ticks.

**Reporting.** Report the split, task names and observed n per arm verbatim.
Include failed and capped runs, not just successful ones.
Show the spread across replicates before attributing a difference to a tool.
Record every comparison, including no observed difference, with the mechanism tested.
These tasks do not separate the effects of the three shorter descriptions.
