# Terminal UI

`agent6 tui` opens the full-screen interface on this repository; `agent6 tui <id>` opens one session.
This page shows it screen by screen; the plain CLI is on [Command line](cli.md).
Every image is from a recorded run; click to enlarge.

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/01-hub.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

## Hub

The new-work composer (`n`) picks the mode, the checkout (this one or a new worktree), the preset and the model; a second row picks the plan to run from and a spend cap in USD.
A plan's **Run this plan** opens the same composer with the plan chosen.
On a terminal 140 columns wide or more the table adds a run's branch; `R` opens the composer with `/review <branch>` for the selected run, and `m` merges it after a confirm that names the unmerged runs that changed some of the same paths.
The composer accepts `/parallel [SPEC --] TASK` ([grammar and examples](usage.md#parallel-directive)).
A session that starts waiting on an answer or ends is announced with a toast and a desktop notification (`notify-send`, when present).

![The hub](screenshots/out/01-hub.png)

## Files

The Files screen opens from the hub with `F` and shows the working tree's files.
The tree sits beside the file; `t` folds it away for a full-width preview or restores it.
A Markdown file opens rendered with its outline hidden; `o` shows or hides the outline.
`s` switches between rendered Markdown and source.
The tree and outline choices last until Files closes.
The footer and `?` help list these keys.

## Conversation

Opening a run lands on its conversation: the task, the model's reasoning and every tool call, following live.
`Esc` or `Ctrl+K` returns to the hub, the TUI's session switcher.
The composer takes the [steer directives](usage.md#follow-and-steer-a-run).
On an ended child, and on an ended run whose worktree is gone or whose manifest cannot be read, the composer says why, as `agent6 sessions show` does; Enter resumes nothing and **Run > Resume this session** is dimmed.
**Run > Fork this session** continues a child in a worktree of its own.
When the manifest cannot be read, **Run > Fork this session** is dimmed too, as is the hub's **Merge selected run** for it.

- `Ctrl+C` copies the selection, or the whole transcript when nothing is selected
- `Ctrl+Z` leaves the view: a run started with `agent6 run --tui` detaches after its current turn (`agent6 attach` reattaches); one opened with `attach --tui` or from the hub keeps running as it was

- on a terminal narrower than a modal's button row the later buttons are off screen; the question modal's fields and Ctrl+S still answer

![A run transcript](screenshots/out/05-transcript.png)

## Run dashboard

`Ctrl+D` toggles the dashboard: task graph beside live reasoning, tool calls with results, event log and latest commit diff side by side.
The children table refreshes every second and keeps the same child selected when rows move.
Each row shows the child's number, id, task, model route, status, current tool, turn, tokens, spend and elapsed seconds.
The result column names the parent chain a child joined, or the ref and commit count for stopped or retained work.
It also shows why a launch failed or how a reviewer ranked the attempts.

- `Tab` to the table and use `↑` or `↓` to select a child.
- Use `←` or `→` to read columns beyond the edge of the screen.
- `Enter` opens the selected child's session if it has one.
- `m` opens a message for the selected live child.
  Press `Enter` to queue the text for its next turn.
- `x` asks for confirmation before requesting an immediate stop for the selected live child.

Both prompts name the child.
Neither sends a request if the child ends or its worker changes while the prompt is open.

![The run dashboard](screenshots/out/02-run-dashboard.png)

## Findings

Open Findings from the hub's View menu (`v`) or the run dashboard's View menu (`Ctrl+F`).

![The findings screen](screenshots/out/06-findings.png)

## Receipts

Open **Receipt** (`e`) from the Findings screen to inspect [recorded evidence](usage.md#session-receipts).

![A receipt](screenshots/out/07-receipt.png)

## Code-review rule sources

See [Code-review rules](usage.md#code-review-rules) for discovery and limits.

## Event log

![The event log](screenshots/out/09-logs.png)

## Configuration

The config page shows every setting, its effective value, and the layer that set it: `default`, `preset`, `global`, `repo`, or `flag` (a `--config FILE`).
Editing a role's model opens one `provider/model` field with live suggestions; unlisted routes are accepted.

In a named-entry form, a blank field keeps an existing value or uses the schema default.

**Remove entry** (`x`) confirms the selected entry's name and config layer before removing the whole entry, including its subtables; on the reviewer roster it picks a reviewer by position.
It removes from the highest layer holding the entry, and accepts only global or repo config.
Values from lower layers may become visible.
The edit popup of an entry's row offers the same Add and Remove for its table.

**Preset…** (`p`) opens the `agent6.preset` setting: the rows show the config under the selected preset, and a save, an unset or an entry change writes `[presets.<name>]` in the chosen layer; **Add preset…** (`P`) creates a table and selects it; **Remove preset** removes the selected table and resets `preset` ([Presets](config.md#presets)).
With `none` picked, edits go to the global and repo config; entries are edited only then.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

## Watching a state machine

An [agent state machine](state-machines.md) runs in the terminal like anything else: author the file, read its graph, watch it execute.
Here `code-fixer` runs a fix loop: an agent state edits the repo to make a failing check pass, a tool state re-runs it, and the machine routes on the result until the check is green or the attempt budget is spent.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/machine-demo.webm" type="video/webm">
</video>
