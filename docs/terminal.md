# Terminal UI

`agent6 tui` opens the full-screen interface on this repository; `agent6 tui <id>` opens one session.
This page shows it screen by screen, then the plain CLI.
Every image is from a recorded run; click to enlarge.

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/01-hub.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

## Hub

The new-work composer accepts `/parallel [SPEC --] TASK` ([grammar and examples](usage.md#parallel-directive)).

![The hub](screenshots/out/01-hub.png)

## Files

The Files screen opens from the hub and shows the working tree's files.
A Markdown file opens rendered, and `s` shows its source.

## Conversation

Opening a run lands on its conversation: the task, the model's reasoning and every tool call, following live.
The composer takes the [steer directives](usage.md#follow-and-steer-a-run).

- `Ctrl+C` copies the selection, or the whole transcript when nothing is selected
- `Ctrl+Z` leaves the view: a run started with `agent6 run --tui` detaches after its current turn (`agent6 attach` reattaches); one opened with `attach --tui` or from the hub keeps running as it was

- on a terminal narrower than a modal's button row the later buttons are off screen; the question modal's fields and Ctrl+S still answer

![A run transcript](screenshots/out/05-transcript.png)

## Run dashboard

`Ctrl+D` toggles the dashboard: task graph beside live reasoning, tool calls with results, event log and latest commit diff side by side.

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

**Remove entry** (`x`) confirms the selected entry's name and config layer before removing the whole entry, including its subtables.
It removes from the highest layer holding the entry, and accepts only global or repo config.
Values from lower layers may become visible.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

## Without the TUI

`agent6 run` executes in the foreground; Ctrl-C opens its pause menu to steer it.

- the menu Tab-completes its commands; Up recalls, Ctrl-R searches past messages; a steer sent from another surface while it is open is taken as the answer
    - the [steer directives](usage.md#follow-and-steer-a-run) plus `/status`, `/tasks`, `/answer`, `/shells`, `/restate`, `/continue`, `/stop`, `/exit`, `/detach`, `/help`
- `/detach` hands the run to the background after its current turn (`agent6 attach` reattaches); Ctrl-Z prints the run's state and never suspends it
- `/exit` leaves the run resumable; `/stop`, or Ctrl-C at the pause prompt, stops it now, as `agent6 stop ID` does
- `run -i` prompts after every commit: `/continue` (bare Enter), `/cost`, `/diff`, `/watch`, `/mcp`, `/init`, `/undo`, `/help`, `/quit`, `/exit`
- hub-started runs run detached; `agent6 attach` follows them (conversation by default; `--tui`, `--raw`)

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/temps-demo.webm" type="video/webm">
</video>

## Watching a state machine

An [agent state machine](state-machines.md) runs in the terminal like anything else: author the file, read its graph, watch it execute.
Here `code-fixer` runs a fix loop: an agent state edits the repo to make a failing check pass, a tool state re-runs it, and the machine routes on the result until the check is green or the attempt budget is spent.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/machine-demo.webm" type="video/webm">
</video>
