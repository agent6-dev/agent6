# Terminal UI

The full-screen TUI screen by screen, then the plain CLI.
Every image is from a recorded run; click to enlarge.

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/01-hub.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

## Hub

`agent6 tui` lists every session for the repository in five columns: updated, status (the mode folded in), cost, id, and task.
Below 100 columns `updated` keeps only the time for today's runs and only the date for older ones, and a status drops its reason; below 80, cost hides, so the task keeps its room.

- Enter opens a run
- `n`: an empty conversation to start a run, plan, or ask (mode, preset and model picked above the composer; the model box shows the model the config resolves for the mode and preset, re-resolved whenever either changes, and a pick overrides it for this run; a refusal shows there with the text kept)
- `l`: the selected run's event log; `m`: merge its branch (`sessions merge`); `d`: delete its history (`sessions rm`), both after a confirm
- Space: expand or fold a fan-out's lanes (one row with its lane count otherwise)
- `c`: the config page; `M`: the machines screen; `r`: refresh; `?`: the keys; `q`: quit

![The hub](screenshots/out/01-hub.png)

## Conversation

Opening a run lands on its conversation, the text `agent6 sessions transcript` prints: the task, the model's reasoning, and every tool call with its input and output, following live.
Tool input and output are clipped to the salient lines by default; Detail cycles hidden, collapsed, expanded.
A live run keeps a steer bar at the bottom.
Above it, a live pane streams a model call while it is in flight, and lists the tool calls in flight (`→ run_command  sleep 60  · running`) until their results land in the transcript.
An approval shows inline at the conversation's tail (the command, fixed-width), with an answer row docked above the bar.
The keys are the CLI prompt's, on every surface: `y` allow, `a` allow all this session, `n` deny, `d` deny all this session.
Once answered, it collapses to one dim line.
The dashboard docks the same row, carrying the command (it has no transcript for it); a machine screen, which has no composer, opens a dialog instead.
No approval takes the focus or a key: the composer keeps both, so a message typed as one arrives is a message.
Tab (or a click) moves the focus off the composer, and the letters answer wherever it lands: the transcript, a pane, or the row itself, where every answer is its own tab stop and Enter answers the focused one.
Answering from the row leaves the focus there, so the next approval answers straight away.
A modal's buttons sit in one row; on a terminal narrower than the row the later buttons are off screen, and the approval keys, the question modal's answer fields and Ctrl+S still answer.

- `Ctrl+T` cycles the thinking and tool detail: hidden, collapsed, expanded
- `Ctrl+C` copies the selection, or the whole transcript when nothing is selected
- `Ctrl+Z` leaves the view: a run `agent6 run --tui` fronts detaches to the background after its current step (`agent6 attach` reattaches); one opened with `attach --tui` or from the hub keeps running as it was
- `Ctrl+_` undoes typing in the composer (`Ctrl+Z` is the detach key everywhere in the view)

![A run transcript](screenshots/out/05-transcript.png)

## Run dashboard

`Ctrl+D` toggles the dashboard: task graph beside live reasoning, tool calls with results, event log and latest commit diff side by side.
Before the first model call, the header names the role and model from the manifest and says `starting`; the spinner runs only while a model call is in flight.

- the diff pane opens on the latest commit; its selector walks the run's per-step commits (newest first), `cumulative` shows the chain up to that step; the task tree and the cost line follow the selected step; a run whose model owns git has no chain and the pane says so
- the composer bar runs along the foot: type to steer, or to resume a finished run; the row above it picks the preset and model the next leg continues under (each first entry names what a resume without flags runs under)
- `/` completes the steer directives; Ctrl-R searches the session's past messages
- `/shells` lists the run's background commands and how they ended; `/restate` replays the conversation since your last message
- `/task <text>` adds work to the run's task graph instead of steering it: the turn in flight never sees it, and the run works it once its open tasks drain (`agent6 steer ID "/task <text>"` does the same from a script or another machine)
- the View menu maximizes the focused pane
- on a terminal under 28 rows the dashboard shows one pane row at a time: the row holding focus, else the log and diff, with a summary line of the tool calls; Tab reaches the folded panes and unfolds them
- on a finished plan, Run > Run this plan starts `agent6 run --from` on it and opens the new run; the plan session stays as it was

![The run dashboard](screenshots/out/02-run-dashboard.png)

## Event log

The View menu's Full log opens the structural events of the run's log, formatted as the dashboard's log pane formats them, scrollable over the whole run.

![The event log](screenshots/out/09-logs.png)

## Configuration

The config page shows every setting, its effective value, and the layer that set it: `default`, `preset`, `global`, `repo`, or `flag` (a `--config FILE`).
`/` filters by name.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

## Without the TUI

`agent6 run` executes in the foreground; Ctrl-C opens its pause menu to steer it.

- the pause menu Tab-completes its commands; Up recalls, Ctrl-R searches past messages
    - `/status`, `/tasks`, `/pin`, `/compact`, `/parallel`, `/btw`, `/task`, `/shells`, `/restate`, `/undo`, `/continue`, `/stop`, `/exit`, `/detach`, `/help`
- a steer sent from another surface (`agent6 steer`, the web or TUI composer) while the menu is open is taken as the answer
- `/detach` in the menu hands the run to the background after its current step (`agent6 attach` reattaches); Ctrl-Z prints the run's state and stands an armed pause down, it never suspends the run (a suspended agent would lose its live provider stream)
- `/exit` in the menu, the fallback prompt, or the `run -i` REPL stops the run and leaves without the follow-up prompt (`agent6 resume` continues it)
- `/stop` in the menu, or Ctrl-C at the pause prompt itself, stops the run now (a third Ctrl-C without the prompt does the same); `agent6 stop ID` from another terminal is the same stop
- `run -i` prompts after every commit: `/continue` (bare Enter), `/cost`, `/diff`, `/watch`, `/mcp`, `/init`, `/undo`, `/help`, `/quit`, `/exit`
- closing a viewer opened with `agent6 attach --tui` (Ctrl-Z) leaves the run as it was
- TUI/web-hub runs start detached; `agent6 attach` covers both kinds: conversation by default, `--tui` full screen, `--json` one-shot snapshot, and `--raw` line tail for runs

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/temps-demo.webm" type="video/webm">
</video>

## Watching a state machine

An [agent state machine](state-machines.md) runs in the terminal like anything else: author the file, read its graph, watch it execute.
Here `code-fixer` runs a fix-loop.
An agent state edits the repo to make a failing check pass.
A tool state re-runs the check.
The machine routes on the result until the check is green or the attempt budget is spent.
The agent's reasoning streams live, as in a run.

- the machines screen (`M` on the hub): `v` (or Enter) opens the parsed file, `R` runs it, `w` watches its instance, `c` creates a draft, `r` refreshes
- the watch screen (also `agent6 attach --tui <id>`): `s` steers the current agent state, `m` messages a waiting instance (`machine poke`), `x` stops it at the next transition

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/machine-demo.webm" type="video/webm">
</video>

---

The screenshots and videos are regenerated from recorded runs by the [pages workflow](https://github.com/agent6-dev/agent6/blob/master/.github/workflows/pages.yml), so they track the current UI.
