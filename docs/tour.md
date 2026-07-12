# Tour

A walk through the terminal UI. Every image is from a recorded run; click to enlarge.

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/01-hub.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

## Hub

`agent6 tui` lists every run for the repository with its mode, status, last activity, and
cost. Enter opens a run; `c` opens the config page; `?` lists the keys.

![The hub](screenshots/out/01-hub.png)

## Conversation

Opening a run lands on its conversation (also `agent6 runs transcript`): the task, the
model's reasoning, and every tool call with its complete input and output, following
live. A live run keeps a steer bar at the bottom.

![A run transcript](screenshots/out/05-transcript.png)

## Run dashboard

`Ctrl+D` toggles the dashboard: the task graph beside the model's live reasoning, then
the tool calls with their results, and the event log and latest commit diff side by
side. The composer bar runs along the foot (type to steer, or to resume a finished
run with a follow-up), and the View menu maximizes the focused pane to full screen.

![The run dashboard](screenshots/out/02-run-dashboard.png)

## Event log

The View menu's Full log opens the JSONL event stream the dashboard is built from,
scrollable over the whole run.

![The event log](screenshots/out/09-logs.png)

## Configuration

The config page shows every setting, its effective value, and where that value came from
(a built-in default, the global config, or the per-repo config). `/` filters by name.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

## Watching a state machine

Beyond one-shot runs, agent6 runs editable state machines: a `.asm.toml` of tool, branch,
agent, and wait states driven over a journal. Author one, read its graph, and watch it
execute. Here `code-fixer` runs a fix-loop: an agent state edits the repo to make a
failing check pass, a tool state re-runs the check, and the machine routes on the
result until it is green or the attempt budget is spent, with the agent's reasoning
streamed live like a run.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/machine-demo.webm" type="video/webm">
</video>

## From the terminal

For terminal-first workflows, `agent6 run` executes in the foreground: steer it with
Ctrl-C, no TUI required. Runs started from the TUI or web hub are detached instead, and
`agent6 attach` attaches to either kind: a plain no-deps line tail by default, `--tui` for
the full-screen TUI, `--json` for a one-shot snapshot of the same state.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/cli-demo.webm" type="video/webm">
</video>

---

These are regenerated from recorded runs by the
[pages workflow](https://github.com/agent6-dev/agent6/blob/master/.github/workflows/pages.yml),
so they track the current UI.
