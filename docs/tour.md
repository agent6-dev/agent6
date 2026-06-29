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

## Run dashboard

Opening a run shows the task graph beside the model's live reasoning, then the tool calls
with their results, and the event log and latest commit diff side by side. A budget bar
runs along the foot, and `f` maximizes the focused pane to full screen.

![The run dashboard](screenshots/out/02-run-dashboard.png)

## Transcript

`t` (or `agent6 runs transcript`) renders the full conversation: the task, the model's
reasoning, and every tool call with its complete input and output.

![A run transcript](screenshots/out/05-transcript.png)

## Event log

`l` opens the JSONL event stream the dashboard is built from, scrollable over the whole
run.

![The event log](screenshots/out/09-logs.png)

## Configuration

The config page shows every setting, its effective value, and where that value came from
(a built-in default, the global config, or the per-repo config). `/` filters by name.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

---

These are regenerated from recorded runs by the
[pages workflow](https://github.com/agent6-dev/agent6/blob/master/.github/workflows/pages.yml),
so they track the current UI.
