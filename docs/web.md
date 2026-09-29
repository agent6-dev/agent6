# Web UI

`agent6 web` serves a browser front-end for driving agent6 from a desktop or a phone: start and watch runs, steer them, answer their prompts, and run state machines.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

The same UI on a phone (single column, bottom nav):

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       style="max-width: 390px">
  <source src="/screenshots/out/web-phone.webm" type="video/webm">
</video>

## Start the server

```bash
agent6 web              # serve the hub on http://127.0.0.1:7658
agent6 web <session-id> # open a session on load
agent6 web <machine>    # open a machine instance on load
agent6 web <draft>      # open a `machine create` draft on load (read-only)
```

`--host` / `--port` override the [`[web]`](config.md#web) config for one invocation.
Stop it with Ctrl-C.

## Pages

Every page docks its text entry at the bottom, like a chat: Enter sends, Shift+Enter inserts a newline.
Buttons do what the CLI command of the same name does, and prompts are answered inline on every page that shows a run.

- **Sessions page**: every session (mode, status, last activity, cost), a fan-out's lanes folded under its row
    - the composer starts a run, plan or ask under a chosen preset and model (the model box shows what the config resolves; a pick overrides it)
    - `more…` holds **Prune merged runs** (`sessions prune`, or with `--delete-squashed` after a confirm) and **Clear saved asks**
- **Machines page**: instances, `machine create` drafts, and cards that run an authored machine file; the composer creates a new one
- **Session view** (live over SSE): the conversation, the same folded transcript the CLI and TUI render, with the turn in flight streaming underneath; a detail toggle cycles collapsed, expanded, hidden
    - a resizable drawer holds the run's context: overview, plan.md, task graph, budget, tool calls, background shells, latest commit diff, event log
    - the Latest commit widget selects any per-step commit (cumulative toggle) and the other widgets follow it; a model-controlled run has no chain and says so
    - the composer steers a live run or resumes an ended one under the preset and model picked above it, and takes the [steer directives](usage.md#follow-and-steer-a-run); Ctrl+Enter sends a live steer as `/now`, Ctrl-R searches past messages
    - buttons: stop now or after the step, compact, fork (a new run at the latest checkpoint, started from its composer), merge, delete history, run a finished plan (`run --from`), and review (`sessions review`, minutes; refused while the run is live)
    - "Allow all" on a prompt appears only where it would grant more than the one call
- **Machine view**: the state overview, the path taken and the current agent state's conversation; the entry submits as **Steer** (into the current state) or **Message** (a `poke` a waiting machine's next tool reads); **Stop** parks the instance at its next transition and `machine run` resumes it
- **Config page**: every setting with value and source, filterable; click a row to set it, with the choices an enum or a provider's model listing offers; Add provider… adds or updates a `[providers.<name>]` block; secrets never shown

Start a machine on the Machines page and watch the current state stream, answering its approvals and questions in place:

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-machine.webm" type="video/webm">
</video>

## Layout

- desktop: a nav rail, the run view a fixed pane with the drawer and conversation scrolling inside it
- phone: a top bar (theme toggle), bottom tab nav, the composer docked above it, and the run view one widget at a time, switched from the top-bar menu

## Notifications and installing (PWA)

The page installs as an app (phone home-screen icon or desktop window).
**🔔 Notifications** on a machine view grants permission; `machine.notify` and a machine's end then pop OS notifications, except on a backgrounded phone.
For a phone not open on the page, point [`[machine.notify].on_event`](config.md#machinenotify-optional) at a push service.

## The HTTP API

The page reads the same wire form as `agent6 attach --json`:

```bash
curl -s localhost:7658/api/hub                       # hub state
curl -s localhost:7658/api/session/<id>              # a run's state, as JSON
curl -s localhost:7658/api/session/<id>/conversation # the folded conversation
curl -s localhost:7658/api/machine/<name>            # a machine's state, as JSON
curl -s localhost:7658/api/config                    # effective config
curl -sN localhost:7658/api/session/<id>/events      # SSE: a snapshot per change
```

- reads: `/api/meta`, `/api/hub`, `/api/routes?mode=&preset=`, `/api/config` with `/suggest/<key>` and `/provider_choices`, `/api/session/<id>` (what `attach --json` prints; `?step=<sha>` folds up to that commit) with `/conversation`, `/restate`, `/diff` (`?sha=&cumulative=1`) and `/events`, `/api/machine/<name>` with `/reasoning`, `/conversation` and `/events`, `/api/draft/<name>` with `/conversation`, `/diff` and `/events`
- writes, small JSON `POST`s: `/api/new`, `/api/session/<id>/{steer,approve,answer,merge,undo,fork,resume,run_plan,review,stop,compact,rm}`, `/api/machine/<name>/{poke,stop,steer,approve,answer}`, `/api/sessions/{prune,rm_asks}`, `/api/config`, `/api/config/provider`, `/api/machine/{create,run}`
- every write goes through the same spawn and answer-file contracts as the CLI; machine names and answer ids validate to one path component
- the page and its PWA assets: `/`, `/manifest.webmanifest`, `/sw.js`, `/icon.svg`, `/favicon.svg`

## Remote access (Tailscale)

The server binds `127.0.0.1` by default and has no app-level auth.
For remote access, put [Tailscale](https://tailscale.com) in front of the loopback bind:

```bash
agent6 web                # keep it on 127.0.0.1:7658
tailscale serve --bg 7658 # HTTPS + WireGuard, reachable on your tailnet
```

- the tailnet identity is the access control and `tailscale serve` terminates HTTPS; agent6 handles no tokens or passwords
- a non-loopback bind exposes the write surface (spawn runs, answer prompts) to anyone reaching the port, so it refuses without the opt-in: `[web].allow_non_loopback = true` for [`[web].host`](config.md#web), `--allow-non-loopback` for `--host`
