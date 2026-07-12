# Web UI

`agent6 web` serves a browser front-end for driving agent6 from a desktop or a
phone: watch a run stream, steer it, approve prompts, answer questions, read the
conversation, and browse, create, run, and watch state machines.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

The same UI on a phone (single column, bottom nav):

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       style="max-width: 390px">
  <source src="/screenshots/out/web-phone.webm" type="video/webm">
</video>

## Run it

```bash
agent6 web            # serve the hub on http://127.0.0.1:7658
agent6 web <run-id>   # open a run on load
agent6 web <machine>  # open a machine instance on load
```

`--host` / `--port` override the [`[web]`](config.md#web) config for one
invocation. Stop it with Ctrl-C.

## What you can do

- **Hub**: every run (mode, status, last activity, cost) and machine instance;
  start new work (run / plan / ask); run an authored machine or create one;
  prune merged run branches.
- **Run view** (live over SSE): the conversation front and center — the same
  folded transcript the CLI stream and the TUI render (reasoning, every tool
  call with its result, commits, the verdict), with the in-progress turn
  streaming underneath. A detail toggle cycles collapsed / expanded / hidden,
  and any clipped item (a long tool result, folded reasoning) expands on click.
  Alongside it: the task graph, budget, clipped tool-call table (hover for the
  full args and result), latest commit diff, and the raw event log. A composer
  under the conversation steers a live run (Enter sends, Shift+Enter newline)
  and, once the run has ended, resumes it with the typed follow-up. Stop now,
  stop after the current step, compact the context, merge the branch, approve
  `run_command` prompts, and answer `ask_user` questions inline.
- **Conversation**: the same view full-page, following the run live.
- **Machines**: the state overview, the path taken, and the current agent
  state's conversation. Steer, approve, and answer the current agent state's
  prompts inline (same controls as a run); send a message to a waiting machine
  (a `poke` payload the next tool reads); and see `machine.notify`/end as
  ephemeral banners and OS notifications.
- **Config**: every setting with its value and source, filterable, click a row
  to set it. Secrets are never shown.

The layout reflows: multi-pane on a wide screen, a single column with a bottom
nav on a phone.

## Notifications and installing (PWA)

The page installs as an app (a phone home-screen icon or a desktop window). Click
**🔔 Notifications** on a machine view to grant permission; a `machine.notify`
message or a machine finishing then pops an OS notification — foreground on any
device, and backgrounded on desktop (a backgrounded phone won't wake, which is
expected). A notification never clears or blocks the send/answer inputs: one
popping mid-type keeps your text and focus. For a phone in your pocket, point the
operator notify hook `[machine.notify].on_event` (see [config.md](config.md)) at
a push service you already use.

## How it talks to the server

The page reads the same wire form as `agent6 watch --json`:

```bash
curl -s localhost:7658/api/hub                 # runs + machines + machine files
curl -s localhost:7658/api/run/<id>            # a run's state, as JSON
curl -s localhost:7658/api/run/<id>/conversation # the folded conversation items
curl -s localhost:7658/api/machine/<name>      # a machine's state, as JSON
curl -s localhost:7658/api/config              # effective config (no secrets)
curl -sN localhost:7658/api/run/<id>/events    # SSE: a fresh snapshot per change
```

`curl /api/run/<id>` returns exactly what `agent6 watch <id> --json` prints.
Writes are small JSON `POST`s (`/api/new`,
`/api/run/<id>/{steer,approve,answer,merge,resume,stop_step,compact}`,
`/api/machine/<name>/{poke,steer,approve,answer}`, `/api/runs/prune`,
`/api/config`, `/api/machine/{create,run}`) that only ever drive the typed spawn /
answer-file contracts, never arbitrary execution. A machine's `approve`/`answer`/
`steer` land in the current agent state's per-state dir; `poke` drops a signal
(with an optional `message`/`data` payload) on the instance. The machine name and
every answer id are validated to a single path component, so a request cannot
traverse out of the instance dir.

## Remote access (Tailscale)

The server binds `127.0.0.1` by default and has no app-level auth. For remote
access, put [Tailscale](https://tailscale.com) in front of the loopback bind:

```bash
agent6 web                       # keep it on 127.0.0.1:7658
tailscale serve --bg 7658        # HTTPS + WireGuard, reachable on your tailnet
```

The tailnet (WireGuard) identity is the access control: only devices on your
tailnet reach it, over an encrypted tunnel, and `tailscale serve` terminates
HTTPS. agent6 itself handles no tokens or passwords.

Binding a non-loopback address exposes the write surface, spawning runs and
answering prompts, to anyone who can reach the port. It is refused unless you
opt in, whether the host comes from [`[web].host`](config.md#web) (needs
`[web].allow_non_loopback = true`) or `--host` (needs `--allow-non-loopback`), so
a copied config or command cannot silently expose you. Prefer `tailscale serve`
in front of a loopback bind over any raw non-loopback bind.
