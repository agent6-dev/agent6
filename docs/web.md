# Web UI

`agent6 web` serves a browser front-end with near-full parity to the TUI, so a
run is fully drivable from a desktop or a phone: watch it stream, steer it,
approve prompts, answer questions, read the transcript, and browse, create, run,
and watch state machines.

It is zero-dependency: a stdlib HTTP server and one self-contained HTML/CSS/JS
page, no framework and no build step. Like the CLI and TUI, it is a thin renderer
of the shared view-model, so nothing about how a run is folded or driven is
re-derived here.

## Run it

```bash
agent6 web            # serve the hub on http://127.0.0.1:8901
agent6 web <run-id>   # open a run on load
agent6 web <machine>  # open a machine instance on load
```

`--host` / `--port` override the [`[web]`](config.md#web) config for one
invocation. Stop it with Ctrl-C.

## What you can do

- **Hub**: every run (mode, status, last activity, cost) and machine instance;
  start new work (run / plan / ask); run an authored machine or create one;
  prune merged run branches.
- **Run dashboard** (live over SSE): the task graph, the model's streamed
  reasoning, tool calls and results, the event log, the latest commit diff, and a
  budget bar. Steer the run, merge its branch, approve `run_command` prompts, and
  answer `ask_user` questions inline.
- **Transcript**: the full provider-agnostic conversation.
- **Machines**: the state overview, the path taken, and the current agent
  state's live reasoning.
- **Config**: every setting with its value and source, filterable, click a row
  to set it. Secrets are never shown.

The layout reflows: multi-pane on a wide screen, a single column with a bottom
nav on a phone.

## How it talks to the server

The page reads the same wire form as `agent6 watch --json`:

```bash
curl -s localhost:8901/api/hub                 # runs + machines + machine files
curl -s localhost:8901/api/run/<id>            # a run's folded RunState
curl -s localhost:8901/api/run/<id>/transcript # the conversation turns
curl -s localhost:8901/api/machine/<name>      # a machine's folded MachineState
curl -s localhost:8901/api/config              # effective config (no secrets)
curl -sN localhost:8901/api/run/<id>/events    # SSE: a fresh snapshot per change
```

`curl /api/run/<id>` returns exactly what `agent6 watch <id> --json` prints.
Writes are small JSON `POST`s (`/api/new`, `/api/run/<id>/{steer,approve,answer,merge}`,
`/api/runs/prune`, `/api/config`, `/api/machine/{create,run}`) that only ever
drive the typed spawn / answer-file contracts, never arbitrary execution.

## Remote access (Tailscale)

The server binds `127.0.0.1` by default and has no app-level auth. For remote
access, put [Tailscale](https://tailscale.com) in front of the loopback bind:

```bash
agent6 web                       # keep it on 127.0.0.1:8901
tailscale serve --bg 8901        # HTTPS + WireGuard, reachable on your tailnet
```

The tailnet (WireGuard) identity is the access control: only devices on your
tailnet reach it, over an encrypted tunnel, and `tailscale serve` terminates
HTTPS. This is what keeps the front-end zero-dependency, no tokens or password
handling in agent6 itself.

Binding a non-loopback address directly (`--host` or `[web].host`) exposes the
write surface, spawning runs and answering prompts, to anyone who can reach the
port. It is off by default and gated behind
[`[web].allow_non_loopback = true`](config.md#web). Prefer `tailscale serve` over
a raw non-loopback bind.
