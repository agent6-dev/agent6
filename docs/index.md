---
title: agent6
hide:
  - toc
---

<div class="a6-hero" markdown>

# agent6

<p class="a6-tagline">A coding agent that jails every command and leaves your branch untouched.</p>

<div class="a6-cta" markdown>
[:material-github: GitHub](https://github.com/agent6-dev/agent6){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/agent6/){ .md-button }
</div>

</div>

<div class="a6-shot" markdown>
![The run dashboard: task graph, budget, tool calls, reasoning, log, and diff](screenshots/out/02-run-dashboard.png)
</div>

```sh
uv tool install agent6                 # or: pipx install agent6
agent6 connect                         # pick a provider, paste a key (once)
cd your-repo
agent6 run "add a --json output mode to the CLI"
```

<div class="a6-grid" markdown>

<div class="a6-card" markdown>
### Jailed commands
Every command the model runs goes through a sandbox that bounds what it reads, writes and reaches on the network.
`auto` picks the strongest level the host allows.
[Security](security.md)
</div>

<div class="a6-card" markdown>
### Your branch stays yours
Each step commits to the run's own hidden ref; HEAD, the index and your branch are untouched until `sessions merge` lands the work.
`resume` continues a run, `fork` branches it at any turn.
</div>

<div class="a6-card" markdown>
### A verify gate
The repo's test command, inferred when unset, certifies the tree before a run may finish.
Every surface shows the same green or red.
</div>

<div class="a6-card" markdown>
### One run, four front-ends
The CLI, the [terminal UI](terminal.md), a [browser](web.md) on a desktop or phone, and an [editor over ACP](acp.md) drive the same runs.
`attach` follows a live run; `steer` queues an instruction from a script or a cron job.
</div>

<div class="a6-card" markdown>
### State machines for long work
`.asm.toml` files you review, edit, run, watch and replay, with waits, operator input and steering built in.
[State machines](state-machines.md)
</div>

<div class="a6-card" markdown>
### Secure by default
`network = "auto"`, `run_commands = "ask"`, `protect_git = true`, a fixed tool surface, eight runtime dependencies, no telemetry.
[Configuration](config.md)
</div>

</div>

## The terminal UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/02-run-dashboard.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

`agent6 run` streams the run in your terminal; `agent6 tui` opens the hub, every run for the repository with its mode, status and cost, and from there the live conversation and the dashboard (Ctrl+D).
The [terminal UI](terminal.md) page has a still of each screen.

## The web UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

`agent6 web` serves the same views in a browser, from a desktop or a phone: start a run, watch it stream, steer it, approve prompts, answer questions, read the transcript, browse and run state machines.
It binds `127.0.0.1`; put `tailscale serve` in front for encrypted remote access.
See [the web UI](web.md).

[Installation](installation.md) covers requirements, shell completion and building from source.
[Usage](usage.md) covers the first run, inspecting it and recovering one that went wrong.
