---
title: agent6
hide:
  - toc
---

<div class="a6-hero" markdown>

# agent6

<p class="a6-tagline">A coding agent that jails model commands and uses editable state
machines for long-running tasks.</p>

<div class="a6-cta" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[:material-github: GitHub](https://github.com/agent6-dev/agent6){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/agent6/){ .md-button }
</div>

</div>

<div class="a6-shot" markdown>
![The run dashboard: task graph, budget, tool calls, reasoning, log, and diff](screenshots/out/02-run-dashboard.png)
</div>

The model can write code and ask to run commands, but those commands go through a jail with
restricted filesystem and network access. Long-running workflows can be written, reviewed,
edited, resumed, and replayed as declarative state machines instead of being left to an
open-ended agent loop.

<div class="a6-grid" markdown>

<div class="a6-card" markdown>
### Per-command sandbox
Each command the model runs is jailed on its own with Landlock + seccomp; the default
`strict` profile adds user namespaces + `pivot_root`, a read-only `.git`, and egress
limited to your provider. Not one coarse container around everything.
</div>

<div class="a6-card" markdown>
### Open-weight or frontier
Works with Anthropic and any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama,
vLLM, llama.cpp, LM Studio), and the prompts and tools are tuned to stay usable on
cheaper open-weight models.
</div>

<div class="a6-card" markdown>
### Resumable and forkable
State is snapshotted before every model call and committed after every passing step.
Resume an interrupted run from its snapshot, or fork a new run from any past turn while
the original stays intact.
</div>

<div class="a6-card" markdown>
### Plan, run, review, ask
Read-only planning and repository Q&A, a diff-review panel, and the run loop, each its own
command. A live terminal dashboard, full transcripts, and searchable run history.
</div>

<div class="a6-card" markdown>
### State machines
`agent6 machine` composes longer automated tasks from runs, sandboxed tool calls, waits,
and branches: drafted by the model, reviewed by you, journaled, and replayable.
</div>

<div class="a6-card" markdown>
### Small, fixed tool surface
The model's tools are a fixed set declared in one file. The only way to add more is an
operator-configured MCP server, off by default. No telemetry, no auto-update.
</div>

</div>

## The terminal UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/02-run-dashboard.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

`agent6 run` is headless by default: a scrolling event stream in your terminal, the CLI
mode. `agent6 tui` opens the hub instead: every run for the repository, with its mode,
status, and cost, where you open a run to watch the dashboard, read the full transcript,
or scroll the event log. `agent6 run --tui` jumps straight to that dashboard; `-i` drives
the run from a stdin REPL. The [tour](tour.md) has a still of each screen.

## The web UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

`agent6 web` serves the same views in a browser: start a run and watch it stream,
steer it, approve prompts, answer questions, read the transcript, and browse and
run state machines, from a desktop or a phone. It binds `127.0.0.1`; put
`tailscale serve` in front for encrypted remote access. See [the web UI](web.md).

## Install

```sh
uv tool install agent6        # or: pipx install agent6
```

agent6 needs Linux for the sandbox (kernel 6.7 or newer for the network rules), Python
3.12 or newer, and an API key for at least one provider. macOS runs unsandboxed behind a
startup warning; on Windows use WSL. See [installation](installation.md) for the full
requirements.

## Run

```sh
agent6 connect                       # pick a provider, paste an API key (once)
agent6 model worker anthropic claude-sonnet-4-6

cd your-repo
agent6 run "add a --json output mode to the CLI"
```

agent6 infers a verify command when you have not set one, commits each step that passes
it, and stops when the run finishes or a budget ceiling is hit. The
[getting started](getting-started.md) guide covers the first run and recovering one that
went wrong.
