---
title: agent6
hide:
  - toc
---

<div class="a6-hero" markdown>

# agent6

<p class="a6-tagline">A sandboxed coding agent for Linux, tuned to stay effective even on
affordable open-weight models like Kimi, GLM, and Qwen, as well as Claude.</p>

<div class="a6-cta" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[:material-github: GitHub](https://github.com/agent6-dev/agent6){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/agent6/){ .md-button }
</div>

</div>

<div class="a6-shot" markdown>
![The run dashboard: task graph, budget, tool calls, reasoning, log, and diff](screenshots/out/02-run-dashboard.png)
</div>

Run agent6 with a weaker or untrusted model on any repository: it cannot escape the
workspace, reach the network beyond your model provider, or corrupt git history. The model
reads, searches, and edits files, runs the project's verify command, and commits each step
that passes; state is snapshotted before every model call, so an interrupted or wrong run
is resumable.

<div class="a6-grid" markdown>

<div class="a6-card" markdown>
### Per-command sandbox
Each command the model runs is jailed on its own (user namespaces, Landlock, seccomp,
`pivot_root`), with `.git` rebound read-only and egress limited to your provider. Not one
coarse container around everything.
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
       poster="screenshots/out/02-run-dashboard.png">
  <source src="screenshots/out/tour.webm" type="video/webm">
</video>

`agent6 tui` opens the hub: every run for the repository, with its mode, status, and
cost. Open a run to watch the dashboard, read the full transcript, or scroll the event
log. `agent6 run` opens the dashboard directly; `--no-tui` and `-i` (a stdin REPL) opt
out. The [tour](tour.md) has a still of each screen.

## Install

```sh
uv tool install agent6        # or: pipx install agent6
```

agent6 needs Linux for the sandbox (kernel 6.7 or newer for the network rules), Python
3.12 or newer, and an API key for at least one provider. macOS and Windows run
unsandboxed behind a startup warning. See [installation](installation.md) for the full
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
