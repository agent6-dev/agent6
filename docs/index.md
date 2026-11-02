---
title: agent6
hide:
  - toc
---

<div class="a6-hero" markdown>

# agent6

<p class="a6-tagline">A coding agent with sandboxed commands, resumable sessions, and editable workflows.</p>

<div class="a6-cta" markdown>

[:material-github: GitHub](https://github.com/agent6-dev/agent6){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/agent6/){ .md-button }

</div>

</div>

<div class="a6-grid" markdown>

<div markdown>

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="screenshots/out/02-run-dashboard.png">
  <source src="screenshots/out/hero-tui.webm" type="video/webm">
  <a href="screenshots/out/hero-tui.webm">Watch the terminal demo</a>.
</video>

[**agent6 tui**](terminal.md)<br>Terminal interface

</div>

<div markdown>

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="screenshots/out/web-shot.png">
  <source src="screenshots/out/hero-web.webm" type="video/webm">
  <a href="screenshots/out/hero-web.webm">Watch the browser demo</a>.
</video>

[**agent6 web**](web.md)<br>Browser interface for desktop or phone

</div>

</div>

Also available as a [streaming CLI](screenshots/out/hero-cli.webm) and through [ACP editor integration](acp.md).

## [Installation](installation.md)

Sandboxing supports **Linux on x86_64 and aarch64**; macOS runs without it, and Windows users should use WSL.

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/how-to/install-pipx.html), then connect a provider and select a model:

```sh
uv tool install agent6      # or pipx install agent6
agent6 connect              # configure a provider
agent6 model worker         # choose a model
```

Use Anthropic or any OpenAI-compatible endpoint, including OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, and LM Studio.
A provider entry can also use the Responses API (`providers.<name>.api_format = "responses"`) or sign in at a [configured OAuth issuer](config.md#configured-oauth).
For subscriptions, replace `agent6 connect` with `agent6 connect chatgpt` or `agent6 connect claude`; the latter uses your installed, signed-in Claude Code binary.
`agent6 completions` installs tab completion for your shell.

## [Usage](usage.md)

A run edits your current checkout.
Checkpoint commits go on a separate `agent6/<id>` branch; checkpointing leaves your current branch's history and index in place.
For a separate checkout, use `agent6 run --worktree "your task"`, which starts from committed HEAD without copying uncommitted changes or untracked dependencies.

```sh
cd your-repo
agent6 check                # check configuration and sandbox support
agent6 run "your task here"
```

Commands ask for approval by default.
After the task finishes, enter another instruction or type `/exit`.

Inspect the latest session, then merge its recorded changes when ready:

```sh
agent6 sessions show        # status and verification result
agent6 sessions diff        # recorded changes
agent6 sessions merge       # merge into the run's starting branch
```

Other commands:

| Command | Purpose |
| --- | --- |
| `agent6 plan "task"` | plan a change; `agent6 run --from <plan-id>` carries it out |
| `agent6 ask "question"` | ask about the repository |
| `agent6 review` | review a diff |
| `agent6 attach <id>` | follow a live session |
| `agent6 steer <id> "instruction"` | direct the next steps |
| `agent6 sessions findings <id>` | list a session's review findings |
| `agent6 sessions receipt <id>` | print the recorded tests, review outcome and how to revert |
| `agent6 stop <id> --after-turn` | stop after the current turn |
| `agent6 resume <id>` | continue an interrupted run |
| `agent6 fork <id> --at-turn 7` | fork a coding run from an earlier turn into a separate worktree |
| `agent6 telemetry export` | export session journals as OpenTelemetry signals or a Prometheus textfile |
| `agent6 pack` | copy your config, secrets, sessions and memory into the workspace before a sync |
| `agent6 unpack` | move the packed files into the new home |

To give a finished run more work, use `agent6 resume <id> --steer "next task"`.

## Features

<div class="a6-grid" markdown>

<div class="a6-card" markdown>

### Sandboxed commands

Runs the model's commands in a Linux sandbox with the file and network access you set.
Inside it, commands cannot read your provider keys.
[Security model](security.md)

</div>

<div class="a6-card" markdown>

### Verification and review

Verifies work with a configured or inferred command and reports whether the result passed, failed, or remains unverified.
Reviews the diff with a panel of models you configure and keeps each finding open until you resolve it.
[Session findings](usage.md#session-findings)

</div>

<div class="a6-card" markdown>

### Resume and fork

Saves conversations, tool activity, and Git checkpoints so you can inspect, resume, or fork a session from an earlier turn.
Pinned instructions survive compaction; repository memory carries facts across runs.
[Session recovery](usage.md#when-a-run-goes-wrong)

</div>

<div class="a6-card" markdown>

### Terminal, browser, and editor

The CLI, terminal UI, browser UI, and editor integration share the same engine and session records.
The browser also shows the repository's commit graph, diffs, and files with blame.
[Follow and steer a run](usage.md#follow-and-steer-a-run)

</div>

<div class="a6-card" markdown>

### Editable workflows

Defines long-running workflows in editable TOML state machines, combining agent runs, tool calls, waits, and branches with offline tests, crash recovery, and replay from recorded results.
[Agent state machines](state-machines.md)

</div>

<div class="a6-card" markdown>

### Parallel attempts

Runs parallel attempts in isolated clones with `--parallel N` or a list of models, then ranks the results for you to choose which to merge.
[Parallel runs](architecture.md#parallel-runs)

</div>

<div class="a6-card" markdown>

### Instructions, skills, and MCP

Supports repository `AGENTS.md` instructions, SKILL.md packs, MCP servers, and background commands for development servers and watchers.
[Skills](config.md#skills)

</div>

<div class="a6-card" markdown>

### Models and budgets

The worker, planner, and reviewer can use different models and reasoning effort.
Budgets limit spending, tokens for unpriced calls, and supported subscription usage.
[Budgets](config.md#budget)

</div>

</div>

No telemetry unless you [configure an export](usage.md#telemetry-export), and no automatic updates.
