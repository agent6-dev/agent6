---
title: agent6
hide:
  - toc
---

<div class="a6-hero" markdown>

# agent6

<p class="a6-tagline">A coding agent with sandboxed commands, resumable sessions, and editable workflows.</p>

Give agent6 a task in a Git repository: fix a bug, add a feature, review a change, or explain the code.
Use a model you choose and follow its work from your terminal, browser, or editor.
For longer workflows, define the steps in a TOML state machine you can review, test, and replay.

<div class="a6-cta" markdown>

[:material-play: Get started](#get-started){ .md-button .md-button--primary }
[:material-github: GitHub](https://github.com/agent6-dev/agent6){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/agent6/){ .md-button }

</div>

</div>

<div class="a6-shot" markdown>

![The terminal dashboard showing the task graph, usage, tool calls, event log, and diff](screenshots/out/02-run-dashboard.png)

</div>

## Get started

Requires **Python 3.12+**, **Git 2.40+**, and access to a model through an API, local server, or supported subscription.
The sandbox supports **Linux on x86_64 and aarch64**; macOS runs without it, and Windows users should use WSL.
See [installation](installation.md) for platform setup and installing [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/how-to/install-pipx/).

```sh
uv tool install agent6        # or: pipx install agent6
agent6 completions            # tab completion for your shell
agent6 connect                # connect a provider
agent6 model worker           # choose its model interactively
cd your-repo
agent6 check                  # check provider config and sandbox support
agent6 run "your task here"
```

For a subscription, replace `agent6 connect` with `agent6 connect chatgpt` or `agent6 connect claude`.
The Claude route uses your installed, signed-in Claude Code binary.

The run streams in your terminal and asks before running commands by default.
It edits your current working tree and saves checkpoint commits on a separate `agent6/<id>` branch by default.
Checkpointing leaves your current branch's history and index in place; `sessions merge` brings the recorded work onto that branch.
At the end, read the verification result and type `/exit` to leave the interactive session.

```sh
agent6 sessions show          # status and verification result of the latest run
agent6 sessions diff          # inspect its changes
agent6 sessions merge         # merge them into the run's starting branch
```

Use `agent6 run --worktree "your task"` for a separate checkout.
It starts from committed files at HEAD, without your uncommitted changes or untracked dependencies; [isolated workspaces](usage.md#isolated-workspaces) explains setup.
The [usage guide](usage.md) continues through planning, review, recovery, and configuration.

## Features

<div class="a6-grid" markdown>

<div class="a6-card" markdown>

### Command sandboxing

Set the filesystem and network access available to model commands, with approvals enabled by default.
`agent6 check sandbox` lists file and command access by path, reports network and approval settings, and probes the jail.
`sandbox.isolation = "auto"` selects `full`, `partial` or `none` and warns when it falls back.
[Security model](security.md)

</div>

<div class="a6-card" markdown>

### Resume and fork

Saved session state, transcripts, and Git checkpoints let you continue interrupted work or try another approach from an earlier turn.
Forking a coding run creates a separate worktree.
[Session recovery](usage.md#when-a-run-goes-wrong)

</div>

<div class="a6-card" markdown>

### Terminal, browser, and editor

The CLI, terminal UI, browser UI, and editor integration share the same engine and session records.
The terminal and browser interfaces let you follow live runs, answer questions, and steer work; the browser also reads the repository's history and files.
Run headers show the model, resolved isolation level, command approval setting, and verification command.
[Interfaces and live controls](usage.md#follow-and-steer-a-run)

</div>

<div class="a6-card" markdown>

### Context and progress

Inspect the task graph, tool calls, usage, and compaction events.
Pin instructions across compaction, request a focused summary, and keep repository memory across runs.
[Context and memory](architecture.md#a-run)

</div>

<div class="a6-card" markdown>

### Parallel attempts

Try a task in isolated clones with the same model or several different models.
`--parallel` produces a ranked comparison, and you choose which result to merge.
[Parallel runs](architecture.md#parallel-runs)

</div>

<div class="a6-card" markdown>

### Editable workflows

Compose agent runs, tool calls, waits, and branches in `.asm.toml` state machines.
Review and test the workflow before running it, then inspect or replay its recorded transitions.
[Agent state machines](state-machines.md)

</div>

</div>

## Project checks

Configure a verification command or let agent6 infer one from the repository.
A fresh run runs it before it edits anything.
A run ends passed, failed or unverified, and a scoped gate says so in the result.
See [verification](usage.md#your-first-run) for inference, retries and runs without a usable gate.

## The terminal UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/02-run-dashboard.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
  <a href="/screenshots/out/tour.webm">Watch the terminal UI demo</a>.
</video>

`agent6 run` streams the conversation in your terminal.
`agent6 tui` opens the repository's session hub; select a run to follow its conversation or open the dashboard with Ctrl+D.
Inspect tools, the task graph, verification, usage, and changes while the run proceeds.
See the [terminal UI guide](terminal.md).

## The web UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
  <a href="/screenshots/out/web-desktop.webm">Watch the web UI demo</a>.
</video>

`agent6 web` lets you start and follow runs, approve commands, answer questions, steer work, and browse state machines from a desktop or phone.
By default, it binds to `127.0.0.1` and prints a sign-in URL and secret.
Follow the [web UI guide](web.md) for remote access, including Tailscale setup.
For an editor, see [Agent Client Protocol (ACP) integration](acp.md).

## Models and everyday tools

Use Anthropic, OpenAI, OpenRouter, or another OpenAI-compatible endpoint, including Ollama, vLLM, llama.cpp, and LM Studio.
ChatGPT and Claude Code subscription routes are also supported.
Set the model and reasoning effort for the worker, planner, and reviewer separately; unset roles use the worker.
See [configuration](config.md) for provider setup and model options.

| Capability | What you can do |
| --- | --- |
| Plan and ask | Explore a repository without editing its files; start a run from a saved plan. |
| Code review | Review a diff with `agent6 review`, or enable the optional review panel during a run. |
| Skills and instructions | Use repository `AGENTS.md` instructions and `SKILL.md` packs through `/name` or `--skill`. |
| Model Context Protocol (MCP) | Connect servers you configure; MCP is off by default, and locally spawned servers are sandboxed by default. |
| Background commands | Keep a development server or watcher running while the agent continues, and read its output or stop it. |
| Usage limits | Track cost and tokens, and set spending limits, token limits for unpriced calls, and usage limits for supported subscriptions. |
| Live controls | Attach, steer, stop, and resume sessions from the CLI, including instructions sent by scripts. |

The [usage guide](usage.md) covers these commands, and [budgets](config.md#budget) explains how usage is measured and limited.

## Workflow control

An agent state machine makes the sequence of work explicit: which agent or tool runs, what result selects the next step, and when to wait or ask for input.
States execute one at a time; `run --parallel` is a separate way to compare concurrent attempts.
Offline replay follows the recorded transitions without calling models or rerunning tools.
See [state machines](state-machines.md) for examples, validation, and crash-recovery semantics.

