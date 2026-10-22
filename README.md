# agent6

**A coding agent with sandboxed commands, resumable sessions, and editable workflows.**

Give agent6 a task in a Git repository: fix a bug, add a feature, review a change, or explain the code.
It uses a model you choose and saves the conversation, tool activity, and Git checkpoints so you can inspect the result or continue the work.
For longer workflows, define the steps in a TOML state machine you can review, test, and replay.

[Documentation](https://agent6.dev/) · [Quick start](#quick-start) · [Security model](https://agent6.dev/security/)

[![agent6's terminal UI showing a coding task, command approval, verification, and the resulting changes](https://agent6.dev/screenshots/out/hero-tui.gif)](https://agent6.dev/screenshots/out/hero-tui.gif)

More demos: [CLI](https://agent6.dev/screenshots/out/hero-cli.gif) · [Web UI](https://agent6.dev/screenshots/out/hero-web.gif)

## Quick start

Requires **Python 3.12+**, **Git 2.40+**, and access to a model through an API, local server, or supported subscription.
The sandbox supports **Linux on x86_64 and aarch64**.
On macOS, commands run without the Linux sandbox; on Windows, use WSL.

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) or use `pipx install agent6`.
Replace `your-repo` and the example task with your project and the change you want:

```sh
uv tool install agent6
agent6 completions            # tab completion for your shell
agent6 connect                # connect a provider
agent6 model worker           # choose its model interactively
cd your-repo
agent6 check                  # check provider config and sandbox support
agent6 run "add a --json output mode to the CLI"
```

For a subscription, use `agent6 connect chatgpt` or `agent6 connect claude` in place of `agent6 connect`.
The Claude route uses your installed, signed-in Claude Code binary.
Then select the worker model as above.

The run streams in your terminal and asks before running commands by default.
It edits your current working tree and saves checkpoint commits on a separate `agent6/<id>` branch by default.
Checkpointing leaves your current branch's history and index in place; `sessions merge` brings the recorded work onto that branch.
At the end, read the verification result and diff; type `/exit` to leave the interactive session.

```sh
agent6 sessions show          # status and verification result of the latest run
agent6 sessions diff          # inspect its changes
agent6 sessions merge         # merge them into the run's starting branch
```

To keep your current checkout unchanged, use `agent6 run --worktree "your task"`.
This starts from the committed files at HEAD; uncommitted changes and untracked dependencies are not copied.
See [isolated workspaces](https://agent6.dev/usage/#isolated-workspaces) for setup commands, and [installation](https://agent6.dev/installation/) for PATH and platform setup.

## Features

- **Command sandboxing:** control what model commands can read, write, and reach on the network, with approvals enabled by default and diagnostics that show the effective boundaries ([security](https://agent6.dev/security/)).
- **Resume and fork:** saved session state, transcripts, and Git checkpoints let you continue interrupted work or branch from an earlier turn into a separate worktree ([recovery](https://agent6.dev/usage/#when-a-run-goes-wrong)).
- **Project checks:** configure a verification command or let agent6 infer one, then see whether the final work passed, failed, or remains unverified ([verification](https://agent6.dev/usage/#your-first-run)).
- **Terminal, browser, and editor:** the CLI, [terminal UI](https://agent6.dev/terminal/), [browser UI](https://agent6.dev/web/) on desktop or phone, and [editor integration through the Agent Client Protocol (ACP)](https://agent6.dev/acp/) share the same engine and session records.
- **Visible context and progress:** inspect the task graph, tool calls, usage, and compaction events; `/pin` preserves an instruction through compaction, `/compact [focus]` requests a summary, and repository memory carries facts across runs ([architecture](https://agent6.dev/architecture/#a-run)).
- **Parallel attempts:** run the same task in isolated clones with `--parallel N` or a list of models, compare the results in a ranked report, and choose which result to merge ([parallel runs](https://agent6.dev/architecture/#parallel-runs)).
- **Editable state machines:** compose agent runs, tool calls, waits, and branches in `.asm.toml` files, with validation, offline tests, journaled transitions, and replay from recorded results ([state machines](https://agent6.dev/state-machines/)).

## Inspect and direct a run

Run `agent6 tui` for the terminal dashboard or `agent6 web` for the browser UI.
Both show sessions, conversations, tool activity, verification results, and usage, and let you answer prompts and steer work.
Run headers show the model, resolved isolation level, command approval setting, and verification command.
The web UI also supports access from another device; the [web guide](https://agent6.dev/web/) covers authentication and remote access.

From another terminal, use the session ID printed by the run:

```sh
agent6 attach <session-id>
agent6 steer <session-id> "keep the public API unchanged"
agent6 stop <session-id> --after-step
agent6 resume <session-id>
```

To continue an already finished run, use `agent6 resume <session-id> --steer "your next task"`.
To explore a different approach, use `agent6 fork <session-id> --at-turn 7 --steer "try a different approach"`.
The [usage guide](https://agent6.dev/usage/) covers transcripts, history search, side questions with `/btw`, and commands inside a run's sandbox.

## Models and integrations

Use Anthropic, OpenAI, OpenRouter, or another OpenAI-compatible endpoint, including local servers such as Ollama, vLLM, llama.cpp, and LM Studio.
ChatGPT and Claude Code subscription routes are also supported.
Choose the model and reasoning effort for the worker, planner, and reviewer separately; unset roles fall back to the worker ([configuration](https://agent6.dev/config/)).

- **Plan and ask:** `agent6 plan` and `agent6 ask` explore a repository without editing its files; `agent6 run --from <plan-id>` uses a saved plan.
- **Code review:** `agent6 review` reviews a diff; an optional review panel can also check work during a run.
- **Skills and instructions:** use `AGENTS.md` for repository instructions and `SKILL.md` packs through `/name` or `--skill`.
- **Model Context Protocol (MCP):** connect servers you configure; MCP is off by default, and locally spawned servers are sandboxed by default.
- **Background commands:** keep a development server or watcher running while the agent continues, with controls to read its output or stop it.
- **Usage limits:** track cost and tokens, and set spending limits, token limits for unpriced calls, and usage limits for supported subscriptions ([budgets](https://agent6.dev/config/#budget)).

## Security and verification

agent6 treats model output as untrusted.
The model's tools are a fixed built-in set plus the MCP servers you enable.
Its Linux sandbox restricts command execution; the main agent process remains trusted and connects to your configured model providers.
`sandbox.isolation` sets the level: `full` is a namespace-based Linux jail, `partial` is Landlock and seccomp without namespaces, `none` leaves commands unconfined.
The default, `auto`, selects the strongest available level and warns when it falls back.
`sandbox.network` is set separately: under `full` it defaults to the run's private network and can use the host network; `partial` and `none` use the host network.
Run `agent6 check sandbox` to inspect file and command access by path, network and approval settings, and live probe results.
Read the [security model](https://agent6.dev/security/) for the boundaries and host requirements at each level.

By default, a fresh run executes an available, permitted verification command before editing.
A verification pass means the selected checks passed for the tested tree and scope.
Runs can end with failed or unverified work, and scoped checks are identified in the result.
Review that result and the diff before merging.

The defaults are `sandbox.network = "auto"`, `sandbox.run_commands = "ask"` and `sandbox.protect_git = "auto"`.
agent6 has eight runtime dependencies, no telemetry and no automatic updates.

## Development and feedback

agent6 is [Apache-2.0 licensed](LICENSE) and is under active development before 1.0; configuration, commands, and stored formats can change.
See [development conventions](AGENTS.md) and the repository's [benchmark and evaluation suites](https://github.com/agent6-dev/agent6/tree/master/bench), which compare models and harness settings with task graders, repeated runs, and usage measurements.

If you try it, [open an issue](https://github.com/agent6-dev/agent6/issues) with the task you attempted, what you expected, and where setup or the run became confusing or failed.
For vulnerabilities, use [private security reporting](https://github.com/agent6-dev/agent6/security/advisories/new).
