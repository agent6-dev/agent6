# agent6

A coding agent that jails model commands and uses editable state machines for long-running tasks.

The model can write code and ask to run commands, but those commands go through a jail with
restricted filesystem and network access. Long-running workflows can be written, reviewed,
edited, resumed, and replayed as declarative state machines instead of being left to an
open-ended agent loop.

**Full documentation: [agent6.dev](https://agent6.dev)**

<table>
  <tr>
    <td align="center" width="34%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-tui.gif"><img src="https://agent6.dev/screenshots/out/hero-tui.gif" alt="the run TUI: conversation streaming, an approval modal, verify + auto-commit, the hub receipt"></a>
      <br><sub><b>the TUI</b><br>the full agent, as a live dashboard</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-cli.gif"><img src="https://agent6.dev/screenshots/out/hero-cli.gif" alt="the CLI: a failing suite, one command, the run streams to a green verify and a diff"></a>
      <br><sub><b>the CLI</b><br>the full agent, in any terminal</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-web.gif"><img src="https://agent6.dev/screenshots/out/hero-web.gif" alt="the web UI: the hub, a session view with expanded tool detail, the sandbox config"></a>
      <br><sub><b>the web UI</b><br>the full agent, desktop or phone</sub>
    </td>
  </tr>
</table>

## Features

- LLM-chosen commands run in a kernel sandbox (Landlock + seccomp); the default
  `strict` isolation adds user namespaces + `pivot_root`, rebinds `.git`
  read-only, and gives jailed commands no network
- Works with Anthropic and any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama,
  vLLM, llama.cpp, LM Studio)
- Per-step git commits, snapshot-resumable runs, per-turn forkable checkpoints, a hard
  metered USD budget with a token fallback for calls the meter cannot price
- Plan, run, review, and ask modes; a live terminal dashboard, a zero-dependency
  browser UI (`agent6 web`, phone-friendly), and an editor-driven ACP agent
  (`agent6 acp`); persistent transcripts and a searchable run history
- Sessions build on each other: `--from <session-id>` seeds a new run or ask with
  another session's context (the source is untouched -- keeping a session's mode is
  `fork`), a session can read this project's other sessions, and `/btw <question>`
  asks a one-off question beside a live run without interrupting it -- the answer
  prints whole at the next turn boundary and stays resumable
- Long jobs do not hold a turn open: `run_command` with `background: true` starts a slow build or a
  watcher, `read_background` polls it, `/shells` lists what a run started and how each
  one ended, and nothing a run started outlives it
- Context compaction: every surface shows what left the
  model's context, and the conversation view shows the summary a restart
  continued from; `/compact [focus]` compacts on demand, `/pin <text>` makes an
  instruction survive compaction verbatim, and `agent6 memory pin` keeps a
  load-bearing memory in every run's prompt (trimmed last, under the block's cap)
- State machines (`agent6 machine`) for long-running automated tasks: LLM-drafted,
  operator-reviewed, journaled, and replayable; they can pause for operator input,
  accept events, be steered from any front-end, and notify you when they need attention
- Skills: install standard SKILL.md packs (superpowers, caveman, any
  agentskills.io repo) with `agent6 skills install <url>`; they index into the
  system prompt, load on demand via a read-only tool, and fire as `/name`
  pause-menu commands or `run --skill`; nothing in a skill is ever executed.
  The format is shared with Claude Code and Pi, and `[skills].extra_dirs` loads
  an existing `~/.claude/skills`-style collection in place. Repo instructions
  are read from `AGENTS.md` (a repo using `CLAUDE.md` can symlink it)
- Small, fixed LLM tool surface; the only extension point is operator-configured MCP
  servers, off by default
- Eight runtime dependencies, no telemetry, no auto-update
- Parallel fan-out (`agent6 run --parallel N|model-a,model-b`): N isolated clone-based
  lanes run independently, each an ordinary sandboxed run; results auto-compare
  (reviewer-model judge, else verify+cost) into a ranked report. Nothing auto-merges;
  `agent6 sessions merge <id>` picks a winner. `agent6 sessions compare <id> <id> ...` runs the
  same ranked comparison over any past runs. The web/TUI composer and a live-run
  steer share one grammar, `/parallel [N|models] <task>` (repeat the token to
  queue more tasks), to dispatch and join a sibling group mid-conversation

## Install

From [PyPI](https://pypi.org/project/agent6/) with
[uv](https://docs.astral.sh/uv/getting-started/installation/) or
[pipx](https://pipx.pypa.io/stable/how-to/install-pipx/):

```bash
uv tool install agent6        # or: pipx install agent6
```

agent6 needs **Linux** for the sandbox, **Python 3.12+**, and
an API key for at least one provider. macOS runs unsandboxed behind a warning; on Windows
use WSL. See [installation](https://agent6.dev/installation/) for the full requirements and
building from source.

## Quick start

```bash
# Connect a provider once (stored in ~/.config/agent6/, key in a 0600 secrets file).
# If already connected, skip both; `agent6 check` verifies it.
agent6 connect                # interactive: pick provider, paste API key
agent6 model worker anthropic claude-sonnet-4-6

# Run the agent on a task. agent6 infers a verify command if you haven't set one.
cd your-repo
agent6 run "add a --json output mode to the CLI"

# Watch and drive runs from a terminal, a full-screen TUI, a browser, or an editor.
agent6 attach <session-id>        # follow + answer a run live (default: conversation view; --raw for the event stream)
agent6 tui                    # full-screen dashboard hub
agent6 web                    # browser UI on http://127.0.0.1:7658 (phone-friendly)
agent6 acp                    # speak ACP on stdio; an editor spawns this

# Audit the effective config, pre-flight the sandbox, resume or fork a run.
agent6 config show
agent6 check
agent6 resume <session-id>
agent6 fork <session-id> --at-turn 7
```

See [getting started](https://agent6.dev/getting-started/) for the
full command tour, [the web UI](https://agent6.dev/web/) for driving runs from a phone,
[configuration](https://agent6.dev/config/) for every field, and the
[security model](https://agent6.dev/security/) for what the sandbox enforces.

Config is layered: built-in secure defaults, then the global `~/.config/agent6/config.toml`,
then the per-repo config (out of the workspace, per-machine, not committed), then an
explicit `--config FILE`. Every field has a default; security-sensitive fields default to
the safe value (`network = "auto"`,
`run_commands = "ask"`, `protect_git = true`), and `git_ops.py`
refuses `push`, `--force`, and history rewrites unconditionally.

## Benchmarks

Reproducible harnesses live under [bench/](bench/). The main one is the
cross-model sweep ([bench/sweep](bench/sweep/)): replicated runs on real-world
tasks, scored out-of-band by each project's own test suite, reported with
confidence intervals for success rate and cost plus a latency comparison.
Also there: real-world
SWE-bench-Lite-style tasks, head-to-head runs against Claude Code / opencode /
aider, `machine create` validation, and a perf-optimization harness. The
recorded numbers in those are mostly small-n exploratory runs; re-run before
quoting.

## Contributing

Read [AGENTS.md](AGENTS.md) first. The repo's verify command decides whether a change is
landable:

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

Adding a tool, loosening a security default, dialling a new network destination, or
changing the jail (`src/agent6/jail/`) requires a `Security review note:` paragraph in
the commit message.

## License

[Apache-2.0](LICENSE).
