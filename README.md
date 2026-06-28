# agent6

A sandboxed coding agent for Linux, tuned to stay effective on affordable open-weight
models like Kimi, GLM, and Qwen, as well as Claude. The LLM is treated as adversarial:
every command it spawns runs inside a custom Rust launcher (`agent6-jail`) built on user
namespaces, Landlock, seccomp, `pivot_root`, `capset(0)`, and `NO_NEW_PRIVS`, so you can
point a weaker or untrusted model at a real repository and it cannot escape the workspace,
reach the network beyond the provider endpoint, or corrupt git history. Runs commit per
step and are resumable and forkable, so an interrupted or wrong turn is never a dead end.

**Full documentation: [agent6.dev](https://agent6.dev)**

[![The agent6 hub](https://agent6.dev/screenshots/out/01-hub.png)](https://agent6.dev/tour/)

## Features

- Sandboxed execution for every LLM-chosen child process, jailed individually
  (Landlock + seccomp + `pivot_root`), with `.git` rebound read-only and egress confined
  to your provider
- Works with Anthropic and any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama,
  vLLM, llama.cpp, LM Studio)
- Per-step git commits, snapshot-resumable runs, per-turn forkable checkpoints, USD and
  token budgets with hard stops
- Plan, run, review, and ask modes; a live terminal dashboard; persistent transcripts and
  a searchable run history
- State machines (`agent6 machine`) for long-running automated tasks: LLM-drafted,
  operator-reviewed, journaled, and replayable
- Small, fixed LLM tool surface; the only extension point is operator-configured MCP
  servers, off by default
- Eight runtime dependencies, no telemetry, no auto-update

## Install

From [PyPI](https://pypi.org/project/agent6/) with
[uv](https://docs.astral.sh/uv/getting-started/installation/) or
[pipx](https://pipx.pypa.io/stable/how-to/install-pipx/):

```bash
uv tool install agent6        # or: pipx install agent6
```

agent6 needs **Linux** for the sandbox (kernel 6.7+ for TCP rules), **Python 3.12+**, and
an API key for at least one provider. macOS and Windows run unsandboxed behind a warning.
See [installation](https://agent6.dev/installation/) for the full requirements and building
from source.

## Quick start

```bash
# Connect a provider once (stored in ~/.config/agent6/, key in a 0600 secrets file).
agent6 connect                # interactive: pick provider, paste API key
agent6 model worker anthropic claude-sonnet-4-6

# Run the agent on a task. agent6 infers a verify command if you haven't set one.
cd your-repo
agent6 run "add a --json output mode to the CLI"

# Audit the effective config, pre-flight the sandbox, resume or fork a run.
agent6 config show
agent6 check
agent6 resume <run-id>
agent6 fork <run-id> --at-turn 7
```

That is the whole loop. See [getting started](https://agent6.dev/getting-started/) for the
full command tour, [configuration](https://agent6.dev/config/) for every field, and the
[security model](https://agent6.dev/security/) for what the sandbox enforces.

Config is layered: built-in secure defaults, then the global `~/.config/agent6/config.toml`,
then the per-repo config (out of the workspace, per-machine, not committed), then an
explicit `--config FILE`. Every field has a default; security-sensitive fields default to
the safe value (`agent_network = "providers"`, `tool_network = "block"`,
`run_commands = "ask"`, `protect_git = true`, `git.allow_* = false`), and `git_ops.py`
refuses `push`, `--force`, and history rewrites unconditionally.

## Benchmarks

Reproducible harnesses live under [bench/](bench/): real-world SWE-bench-Lite-style tasks,
head-to-head runs against Claude Code / opencode / aider, `machine create` validation, and
a perf-optimization harness. See each directory's README for recorded numbers (single runs,
no variance measured; re-run before quoting).

## Contributing

Read [AGENTS.md](AGENTS.md) first. The repo's verify command decides whether a change is
landable:

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

Changes under `sandbox/`, `tools/`, `git_ops.py`, `providers/`, or `graph/curator` must
include a security review note in the commit message.

## License

[Apache-2.0](LICENSE).
