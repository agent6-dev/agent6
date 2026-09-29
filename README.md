# agent6

**A coding agent that jails every command and leaves your branch untouched.**
Each command the model runs goes through a sandbox, each step commits to the run's own hidden ref, and a verify gate certifies the tree before a run may finish.

**Documentation: [agent6.dev](https://agent6.dev)**

<table>
  <tr>
    <td align="center" width="34%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-tui.gif"><img src="https://agent6.dev/screenshots/out/hero-tui.gif" alt="the run TUI: conversation streaming, an approval answered inline, verify + auto-commit, the hub receipt"></a>
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

## Install

```bash
uv tool install agent6        # or: pipx install agent6
agent6 connect                # pick a provider, paste a key; or `connect chatgpt` / `connect claude` for a subscription
cd your-repo
agent6 run "add a --json output mode to the CLI"
```

Linux (x86_64/aarch64) with Python 3.12+; other platforms run without the sandbox behind a warning.
[Installation](https://agent6.dev/installation/) covers shell completion, PATH and building from source.

## Why agent6

- **Jailed commands.** Every command the model asks to run goes through a jail that bounds what it reads, writes and reaches on the network; `auto` picks the strongest level the host allows ([Security](https://agent6.dev/security/)).
- **Your branch stays yours.** Each step commits to the run's own ref, so HEAD, the index and your branch are untouched until `sessions merge` lands the work; `resume` continues a run and `fork` branches it at any turn.
- **A verify gate.** The repo's test command, inferred when unset, certifies the tree before a run may finish, and every surface shows the same green or red.
- **One run, four front-ends.** The CLI, the TUI, a [browser](https://agent6.dev/web/) on a desktop or phone, and an [editor over ACP](https://agent6.dev/acp/) all drive the same runs; `attach` follows one live and `steer` queues an instruction from a script or a cron job.
- **State machines for long work.** `.asm.toml` files you review, edit, run, watch and replay, with waits, operator input and steering built in ([State machines](https://agent6.dev/state-machines/)).
- **Secure by default.** `network = "auto"`, `run_commands = "ask"`, `protect_git = true`; a fixed tool surface, eight runtime dependencies, no telemetry, no auto-update.

Providers: Anthropic, any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, LM Studio), a ChatGPT subscription, or the signed-in Claude Code binary, with model and reasoning effort set per role ([Configuration](https://agent6.dev/config/)).

## Learn more

[Usage](https://agent6.dev/usage/) walks through a first run, inspecting it and recovering one that went wrong; [Architecture](https://agent6.dev/architecture/) explains the loop, the task graph, compaction, memory and `--parallel` lanes.
