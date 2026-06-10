# Configuration

agent6 is **secure by default**: every field has a default (security-sensitive
ones default to the safe value), so you only set what you want to change. This
is the field reference; for the security model behind the `[sandbox]` and `[git]`
fields see **[SECURITY.md](SECURITY.md)**.

## Where config lives (layered, lowest precedence first)

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | — | (secure defaults, always present) |
| global *(default location)* | `$XDG_CONFIG_HOME/agent6/config.toml` (default `~/.config/agent6/config.toml`; or set `AGENT6_CONFIG_HOME`) | `agent6 connect`, `agent6 model` |
| per-repo *(override)* | `./.agent6/config.toml` | `agent6 init`, `agent6 config set --repo` |
| explicit | `--config FILE` | `agent6 run --config FILE` |

A per-repo config can be empty when the global config supplies a provider +
model; the one thing a repo always needs is its `workflow.verify_command`.

## Creating & inspecting

- `agent6 connect` — add a provider + API key (stored `0600`), global.
- `agent6 model <role> <provider> <model> [--thinking off|low|medium|high]`.
- `agent6 init` — scaffold `.agent6/config.toml` + `AGENTS.md` in a repo.
- `agent6 config show` — every effective value and which layer set it.
- `agent6 config get|set|unset|add|remove <dotted.key> [value]` — edit one leaf
  (`--repo`, or `--machine FILE` for a machine `[config]` overlay). Every edit is
  re-validated and rolled back if it would produce an invalid config.
- `agent6 config fill [--repo]` — materialize every resolved value into one file.
- `agent6 check` — validate config + sandbox + provider keys without running.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version (must be `1`). |
| `workspace_subdir` | `".agent6"` | In-repo directory for config + run state (`config.toml`, `runs/`, `machines/`, `memories/`). **Global-config only** (a repo can't rename the directory it lives in); a bare name — no slashes, `..`, or absolute paths. |

## `[providers.<name>]`

One backend per block; `<name>` is yours to pick and is referenced from
`[models.<role>]`. At least one provider is required.

| Field | Default | `kind` | Meaning |
|---|---|---|---|
| `kind` | *(required)* | — | `"anthropic"` (Anthropic Messages) or `"openai"` (any OpenAI Chat-Completions-compatible endpoint: OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp). |
| `api_key_env` | none | both | Env var holding the API key. Omit to use the key stored by `agent6 connect`, or for unauthenticated local endpoints. The env var (when set) takes precedence. |
| `http_timeout_s` | `600.0` | both | Per-HTTP-call timeout (connect + read), seconds. Lower it to fail fast on a stuck endpoint. |
| `prompt_caching` | `true` | anthropic | Enable Anthropic prompt caching. |
| `base_url` | `https://api.openai.com/v1` | openai | Endpoint base URL (e.g. `https://openrouter.ai/api/v1`, `http://localhost:11434/v1`). |
| `extra_headers` | `{}` | openai | Extra HTTP headers sent on every request (e.g. OpenRouter's `HTTP-Referer` / `X-Title`). |
| `extra_body` | `{}` | openai | Provider-specific JSON merged into every request body (keys override computed fields). See below. |

Each endpoint gets its own block, so OpenAI and OpenRouter run side-by-side
under different `<name>`s.

### OpenRouter routing & caching (`extra_body`)

OpenRouter fans a model across multiple backends whose **speed and prompt
caching differ a lot** — and its default routing is not deterministic, so the
re-sent system prompt may or may not be cached call-to-call. Pin the behaviour
with `extra_body.provider` ([OpenRouter routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
kind = "openai"
base_url = "https://openrouter.ai/api/v1"
# Prefer the fastest backend; for kimi-k2.6 this lands on one that caches the
# prompt prefix, so the re-sent system prompt is near-free (cache_r in the cost
# summary jumps from ~0 to most of the input). Recommended.
extra_body = { provider = { sort = "throughput" } }
# Alternatives: pin a specific backend  { order = ["DeepInfra"], allow_fallbacks = true }
#               cap price               { max_price = { prompt = 1, completion = 2 } }
```

This is the lever to **pay for a faster/caching backend**. Caching matters more
than payload size: the large per-call input is the same prefix every turn, so a
caching backend makes it cheap without trimming anything. Watch `cache_r` in the
run's cost summary to confirm it's engaging.

## `[models.<role>]`

Role routing. Three roles, all optional: **`worker`** drives `agent6 run` /
`resume` (its model's pricing also drives the USD→token budget conversion);
**`planner`** drives `agent6 plan`; **`reviewer`** drives `agent6 review` + the
in-loop critic. `planner` and `reviewer` fall back to `worker` when unset. Any
provider may serve any role (cross-vendor mixes are fine).

| Field | Default | Meaning |
|---|---|---|
| `provider` | *(required)* | A `[providers.*]` name. |
| `model` | *(required)* | Model id at that provider. |
| `temperature` | `0.0` | Sampling temperature pinned per call (`0.0`–`2.0`). `0.0` keeps the tool-use loop stable; some open-weights models degenerate at high temperature. |
| `thinking` | none | Reasoning effort: `off` / `low` / `medium` / `high`. OpenAI-compatible reasoning models get a reasoning-effort knob; Anthropic maps it onto an extended-thinking budget (low/medium/high ≈ 4k/8k/16k tokens). Non-reasoning models ignore it. |

## `[sandbox]`

The security boundary. Profiles and the network model are specified in
**[SECURITY.md](SECURITY.md)** (§3 profiles, §1b/§8 network); this is a summary.

| Field | Default | Meaning |
|---|---|---|
| `profile` | `"auto"` | `auto` / `strict` / `hardened` → resolves to an effective `strict` / `hardened` / `none` (SECURITY §3). |
| `agent_network` | `"providers"` | The agent's own egress: `providers` / `local` / `open` (SECURITY §1b). |
| `tool_network` | `"block"` | Jailed-command egress: `block` / `only_explicit_states` / `allow` (SECURITY §8). |
| `allow_urls` | `[]` | Extra agent egress hosts under `agent_network = "providers"` (`host`, `host:port`, or URL). Edit with `agent6 config add/remove sandbox.allow_urls <host>`. |
| `run_commands` | `"ask"` | Whether the LLM gets `run_command`: `yes` / `no` / `ask`. |
| `protect_git` | `true` | Re-bind `.git/` read-only in every jail. |
| `protect_agent6` | `true` | Re-bind the `.agent6/` directory (config + run state) read-only in every jail. |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `require_clean_worktree` | `true` | Refuse to start on a dirty worktree. |
| `auto_stash` | `false` | Stash before, restore after the run. |
| `branch_per_run` | `true` | Cut a fresh `agent6/<slug>` branch off HEAD (else stay on the current branch and remember the starting sha). |
| `commit_strategy` | `"per_step"` | End-of-run finalization: `per_step` (keep N commits) / `squash` (one combined commit) / `stage` (leave staged) / `none` (leave unstaged). All strategies commit per-step *during* the run. |
| `run_repo_hooks` | `false` | Whether the repo's own `.git/hooks/*` run during agent6's git ops (notably the per-step auto-commit). Default off: a repo hook is repo-controlled code that runs on the **host** (outside the jail), so honoring it on agent6's commit is a host-RCE vector for an untrusted repo — and the `verify_command` is agent6's real success gate. Set true to honor the repo's hooks (you trust the repo). `core.fsmonitor`/`diff.external` are always neutralized regardless. |
| `allow_push` / `allow_force` / `allow_history_rewrite` | `false` | Reserved. `git_ops.py` refuses push / `--force` / history rewrite / `reset --hard` **unconditionally** regardless of these (SECURITY §5). |

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` / `email` | none | Override the commit identity (else use the project's `git config`). `agent6 run` refuses to start with no resolvable identity. |
| `coauthor` | none | Append a `Co-authored-by:` trailer, e.g. `"Alice <alice@example.com>"`. |

## `[workflow]`

| Field | Default | Meaning |
|---|---|---|
| `verify_command` | `[]` | argv defining "a step succeeded". **Required for `agent6 run`** (checked at run time); `plan` / `review` don't need it. |
| `verify_timeout_s` | `600.0` | Per-call timeout for `verify_command` / `metric.command`. |
| `critic` | `"off"` | In-loop critic (runs the `reviewer` model): `off` / `on_verify_fail` / `before_finish` / `periodic`. |
| `critic_period` | `10` | Iterations between critiques when `critic = "periodic"`. |
| `revise_prompt` | `"off"` | One-shot task-prompt revision before the loop: `off` / `auto` / `interactive`. |
| `compact_drop_at_chars` | `256000` | Tier-1 compaction: replace oldest tool-results with a placeholder. |
| `compact_summarise_at_chars` | `768000` | Tier-2 compaction: summarise elided history + restart (the task DAG survives). |
| `context_summary_max_tokens` | `2048` | Cap on the tier-2 summary. |

### `[workflow.metric]` (optional)

A continuous score for tasks with a measurable goal. `command` runs in the jail
like `verify_command`; `pattern`'s first capture group is parsed as a number.

| Field | Default | Meaning |
|---|---|---|
| `command` | *(required)* | argv to run for the metric. |
| `pattern` | *(required)* | Regex; first capture group = the numeric metric. |
| `goal` | *(required)* | `"minimize"` or `"maximize"`. |

## `[budget]`

Hard stops; on hit the run aborts (exit 3) and is resumable (raise the limit and
`agent6 resume <run-id>`).

| Field | Default | Meaning |
|---|---|---|
| `max_input_tokens` | `2000000` | Input-token ceiling. |
| `max_output_tokens` | `200000` | Output-token ceiling. |
| `max_usd` | `0.0` (off) | Optional USD ceiling; converted to token caps at load (worker-model pricing) **and** enforced at runtime as an exact dollar ceiling that includes cache_read/cache_creation cost (which the token caps omit). |

Override per-run from the CLI without editing config: `agent6 run --max-usd 5`,
`--max-input-tokens`, `--max-output-tokens` (on `run`, `plan`, `resume`).

## `[notify]` (optional)

Runs an operator-controlled argv after `agent6 run` / `resume` (success or
failure), **outside the jail** as your user, with `AGENT6_RUN_ID`,
`AGENT6_RUN_DIR`, `AGENT6_RUN_OK` (`1`/`0`), `AGENT6_RUN_REASON` set.

| Field | Default | Meaning |
|---|---|---|
| `on_complete` | `[]` | argv to run (empty = disabled). |
| `timeout_s` | `30.0` | Hook timeout. |

## `[mcp]` + `[[mcp.servers]]` (optional)

Spawn Model Context Protocol servers at run start; their tools appear to the LLM
as `mcp__<name>__<tool>`. Each server runs **as your user, outside the jail**
(the `command` is operator-controlled, never LLM-influenced); the LLM *can*
influence the arguments it passes, so audit each server like a `run_command`
allow-list.

| Field | Default | Meaning |
|---|---|---|
| `mcp.enabled` | `false` | Master switch; `false` means zero `mcp__*` tools. |
| `servers[].name` | *(required)* | Tool prefix (`mcp__<name>__<tool>`). |
| `servers[].command` | *(required)* | argv for the stdio JSON-RPC server. |
| `servers[].enabled` | `true` | Per-server toggle. |
| `servers[].startup_timeout_s` | `10.0` | `initialize` + `tools/list` handshake budget. |
| `servers[].call_timeout_s` | `60.0` | Per `tools/call` timeout. |

---

## Environment variables

| Variable | Effect |
|---|---|
| `AGENT6_CONFIG_HOME` | Override the global config directory (default `$XDG_CONFIG_HOME/agent6`). |
| `AGENT6_CACHE_HOME` | Override the cache directory (model-list cache, etc.). |
| `AGENT6_JAIL_BIN` | Path to a specific `agent6-jail` binary (else the bundled one). |
| `AGENT6_ALLOW_ROOT` | `1` permits running as root (same as `--allow-root`). |

A provider's `api_key_env` (default `ANTHROPIC_API_KEY` etc.) supplies its key.
A few additional `AGENT6_*` toggles exist for testing/advanced use; see the
source if you need them.
