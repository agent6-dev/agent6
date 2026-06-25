# Configuration

agent6 is **secure by default**: every field has a default (security-sensitive
ones default to the safe value), so you only set what you want to change. This
is the field reference; for the security model behind the `[sandbox]` and `[git]`
fields see [SECURITY.md](SECURITY.md).

## Where config lives (layered, lowest precedence first)

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | — | (secure defaults, always present) |
| global *(default location)* | `$XDG_CONFIG_HOME/agent6/config.toml` (default `~/.config/agent6/config.toml`; or set `AGENT6_CONFIG_HOME`) | `agent6 connect`, `agent6 model` |
| per-repo *(override)* | `<state-dir>/<repo-id>/config.toml` | `agent6 init`, `agent6 config set --repo` |
| explicit | `--config FILE` | `agent6 run --config FILE` |

The per-repo config lives in the per-repo state dir out of the workspace
(`$XDG_STATE_HOME/agent6/<repo-id>/`; see `[agent6].state_dir` below), so it
is per-machine and not committed or shared. It can be empty -- even absent --
when the global config supplies a provider + model. `workflow.verify_command`
is optional: `agent6 run`/`plan` infer one per run when it is unset (see the
table below), so a repo needs nothing repo-specific to run.

## Creating & inspecting

- `agent6 connect`: add a provider + API key (stored `0600`), global.
- `agent6 model <role> <provider> <model> [--thinking off|low|medium|high]`.
- `agent6 init`: an OPTIONAL, granular setup wizard. Step by step it creates the
  per-repo `config.toml` (in the state dir) if missing, sets a `verify_command`
  inferred from the repo, adds `.gitignore` entries, and creates/updates
  `AGENTS.md` -- each step asks first and never overwrites your files.
- `agent6 config show`: every effective value and which layer set it.
- `agent6 config get|set|unset|add|remove <dotted.key> [value]`: edit one leaf
  (`--repo`, or `--machine FILE` for a machine `[config]` overlay). Every edit is
  re-validated and rolled back if it would produce an invalid config.
- `agent6 config fill [--repo]`: materialize every resolved value into one file.
- `agent6 check`: validate config + sandbox + provider keys without running.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version (must be `1`). |
| `state_dir` | `"$XDG_STATE_HOME/agent6"` | Absolute base path for all per-repo state, out of the workspace. Each repo gets `<state_dir>/<repo-id>/` (`<repo-id>` = `<folder>-<short hash of the repo's canonical path>`) holding `config.toml`, `runs/`, `machines/`, `memories/`. **Global-config only** (it locates the per-repo config). Override with the `AGENT6_STATE_HOME` env var. Devcontainer tip: the XDG state base is inside the container and ephemeral, so mount a volume at the state dir or set this to a persisted out-of-cwd path to keep run state across rebuilds. |

## `[providers.<name>]`

One backend per block; `<name>` is yours to pick and is referenced from
`[models.<role>]`. At least one provider is required. Three orthogonal choices
describe any backend:

- **`api_format`** — the wire dialect (the only field that selects code).
- **`deployment`** — a named profile for the URL / model-placement / version
  quirks of *where* that format is hosted.
- **auth** — `auth_style` plus a static `api_key_env` or a refreshable
  `token_command`.

So Claude-on-Vertex and Gemini-on-Vertex differ only in `api_format` (both
`deployment = "vertex"`). A minimal block is just `api_format` (plus `base_url`
for a non-default host); everything else defaults.

| Field | Default | Meaning |
|---|---|---|
| `api_format` | *(required)* | `"anthropic"` (Anthropic Messages) or `"openai"` (OpenAI Chat Completions: OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint, …). |
| `deployment` | `"direct"` | `"direct"`, `"vertex"` (Vertex AI), or `"azure"` (Azure OpenAI; `openai` only). Selects the URL shape + model/version placement. |
| `base_url` | per (format, deployment) | Endpoint host+path prefix. Defaults to the official endpoint for `direct`; **required** for vertex/azure (it carries project/resource/region). Its host is the only network destination the confined agent may dial. |
| `auth_style` | per (format, deployment) | `"x_api_key"` (Anthropic header), `"bearer"` (`Authorization: Bearer`), `"api_key_header"` (Azure's `api-key`), or `"none"` (local). Defaulted from format+deployment, so you rarely set it. |
| `api_key_env` | none | Env var holding the key. Omit to use the key stored by `agent6 connect` (secrets.toml), or for unauthenticated local endpoints. The env var (when set) takes precedence. |
| `token_command` | none | Command (argv) run to mint a short-lived bearer (printed to stdout) instead of a static key. Re-run on a TTL and once on a `401`/`403`. Takes precedence over `api_key_env`. See below. |
| `token_command_ttl_s` | `300.0` | Seconds to cache `token_command` output before re-running it. |
| `extra_headers` | `{}` | Extra HTTP headers on every request (e.g. OpenRouter's `HTTP-Referer` / `X-Title`). Not for secrets — use the auth fields. |
| `extra_body` | `{}` | Provider-specific JSON merged into every request body (load-bearing keys filtered). See below. |
| `extra_query` | `{}` | Extra URL query params (e.g. Azure's `api-version`). No secrets here. |
| `prompt_caching` | `true` | (`anthropic`) Enable Anthropic prompt caching. |
| `http_timeout_s` | `600.0` | Per-HTTP-call timeout (connect + read), seconds. Lower it to fail fast on a stuck endpoint. |

Each endpoint gets its own block, so OpenAI and OpenRouter run side-by-side
under different `<name>`s.

### Deployments

```toml
# Anthropic direct (default) — equivalent to a bare api_format = "anthropic"
[providers.anthropic]
api_format = "anthropic"

# Gemini on Vertex (OpenAI-compatible endpoint)
[providers.vertex-gemini]
api_format = "openai"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/endpoints/openapi"
token_command = ["gcloud", "auth", "print-access-token"]

# Claude on Vertex (Anthropic Messages over Vertex: model goes in the URL,
# anthropic_version in the body — handled for you by deployment = "vertex")
[providers.vertex-claude]
api_format = "anthropic"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/publishers/anthropic/models"
token_command = ["gcloud", "auth", "print-access-token"]

# Azure OpenAI (the model id IS the deployment name; api-version is required)
[providers.azure]
api_format = "openai"
deployment = "azure"
base_url = "https://RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
extra_query = { "api-version" = "2024-06-01" }
```

### OpenRouter routing & caching (`extra_body`)

OpenRouter fans a model across multiple backends whose speed and prompt
caching differ a lot. Its default routing is not deterministic, so the
re-sent system prompt may or may not be cached call-to-call. Pin the behaviour
with `extra_body.provider` ([OpenRouter routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
api_format = "openai"
base_url = "https://openrouter.ai/api/v1"
# Prefer the fastest backend; for kimi-k2.6 this lands on one that caches the
# prompt prefix, so the re-sent system prompt is near-free (cache_r in the cost
# summary jumps from ~0 to most of the input). Recommended.
extra_body = { provider = { sort = "throughput" } }
# Alternatives: pin a specific backend  { order = ["DeepInfra"], allow_fallbacks = true }
#               cap price               { max_price = { prompt = 1, completion = 2 } }
```

Set it without hand-editing: it's a table value, so pass the whole thing (the
CLI completes common presets after `extra_body`):

```bash
agent6 config set providers.openrouter.extra_body '{ provider = { sort = "throughput" } }'
```

This is the lever to pay for a faster/caching backend. Caching matters more
than payload size: the large per-call input is the same prefix every turn, so a
caching backend makes it cheap without trimming anything. Watch `cache_r` in the
run's cost summary to confirm it's engaging.

### Short-lived bearer tokens (`token_command`)

Some endpoints authenticate with a short-lived bearer that has to be refreshed
rather than a static key (Vertex's Google OAuth access token, internal OIDC/STS
gateways). Point `token_command` at any command that prints a current token to
stdout — e.g. `["gcloud", "auth", "print-access-token"]`, or for Azure AD
`["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"]`.
agent6 runs it, caches the token for `token_command_ttl_s`, re-runs it when that
elapses, and re-runs it once more on a `401`/`403` so an expired token
self-heals. It works for either `api_format` (the Deployments examples above use
it for Vertex). `token_command` takes precedence over `api_key_env`.

The command runs in agent6's own process (outside any run sandbox) with your
environment, the same trust level as an `[[mcp.servers]]` command, so it is an
operator-only knob. Whatever it prints to stdout is sent as the auth header
(`Authorization: Bearer <token>` for `auth_style = "bearer"`); a non-zero exit,
a timeout, or empty output surfaces as a
provider error.

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
[SECURITY.md](SECURITY.md) (§3 profiles, §1b/§8 network); this is a summary.

| Field | Default | Meaning |
|---|---|---|
| `profile` | `"auto"` | `auto` / `strict` / `hardened` → effective `strict` / `hardened` / `none`; or explicit `none` to run UNSANDBOXED, allowed only inside a detected container (else refused on a bare host unless `AGENT6_ALLOW_NO_SANDBOX=1`). See SECURITY §3. |
| `agent_network` | `"providers"` | The agent's own egress: `providers` / `local` / `open` (SECURITY §1b). |
| `tool_network` | `"block"` | Jailed-command egress: `block` / `only_explicit_states` / `allow` (SECURITY §8). |
| `allow_urls` | `[]` | Extra agent egress hosts under `agent_network = "providers"` (`host`, `host:port`, or URL). Edit with `agent6 config add/remove sandbox.allow_urls <host>`. |
| `run_commands` | `"ask"` | Whether the LLM gets `run_command`: `yes` / `no` / `ask`. |
| `protect_git` | `true` | Strict only: re-bind `.git/` read-only in the jail. On the hardened profile the cwd is blanket read-write (no mount namespace to carve), so `.git` is writable by jailed commands there; that is gated by `run_commands`, recoverable (branch-per-run, commits through git_ops), and bounded by the surrounding container. |
| `extra_read_paths` | `[]` | Extra absolute paths a jailed command may **read + execute** (not write), beyond the system defaults (`/usr /bin /lib …`) and the workspace. For a project whose toolchain/interpreter lives outside the repo — a system conda/virtualenv, a Go/Rust/Node toolchain, a shared data dir. Granted under `hardened` **and** `strict`. Loosens confinement (the child can read more of the host), so list only what the build/test needs. |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `require_clean_worktree` | `true` | Refuse to start on a dirty worktree. |
| `auto_stash` | `false` | Stash before, restore after the run. |
| `branch_per_run` | `true` | Cut a fresh `agent6/<slug>` branch off HEAD (else stay on the current branch and remember the starting sha). |
| `commit_strategy` | `"per_step"` | End-of-run finalization: `per_step` (keep N commits) / `squash` (one combined commit) / `stage` (leave staged) / `none` (leave unstaged). All strategies commit per-step *during* the run. |
| `run_repo_hooks` | `false` | Whether the repo's own `.git/hooks/*` run during agent6's git ops (notably the per-step auto-commit). Default off: a repo hook is repo-controlled code that runs on the host (outside the jail), so honoring it on agent6's commit is a host-RCE vector for an untrusted repo, and the `verify_command` is agent6's real success gate. Set true to honor the repo's hooks (you trust the repo). `core.fsmonitor`/`diff.external` are always neutralized regardless. |
| `allow_push` / `allow_force` / `allow_history_rewrite` | `false` | Reserved. `git_ops.py` refuses push / `--force` / history rewrite / `reset --hard` unconditionally regardless of these (SECURITY §5). |

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` / `email` | none | Override the commit identity (else use the project's `git config`). `agent6 run` refuses to start with no resolvable identity. |
| `coauthor` | none | Append a `Co-authored-by:` trailer, e.g. `"Alice <alice@example.com>"`. |

## `[workflow]`

| Field | Default | Meaning |
|---|---|---|
| `verify_command` | `[]` | argv defining "a step succeeded" (run with NO shell — wrap a pipeline as `["sh","-c","a && b"]`). **Optional**: when unset, `agent6 run`/`plan` infer one per run — AGENTS.md `## Verify command` section → repo manifests (package.json/Makefile/pyproject/Cargo/go.mod) → a cheap model call — inject it in-memory (never written to config) and print it. With none inferable, the run is *gateless* (per-step commits, no green gate). Set it to pin a deterministic one. |
| `verify_timeout_s` | `600.0` | Per-call timeout for `verify_command` / `metric.command`. |
| `critic` | `"off"` | Trigger for the in-loop **adversarial review panel** (below): `off` / `on_verify_fail` / `before_finish` / `periodic`. |
| `critic_period` | `10` | Iterations between reviews when `critic = "periodic"`. |
| `review_panel_size` | `1` | Number of reviewer seats (personas cycled). |
| `review_personas` | `[]` | Per-seat stances, e.g. `["security","correctness","tests"]`; empty = a built-in set. |
| `review_decision` | `"advisory"` | `advisory` (inject findings, never blocks) / `veto` / `quorum` / `all`. Only gates in-loop. |
| `review_quorum` | `2` | K for `quorum` (counts **distinct models**, so same-model seats can't fake a quorum). |
| `review_tier` | `"diff"` | `diff` (one grounded call over the diff) or `explore` (read-only tool-using reviewer that reads the broader repo to catch cross-file impact). |
| `review_concurrency` | `1` | In-loop seat parallelism (post-hoc `agent6 review` is always parallel). |
| `review_max_total_rejections` | `4` | Per-run blocks before the gate auto-disarms to advisory (anti-stall). |
| `review_budget_fraction` | `0.25` | Max run-budget fraction the panel may spend. |
| `review_seats` | `[]` | Explicit `"persona@provider/model"` seats → **distinct models per seat** (overrides size/personas). A bare `"persona"` routes via `[models.reviewer]`. |
| `revise_prompt` | `"off"` | One-shot task-prompt revision before the loop: `off` / `auto` / `interactive`. |
| `profile` | `""` | Named **config profile** (below). The `--profile` CLI flag overrides it. |
| `compact_drop_at_chars` | _adaptive_ | Tier-1 compaction: replace oldest tool-results with a placeholder. Default (unset) sizes from the worker model's context window (~45% of it); set BOTH `compact_*` to pin. |
| `compact_summarise_at_chars` | _adaptive_ | Tier-2 compaction: summarise elided history + restart (the task DAG survives). Default (unset) ~80% of the model's context window; the historical 256k/768k apply when the window is unknown. |
| `context_summary_max_tokens` | `2048` | Cap on the tier-2 summary. |
| `structural_priors` | `true` | Include the enriched `<repo-priors>` block (ranked hot symbols, git co-change, tree-sitter outline) in the system prompt. Set `false` for a leaner/cheaper prompt. |
| `system_prompt_file` | `""` | ADVANCED: replace run-mode's static base prompt with this file's contents (the dynamic verify / budget / repo-priors blocks still append). Validated to exist at config load; a startup warning fires if it omits the core tool names. |

### Adversarial review panel

When `critic != "off"`, the in-loop second opinion is a **grounded review panel**:
`review_panel_size` reviewers (the `reviewer` model, or distinct models via
`review_seats`) each scrutinize the run's diff (+ the last verify result) and
return structured findings. The same panel runs post-hoc, read-only, via
`agent6 review --reviewers N [--personas a,b,c]`.

What keeps it from false-blocking correct work (the trap that retired the old
in-loop reviewer): grounding is **mechanical, not prose**. A reviewer's `block`
only gates if its `file:line` is actually in the diff AND its category is in a
fixed allowed-block set (security / sandbox-bypass / off-topic-edit / data-loss /
verify-uncovered-correctness). Taste, naming, "missing test", and uncited claims
are downgraded to advisory and can never stall the run; a per-run rejection cap
disarms the gate after a few blocks. `advisory` (the default) only injects
findings as guidance — enable `veto`/`quorum` to gate.

### Config profiles

A profile presets many settings at once so a task picks a strategy with one knob.
Select with `--profile <name>` (on `run`/`plan`/`ask`), `[workflow] profile =
"<name>"`, or the TUI new-work chooser. A profile **overrides** config rather than
being a baseline: its preset is injected just above the config layer that selected
it, so precedence (low → high) is

    defaults < global config < [profile via global workflow.profile]
            < repo config < [profile via repo workflow.profile]
            < [profile via --profile flag] < --config FILE

The most-specific source wins (`--profile` flag, else repo `[workflow].profile`,
else global; the presets never stack), and the profile beats the config at its
scope — but a more-specific config layer, an explicit `--config FILE`, or an
individual flag still beats the profile.

| Profile | Bundles |
|---|---|
| `quick` | review off, tighter output budget — fast/cheap. |
| `standard` | the plain defaults (no review). The default. |
| `ultra` | a 3-seat grounded quorum panel — thorough review. |
| `paranoid` | 5 explore-tier seats, `before_finish` veto, bigger budget. |

Define your own with a `[profiles.<name>]` table (a partial config); it wins over
a built-in of the same name. Example:

```toml
[profiles.myteam.workflow]
critic = "before_finish"
review_panel_size = 3
review_decision = "veto"
review_seats = ["security@anthropic/claude-opus-4-8", "correctness@openrouter/moonshotai/kimi-k2"]
```

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
| `max_input_tokens` | `2000000` | Input-token ceiling. Exact and always enforced. |
| `max_output_tokens` | `200000` | Output-token ceiling. Exact and always enforced. |
| `best_effort_usd_limit` | `0.0` (off) | Dollar-denominated bound, enforced where price data exists. |

Token ceilings are the authoritative constraint. When the worker model's
price is cached, `best_effort_usd_limit` converts to token ceilings at load
(the lower wins per axis); at runtime the run also stops when estimated
spend (reported cost, else price times tokens, cache included) crosses it.
With no price and no reported cost it does nothing, hence best effort.

Override per-run from the CLI without editing config: `agent6 run --max-usd 5`,
`--max-input-tokens`, `--max-output-tokens` (on `run`, `plan`, `resume`).
Passing `--max-usd` explicitly refuses to start when the worker model has no
price data, since the flag could not be honored.

## `[machine]`

State-machine runtime knobs (`agent6 machine run`).

| Field | Default | Meaning |
|---|---|---|
| `snapshot_keep` | `5` | Recent blackboard snapshots retained per machine instance. Recovery reads only the latest and `machine replay` rebuilds from the journal, so older snapshots are an audit convenience. `0` keeps every snapshot (one file per transition; budget disk for long-running machines). |

## `[notify]` (optional)

Runs an operator-controlled argv after `agent6 run` / `resume` (success or
failure), outside the jail as your user, with `AGENT6_RUN_ID`,
`AGENT6_RUN_DIR`, `AGENT6_RUN_OK` (`1`/`0`), `AGENT6_RUN_REASON` set.

| Field | Default | Meaning |
|---|---|---|
| `on_complete` | `[]` | argv to run (empty = disabled). |
| `timeout_s` | `30.0` | Hook timeout. |

## `[mcp]` + `[[mcp.servers]]` (optional)

Spawn Model Context Protocol servers at run start; their tools appear to the LLM
as `mcp__<name>__<tool>`. Each server runs as your user, outside the jail
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
