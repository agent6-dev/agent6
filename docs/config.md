# Configuration

agent6 is **secure by default**: every field has a default (security-sensitive
ones default to the safe value), so you only set what you want to change. This
is the field reference; for the security model behind the `[sandbox]` and `[git]`
fields see [security.md](security.md).

## Where config lives (layered, lowest precedence first)

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | (none) | (secure defaults, always present) |
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
  (`--repo`, or `--machine-file FILE` for a machine `[config]` overlay). Every edit is
  re-validated and rolled back if it would produce an invalid config.
- `agent6 config fill [--repo]`: materialize every resolved value into one file.
- `agent6 config fix`: drop invalid entries (unknown keys, stale values left by a
  schema change) from the global and repo config, printing each and whether it was
  global or repo. Use `--machine-file FILE` to repair a machine `[config]` overlay
  instead. Removals apply immediately (no dry-run); an entry it cannot drop as a
  plain leaf (a non-absolute `state_dir`) is reported, not silently left.
- `agent6 check`: validate config + sandbox + provider keys without running.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version (must be `1`). |
| `state_dir` | `"$XDG_STATE_HOME/agent6"` | Absolute base path for all per-repo state, out of the workspace. Each repo gets `<state_dir>/<repo-id>/` (`<repo-id>` = `<folder>-<short hash of the repo's canonical path>`) holding `config.toml`, `runs/`, `machines/`, `memories/`, and `lineage.jsonl` (the fork forest: one `{child,parent,turn,sha}` edge per line). **Global-config only** (it locates the per-repo config). Override with the `AGENT6_STATE_HOME` env var. Devcontainer tip: the XDG state base is inside the container and ephemeral, so mount a volume at the state dir or set this to a persisted out-of-cwd path to keep run state across rebuilds. |

## `[providers.<name>]`

One backend per block; `<name>` is yours to pick and is referenced from
`[models.<role>]`. At least one provider is required. Three orthogonal choices
describe any backend:

- **`api_format`**: the wire dialect (the only field that selects code).
- **`deployment`**: a named profile for the URL / model-placement / version
  quirks of *where* that format is hosted.
- **auth**: `auth_style` plus a static `api_key_env` or a refreshable
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
| `extra_headers` | `{}` | Extra HTTP headers on every request (e.g. OpenRouter's `HTTP-Referer` / `X-Title`). Not for secrets; use the auth fields. |
| `extra_body` | `{}` | Provider-specific JSON merged into every request body (load-bearing keys filtered). See below. |
| `extra_query` | `{}` | Extra URL query params (e.g. Azure's `api-version`). No secrets here. |
| `prompt_caching` | `true` | (`anthropic`) Enable Anthropic prompt caching: the system prompt, tool list, and (via rolling breakpoints the loop advances each turn) the growing conversation are all re-read from cache at 0.1x input price. |
| `http_timeout_s` | `600.0` | Per-HTTP-call timeout (connect + read), seconds. Lower it to fail fast on a stuck endpoint. |

Each endpoint gets its own block, so OpenAI and OpenRouter run side-by-side
under different `<name>`s.

### Deployments

```toml
# Anthropic direct (default): equivalent to a bare api_format = "anthropic"
[providers.anthropic]
api_format = "anthropic"

# Gemini on Vertex (OpenAI-compatible endpoint)
[providers.vertex-gemini]
api_format = "openai"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/endpoints/openapi"
token_command = ["gcloud", "auth", "print-access-token"]

# Claude on Vertex (Anthropic Messages over Vertex: model goes in the URL,
# anthropic_version in the body, handled for you by deployment = "vertex")
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
stdout, e.g. `["gcloud", "auth", "print-access-token"]`, or for Azure AD
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
[security.md](security.md) (§3 profiles, §1b/§8 network); this is a summary.

| Field | Default | Meaning |
|---|---|---|
| `profile` | `"auto"` | `auto` picks the strongest profile the host supports (`strict`, else `hardened`; never `none` on Linux); explicit `strict`/`hardened` are refused where unsupported, never downgraded. Explicit `none` runs UNSANDBOXED (self-authorizing, loud warning); the per-invocation forms are `--dangerously-disable-sandbox` / `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`. See SECURITY §3. |
| `agent_network` | `"providers"` | The agent's own egress: `providers` / `local` / `open` (SECURITY §1b). |
| `tool_network` | `"block"` | Jailed-command egress: `block` / `only_explicit_states` / `allow` (SECURITY §8). |
| `allow_urls` | `[]` | Extra agent egress hosts under `agent_network = "providers"` (`host`, `host:port`, or URL). Edit with `agent6 config add/remove sandbox.allow_urls <host>`. |
| `run_commands` | `"ask"` | Whether the LLM gets `run_command`: `yes` (auto-approve; also `--auto-approve`) / `no` (withheld) / `ask` (prompt each call). `yes` skips approval; confinement still depends on `sandbox.profile`. |
| `protect_git` | `true` | Strict only: re-bind `.git/` read-only in the jail. On the hardened profile the cwd is blanket read-write (no mount namespace to carve), so `.git` is writable by jailed commands there; that is gated by `run_commands`, recoverable (branch-per-run, commits through git_ops), and bounded by the surrounding container. |
| `extra_read_paths` | `[]` | Extra absolute paths a jailed command may **read + execute** (not write), beyond the system defaults (`/usr /bin /lib …`) and the workspace. Use it for a project whose toolchain/interpreter lives outside the repo: a system conda/virtualenv, a Go/Rust/Node toolchain, a shared data dir. Granted under `hardened` **and** `strict`. Loosens confinement (the child can read more of the host), so list only what the build/test needs. |
| `memory_limit_mb` | `4096` | Per-process memory cap (MiB) on every jailed child, applied as `RLIMIT_DATA` and inherited by the child's descendants. A runaway allocation fails with ENOMEM (Python `MemoryError`) that the agent handles as an ordinary failed command, instead of driving the host to the OOM killer. `0` disables. Per process, not per tree; no effect under profile `none`. Raise it when a legitimate build/test needs more than 4 GiB in one process. |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `require_clean_worktree` | `true` | Refuse to start on a dirty worktree. |
| `auto_stash` | `false` | Stash uncommitted changes before the run. Restored at run end per `auto_stash_pop`; otherwise agent6 prints how to `git stash pop` them (never silently left). |
| `auto_stash_pop` | `false` | When `auto_stash` stashed changes, pop them back at run end if it is safe: clean worktree and a conflict-free apply, switching back to the base branch first under `branch_per_run`. On any conflict or doubt, leave the stash and print how to restore it. Never `reset --hard`. |
| `branch_per_run` | `true` | Cut a fresh `agent6/<slug>` branch off HEAD (else stay on the current branch and remember the starting sha). |
| `branch_from` | `"current"` | Where the run branch is cut from when you are **not** on the base branch (e.g. still on a previous run's `agent6/*` branch): `current` cuts from HEAD, stacking on it (serial runs pile up); `base` cuts from the base line (the nearest non-run branch this branch descends from), so each run starts clean; `ask` prompts (base / stack / abort), non-interactive falling back to `base`. No effect when you are already on a base branch. |
| `merge_strategy` | `"squash"` | Default strategy for `agent6 runs merge`: `squash` (one combined commit), `merge` (a --no-ff merge keeping the per-step history), or `ff` (fast-forward only). The per-step commits always happen on the run branch during the run; this only governs how they consolidate onto your branch. |
| `auto_merge` | `false` | After a successful run, run `merge_strategy` automatically to land the run branch on its base (what `agent6 runs merge` does, for you). Requires `branch_per_run` (config refuses the pair otherwise: without a run branch there is nothing to merge). On conflict the run branch is left intact with instructions. With `auto_stash_pop`, the merge lands first, then your stashed pre-run changes. |
| `auto_prune` | `false` | After `auto_merge`, delete the run branch when `git branch -d` can (reachable-merged, i.e. `merge`/`ff` strategies). A squash-merged branch is unreachable, so it is reported with the `git branch -D` to remove it by hand, never force-deleted. Requires `auto_merge` (config refuses it otherwise). With both on, run branches stop accumulating. |
| `run_repo_hooks` | `false` | Whether the repo's own `.git/hooks/*` run during agent6's git ops (notably the per-step auto-commit). Default off: a repo hook is repo-controlled code that runs on the host (outside the jail), so honoring it on agent6's commit is a host-RCE vector for an untrusted repo, and the `verify_command` is agent6's real success gate. Set true to honor the repo's hooks (you trust the repo). `core.fsmonitor`/`diff.external` are always neutralized regardless. |
| `allow_push` / `allow_force` / `allow_history_rewrite` | `false` | Reserved. `git_ops.py` refuses push / `--force` / history rewrite / `reset --hard` unconditionally regardless of these (SECURITY §5). |

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` / `email` | none | Override the commit identity (else use the project's `git config`). `agent6 run` refuses to start with no resolvable identity. |
| `coauthor` | none | Append a `Co-authored-by:` trailer, e.g. `"Alice <alice@example.com>"`. |

## `profile` (top-level)

| Field | Default | Meaning |
|---|---|---|
| `profile` | `""` | Named **config profile** (see [Config profiles](#config-profiles)). A bare top-level key (not inside any section) because it overrides every section. The `--profile` CLI flag overrides it. |

## `[workflow]`

| Field | Default | Meaning |
|---|---|---|
| `verify_command` | `[]` | argv defining "a step succeeded" (run with no shell; wrap a pipeline as `["sh","-c","a && b"]`). **Optional**: when unset, `agent6 run`/`plan` infer one per run (AGENTS.md `## Verify command` section → repo manifests (package.json/Makefile/pyproject/Cargo/go.mod) → a cheap model call), inject it in-memory (never written to config), and print it. With none inferable, the run starts *gateless* (per-step commits, no green gate); if it then creates a recognizable project, the deterministic tiers re-run at each commit and the first hit whose runner resolves on the jail PATH is adopted for the rest of the run. Set it to pin a deterministic one. |
| `verify_timeout_s` | `600.0` | Per-call timeout for `verify_command` / `metric.command`. |
| `require_verify_to_finish` | `false` | When true, `finish_run` is refused while the last verify is red (or a verify command is configured but never run): the worker must get verify green or explicitly stop. Bounded (a few nudges, then honoured). Independent of this flag, a finish over a red/stale verify is always reported honestly (`run.end all_passed=false` -> "finished", never "passed"). |
| `spec_recheck_on_finish` | `false` | Bounce the FIRST `finish_run` over a green verify once, directing a re-check of every spec requirement. Measured A/B (bench/coreagent eventflow + textkit, n=6/arm, 3 models): no score gain beyond noise anywhere, a score DROP on mistral-small, and +38-88% cost from the extra verification loop the bounce triggers. Injecting a full debugging-methodology skill helps where this generic bounce does not; prefer that. Kept off; candidate for removal. |

## `[review]`

The in-loop critic trigger plus the adversarial review panel (below).

| Field | Default | Meaning |
|---|---|---|
| `trigger` | `"off"` | Trigger for the in-loop **adversarial review panel**: `off` / `on_verify_fail` / `before_finish` / `periodic`. |
| `period` | `10` | Iterations between reviews when `trigger = "periodic"`. |
| `panel_size` | `1` | Number of reviewer seats (personas cycled). |
| `personas` | `[]` | Per-seat stances, e.g. `["security","correctness","tests"]`; empty = a built-in set. |
| `decision` | `"advisory"` | `advisory` (inject findings, never blocks) / `veto` / `quorum` / `all`. Only gates in-loop. |
| `quorum` | `2` | K for `quorum` (counts **distinct models**, so same-model seats can't fake a quorum). |
| `tier` | `"diff"` | `diff` (one grounded call over the diff) or `explore` (read-only tool-using reviewer that reads the broader repo to catch cross-file impact). |
| `concurrency` | `1` | In-loop seat parallelism (post-hoc `agent6 review` is always parallel). |
| `max_total_rejections` | `4` | Per-run blocks before the gate auto-disarms to advisory (anti-stall). |
| `budget_fraction` | `0.25` | Max run-budget fraction the panel may spend. |
| `seats` | `[]` | Explicit `"persona@provider/model"` seats → **distinct models per seat** (overrides size/personas). A bare `"persona"` routes via `[models.reviewer]`. |

### Adversarial review panel

When `trigger != "off"`, the in-loop second opinion is a **grounded review panel**:
`panel_size` reviewers (the `reviewer` model, or distinct models via
`seats`) each scrutinize the run's diff (+ the last verify result) and
return structured findings. The same panel runs post-hoc, read-only, via
`agent6 review --reviewers N [--personas a,b,c]`.

What keeps it from false-blocking correct work (the trap that retired the old
in-loop reviewer): grounding is **mechanical, not prose**. A reviewer's `block`
only gates if its `file:line` is actually in the diff AND its category is in a
fixed allowed-block set (security / sandbox-bypass / off-topic-edit / data-loss /
verify-uncovered-correctness). Taste, naming, "missing test", and uncited claims
are downgraded to advisory and can never stall the run; a per-run rejection cap
disarms the gate after a few blocks. `advisory` (the default) only injects
findings as guidance. Enable `veto`/`quorum` to gate.

## `[context]`

Tiered context-compaction thresholds (approximate chars; tokens ≈ chars/4).

| Field | Default | Meaning |
|---|---|---|
| `drop_at_chars` | _adaptive_ | Tier-1 compaction: replace oldest tool-results with a placeholder. Default (unset) sizes from the worker model's context window (~45% of it); set BOTH thresholds to pin. |
| `summarise_at_chars` | _adaptive_ | Tier-2 compaction: summarise elided history + restart (the task DAG survives). Default (unset) ~80% of the model's context window; the historical 256k/768k apply when the window is unknown. Must be greater than `drop_at_chars`. |
| `summary_max_tokens` | `2048` | Cap on the tier-2 summary (also caps a gist distillation call). |
| `elision_gists` | `true` | Tier-1 decays a large `read_file` result to a placeholder carrying a model-written gist of the file (one batched reviewer-model call per drop event) before the bare marker; under continued pressure gists demote to bare so the byte bound holds. `false` = straight to bare markers, no distiller calls. |

## `[prompt]`

| Field | Default | Meaning |
|---|---|---|
| `system_prompt_file` | `""` | ADVANCED: replace run-mode's static base prompt with this file's contents (the dynamic verify / budget / repo-priors blocks still append). Validated to exist at config load; a startup warning fires if it omits the core tool names. |
| `structural_priors` | `true` | Include the enriched `<repo-priors>` block (ranked hot symbols, git co-change, tree-sitter outline) in the system prompt. Set `false` for a leaner/cheaper prompt. |
| `revise_prompt` | `"off"` | One-shot task-prompt revision before the loop: `off` / `auto` / `interactive`. |
| `decompose` | `"auto"` | Front-load task decomposition (run mode): `"auto"` \| `"on"` \| `"off"`. When on, swaps the "DAG is optional" guidance for a "decompose first" directive: the worker lays the task out as ordered subtasks before editing, then the surface-current-task + finish-gate machinery walks it one focused subtask at a time. Helps a small model that under-finishes multi-component implementation tasks (measured: mistral-small-3.2-24b textkit +0.53, rpn +0.13 score; flat on a debug task, where dropping-components isn't the failure mode); a capable model decomposes implicitly and only pays the overhead (~2-4x turns/cost). `"auto"` (default) resolves per worker model from the capability registry (`models/registry.py`): on only for model families with a measured win, off for everything else; `agent6 config show` displays the resolved value. `--decompose` on `agent6 run` forces it on for one run. No effect on plan/ask/machine/agent modes. |

## `[skills]`

Operator-installed SKILL.md packs (the agentskills.io format superpowers,
caveman, and most skill repos ship). Installed skills live under
`$XDG_DATA_HOME/agent6/skills/<name>/`; `agent6 skills install <url>` accepts a
direct SKILL.md URL, a git repository URL (installs every `skills/*/SKILL.md`),
or a local path. Installed means enabled: run mode lists each enabled skill's
name + description in a `<skills>` system-prompt index and the worker loads
full content on demand with the read-only `use_skill` tool. Enabled skills also
register as `/<name>` pause-menu commands (built-ins always win collisions) and
work with `agent6 run --skill <name>`. See [security.md](security.md) for the
trust model.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch: off = no index block, no `use_skill` tool, no slash commands. |
| `extra_dirs` | `[]` | Additional skill directories scanned BEFORE the installed dir (a local checkout under development wins over an installed copy). |
| `state` | `{}` | Per-skill exceptions, one value per skill: `"disabled"` drops it everywhere; `"always"` injects the full SKILL.md text into the system prompt instead of indexing it. Absent = enabled. Layers merge the map key-wise, so a repo config can flip one skill. `agent6 skills enable/disable [--repo]` write it. |

Measured while building (2026-07-10, n=3 + controls): on small open models
(qwen3-coder-30b, mistral-small-3.2) the passive index alone never triggered an
organic `use_skill` call, and system-prompt style instructions produced zero
compliance (byte-verified delivery, MOOSE positive control). `always`, `/name`,
and `--skill` are the reliable delivery paths for such models; an irrelevant
index measurably distracted mistral-small. Prefer a small index on weak models.

### Config profiles

A profile presets many settings at once so a task picks a strategy with one knob.
Select with `--profile <name>` (on `run`/`plan`/`ask`), the top-level `profile =
"<name>"` key, or the TUI new-work chooser. A profile **overrides** config rather than
being a baseline: its preset is injected just above the config layer that selected
it, so precedence (low → high) is

    defaults < global config < [profile via global profile field]
            < repo config < [profile via repo profile field]
            < [profile via --profile flag] < --config FILE

The most-specific source wins (`--profile` flag, else repo's `profile`,
else global's; the presets never stack), and the profile beats the config at its
scope. But a more-specific config layer, an explicit `--config FILE`, or an
individual flag still beats the profile.

| Profile | Bundles |
|---|---|
| `quick` | review off, tighter output budget; fast/cheap. |
| `standard` | the plain defaults (no review). The default. |
| `ultra` | a 3-seat grounded `before_finish` veto panel; thorough review. |
| `paranoid` | 5 explore-tier seats, `before_finish` veto, bigger budget. |

Define your own with a `[profiles.<name>]` table (a partial config); it wins over
a built-in of the same name. Example:

```toml
profile = "myteam"

[profiles.myteam.review]
trigger = "before_finish"
panel_size = 3
decision = "veto"
seats = ["security@anthropic/claude-opus-4-8", "correctness@openrouter/moonshotai/kimi-k2"]
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

The token ceilings are exact and always enforced. When the worker model's
price is cached, `best_effort_usd_limit` also converts to token ceilings at
load (the lower wins per axis), sizing each axis to the full dollar budget;
the run then stops when estimated spend (reported cost, else price times
tokens, cache included) crosses the limit. That combined, cache-inclusive USD
check is the authoritative bound on a USD-budgeted run -- it trips before
either full-budget axis cap, so an output-heavy workload (e.g. a reasoning
model whose hidden reasoning dominates output) can spend the whole budget
instead of halting once a ratio-split output cap is hit. With no price and no
reported cost the USD limit does nothing, hence best effort.

Price data comes from provider model listings (today OpenRouter's; Anthropic's
API publishes none), cached under `$XDG_CACHE_HOME/agent6/models/`. A
direct-Anthropic model id is priced via its OpenRouter listing when that cache
is present (`claude-opus-4-8` → `anthropic/claude-opus-4.8`, same list
prices), so USD budgets and cost summaries work on direct Anthropic runs too.

Override per-run from the CLI without editing config: `agent6 run --max-usd 5`,
`--max-input-tokens`, `--max-output-tokens` (on `run`, `plan`, `resume`).
Passing `--max-usd` explicitly refuses to start when the worker model has no
price data, since the flag could not be honored.

## `[machine]`

State-machine runtime knobs (`agent6 machine run`).

| Field | Default | Meaning |
|---|---|---|
| `snapshot_keep` | `5` | Recent blackboard snapshots retained per machine instance. Recovery reads only the latest and `machine replay` rebuilds from the journal, so older snapshots are an audit convenience. `0` keeps every snapshot (one file per transition; budget disk for long-running machines). |

### `[machine.notify]` (optional)

Operator notify hook for a running machine, the out-of-band channel for a
phone in a pocket. Runs an operator-controlled argv on the host outside the
jail on every `machine.notify` (a state's `notify` message) and on the
terminal `machine.end`, with `AGENT6_MACHINE_ID`, `AGENT6_MACHINE_DIR`,
`AGENT6_MACHINE_EVENT` (`notify`/`end`), `AGENT6_MACHINE_STATE`,
`AGENT6_MACHINE_MESSAGE`, and `AGENT6_MACHINE_LEVEL` (the level for a notify,
the `ok`/`failed` status for an end). Set it only in the global/repo config,
never in a machine `[config]` overlay (rejected at load); the argv is never
LLM output. Fan out to your own push channel (ntfy/Pushover/email/Telegram).

| Field | Default | Meaning |
|---|---|---|
| `on_event` | `[]` | argv to run on each notify/end (empty = disabled). |
| `timeout_s` | `30.0` | Hook timeout. |

## `[notify]` (optional)

Runs an operator-controlled argv after `agent6 run` / `resume` (success or
failure), outside the jail as your user, with `AGENT6_RUN_ID`,
`AGENT6_RUN_DIR`, `AGENT6_RUN_OK` (`1`/`0`), `AGENT6_RUN_REASON` set.

| Field | Default | Meaning |
|---|---|---|
| `on_complete` | `[]` | argv to run (empty = disabled). |
| `timeout_s` | `30.0` | Hook timeout. |

## `[web]`

Bind for `agent6 web`, the browser front-end (see [the web UI](web.md)). Secure
by default: loopback only. Remote access is expected behind `tailscale serve`, so
there is no app-level auth; binding a non-loopback host exposes the write surface
(spawn runs, answer prompts) and requires an explicit opt-in.

| Field | Default | Meaning |
|---|---|---|
| `web.host` | `127.0.0.1` | Bind address. A non-loopback value requires `allow_non_loopback = true`. |
| `web.port` | `7658` | Listen port. |
| `web.allow_non_loopback` | `false` | Opt-in to bind a non-loopback host. Off by default so a typo or copied config can never silently expose the agent. Prefer `tailscale serve` in front of a `127.0.0.1` bind instead. |

## `[parallel]`

Fan-out defaults for `agent6 run --parallel N` (or `--parallel model-a,model-b`).
Each lane is a disposable clone of the repo that runs independently and lands its
own `agent6/<id>` branch; the orchestrator symlinks the live lanes into `agent6
runs` for visibility, then imports each and prints a ranked auto-comparison.
Nothing is merged for you. `--max-usd` is per lane: total spend is up to
`--max-usd` x lane count, and the orchestrator prints the
`$X/lane x N = $Y total` line before spawning. `--auto-approve` forwards the
same way: every lane inherits it, so a lane never sits on a `run_commands=ask`
approval nothing detached can answer.

| Field | Default | Meaning |
|---|---|---|
| `parallel.max_lanes` | `4` | Hard cap on lanes per fan-out; `--parallel` over this refuses up front. |
| `parallel.workdir` | `""` | Base dir for lane clones (`<workdir>/<fanout-id>/lane-<i>`). `""` = `<cache_dir>/parallel`, cleaned up after import. |

A live run can also dispatch subordinate lanes mid-run: steer it (Ctrl-C) with a
message starting with `/parallel [spec] <task>` (repeat the token to queue more
tasks in one message; a spec is a lane count or model list, omitted = one lane;
a first token with a comma or slash reads as the spec, a bare model name reads
as task text).
The coordinator commits its worktree, clones committed HEAD into each expanded
lane, runs them to completion, joins each branch back in order (a conflict is
reported for you to merge, never aborts the run), and continues informed by a
per-lane summary. Lanes reuse the `[parallel]` workdir/cache above.
Dispatch is depth 1: a lane can never itself fan out or dispatch. When the
front-end does not wire a dispatcher (headless, plan/ask), `/parallel` answers
"not available" and the run continues.

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

A provider's `api_key_env`, when set, names the environment variable that
supplies its key; omit it to read the key from `secrets.toml`.
A few additional `AGENT6_*` toggles exist for testing/advanced use; see the
source if you need them.
