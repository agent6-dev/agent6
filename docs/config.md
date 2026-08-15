<!-- Generated from docs/config_template.md by docs/gen_config.py; edit those, then regenerate. -->
# Configuration

agent6 is **secure by default**: every field has a default (security-sensitive
ones default to the safe value), so you only set what you want to change. This
is the field reference; the security model behind `[sandbox]` and `[git]` is
[security.md](security.md).

## Where config lives (layered, lowest precedence first)

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | (none) | (secure defaults, always present) |
| global *(default location)* | `$XDG_CONFIG_HOME/agent6/config.toml` (`AGENT6_CONFIG_HOME` overrides) | `agent6 connect`, `agent6 model` |
| per-repo *(override)* | `<state-dir>/<repo-id>/config.toml` | `agent6 init`, `agent6 config set --repo` |
| explicit | `--config FILE` | `agent6 run --config FILE` |

The per-repo config lives in the state dir, out of the workspace: per-machine,
never committed. It can be empty or absent when the global config supplies a
provider + model; `workflow.verify_command` is inferred per run when unset.

## Creating & inspecting

- `agent6 connect`: add a provider + API key (stored `0600`), global.
- `agent6 model <role> <provider> <model> [--thinking off|low|medium|high]`.
- `agent6 init`: optional setup wizard (per-repo config, inferred
  `verify_command`, `.gitignore`, `AGENTS.md`); every step asks first.
- `agent6 config show`: every effective value and which layer set it.
  `--descriptions` adds each value's meaning under its row; `config show
  <key>` prints one key untruncated, meaning included.
- `agent6 config get|set|unset|add|remove <dotted.key> [value]` (`--repo`, or
  `--machine-file FILE` for a machine `[config]` overlay). Every edit is
  re-validated and rolled back if invalid. A sibling pair that must move
  together is set as one inline table:
  `agent6 config set context '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'`.
- Writes are atomic and the edit lock FAILS OPEN (a blocked lock never blocks
  the write; worst case one lost update, the error says "kept as written").
  A config file that is a symlink is followed only when you own the target.
- `agent6 config fill`: materialize defaults + global config into the global
  file. The repo layer and any selected preset are left as-is.
- `agent6 config fix`: drop invalid entries (unknown keys, stale values),
  naming each; `--machine-file FILE` repairs an overlay instead.
- `agent6 check`: validate config + sandbox + provider keys without running.
  `config show` prints what is SET (`network = "auto"`); `check` prints what
  that resolved to on this host, for the isolation level, the commands'
  network, and each MCP server's network and `approve`.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version (must be `1`). |
| `state_dir` | `"$XDG_STATE_HOME/agent6"` | Absolute base for all per-repo state (`<state_dir>/<repo-id>/`), out of the workspace. Global-config only; when set it wins over `AGENT6_STATE_HOME` and the XDG default. In a devcontainer the default is ephemeral: point it at a persisted volume to keep runs across rebuilds. |

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`. Three
orthogonal choices describe any backend: **`api_format`** (the wire dialect,
the only field that selects code), **`deployment`** (URL/placement quirks of
where it is hosted), and **auth** (`auth_style` + `api_key_env` or
`token_command`). A minimal block is just `api_format` (plus `base_url` for a
non-default host).

| Field | Default | Meaning |
|---|---|---|
| `api_format` | *(required)* | `"anthropic"` (Messages) or `"openai"` (Chat Completions: OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint, …). |
| `deployment` | `"direct"` | `"direct"`, `"vertex"`, or `"azure"` (`openai` only). Selects URL shape + model/version placement. |
| `base_url` | per (format, deployment) | Endpoint host + path prefix; required for vertex/azure. Its host is the only network destination the agent dials for this provider. |
| `auth_style` | per (format, deployment) | `"x_api_key"`, `"bearer"`, `"api_key_header"` (Azure), or `"none"` (local). Rarely set by hand. |
| `api_key_env` | none | Env var holding the key (wins over `secrets.toml`). Omit for `agent6 connect` keys or unauthenticated local endpoints. |
| `token_command` | none | argv that prints a short-lived bearer to stdout; re-run on TTL and once on `401`/`403`. Wins over `api_key_env`. |
| `token_command_ttl_s` | `300.0` | Seconds to cache `token_command` output. |
| `extra_headers` | `{}` | Extra HTTP headers on every request. Not for secrets. |
| `extra_body` | `{}` | Provider-specific JSON merged into every request body LAST, so tuning keys (max_tokens, temperature) win; the structural keys agent6 owns (messages, model, stream, tools, tool choice, response shape) are filtered. Values must be JSON-shaped: a TOML date/time is refused. e.g. OpenRouter routing options. |
| `extra_query` | `{}` | Extra URL query params (e.g. Azure's `api-version`). |
| `http_timeout_s` | `600.0` | Per-HTTP-call timeout (read/write; connect is bounded at 20s). |
| `prompt_caching` | `true` | (`anthropic`) Prompt caching: system prompt, tools, and the growing conversation re-read at 0.1x input price. |

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

# Claude on Vertex (model in the URL, anthropic_version in the body: handled
# by deployment = "vertex")
[providers.vertex-claude]
api_format = "anthropic"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/publishers/anthropic/models"
token_command = ["gcloud", "auth", "print-access-token"]

# Azure OpenAI (the model id IS the deployment name; api-version required)
[providers.azure]
api_format = "openai"
deployment = "azure"
base_url = "https://RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
extra_query = { "api-version" = "2024-06-01" }
```

### OpenRouter routing & caching (`extra_body`)

OpenRouter's default routing is not deterministic, so prompt caching may or
may not engage call-to-call. Pin it with `extra_body.provider`
([routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
api_format = "openai"
base_url = "https://openrouter.ai/api/v1"
extra_body = { provider = { sort = "throughput" } }  # prefer fast, caching backends
# Alternatives: { order = ["DeepInfra"], allow_fallbacks = true }
#               { max_price = { prompt = 1, completion = 2 } }
```

```bash
agent6 config set providers.openrouter.extra_body '{ provider = { sort = "throughput" } }'
```

Caching matters more than payload size: the large per-call input is the same
prefix every turn. Watch `cache_r` in the cost summary to confirm it engages.

### Short-lived bearer tokens (`token_command`)

For endpoints authenticated by a refreshed bearer rather than a static key
(Vertex OAuth, OIDC/STS gateways): point `token_command` at anything that
prints a current token, e.g. `["gcloud", "auth", "print-access-token"]` or
`["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"]`.
Cached `token_command_ttl_s` seconds, re-run once on `401`/`403`. It runs in
agent6's own process with your environment (operator-only, same trust as an
MCP `command`); non-zero exit, timeout, or empty output surfaces as a provider
error.

## `[models.<role>]`

Role routing. **`worker`** drives `run`/`resume` (its pricing also drives the
USD→token budget conversion); **`planner`** drives `plan`; **`reviewer`**
drives `review` + the in-loop critic. `planner`/`reviewer` fall back to
`worker`. Cross-vendor mixes are fine.

<!-- the three roles are the same shape, so the table is rendered once -->
| Field | Default | Meaning |
|---|---|---|
| `provider` | *(required)* | A `[providers.*]` name. |
| `model` | *(required)* | Model id at that provider. |
| `temperature` | `0.0` | Pinned per call (`0.0` to `2.0`). `0.0` keeps tool use stable. |
| `thinking` | none | Reasoning effort: `off`/`low`/`medium`/`high`. Anthropic maps it to a thinking budget (≈ 4k/8k/16k tokens); non-reasoning models ignore it. |

## `[sandbox]`

The security boundary. The model:
[Isolation-level selection](security.md#3-isolation-level-selection) and
[State-machine egress](security.md#8-state-machine-egress-script-bundles) in
security.md. This is the field summary.

| Field | Default | Meaning |
|---|---|---|
| `isolation` | `"auto"` | `auto` picks the strongest the host supports (`strict`, else `hardened`; `none` only when the host offers no confinement, loudly). Explicit `strict`/`hardened` refuse where unsupported, never downgrade. Explicit `none` runs UNSANDBOXED (also `--dangerously-disable-sandbox` / `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`). |
| `network` | `"auto"` | Which network jailed commands join. `auto`: the run's PRIVATE network (commands reach each other, nothing off the box, nothing outside reaches in), enforced on `strict`, degraded to the host's with a warning on `hardened`/`none`. `session`: the same, refusing where unenforceable. `only_explicit_states`: strict-only; machine `tool` states opt in. `host`: the machine's network. No per-command `none`: a run's commands share one launcher. |
| `run_commands` | `"ask"` | May the LLM run commands (`run_command`, `run_verify_command`, `stop_background`: one decision for all three): `yes` auto-approves, `no` withholds the tools (and the verify gate with them), `ask` prompts per call with session-wide allow/deny answers. `agent6 ask`/`plan` clamp `yes` to `ask`. Per-invocation: `--auto-approve` (never over a configured `no`), `--no-commands`. A run that cannot ask anyone refuses to start. |
| `fetch_hosts` | `[]` | Hosts the `fetch` tool reads WITHOUT asking; any other host prompts, and an absent operator is a no. Empty = every fetch prompts; `["*"]` = any host, written down as a choice; a leading dot allows subdomains (`.readthedocs.io`). HOSTS, not URL prefixes. The rest of fetch is fixed (https only, 1 MiB cap, redirects returned not followed: security.md, Fixed tool surface). Hidden when `network = "host"`; withheld from machine/agent states. A GET can still encode data in its path, hence the empty default. |
| `protect_git` | `true` | Keep `.git/` unwritable by jailed commands (else one can plant a git filter that agent6's host-side auto-commit executes). STRICT-ONLY: a read-only bind needs a mount namespace (security.md, Git invariants). On `hardened` the default degrades with a warning; an explicit `true` refuses. The in-process edit tools refuse `.git` writes everywhere regardless. |
| `memory_limit_mb` | `0` (off) | `RLIMIT_DATA` cap (MiB) per jailed process (inherited). Off by default: the kernel already handles a memory bomb, and a cap costs real builds more than it buys. Set one to bound a specific task; a runaway then fails as an ordinary command error. |
| `extra_read_paths` | `[]` | Extra absolute paths **the run** may **read + execute**, at their real locations: a toolchain or interpreter outside the repo (conda, Go/Rust/Node, a shared data dir). Mounted for jailed commands, readable by the in-process tools (name one with an absolute path). Loosens confinement; list only what the build needs. |
| `extra_write_paths` | `[]` | Extra absolute paths **the run** may **read + write**, at their real locations: a build cache, an output dir, a sibling checkout the task edits. Write implies read. List only what the task writes. |
| `hide_paths` | `[]` | Paths **the run** may never read or write, even under a broader grant. agent6's config dir and state base are always hidden, so an `extra_read_paths` grant of `$HOME` never exposes `secrets.toml` or your run history (the data dir and cache are not: installed skills stay usable). Enforced twice: the in-process tools refuse them at **every** isolation level (`none` included), and jailed commands see them masked (a dir appears empty, a file reads empty). Masking needs the mount namespace: on `hardened` an entry it cannot mask refuses the run, and a grant exposing the always-hidden dirs warns loudly instead. |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `require_clean_worktree` | `true` | Refuse to start on a dirty worktree. |
| `auto_stash` | `false` | Stash uncommitted changes before the run; restored per `auto_stash_pop`, else the `git stash apply <sha>` line is printed (by sha, never silently left). |
| `auto_stash_pop` | `false` | Pop the stash back at run end when safe (clean tree, conflict-free apply). On any doubt, leave it and print how to restore. Never `reset --hard`. |
| `branch_per_run` | `true` | Also advance a visible `agent6/<id>` branch to the run's chain tip (else the hidden `refs/agent6/<id>/head` ref only). Forced on for `--parallel` lanes (work is imported by branch). |
| `commit_per_step` | `true` | Per-step commits onto the run's detached chain (a temp index; HEAD, your index, and your checkout are never touched). Off: agent6 never commits; work stays only in the worktree, and resume-from-git, `sessions diff`/`merge`, and `/parallel` dispatch from a changed tree degrade. |
| `merge_strategy` | `"squash"` | `agent6 sessions merge` default: `squash` (one commit), `merge` (--no-ff, keeps per-step history), `ff`. Governs consolidation only; per-step commits always land on the run's chain. |
| `auto_merge` | `false` | After a run with nothing red, land the run's work on its base automatically (never over a red/stale verify). With `branch_per_run` off it merges the hidden chain ref. On conflict nothing moves and instructions are printed. |
| `auto_prune` | `false` | After `auto_merge`, delete the run branch when `git branch -d` can (merge/ff). A squash-merged branch is reported with the `-D` line, never force-deleted. Requires `auto_merge`; no-op without a run branch. |
| `run_repo_hooks` | `false` | Run the repo's own `.git/hooks/*` during agent6's git ops. Off: a repo hook is repo-controlled host code, an RCE vector on an untrusted repo. `core.fsmonitor`/`diff.external` are always neutralized. |
| `run_repo_filters` | `false` | Honor the repo's own content drivers (`filter.<n>.clean/smudge/process`, `merge.<n>.driver`) during agent6's git ops. Off: a driver defined in `.git/config` is repo-controlled host code that runs on the auto-commit's `git add` (or a chain merge), the same RCE class as a hook; agent6 neutralizes each by name. Turn on to support **Git-LFS** (its clean/smudge filters are exactly these). |

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` | none | Override the commit identity (else the project's `git config`). `agent6 run` refuses to start with no resolvable identity. |
| `email` | none | Override the commit identity (else the project's `git config`). `agent6 run` refuses to start with no resolvable identity. |
| `trailer` | `""` | Appended to every commit agent6 makes, e.g. `"Assisted-by: agent6:{model}"` or `"Co-authored-by: agent6:{model} <noreply@agent6.dev>"`. `{model}` = the model(s) that wrote the code, `", "`-joined when several contributed. |

### `[git.commit.checkpoint]` and `[git.commit.squash]`

| Field | Default | Meaning |
|---|---|---|
| `checkpoint.message` | `"agent6"` | Per-step message style: `agent6` (`agent6 iter N:`), `conventional` (derived from the diff, no model call), or `model` (model-written, degrading to `agent6` on failure). |
| `squash.message` | `"agent6"` | Squash-commit style: checkpoint's styles plus `combine` (git's concatenated per-step log). |

## `preset` (top-level)

| Field | Default | Meaning |
|---|---|---|
| `preset` | `""` | Named strategy preset: fills many settings at once (see the Presets section). Top-level because it overrides every section. `agent6 config set preset <name>` (`--repo`); `--preset` overrides per run. |

## `[workflow]`

| Field | Default | Meaning |
|---|---|---|
| `verify_command` | `[]` | argv defining "a step succeeded" (no shell; wrap a pipeline as `["sh","-c","a && b"]`). Optional: unset infers per run (AGENTS.md `## Verify command`, then repo manifests, then a model call over the manifests -- skipped when there are none), injected in-memory and printed. None inferable = the run starts gateless; a recognizable project created mid-run adopts the first resolvable inferred gate. Set it to pin one. |
| `verify_timeout_s` | `600.0` | Per-call timeout for `verify_command` / `metric.command`. The operator's gate needs a verdict, so it is bounded; a model-chosen `run_command` is not (see `command_checkin_s`). |
| `command_checkin_s` | `900.0` | How long a model's `run_command` may run before it is **handed back** as a background job. Not a timeout: nothing is killed, the command keeps running, and the model is told (`returncode: null`, `still_running: true`, a `background_id`) so it can wait with `read_background`, stop it, or carry on. `0` disables the hand-back, which is right when a human is watching and can interrupt. |
| `require_verify_to_finish` | `false` | Refuse `finish_session` while the last verify is red or never ran (bounded nudges). Regardless, a finish over red is always reported "finished", never "passed". |
| `spec_recheck_on_finish` | `false` | Bounce the first finish over a green verify once for a spec re-check. Measured (n=6/arm, 3 models): no gain beyond noise, one score drop, +38-88% cost. Kept off; candidate for removal. |

## `[review]`

| Field | Default | Meaning |
|---|---|---|
| `trigger` | `"off"` | In-loop review panel trigger: `off` / `on_verify_fail` / `before_finish` / `periodic`. |
| `period` | `10` | Iterations between reviews for `periodic`. |
| `decision` | `"advisory"` | `advisory` (inject findings, never block) / `veto` / `quorum` / `all`. |
| `quorum` | `2` | K for `quorum`; counts distinct MODELS, so same-model seats can't fake it. |
| `max_total_rejections` | `4` | Per-run blocks before the gate auto-disarms to advisory. |
| `budget_fraction` | `0.25` | Skip the in-loop panel once remaining budget falls below this fraction. |
| `seats` | `[]` | Panel roster: `"persona"` routes via `[models.reviewer]`; `"persona@provider/model"` pins a model per seat. `agent6 review --reviewers N [--personas …]` synthesizes an equivalent. |
| `concurrency` | `1` | In-loop seat parallelism (post-hoc `agent6 review` is always parallel). |
| `tier` | `"diff"` | `diff` (one grounded call over the diff) or `explore` (read-only tool-using reviewer, cross-file). |

Grounding is mechanical, not prose: a `block` gates only if its `file:line`
is in the diff AND its category is in a fixed allowed set (security /
sandbox-bypass / off-topic-edit / data-loss / verify-uncovered-correctness);
everything else is advisory and cannot stall the run.

## `[context]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

| Field | Default | Meaning |
|---|---|---|
| `drop_at_chars` | _adaptive_ | Tier 1: oldest tool results become placeholders. Unset sizes from the worker's context window (~45%); set BOTH thresholds to pin. |
| `summarise_at_chars` | _adaptive_ | Tier 2: summarise elided history and restart (the task DAG survives). Unset = the window minus a 16k-token reserve. Must exceed `drop_at_chars`. |
| `keep_recent_chars` | `80000` | Verbatim recent-history tail kept through a tier-2 restart (chars; 0 keeps none). |
| `keep_thinking_turns` | `0` | Drop thinking from assistant turns older than N assistant turns, at tier-1 moments. `0` (default) keeps all thinking, matching pi; Claude Code clears old thinking. Anthropic-format providers only: the openai wire never re-sends thinking. |
| `summary_max_tokens` | `2048` | Cap on the tier-2 summary (and gist distillation calls). |
| `elision_gists` | `true` | Tier 1 decays a large `read_file` to a model-written gist before the bare marker (demoted under continued pressure so the byte bound holds). `false` = straight to bare markers. |

## `[prompt]`

| Field | Default | Meaning |
|---|---|---|
| `system_prompt_file` | `""` | ADVANCED: replace run-mode's static base prompt with this file (dynamic blocks still append). Warned at startup if core tool names are missing. |
| `structural_priors` | `true` | Include the `<repo-priors>` block (hot symbols, co-change, outline). `false` for a leaner prompt. |
| `revise_prompt` | `"off"` | One-shot task-prompt revision before the loop: `off` / `auto` / `interactive`. |
| `decompose` | `"auto"` | Front-load task decomposition (run mode): `on` helps small models that under-finish multi-part tasks (measured on mistral-small; capable models just pay 2-4x overhead). `auto` resolves per worker model from the capability registry; `config show` displays the resolved value. `--decompose` forces one run. |

## `[skills]`

Operator-installed SKILL.md packs (the agentskills.io format). Installed under
`$XDG_DATA_HOME/agent6/skills/<name>/`; `agent6 skills install <url>` takes a
SKILL.md URL, a git repo (every `skills/*/SKILL.md`), or a local path.
Installed = enabled: an index in the system prompt, on-demand content via
`use_skill`, a `/<name>` pause-menu command, and `run --skill <name>`. The
format is shared with Claude Code and most agentskills.io tooling: point
`extra_dirs` at an existing collection (`~/.claude/skills`, …) or install to
copy. Repo-local skill dirs are deliberately NOT discovered (repo content is
not config). Trust model: [security.md](security.md).

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch: off = no index, no `use_skill`, no slash commands. |
| `extra_dirs` | `[]` | Additional skill dirs, scanned BEFORE the installed dir. |
| `state` | `{}` | Per-skill: `"disabled"` drops it; `"always"` injects the full text into the system prompt. Layers merge key-wise; `agent6 skills enable/disable [--repo]` writes it. |

Measured (2026-07): small and frontier open models alike almost never invoke a
skill organically from the passive index, and no prompt lever made it
reliable. When a skill must apply, use `always`, `/name`, or `--skill`.

### Presets

A preset fills many settings at once. `agent6 config presets` lists them;
select with `--preset <name>`, `agent6 config set preset <name>` (`--repo`),
or the TUI new-work chooser. A preset overrides config at the layer that
selected it (most-specific source wins, presets never stack); a more-specific
config layer, `--config FILE`, or an individual flag still beats it. A
`--config FILE` or machine overlay cannot select one.

| Preset | Bundles |
|---|---|
| `quick` | review off; fast/cheap. |
| `standard` | the plain defaults. The default. |
| `ultra` | a 3-seat grounded `before_finish` veto panel. |
| `paranoid` | 5 explore-tier seats, `before_finish` veto. |

Define your own with a `[presets.<name>]` table (a partial config); a
built-in's name replaces that built-in wholesale:

```toml
preset = "myteam"

[presets.myteam.review]
trigger = "before_finish"
decision = "veto"
seats = ["security@anthropic/claude-opus-4-8", "correctness@openrouter/moonshotai/kimi-k2"]
```

### `[workflow.metric]` (optional)

A continuous score for measurable goals; `command` runs in the jail like
`verify_command`.

| Field | Default | Meaning |
|---|---|---|
| `command` | *(required)* | argv to run. |
| `pattern` | *(required)* | Regex; first capture group = the number. |
| `goal` | *(required)* | `"minimize"` or `"maximize"`. |

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost,
else cached price × tokens) count against `max_usd`; unpriceable calls count
input+output tokens against `max_tokens_fallback`. Both: `-1` unlimited, `0`
refuse that ledger up front, `> 0` the cap.

| Field | Default | Meaning |
|---|---|---|
| `max_usd` | `10.0` | Cap on metered spend (cache-aware, per model). |
| `max_tokens_fallback` | `2000000` | Token cap for UNMETERED calls only (local models, price gaps). |

`--max-usd` / `--max-tokens-fallback` override per run; an explicit
`--max-usd` refuses to start when the worker has no price data. Prices come
from provider listings (OpenRouter's; cached under
`$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its
OpenRouter listing.

## `[machine]`

| Field | Default | Meaning |
|---|---|---|
| `snapshot_keep` | `5` | Blackboard snapshots kept per instance (recovery reads only the latest; `machine replay` rebuilds from the journal). `0` keeps all. |
| `state_log_keep` | `50` | Per-state watchable log dirs kept per instance (`<instance>/states/`); the journal keeps the full transition history regardless. `0` keeps all. |

### `[machine.notify]` (optional)

Operator hook on every `machine.notify` and the terminal `machine.end`: an
operator argv on the host with a minimal env (PATH/HOME/locale/desktop-bus +
`AGENT6_MACHINE_ID/DIR/EVENT/STATE/MESSAGE/LEVEL`), never your full
environment. Global/repo config only (a machine `[config]` overlay setting it
is rejected). Fan out to ntfy/Pushover/email/Telegram yourself.

| Field | Default | Meaning |
|---|---|---|
| `on_event` | `[]` | argv per notify/end (empty = disabled). |
| `timeout_s` | `30.0` | Hook timeout. |

## `[notify]` (optional)

Runs after `run`/`resume` with the same minimal env plus
`AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`. `OK=1` means the agent stopped
deliberately; `VERIFIED` is what the gate said — a hook that wants "green"
reads the second.

| Field | Default | Meaning |
|---|---|---|
| `on_complete` | `[]` | argv to run (empty = disabled). |
| `timeout_s` | `30.0` | Hook timeout. |

## `[web]`

Bind for `agent6 web` ([the web UI](web.md)). Loopback only by default, no
app auth: remote access is expected behind `tailscale serve`.

| Field | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Bind address; non-loopback requires `allow_non_loopback = true`. |
| `port` | `7658` | Listen port. |
| `allow_non_loopback` | `false` | Opt-in for a non-loopback bind, so a typo can never silently expose the write surface. |

## `[parallel]`

Fan-out defaults for `run --parallel N|model-a,model-b`. Each lane is a
disposable clone running its own `agent6/<id>` branch; lanes are imported and
ranked, nothing merges for you. `--max-usd` is per lane (the total is printed
before spawning); `--auto-approve` forwards to every lane. A live run
dispatches lanes the same way via the `/parallel` steer directive (depth 1: a
lane never fans out; headless surfaces without a dispatcher answer "not
available").

| Field | Default | Meaning |
|---|---|---|
| `max_lanes` | `4` | Hard cap per fan-out (1-1024); more refuses up front. |
| `workdir` | `""` | Base dir for lane clones. `""` = `<cache_dir>/parallel`, cleaned up after import. |

## `[mcp]` + `[mcp.servers.<name>]` (optional)

MCP servers, spawned (`command`) or connected (`url`); tools appear as
`mcp__<name>__<tool>`. A server runs as your user outside the jail with a
curated env (never your provider keys; `pass_env` adds named vars). The LLM
influences the ARGUMENTS it passes, so each call is approved like a command
(`approve`), and audit each server like a `run_command` allow-list. `agent6 mcp connect` handshakes first and only then writes the
entry; a server that does not start is skipped with an
`mcp.server_unavailable` journal event, never fatal.

```
agent6 mcp connect files -- npx -y @modelcontextprotocol/server-filesystem .
agent6 mcp connect browser --url http://127.0.0.1:8931/mcp --token-env PW_TOKEN
agent6 mcp list
```

A spawned server is a jailed child like any other: same launcher, same
sandbox a `run_command` gets. Its `[sandbox]` block names what it needs on
top of that.

```toml
[mcp.servers.notes]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "~/notes"]

[mcp.servers.notes.sandbox]
read_paths  = ["~/notes"]   # additive; the system and tool dirs are already there
write_paths = ["~/notes"]
```

A confined spawn also loses the desktop-session addresses
(`DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `DISPLAY`, `WAYLAND_DISPLAY`):
Landlock does not gate unix-socket `connect()`, and an unconfined session
daemon would act on the server's behalf. Anything else that reaches an
unconfined process is still a way out — name the narrowest paths that work.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch; `false` = zero `mcp__*` tools. |
| `servers.<name>.command` | `[]` | argv for a stdio server agent6 spawns. Exactly one of this or `url`. |
| `servers.<name>.url` | `""` | An http(s) endpoint the OPERATOR runs; agent6 only connects, owning none of its environment or confinement. |
| `servers.<name>.token_env` | `""` | For a `url` server: env var holding the bearer. Named, never inlined; never logged. Over plaintext `http://` to a non-loopback host the token is readable on the wire: `mcp connect` asks first, and every run warns. |
| `servers.<name>.enabled` | `true` | Per-server toggle. |
| `servers.<name>.pass_env` | `[]` | Env vars the server needs, BY NAME. Everything else is the curated base. |
| `servers.<name>.approve` | `"ask"` | Ask before each of this server's tool calls, showing the arguments the model chose; `yes` never asks. The session answers are per server: "allow all" covers THIS server for the run (not the command tools, not a sibling server), "deny all" withdraws its tools from the next turn. `--auto-approve` sets `yes` for the run. No `no`: withholding a server's tools is what `enabled = false` says. |
| `servers.<name>.startup_timeout_s` | `10.0` | `initialize` + `tools/list` budget. |
| `servers.<name>.call_timeout_s` | `60.0` | Per `tools/call` timeout. |
| `servers.<name>.httpx_trust_env` | `false` | For a `url` server: forward httpx's `trust_env`, so the connection honors the ambient HTTP(S)_PROXY (and .netrc / SSL_CERT_FILE). Off by default so a local server's bearer token never routes to a proxy; set it for a server reachable only through the environment's proxy. |

### `[mcp.servers.<name>.sandbox]`

| Field | Default | Meaning |
|---|---|---|
| `read_paths` | `[]` | Read+execute paths for this server BEYOND the sandbox a jailed command gets (absolute or `~`). The workspace, system dirs, tool dirs and a writable `/tmp` as `HOME` are already there, so a block names only the server's own data. |
| `write_paths` | `[]` | Paths it may write, likewise additive. |
| `network` | `"auto"` | Which network this server joins. `auto`: one of its own where the host can give a namespace, degrading to the host's with a warning. `none`: the same, refusing rather than running connected. `session`: the RUN's network, so a dev server a background command started answers this server too (a browser server driving the app under test), and still nothing off the box. `host`: the machine's network. |
| `unconfined` | `false` | No sandbox at all, for a server whose job IS arbitrary host access. Contradicts every other field here, so setting both is refused rather than half-applied. |

---

## Reaching a run's network

A run's commands share one network with no route off the box, so a dev server
the agent starts is invisible from here — that is the same property that keeps
it off the internet. Two commands are the way in, both operator-only:

```
agent6 sessions show <id>        # what it is serving, and the command to open it
agent6 forward <id>              # list the ports it is listening on
agent6 forward <id> 3000         # bridge that port to one on this machine
agent6 exec <id> -- curl ...     # run a command in the run's sandbox
```

`exec` runs in the whole sandbox — same mounts, same network — so what you see
is what the agent sees. Neither is reachable by the model.

---

## Environment variables

| Variable | Effect |
|---|---|
| `AGENT6_CONFIG_HOME` | Override the global config directory. |
| `AGENT6_CACHE_HOME` | Override the cache directory. |
| `AGENT6_JAIL_BIN` | Path to a specific `agent6-jail` binary (else bundled). |
| `AGENT6_ALLOW_ROOT` | `1` permits running as root (same as `--allow-root`). |

A provider's `api_key_env` names the env var supplying its key; omit it to
read `secrets.toml`. A few additional `AGENT6_*` toggles exist for
testing/advanced use; see the source.
