# Configuration

Every field has a default and the security-sensitive ones default to the safe value, so you set only what you want to change.
This is the field reference; the [security model](security.md) covers what `[sandbox]` and `[git]` enforce.

## Where config lives

The layers, lowest precedence first:

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | (none) | (secure defaults, always present) |
| global *(default location)* | `$XDG_CONFIG_HOME/agent6/config.toml` | `agent6 connect`, `agent6 model` |
| per-repo *(override)* | `$XDG_STATE_HOME/agent6/<repo-id>/config.toml` | `agent6 init`, `agent6 config set --repo` |
| flag (FILE) | `--config FILE` | `agent6 run --config FILE` |
| machine overlay | a machine file's `[config]` table | `agent6 config set --machine-file FILE` |

`--config FILE` may repeat; later files win.

A selected preset is inserted above the config layer that selected it; [Presets](#presets) gives the precedence rules.
The per-repo config lives in the state dir, out of the workspace: per-machine, never committed.
It can be empty or absent when the global config supplies a provider and model.

## Creating and inspecting

- `agent6 connect`: add a provider and, when needed, credentials (stored `0600`), global by default; `--repo` writes the provider entry to the repo config while credentials stay global.
  An existing block's format wins over a preset's.
- `agent6 model <role> [provider/]model [--effort off|low|medium|high|xhigh|max]`: a model id alone keeps the role's provider; a configured provider name alone opens a model picker on a terminal, or lists its models without one.
  A model change preserves the target layer's other role settings; only an explicit `--effort` replaces effort.
- `agent6 init`: optional setup wizard (per-repo config, inferred `verify_command`, `.gitignore`, `AGENTS.md`); every step asks first unless `--yes` skips the prompts.
- `agent6 config show`: every effective value and which layer set it.
  `--descriptions` adds each value's meaning under its row; `config show <key>...` prints the named keys (or sections), meaning included.
  Text values on the CLI, TUI and web use TOML literals without truncation, including quoted strings and inline tables, so an unmasked value pastes into `config set` unchanged.
  `(unset)` marks the absence of a value; TOML has no null literal.
  Mapping entry values are masked except for fields marked display-safe (`skills.state`, `review.reviewers`).
- `agent6 config set|unset|add|remove <dotted.key> [value]` (`--repo`, or `--machine-file FILE` for a machine `[config]` overlay); `agent6 config get <dotted.key>` prints the effective value and its layer.
  A key component that is not a bare TOML key (`[A-Za-z0-9_-]+`) is written quoted, as in `providers.demo.extra_body."a.b"` (keep those quotes inside a shell-quoted argument).
  `get` accepts a whole mapping or one entry and prints each entry with its own origin.
  `set` and `unset` accept a whole mapping or one entry, such as `skills.state.alpha`.
  Every edit is re-validated; an invalid edit is rolled back when the config lock is held.
  A sibling pair that must move together is set as one inline table: `agent6 config set compaction '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'`.
- Writes are atomic; failure to take the config lock does not block the write, but concurrent edits can lose updates.
  An invalid edit without the lock is reported as "kept as written".
  A symlinked config file is followed only when you own the target, or its nearest existing parent when the target does not exist.
- The [TUI config page](terminal.md#configuration) adds and removes named provider, MCP server and skill state entries through forms.
- `agent6 config set --secret telemetry.headers.<name> <value>` saves a literal header value in the private global `secrets.toml`; `agent6 config unset --secret telemetry.headers.<name>` removes it.
  This target accepts only telemetry header keys and cannot be combined with `--repo` or `--machine-file`.
  Values are never printed by these commands or `config show`, but command-line values can remain in shell history and the process list ([secrets](security.md#8-secrets-and-connect)).
- `agent6 config fill`: materialize defaults + global config into the global file; pass `--force` to replace an existing file.
  The repo layer and any selected preset are left as-is.
- `agent6 config fix`: drop auto-removable invalid entries (unknown keys, stale values) from the global and repo configs and their presets, a global value a repo override masks included, naming each, and report entries that need manual repair; `--machine-file FILE` repairs an overlay instead.
- `agent6 check`: validate config + sandbox + provider keys without starting an agent run.
  `config show` tags resolved values `(adaptive)`: `auto` settings for `sandbox.isolation`, `sandbox.network` and `prompt.decompose`, each role's unset effort, and adaptive compaction thresholds.

---

## `[agent6]`

<!-- config-table: agent6 -->

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`.
Provider names must match `[A-Za-z0-9_-]+`: one or more ASCII letters, digits, hyphens or underscores.

Three orthogonal choices describe an HTTP backend: **`backend`** (the wire dialect), **`deployment`** (URL/placement quirks of where it is hosted), and **auth** (`auth_style` + `api_key_env`, `token_command` or `oauth`).
A minimal block is just `backend` (plus `base_url` for a non-default host).

<!-- config-table: providers providers.<name> -->

### Request headers

`[providers.<name>.extra_headers]` is the one place extra HTTP headers come from.
A literal string (`"x-title" = "agent6"`) rides every API request: model calls, the model listing and the plan reading.
A template table renders from request facts for the request classes it names.
Only a template whose `targets` name `oauth` or `exchange` reaches a configured issuer's credential endpoint.

The facts: `request_id` (fresh per HTTP attempt) and `agent6_version` on every request; `model`, `conversation_id` (an opaque id per role and provider or per review panel and seat, kept across a resume), `initiator` (`user` or `agent`), `is_first_turn`, `is_subagent`, `has_tool_results` and `has_images` on a model call.
`{{` and `}}` are literal braces; a placeholder holds a fact name and nothing else.

The transport owns `authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `cookie`, `host`, `connection`, `content-length`, `transfer-encoding`, `content-type` and `accept`; naming one refuses at config load, as does a control character, two spellings of one header, or a template its target cannot render.

```toml
[providers.work.extra_headers]
"x-title" = "agent6"
"user-agent" = { template = "agent6/{agent6_version}", targets = ["inference", "models"] }
"x-initiator" = { template = "{initiator}" }
"x-vision" = { template = "true", when = "has_images" }
```

<!-- config-table: providers.<name>.extra_headers.<name> -->

### Configured OAuth

`[providers.<name>.oauth]` names an issuer and the public-client registration the operator made there; `agent6 connect <name>` signs in with authorization code and PKCE (`authorization_endpoint`, a loopback redirect or a pasted callback) or the device grant (`device_authorization_endpoint`), and stores the grant in `secrets.toml`.
Every endpoint is https, or plain http on `127.0.0.1` or `[::1]` for a local service; nothing is discovered, no client secret exists, and no redirect is followed.
A stored grant is bound to the entry's transport settings (`backend`, `deployment`, `base_url`, `auth_style`, `extra_query`, `extra_headers`) and its issuer settings: change any of them and the next run refuses until `agent6 connect <name>` runs again.
`[providers.<name>.oauth.exchange]` adds one second exchange when the model endpoint takes a token other than the issuer's: one `GET` or `POST` with the issuer's bearer, whose named top-level fields are the token and its expiry.

<!-- config-table: providers.<name>.oauth providers.<name>.oauth.exchange -->

```toml
[providers.work]
backend = "openai"
base_url = "https://models.example/v1"

[providers.work.oauth]
issuer = "https://login.example"
client_id = "operator-registered-public-client"
scopes = ["model.invoke"]
authorization_endpoint = "https://login.example/authorize"
device_authorization_endpoint = "https://login.example/device"
token_endpoint = "https://login.example/token"
revocation_endpoint = "https://login.example/revoke"
redirect_uri = "http://127.0.0.1:8765/auth/callback"
```

### Deployments

```toml
# Anthropic direct (default)
[providers.anthropic]
backend = "anthropic"

# Gemini on Vertex (OpenAI-compatible endpoint)
[providers.vertex-gemini]
backend = "openai"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/endpoints/openapi"
token_command = ["gcloud", "auth", "print-access-token"]

# Claude on Vertex (model in the URL, anthropic_version in the body: handled
# by deployment = "vertex")
[providers.vertex-claude]
backend = "anthropic"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/publishers/anthropic/models"
token_command = ["gcloud", "auth", "print-access-token"]

# Azure OpenAI (the model id is the deployment name; api-version required)
[providers.azure]
backend = "openai"
deployment = "azure"
base_url = "https://RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
extra_query = { "api-version" = "2024-06-01" }
```

### Responses API endpoint (`backend = "responses"`)

The Responses API on an endpoint the operator names: `/responses` is appended to `base_url`, which has no vendor default, and `direct` is the only deployment.
A key (`agent6 connect <name>`), a `token_command` or a configured [`oauth`](#configured-oauth) issuer authenticates it; usage meters tokens like any keyed endpoint, priced from the catalog when the model is known.

```toml
[providers.work]
backend = "responses"
base_url = "https://models.example/v1"

[models.default]
model = "work/reasoning-model"
```

- The system prompt goes as `instructions`, tools as flat function tools, tool results as `function_call_output`, the output cap as `max_output_tokens`; `store = false` always, and encrypted reasoning items are replayed to this endpoint and model only.
- The role's `effort` goes as `reasoning.effort` (`off` sends `none`; unset sends nothing); a `temperature` the endpoint refuses is dropped for the rest of the run.
- `extra_body` never sets `store`, `previous_response_id`, `conversation`, `background`, `max_output_tokens`, `parallel_tool_calls`, `text`, `response_format` or the structural fields.
- Streaming runs when a surface watches the call; a JSON response serves the rest.

### ChatGPT subscription (`backend = "chatgpt"`)

Uses a ChatGPT plan (Plus/Pro/Business) instead of an API key, over the Codex Responses backend.
The `session-id` header and `prompt_cache_key` carry the [shared conversation id](#request-headers).
Calls outside a conversation omit both.

```bash
agent6 connect chatgpt      # browser sign-in (device code, then paste, when headless)
agent6 model default chatgpt/gpt-5.6-sol
```

- `agent6 connect chatgpt` runs a PKCE OAuth sign-in against OpenAI's fixed OAuth authority (`https://auth.openai.com`) and stores the tokens in `secrets.toml` (0600); they refresh automatically.
- Past the included window, calls draw on purchased credits (real money; auto top-up can buy more).
  `[budget].allow_plan_overage = false` (the default) is a circuit breaker: once a window is exhausted with credits present the run stops at its next boundary; a call already in flight completes, so a boundary-crossing call can spend before the stop.
- Whether these conversations train OpenAI's models follows the ChatGPT account's own data controls (Settings > Data controls > "Improve the model for everyone"); agent6 cannot change that setting.
  agent6 never calls the feedback/rating endpoints, which would opt the rated turns into training regardless of it; there is no rating surface.
- Model names complete from the backend's own listing for the signed-in plan (fetched like other providers' catalogs), and its context windows size compaction.
- An `unsupported_value` refusal naming `reasoning.effort` and listing accepted levels, over HTTP 400 or a stream error frame, silently retries once at the lowest listed effort.
  Later calls on that provider instance keep the accepted levels; they are not persisted.
- `agent6 connect chatgpt --logout` signs out: the grant is revoked at the OAuth authority (best effort) and the tokens leave `secrets.toml`.
- Spend is plan-metered, not dollar-metered: every response carries the account's rate-limit windows.
  `[budget].max_plan_points` caps the plan movement observed during one run (`--max-plan-points` per run).
  Dollar figures stay an authoritative $0.

### Claude Code subscription (`backend = "claude_code"`)

Runs model calls inside the installed, signed-in Claude Code binary: a Claude subscription (Pro/Max) instead of an API key.

```bash
claude auth login                        # once, in Claude Code itself
agent6 connect claude                    # writes [providers.claude], checks the sign-in
agent6 model default claude/claude-sonnet-4-5
```

- agent6's own loop, tools, jail, verify gate, and approvals drive the run; the binary supplies the model.
  Inside it every Claude Code capability is off: built-in tools, hooks, settings, CLAUDE.md, MCP servers, skills, slash commands, session files, auto-memory, auto-compaction.
- agent6 never reads the login under `~/.claude`; a `CLAUDE_CONFIG_DIR` in the environment selects a relocated one.
- Spend is plan-metered: every round reports the account's 5-hour and 7-day windows.
  `[budget].max_plan_points` caps the plan movement observed during one run (`--max-plan-points` per run).
  Dollar figures stay an authoritative $0; the binary's own list-price estimate is not recorded.
- Ignored: `[models.<role>].temperature` and the loop's per-call output-token cap (the binary owns sampling).
  Refused: `effort = "off"` (`claude --effort` has no off value; use `low`).
- Optional roles inherit the default; override one with `agent6 model helper claude/claude-haiku-4-5`.
  Each side call without tools is one short-lived `claude` process.
- One `claude` process serves a worker execution.
  It restarts, replaying the conversation as one user message with retained attachments, on resume, fork, `/undo`, a steer or stop mid-turn, a tier-2 context restart, and when the live context nears the window.
  Tier-1 text compaction shrinks the model's context at that next restart, not before.
  Removing or changing a retained media result forces a restart.
- Claude Code appends the account email to every system prompt it sends; agent6 replaces it with `<operator-email>` in the model's returned text.
- Text-only tool results are capped at 34,000 bytes of UTF-8 for this provider (60,000 elsewhere): Claude Code writes a result over 50,000 bytes under `~/.claude/projects` and hands the model a 2 KB preview of it, and the turn's notices (a verify tail, a review critique, the nudges) ride in the same payload, with 16,000 bytes kept for them.
- Image and PDF attachments use MCP image or resource items, outside the text cap.
  The provider checks text plus notices against the 50,000-byte guard and each complete encoded stdin line against 8 MiB.
  See [Media retention](security.md#media-retention) for the child process and upstream boundary.
- Use a full model id (`claude-sonnet-4-5`, not `sonnet`) so the context window is known for compaction sizing.
- `agent6 connect claude --logout` is refused: agent6 stores no Claude Code credentials; `claude auth logout` signs out.

### OpenRouter routing and caching (`extra_body`)

OpenRouter's default routing is not deterministic, so prompt caching may or may not engage call-to-call.
Pin it with `extra_body.provider` ([routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
backend = "openai"
base_url = "https://openrouter.ai/api/v1"
extra_body = { provider = { sort = "throughput" } }  # prefer fast backends
# Alternatives: { order = ["DeepInfra"], allow_fallbacks = true }
#               { max_price = { prompt = 1, completion = 2 } }
```

```bash
agent6 config set providers.openrouter.extra_body \
  '{ provider = { sort = "throughput" } }'
```

Watch `cache_r` in the cost summary to confirm caching engages.

### Short-lived bearer tokens (`token_command`)

For endpoints authenticated by a refreshed bearer rather than a static key (Vertex OAuth, OIDC/STS gateways): point `token_command` at anything that prints a current token, e.g. `["gcloud", "auth", "print-access-token"]` or `["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"]`.
It runs outside the jail with your environment minus registered provider-key variables and `AGENT6_WEB_SECRET` (operator-only).
Non-zero exit, timeout, or empty output surfaces as a provider error.

## `[models.<role>]`

Role routing:

<!-- config-table: models -->

Every model call requires a `models.default.model` route.
Unset fields in `worker`, `planner`, `reviewer` and `helper` inherit directly from `models.default`.
A role never inherits from another role.
For example, `[models.planner]` with only `effort = "max"` keeps the default route and temperature.
`agent6 config show` attributes inherited leaves to the layer that supplied the default.
A worker-only config refuses with a remedy; rename `[models.worker]` to `[models.default]` to migrate.
A persisted `model` is fully qualified: `model = "openrouter/openai/gpt-4.1-mini"`.
The first segment names a `[providers.*]` entry; everything after the first slash is the model id verbatim.
`effort` and `temperature` are sibling keys.

<!-- config-table: models.default -->

## `[sandbox]`

The field summary; the security model is in security.md: [Sandbox](security.md#2-sandbox) and [Network](security.md#5-network).
For Go's compiled-package cache, see [Go build cache](cli.md#go-build-cache).

`extra_read_paths`, `extra_write_paths` and `hide_paths` take absolute paths with no `..` segment, and `extra_device_paths` only paths under `/dev`; anything else refuses at config load.

<!-- config-table: sandbox -->

## `[container]`

The container `sandbox.isolation = "container"` runs every command in; the runtime is the boundary and agent6 adds no confinement there.
With `exec` empty, agent6 finds this workspace's running Docker or Podman dev container by its labels ([Commands inside a dev container](cli.md#commands-inside-a-dev-container)).
Filling this table alone changes nothing.

<!-- config-table: container -->

## `[git]`

<!-- config-table: git -->

`agent6 sessions merge --method <method>` overrides `git.merge_method` for that merge.

See [Git](security.md#7-git) for where `sandbox.protect_git` enforces `git.control = "agent6"` and which settings warn.

### `[git.commit]`

<!-- config-table: git.commit -->

### `[git.commit.checkpoint]` and `[git.commit.squash]`

<!-- config-table: git.commit.checkpoint git.commit.squash -->

## `preset` (top-level)

<!-- config-table: preset -->

### Presets

A preset fills many settings at once.
`agent6 config presets` lists them; select with `--preset <name>`, `agent6 config set preset <name>` (`--repo`), or the preset picker of the TUI and web new-task composers.
A preset overrides config at the layer that selected it (most-specific source wins, presets never stack); a more-specific config layer, `--config FILE`, or an individual flag still beats it.
A `--config FILE` or machine overlay cannot select one.

The default `preset = "none"` selects no preset.
Set `preset = "none"` in the repo config to stop lookup before a global preset, or pass `--preset none` to override both layers.
This keeps the layers' own settings; it does not reset them to defaults.
An empty string is refused.
The name `none` is reserved, so `[presets.none]` is refused.
The TUI and web config pages edit the selected preset in place: with `preset` set to a name, a save, an unset or an entry change writes `[presets.<name>]` in the chosen layer; with `preset = "none"`, the global and repo config.
**Preset…** opens the `preset` setting, **Add preset…** creates a table and selects it, and **Remove preset** removes the selected table and resets `preset`.

<!-- presets-table -->

Define your own with a `[presets.<name>]` table (a partial config); a built-in's name replaces that built-in wholesale:

```toml
preset = "myteam"

[presets.myteam.review]
trigger = "before_finish"
block_on = "any"
reviewers = [
  { focus = "security", model = "anthropic/claude-opus-4-8" },
  { focus = "correctness", model = "openrouter/moonshotai/kimi-k2", effort = "high" },
]
```

The first write under a built-in's name (`agent6 config set presets.review.sandbox.network host`, or a page edit with `preset = "review"`) copies the built-in's values into the table first, so the rest of it stays.

## `[harness]`

<!-- config-table: harness -->

### `[harness.metric]` (optional)

A continuous score for measurable goals; `command` runs in the jail like `verify_command`.
Set its required fields together: `agent6 config set --repo harness.metric '{ command = ["python3", "bench.py"], pattern = "score: ([0-9.]+)", goal = "minimize" }'`.

<!-- config-table: harness.metric -->

## `[review]`

<!-- config-table: review -->

The [TUI](terminal.md#configuration) and [web](web.md#pages) config pages edit the ordered roster with **Add reviewer…** and positional removal.
Each edit writes the whole list to the global or repo config, including `[]` after the last removal.

Each `reviewers` entry has these fields:

<!-- config-table: review.reviewers.<entry> -->

Set `model = "provider/model"` to pin a route; omit it to inherit the resolved reviewer role.
Every omitted field inherits the resolved reviewer role's value, `model` pinned or not.
Explicit `effort = "off"` and `temperature = 0.0` override inherited values.
`model_quorum` requires at least `min_blocking_models` distinct resolved provider/model routes at config validation.
Different focuses or effort levels on one route still count as one blocking model.

```toml
[review]
trigger = "before_finish"
scope = "diff"
repo_max_turns = 20
interval_turns = 10
min_remaining_budget_fraction = 0.25
block_on = "any"
min_blocking_models = 2
reviewers = [
  { focus = "security" },
  { focus = "correctness", model = "openrouter/moonshotai/kimi-k2", effort = "high" },
]
```

A finding grounds on the path, side and lines it cites against the diff; an ungrounded finding is dropped.
A `block` holds only when its seat's verdict is `block` and its category is one of `security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`, `verify-uncovered-correctness` (the last only when verify passed); otherwise it reads as `warn`, which cannot stall the run.
Findings injected in an earlier round or execution neither repeat nor count toward a later panel's gate while the blob they cite is unchanged.

## `[compaction]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

<!-- config-table: compaction -->

## `[prompt]`

<!-- config-table: prompt -->

## `[skills]`

Operator-installed SKILL.md packs (the agentskills.io format, shared with Claude Code), under `$XDG_DATA_HOME/agent6/skills/<name>/`.

A skill name is 1 to 64 characters of lowercase ASCII letters, digits and hyphens, with no leading, trailing or doubled hyphen, and equals its skill directory name.
Discovery skips a skill that breaks this rule with a warning.

- `agent6 skills install <url>` takes a SKILL.md URL, a git repo (every `skills/*/SKILL.md`) or a local path; `extra_dirs` points at an existing collection (`~/.claude/skills`, …) instead of copying
- installed means enabled: an index in the system prompt, on-demand content via `use_skill`, a `/<name>` pause-menu command, and `run --skill <name>`
- repo-local skill dirs are not discovered (repo content is not config); the trust model is in [security.md](security.md)

<!-- config-table: skills -->

When a skill must apply, use `always`, `/name`, or `--skill`.

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost, else cached price × tokens) count against `max_usd`; a call carrying a plan-usage reading counts percentage points against `max_plan_points`; the rest count input+output tokens against `max_tokens_fallback`.
All three: `-1` unlimited, `0` refuse that ledger up front, `> 0` the cap.

<!-- config-table: budget -->

`--max-usd` / `--max-tokens-fallback` / `--max-plan-points` override per run.
A flag's value is recorded with the run: a resume without that flag runs under it again, and a flag on the resume replaces it for later resumes too.
Purchased-credit spend and API cost share the USD ceiling.
Prices come from provider listings (OpenRouter's; cached under `$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its OpenRouter listing.

## `[machine]`

<!-- config-table: machine -->

## `[notify]` (optional)

`on_event` is one operator argv on the host for every notification, with a minimal env (PATH/HOME/locale/desktop-bus), never your full environment.
`AGENT6_NOTIFY_EVENT` names the event (`session.end`, `machine.notify`, `machine.end`), `AGENT6_NOTIFY_MESSAGE` carries the end reason or the machine's message, and `AGENT6_NOTIFY_LEVEL` reads `ok` or `failed` for an end and `info`, `warn` or `error` for a machine notify.
A session adds `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`: `OK=1` means the agent stopped deliberately; `VERIFIED` is what the gate said; a hook that wants "green" reads the second.
A machine adds `AGENT6_MACHINE_ID/DIR/STATE`.
Global/repo config only (a machine `[config]` overlay setting `on_event` is rejected).
Fan out to ntfy/Pushover/email/Telegram yourself.

<!-- config-table: notify -->

## `[web]`

Bind and optional metrics route for `agent6 web` ([the web UI](web.md)).
Loopback only by default; remote access is expected behind `tailscale serve`.
Every browser [signs in](web.md#sign-in) with the server's secret.

<!-- config-table: web -->

## `[telemetry]`

What `agent6 telemetry export` sends, and where ([Telemetry export](cli.md#telemetry-export)).
The web scrape route requires `metrics = "prometheus"` and the [`web.metrics_route`](#web) opt-in.
Off by default: no signal is selected, nothing is dialled, no socket is opened.
Header values live in `secrets.toml` under `[telemetry.headers]`, never in config.
The [`--secret` target](#creating-and-inspecting) writes and removes them; `[telemetry].headers` selects which names the exporter sends.

<!-- config-table: telemetry -->

## `[parallel]`

Lane limit for `run --parallel N|[provider/]model,...`.
Each lane is a disposable clone running its own `agent6/<id>` branch; lanes are imported and ranked, nothing merges for you.
`--max-usd` is per lane (the total is printed before spawning); `--auto-approve` forwards to every lane.
A live run dispatches lanes the same way via `/parallel [SPEC --] TASK` ([grammar and examples](usage.md#parallel-directive)).
A lane never fans out; headless surfaces without a dispatcher answer "not available".

<!-- config-table: parallel -->

## `[workspace]`

Placement and setup for lane clones, machine state clones, forks and `run --worktree` worktrees.

<!-- config-table: workspace -->

## `[mcp]` and `[mcp.servers.<name>]` (optional)

agent6 as an MCP client; for the other direction (`agent6 mcp serve`, agent6 as a server) see [Editor integration](acp.md#as-an-mcp-server).

MCP servers, spawned (`command`) or connected (`url`); tools appear as `mcp__<name>__<tool>` in run mode only (plan, ask, machine and read_only sessions never offer them: agent6 cannot classify an external tool as read-only).
A spawned server runs as a jailed child by default (its own `[mcp.servers.<name>.sandbox]` policy; `unconfined = true` opts out) with a curated env (never your provider keys; `pass_env` adds named vars); a `url` server is a process you run and confine yourself.
The model chooses the arguments, so each call is approved like a command (`approve`); audit each server like a `run_command` allow-list.

`agent6 mcp add` saves a spawned server unprobed with a warning when `sandbox.isolation` resolves to `none` or a run would refuse the config; otherwise it requires a successful probe before writing the entry.
During a run, a server that does not start is skipped with an `mcp.server_unavailable` journal event, never fatal.
`agent6 check mcp` probes under the run's sandbox with the repository bound read-only and skips a server it cannot hold that way ([the rule](security.md#2-sandbox)).

```
agent6 mcp add files -- npx -y @modelcontextprotocol/server-filesystem .
agent6 mcp add browser --url http://127.0.0.1:8931/mcp --token-env PW_TOKEN
agent6 mcp list
agent6 mcp remove files
```

`agent6 mcp remove <name>` drops the entry from the global config (`--repo` for the per-repo one): `config unset` cannot, and dropping its `command`/`url` is refused.

A spawned server's `[sandbox]` block names what it needs on top of the sandbox a `run_command` gets.

```toml
[mcp.servers.notes]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/you/notes"]

[mcp.servers.notes.sandbox]
extra_write_paths = ["~/notes"]   # additive; includes read access
```

A confined spawn also loses the desktop-session addresses (`DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `DISPLAY`, `WAYLAND_DISPLAY`): Landlock does not gate unix-socket `connect()`, and an unconfined session daemon would act on the server's behalf.
Anything else that reaches an unconfined process is still a way out, so name the narrowest paths that work.

<!-- config-table: mcp mcp.servers.<name> -->

### `[mcp.servers.<name>.sandbox]`

<!-- config-table: mcp.servers.<name>.sandbox -->

---

## Reaching a run's network

When `sandbox.network` resolves to `session`, a run's commands share one network with no route off the box, so a dev server the agent starts is reachable only through these operator-only commands:

```
agent6 sessions show <id>        # what it is serving, and the command to open it (the TUI dashboard and web run headers say the same)
agent6 forward <id>              # list the ports it is listening on
agent6 forward <id> 3000         # bridge that port to one on this machine
agent6 exec <id> -- curl ...     # run a command in the run's sandbox
```

`exec` runs in the whole sandbox (the run's recorded isolation and network, with mounts from your current config), so what you see is what the agent sees.
None of them is reachable by the model.

---

## Environment variables

| Variable | Effect |
|---|---|
| `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME` | The XDG base directories; agent6 lives in an `agent6/` dir under each (config and secrets; per-repo state; the cache; installed skills). |
| `AGENT6_DETACHED_AWAY` | approval and question handling without a terminal or editor; `AGENT6_DETACHED_AWAY=wait`: both parked for a front-end; `AGENT6_DETACHED_AWAY=deny`: approval refused; `AGENT6_DETACHED_AWAY=approve`: approval granted; `deny`: a waiting question is unanswered at once, counted by `sessions show`; `wait`, `approve` or unset: a waiting question waits for a front-end; a queued question (`ask_user` with `wait` false) queues in every mode; hub and machine spawns: `AGENT6_DETACHED_AWAY=wait`; `--parallel` lanes: coordinator's marker, otherwise wait with a terminal to attach from, deny without one |
| `AGENT6_AUTO_APPROVE` | `1` grants every command approval to a machine's agent states, as `--auto-approve` does (a configured `deny` stays deny). |
| `AGENT6_NO_COMMANDS` | `1` withholds every command tool from a machine's agent states, as `--no-commands` does. |
| `AGENT6_JAIL_BIN` | Path to a specific `agent6-jail` binary (else bundled). |
| `AGENT6_DISABLE_SANDBOX` | `1` forces `sandbox.isolation = "none"` for one invocation (same as `--disable-sandbox`). |
| `AGENT6_ALLOW_ROOT` | `1` permits running as root (same as `--allow-root`). |
| `AGENT6_WEB_SECRET` | browser sign-in secret for `agent6 web`; at least 12 characters after lowercasing and joining words with single hyphens; overridden by `--secret`; unset: fresh secret generated and printed per launch |

A provider's `api_key_env` names the env var supplying its key; omit it to read `secrets.toml`.
The bench and development switches are listed in [architecture.md](architecture.md#bench-and-development-switches).
