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

<!-- config-table: agent6 -->

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`. Three
orthogonal choices describe any backend: **`api_format`** (the wire dialect,
the only field that selects code), **`deployment`** (URL/placement quirks of
where it is hosted), and **auth** (`auth_style` + `api_key_env` or
`token_command`). A minimal block is just `api_format` (plus `base_url` for a
non-default host).

<!-- config-table: providers.<name> -->

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
<!-- config-table: models.worker -->

## `[sandbox]`

The security boundary. The model:
[Isolation-level selection](security.md#3-isolation-level-selection) and
[State-machine egress](security.md#8-state-machine-egress-script-bundles) in
security.md. This is the field summary.

<!-- config-table: sandbox -->

## `[git]`

<!-- config-table: git -->

### `[git.commit]`

<!-- config-table: git.commit -->

### `[git.commit.checkpoint]` and `[git.commit.squash]`

<!-- config-table: git.commit.checkpoint git.commit.squash -->

## `preset` (top-level)

<!-- config-table: preset -->

## `[workflow]`

<!-- config-table: workflow -->

## `[review]`

<!-- config-table: review -->

Grounding is mechanical, not prose: a `block` gates only if its `file:line`
is in the diff AND its category is in a fixed allowed set (security /
sandbox-bypass / off-topic-edit / data-loss / verify-uncovered-correctness);
everything else is advisory and cannot stall the run.

## `[context]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

<!-- config-table: context -->

## `[prompt]`

<!-- config-table: prompt -->

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

<!-- config-table: skills -->

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

<!-- config-table: workflow.metric -->

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost,
else cached price × tokens) count against `max_usd`; unpriceable calls count
input+output tokens against `max_tokens_fallback`. Both: `-1` unlimited, `0`
refuse that ledger up front, `> 0` the cap.

<!-- config-table: budget -->

`--max-usd` / `--max-tokens-fallback` override per run; an explicit
`--max-usd` refuses to start when the worker has no price data. Prices come
from provider listings (OpenRouter's; cached under
`$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its
OpenRouter listing.

## `[machine]`

<!-- config-table: machine -->

### `[machine.notify]` (optional)

Operator hook on every `machine.notify` and the terminal `machine.end`: an
operator argv on the host with a minimal env (PATH/HOME/locale/desktop-bus +
`AGENT6_MACHINE_ID/DIR/EVENT/STATE/MESSAGE/LEVEL`), never your full
environment. Global/repo config only (a machine `[config]` overlay setting it
is rejected). Fan out to ntfy/Pushover/email/Telegram yourself.

<!-- config-table: machine.notify -->

## `[notify]` (optional)

Runs after `run`/`resume` with the same minimal env plus
`AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`. `OK=1` means the agent stopped
deliberately; `VERIFIED` is what the gate said — a hook that wants "green"
reads the second.

<!-- config-table: notify -->

## `[web]`

Bind for `agent6 web` ([the web UI](web.md)). Loopback only by default, no
app auth: remote access is expected behind `tailscale serve`.

<!-- config-table: web -->

## `[parallel]`

Fan-out defaults for `run --parallel N|model-a,model-b`. Each lane is a
disposable clone running its own `agent6/<id>` branch; lanes are imported and
ranked, nothing merges for you. `--max-usd` is per lane (the total is printed
before spawning); `--auto-approve` forwards to every lane. A live run
dispatches lanes the same way via the `/parallel` steer directive (depth 1: a
lane never fans out; headless surfaces without a dispatcher answer "not
available").

<!-- config-table: parallel -->

## `[mcp]` + `[mcp.servers.<name>]` (optional)

MCP servers, spawned (`command`) or connected (`url`); tools appear as
`mcp__<name>__<tool>`. A spawned server runs as a jailed child by default
(its own `[mcp.servers.<name>.sandbox]` policy; `unconfined = true` opts
out) with a curated env (never your provider keys; `pass_env` adds named
vars); a `url` server is a process you run and confine yourself. The LLM
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

<!-- config-table: mcp mcp.servers.<name> -->

### `[mcp.servers.<name>.sandbox]`

<!-- config-table: mcp.servers.<name>.sandbox -->

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
