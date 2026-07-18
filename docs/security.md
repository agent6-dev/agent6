# SECURITY

agent6 treats the LLM as untrusted. Concrete claims below, layer by layer, each
with what it means and where it stops.

## Reporting

Open a GitHub issue prefixed `[security]`. Include: agent6 version
(`agent6 --version`), kernel + distro (`uname -a`, `/etc/os-release`),
`agent6 check sandbox` output, and a minimal repro (ideally a failing test under
`tests/security/`).

## Threat model

Adversary: a fully malicious worker model, or an honest model that has
been prompt-injected by a file in the workspace.

We assume the adversary controls:

- the text of every LLM response
- the choice of tool calls and their arguments (within the published
  JSON schema)
- the content of any file the agent reads during the run

We do NOT assume the adversary controls:

- the kernel
- the agent6 binary
- the provider endpoints

Under that adversary, agent6 aims to hold:

1. **No writes outside the workspace.**
2. **No reads outside the workspace and a read-only system set.**
    - The system set (`/usr /bin /sbin /lib /lib64 /etc /dev /proc /tmp`) exists
      so installed toolchains resolve.
    - `hardened` also exposes `$HOME` + `/run` (Landlock can't carve them out);
      `sandbox.extra_read_paths` adds more.
3. **No agent egress except the configured providers** (+ `sandbox.allow_urls`),
   under `sandbox.agent_network = "providers"` (default; §1b).
    - Jailed commands are governed separately by `sandbox.tool_network`
      (default `block`; §8).
4. **agent6's own git never pushes, `--force`s, rewrites history, or `reset
   --hard`s** (§5).
    - This does NOT bind a `git` the model runs via `run_command`; that path is
      bounded by the sandbox (`protect_git` read-only-binds `.git` on `strict`;
      push needs egress).
5. **No persistence after the run:** no daemon, cron, or `.bashrc` write.
    - Children can only write inside the jail's mount namespace.

## Defense layers

### 1. Agent-process Landlock (`hardened` only)

Applied at `run`/`resume` start, before any network object. Restricts the Python
process irrevocably, inherited by every child:

| Landlock rule | Allowed |
|---|---|
| FS read+exec | cwd, `$HOME`, `/usr`, `/etc`, `/tmp`, `/bin` `/sbin` `/lib` `/lib64` `/dev`, `/run` + `/proc` when present |
| FS write | cwd, `/tmp`, the `/dev` char devices, `/proc` when present |
| TCP connect (kernel ≥ 6.7) | the *ports* of configured providers (each `base_url` port, default `443`) |

- **`strict` skips this layer.**
    - Its per-command namespaces + broker (§1b) are stronger, and this would
      break the jail's `pivot_root`/`mount` at Landlock ABI ≥ 7.
- **The read+exec set mirrors the jail child's roots.**
    - The launcher opens each from here to grant the child, so a missing one
      (e.g. `/dev` on merged-`/usr`) makes the child's execve fail EACCES.
- **The network rule filters by port, not host.**
    - It blocks other ports but can't pin egress to a host; use
      `agent_network = "providers"` (§1b) for that.
- **Kernels < 6.7 get FS-only Landlock, with a warning.**
    - Don't run there if the host UID can read exfiltratable credentials.

### 1b. Provider-only egress broker (`agent_network = "providers"`, default)

- **The agent's only path off-host is one hard-wired unix socket per allowed
  provider; the LLM never chooses a destination.**
    - Setup, `strict`-only (needs user namespaces): in the host netns and
      single-threaded, the agent binds one `AF_UNIX` socket per provider
      `host:port` and forks a **broker** that stays in the host netns.
    - The agent then `unshare(CLONE_NEWUSER|CLONE_NEWNET)`s into an empty netns
      (loopback only). Off-host is only those sockets.
    - Per connection the broker dials that socket's fixed `host:port` (resolved
      per-connect, robust to CDN IP churn). TLS is end-to-end, so it sees only
      ciphertext.
    - The allowed set derives uniformly from each provider's effective
      `base_url` host (every `api_format` and `deployment` carries the dialled
      host there; the deployment profile only appends path/model), unioned with
      `sandbox.allow_urls` (`app/egress.py`).
- **Fail-closed: the netns is the boundary, not a filter.**
    - A missing route is no connectivity, never a silent leak; the allow-list is
      fixed at bind time, so the agent can't widen it.
    - Hosts that only support `hardened` refuse rather than run unconfined.
- **`local` pins to loopback providers; `open` skips the broker.**
    - `local` refuses a non-local provider. `machine run` applies the same setup
      per `agent`-state subprocess.
- **A detached resume is spawned from the host, not the empty netns.**
    - **Host spawner** (`sandbox/host_spawn.py`): a helper forked beside the
      broker before isolation spawns `agent6 resume <run_id>` from the host, on
      request over a close-on-exec pipe, exec'ing only the agent6 binary captured
      at fork. Neither argv nor pipe is reachable from LLM output.
    - **Inherited-namespace refusal**: `AGENT6_NETNS_ISOLATED=1` makes a child
      that sees it refuse with the cause, not burn provider retries.
- **`sandbox.allow_urls` widens the allow-list; only the operator can.**
    - Each entry (`host` / `host:port` / URL) gets its own broker socket, same
      properties. Default empty, validated at load, honored only under
      `providers`, agent path only. Last-overlay-wins, so `config show` is
      authoritative.
- **MCP servers get no outbound network under `providers`** (a deliberate limit;
  local `AF_UNIX` helpers are unaffected).

### 2. `agent6-jail` (Rust), for every child command

`apply_edit` is in-process; every `run_verify_command`/`run_command` runs in
`agent6-jail`. Under `strict` it:

- Forks a new user/mount/PID/IPC/UTS/net namespace.
- `pivot_root`s into a minimal bind-mount rootfs on a fresh tmpfs: cwd + private
  `/tmp` writable, system paths (+ `extra_read_paths`) read-only. Operator-tool
  dirs join as read+exec `tool_paths` mounts (standard bin dirs that exist, the
  real dirs their symlinks resolve to, uv-managed CPythons), derived by
  `sandbox.jail.operator_tool_paths`; run_command/verify jails and machine tool
  jails share that one computation, and `machine check` probes the same PATH.
- Exposes curated `/dev` (`null zero urandom random full`); omits `/dev/tty`
  (it would let a child write escape sequences to the parent's terminal).
- Mounts a fresh private `/proc`; if that fails, leaves `/proc` empty (never the
  host's, which would leak process info).
- Applies Landlock FS rules (net confinement is the namespace).
- Installs a seccomp deny-list: dangerous syscalls (ptrace, mount, setns,
  unshare, kexec, bpf, perf, keyctl, module loading, reboot, clock-set, …)
  return `EPERM`, the rest allowed.
- Sets `NO_NEW_PRIVS`, so the kernel ignores setuid bits (`sudo`/setuid can't
  escalate).
- `execve`s the binary and SIGKILLs the group at the wall-clock timeout.

Notes:

- **The memory cap is operational, not a threat-model control.** A per-process
  `RLIMIT_DATA` (`[sandbox].memory_limit_mb`, default 4096, `0` off; not
  `RLIMIT_AS`, so V8/JVM/ASAN keep working) stops one runaway allocation, nothing
  more.
- **No `capset`.** `strict` maps namespaced-root to your uid; `hardened` keeps the
  caller's caps (none for a normal user).
- **`hardened` drops the namespaces + rootfs;** Landlock, seccomp,
  `NO_NEW_PRIVS`, and the timeout remain.
- The policy arrives as JSON on stdin from `run_in_jail`; the Rust side validates
  it against a strict schema and refuses unknown fields.

### 2a. Environment: sudo, packages, provisioning

The jail is one-way: the agent works within the environment you give it and
can't expand it.

- **`sudo` can't escalate, even passwordless.** `NO_NEW_PRIVS` voids setuid, so
  jailed `sudo` fails regardless of any `NOPASSWD` rule.
- **Package installs are impossible.** `apt`/`dnf`/`apk` need all three of root
  (blocked), mirror network (provider-only egress), and `/usr`/`/var` writes
  (denied).
- **Compiling and running host-installed toolchains works.**
    - `run_verify_command` and (when `run_commands` permits) `run_command` run
      jailed; they just can't install new tools, and a networked build step needs
      `tool_network` loosened.
- **Provisioning is operator-first.** Install toolchains, venvs, and deps
  yourself before/outside agent6; widen access via config, never sudo
  (`extra_read_paths`, `tool_network`, `[providers.*].base_url`, all in
  `config show`).
- **Running agent6 as root** (`--allow-root` / `AGENT6_ALLOW_ROOT=1`) **weakens
  the boundary.**
    - `strict` maps inside-root to real root, so jailed children run as real root
      under only Landlock + seccomp + `NO_NEW_PRIVS`.
    - Still no writes outside the workspace and no egress beyond providers, but
      the allowed *reads* now include root-only files (`/etc/shadow` under
      `hardened`; `strict`'s rootfs hides them). Run as your normal user.

### 2b. Host-side subprocess allowlist

Everything the model can influence runs through `run_in_jail` (§2). A fixed set
of modules also shells out directly with `subprocess.run`/`Popen`; each has
fixed argv depending only on operator input, never LLM output.
`tests/security/test_subprocess_allowlist.py` pins the file list; audit with
`rg 'subprocess\.(run|Popen)' src/agent6/`.

- `git_ops.py`: agent6's own git operations (§5).
- `sandbox/detect.py`: probes the host's sandboxing capabilities.
- `sandbox/jail.py`: the jail launcher itself.
- `sandbox/host_spawn.py`: the pre-forked detach helper; spawns
  `agent6 resume <run_id>` in the host namespaces, argv built from the agent6
  exe captured at fork time, requests only from the trusted parent over a
  close-on-exec pipe.
- `tools/lsp.py`: the `ty` language server, exe resolved from PATH.
- `tools/mcp_client.py`: operator-configured `[mcp.servers.*]` server commands.
- `providers/token_command.py`: the operator-configured
  `[providers.*].token_command` that mints a provider bearer; argv from config.
- `ui/spawn.py`: the shared front-end spawn helper; spawns the agent6 CLI
  detached for run/machine launches and captures `runs merge`/`prune`/
  `config set`; argv is the agent6 exe plus operator-chosen args.
- `ui/notify.py`: fires `notify-send` with fixed argv (exe, `--`
  end-of-options, two positional data args, no shell) for the device-present
  machine notification; the message is inert data, never a command or an
  option.
- `ui/cli/` helpers:
    - `$EDITOR` for plan and steer editing.
    - `git diff/log` for the review subcommand and the `runs`/`ask` diff views;
      argv from the run manifest the CLI wrote outside the jail.
    - `rg` for history search.
    - The fixed-argv `python -m agent6.ui.tui` co-process behind `run --tui`.
    - `ui/cli/system_cmds.py`: `cp`/`rm`/`apparmor_parser` via sudo with fixed
      argv for `agent6 system apparmor` (operator host setup).
- `app/` helpers:
    - `app/finalize.py`: the operator `[notify].on_complete` hook fired at
      run end; argv from config.
    - `app/machine/_scriptcheck.py`: ruff/ty with fixed argv to statically read
      generated scripts, which only ever execute via `run_in_jail`.
    - The `machine run` supervisor (`app/machine_agent.py`): spawns each agent
      state as a fixed-argv `python -m agent6.ui.cli.machine_agent` subprocess
      whose request travels in a temp file, never on argv; its operator
      `[machine.notify].on_event` hook (argv from config, fired from
      `app/machine/_preflight.py`) runs on the host with `AGENT6_MACHINE_*`
      env, mirroring `[notify].on_complete`.
    - `ui/cli/skills_cmds.py`: `git clone --depth 1 -- <url>` with fixed argv
      for `agent6 skills install`; the URL is operator-supplied on the CLI and
      nothing fetched is ever executed.
- `ui/tui/clipboard.py`: fixed-argv `tmux set-buffer -w` with the copied
  transcript text as one inert data argument.
- `ui/tui/conversation.py`: the operator's `$PAGER`, argv from the environment,
  transcript text on stdin.

### 3. Profile selection

You set `sandbox.profile`; it resolves against the host to the *effective*
profile. No silent downgrade: a request the host can't meet is refused, and
`auto` reaches `none` only by detecting a non-Linux host.

| `sandbox.profile` | Host | Effective |
|---|---|---|
| `auto` *(default)* | Linux + user namespaces | `strict` |
| `auto` | Linux, no user namespaces | `hardened` |
| `auto` | non-Linux | `none` |
| `strict` | Linux + user namespaces | `strict` |
| `strict` | else | ⛔ refuse |
| `hardened` | Linux | `hardened` |
| `hardened` | non-Linux | ⛔ refuse |
| `none` *(opt-out)* | any | `none` (the environment is the boundary) |

- **strict**: full namespaces + `pivot_root` + Landlock + seccomp + `NO_NEW_PRIVS`.
- **hardened**: Landlock + seccomp + `NO_NEW_PRIVS`, no namespaces.
    - Works in default-seccomp Docker (the container blocks the inner
      `clone(CLONE_NEW*)`); the container is the blast radius.
- **none**: unsandboxed, always with a loud warning.

- **Unsandboxing is explicit and self-authorizing.** `profile = "none"`,
  `--dangerously-disable-sandbox`, or `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`. The
  LLM can't reach argv/env, so setting one is the consent.
- **Sandbox-off + auto-approved `run_command` adds a one-time gate.** For that
  combination only: `Continue? [y/N]` interactively, a warning in CI/`machine
  run`.
- CI should set `strict` to fail loud if the sandbox is weaker than expected.

### 4. Fixed tool surface

- **The LLM only sees the fixed set in `src/agent6/tools/schema.py`.**
    - Structured edits, read-only navigation, fixed-argv verify/metric commands,
      `finish_run`, `ask_user`, a curator task notepad, a cross-run memory
      notepad, and capability-gated `run_command`.
    - No `shell`, no `write_file` (writes go through `apply_edit`, which refuses
      paths outside cwd), no `web_fetch`, no `eval`.
    - Adding a tool needs a security review note ([AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md)).
- **The memory notepad is a prompt-injection persistence channel.**
    - `add_memory`/`invalidate_memory` (run mode) write fixed markdown under
      `<state-dir>/<repo-id>/memories/` (code picks the path; the model supplies
      only a schema-validated scope + text); active notes join later runs' system
      prompt on the same repo.
    - Mitigated: notes are inert data (never executed), the injected block is
      size-capped and framed as untrusted, and the store is operator-auditable
      (`agent6 memory list --all`, `agent6 memory invalidate` keeps the trail).
    - It weakens no boundary here: sandbox/egress/git policy come from config, not
      prompt content.

### 5. Git invariants

- **agent6's own git refuses the destructive ops, by construction.**
    - `git_ops.py` is the only module through which agent6 invokes git; it wraps
      the safe ops (status, add, commit, diff, branch, checkout) and refuses
      `push`, `reset --hard`, `commit --amend`, `rebase`,
      `filter-branch`/`filter-repo`, `branch -D`/`--force`, and any `--force`/`-f`
      on a destructive verb.
    - `git.allow_push`/`allow_force`/`allow_history_rewrite` exist for
      forward-compat but are ignored.
- **A `git` the model runs via `run_command` is bounded by the sandbox, not this
  list.**
    - On `strict`, `protect_git` read-only-binds `.git`, so a rewrite fails and
      `push` has no egress. On `hardened`, `.git` is writable, so the container is
      the boundary.
- **git_ops neutralizes repo-controlled host code in a poisoned `.git/config`.**
    - `core.fsmonitor` and `diff.external` are always off; `.git/hooks/*` run only
      under `git.run_repo_hooks = true` (default false; `core.hooksPath` points
      away so a hook can't fire on agent6's auto-commit).
    - On `strict` this complements `protect_git`'s RO `.git`. On `hardened` the cwd is
      blanket RW (an RO `.git` would break cargo/pytest creating
      `target/`/`.pytest_cache/`), so `.git` is writable there; acceptable, gated
      by `run_commands`, recoverable (branch-per-run, commits via git_ops),
      container is the blast radius.
- **The edit tools refuse writes into an in-repo venv or `site-packages`.**
    - A `pyvenv.cfg` dir or `site-packages` ancestor: a run rewriting an
      editable-install `.pth` would silently corrupt the venv, invisible in `runs
      diff`/merge since venvs are gitignored. Reads stay allowed.
    - Related limit: an editable install records the host path in its `.pth`,
      absent under the jail's `/workspace`, so a `verify_command` importing the
      project can `ModuleNotFoundError`. Fix with pytest `pythonpath`, a
      `conftest.py`, or a non-editable install.

### 5b. Secrets, `connect`, root

- **Provider keys are `0600`, owner-only, and never leave agent6's process.**
    - In `$XDG_CONFIG_HOME/agent6/secrets.toml` (refused if group/other-readable
      or foreign-owned, like an SSH key), or from `[providers.<name>].api_key_env`
      (env wins). Never in transcripts, never in `config show` (redacted), never
      mounted into the jail.
- **`agent6 connect` never executes remote input.**
    - It only prompts locally (`getpass`) and writes config/secrets. It makes one
      read-only `GET` to the provider's key endpoint to confirm auth (status only;
      `--no-verify` to skip).
    - During a run agent6 opens no listening socket (MCP is stdio, the broker is a
      private unix socket); the only accept-side socket is opt-in `agent6 web` (§7).
- **Running as root is refused without an explicit opt-in.**
    - `--allow-root` / `AGENT6_ALLOW_ROOT=1` (+ a banner). Under `sudo`, agent6
      reads the *real* user's config/secrets (from `SUDO_UID`/`SUDO_USER`), not
      root's, and chowns state-dir writes back. It doesn't drop privileges
      in-process: the jail, not the uid, is the boundary.

### 6. Curator + state location

- **An in-process `GraphCurator` owns the task graph.**
    - It validates every mutation against a pydantic schema before writing, and
      holds a per-mutation flock on the run dir. A write-path fault after the
      in-memory update reloads from disk before surfacing, so a later read never
      observes a node that was never persisted.
- **The run directory is safe because of its location, not any single writer.**
    - Per-repo state lives at `$XDG_STATE_HOME/agent6/<repo-id>/` (override with
      `[agent6].state_dir`), outside the cwd jailed commands run on.

### 6b. Parallel lanes (fan-out / coordinator dispatch)

`agent6 run --parallel`, `agent6 runs compare`, and a live run's `/parallel`
steer directive (§ [architecture.md](architecture.md#parallel-runs-fan-out-and-coordinator-dispatch))
each spawn subordinate work. Nothing here loosens the sandbox:

- **Every lane is an ordinary run.** A lane is a plain detached `agent6 run` on
  its own clone: its own jail per `sandbox.profile`, its own egress broker
  (§1b), its own `run_commands` policy. Nothing shares a sandbox or a broker
  socket across lanes or with the parent run.
- **Recursion is blocked by an env guard, not policy.** Every spawned lane
  carries `AGENT6_SUBRUN=1`; both the `--parallel` flag and the coordinator's
  `lane_spawner` wiring refuse when it is set, so a lane can never itself fan
  out or dispatch (depth 1 by construction).
- **A lane's config carries key references, never secret values.** The
  orchestrator writes each lane a `--config` file via `materialize()`, a dump
  of the resolved `Config` model (provider `base_url`, `api_key_env` names,
  etc.) -- `Config` never holds a raw API key. The lane's own process reads
  the same `secrets.toml` / provider env var as any other run, same user, same
  host.
- **No new subprocess call site.** `workflows/subrun.py`, `app/parallel.py`,
  and `ui/cli/parallel.py` add no direct `subprocess` use; lane git plumbing
  (clone/fetch/merge) goes through `git_ops.py` and lane spawning goes through
  `ui/spawn.py`, both already on the §2b allowlist. The
  `tests/security/test_subprocess_allowlist.py` pin needed no new entry.
- **Dirty-tree refusal, not auto-stash.** A lane clones committed HEAD only,
  so `--parallel` refuses a dirty origin under `git.require_clean_worktree`
  (the same policy and message shape `agent6 run` uses) rather than carrying
  uncommitted work into a lane it cannot see.

### 7. No agent-owned network surface (except opt-in `agent6 web`)

- **The loop opens no accept-side socket.**
    - Only outbound HTTPS to the provider; the task graph is an in-process
      curator, no socket.
- **`agent6 web` is the one accept-side socket, and only when you start it.**
    - Loopback (`127.0.0.1`) by default, no app auth (run behind `tailscale
      serve`; the tailnet identity is the access control, see [the web UI](web.md)).
    - A non-loopback bind is refused unless opted in: `[web].host` needs
      `[web].allow_non_loopback = true`, `--host` needs `--allow-non-loopback`.
- **The server renders folded state and drives typed contracts only; it executes
  nothing.**
    - New-work spawns fixed argv with the task behind `--`; machine-run is
      allow-listed to authored files; answers write only the addressed run's
      answer files (run id, answer id, machine target state dir each validated to
      one path component); merge/prune/config-set are fixed agent6 subcommands.
- **State-changing POSTs carry a CSRF guard.**
    - Body must be `Content-Type: application/json` (a cross-site `fetch` with it
      triggers a preflight the server never answers) and any `Origin` must match
      `Host`. Holds on loopback and behind `tailscale serve`.
    - It does NOT cover DNS rebinding (that needs a Host allow-list incompatible
      with the tailnet name).
- **Request framing is bounded.** 1 MiB body cap (413), chunked refused (411),
  any unread-body refusal closes the connection.
- **The machine write surface (`POST /api/machine/<name>/{poke,answer,approve,steer}`)
  uses the same guards.** `poke` writes only the instance signal file (inert JSON
  the next `tool` reads); the others write only the current agent state's
  per-state dir. PWA assets are static; the service worker is a no-op passthrough
  (no Web Push/VAPID).
- No telemetry, no auto-update, no remote control plane.

### 8. State-machine egress + script bundles

- **`machine run` is a supervisor in the host netns that makes no network calls.**
    - Each `agent` state confines its own egress per `agent_network` (the broker,
      §1b); each `tool` state is jailed, so a per-tool `allow_network` sets its
      netns independently.
    - This lets a machine keep agents on the provider API while one reviewed,
      fixed-argv `tool` reaches the network: unlike `run_command` (LLM-chosen
      argv), a `tool` isn't a free exfil channel.

Egress = `agent_network` × `tool_network` × per-tool `allow_network`; the
effective profile decides what's enforceable. "offline" = no egress.

**Agent egress** by `agent_network`:

| `agent_network` | `strict` | `hardened` | `none` |
|---|---|---|---|
| `providers` *(def)* | providers + `allow_urls`, broker-pinned | provider *ports* only (Landlock) | unconfined ⚠ |
| `local` | loopback providers only, broker-pinned | ⛔ refuse | unconfined ⚠ |
| `open` | unconfined | unconfined | unconfined ⚠ |

**Jailed-command egress** (`run_command`, machine `tool`) by `tool_network`
(cells = `strict`):

| jailed command | `block` *(def)* | `only_explicit_states` | `allow` |
|---|---|---|---|
| `run_command` | offline | offline | host network |
| `tool`, `allow_network` `auto`(def)/`block` | offline | offline | offline |
| `tool`, `allow_network = allow` | ⛔ refuse | host network | host network |

**Refusals** (fail-closed):

| Configuration | When |
|---|---|
| `tool_network = allow` without `agent_network = open` | config load ¹ |
| a `tool` sets `allow_network = allow` under `tool_network = block` | machine start |
| `agent_network = local` or `tool_network = only_explicit_states` | run start, `hardened` ² |
| a machine with `tool` states, or a `tool` with `allow_network = block`, under `tool_network = block` | machine start, `hardened` ² |

- ⚠ `none` (non-Linux) is unsandboxed: nothing enforced, nothing refused, loud
  warning.
- ¹ `run_command` runs in the agent process, so it can't reach the network while
  the agent is confined.
- ² per-command isolation needs a netns, so it's `strict`-only; on `hardened` a
  jailed child inherits the agent's Landlock and the cases needing real isolation
  are refused.

More fail-closed properties:

- **Operator-gated policy.** `agent_network`/`tool_network` are read only from the
  operator's config; a machine's `[config]` overlay is rejected at load if it
  declares `[providers.*]`, `[sandbox.*]`, `[profiles.*]`, or `git.run_repo_hooks`.
    - Otherwise a profile preset or a host `[machine.notify]` argv could splice
      into the resolved config, and `run_repo_hooks` would run repo `.git/hooks`
      on the host on a `mode="run"` commit. A `tool` only *declares*
      `allow_network`; honoring `allow` is the operator's call, and every conflict
      is refused at startup naming the state.
- **Bundle confinement.** Scripts live in a reviewed `scripts/` beside the
  `.asm.toml`; `machine check` verifies every entry and static reference resolves
  inside the bundle (escaping symlinks rejected).
    - Scripts are operator-authored and committed, never fetched/generated at run
      time, and the `.asm.toml` + `scripts/` are RO in every jail during a run, so
      a state can't rewrite its own logic or add an `allow_network` flag.
- **Notifications don't widen the agent's surface.** Front-ends render
  `machine.notify` as an overlay, and `attach`/TUI call `notify-send` with a FIXED
  argv (no shell), so a model message is inert data.
    - The out-of-band hook `[machine.notify].on_event` runs an operator argv on the
      host with only `AGENT6_MACHINE_*` env (mirrors `[notify].on_complete`); a
      `[config]` overlay setting `[machine.notify]` is rejected at load. No Web
      Push/VAPID.

## Skills trust model

- **A skill is config, not repo content: install only from trusted sources.**
    - `agent6 skills install <url>` is an operator-initiated CLI fetch (same trust
      class as `connect`); what it installs enters the system prompt/tool results
      verbatim.
- **Nothing in a skill runs at install or load.**
    - Its scripts run only if the model runs them through the jailed command path,
      subject to `run_commands`.
- **`use_skill` is read-only and path-contained.** Serves the skill's own dir only
  (symlinks/`..` resolved first), never the repo or network. Skill dirs aren't
  mounted into the jail; content reaches the model engine-side.
- **Repo-local `.claude/skills/` are deliberately NOT discovered.** Third-party
  repo content must not enter the prompt; only the installed dir +
  `[skills].extra_dirs` are scanned.

## Prompt-injection resilience

[`tests/security/test_prompt_injection.py`](https://github.com/agent6-dev/agent6/blob/master/tests/security/test_prompt_injection.py)
runs an adversarial corpus through the planner/worker/reviewer prompts and
asserts no exfiltration, no out-of-policy tool calls, and no following embedded
instructions to weaken constraints. It's a smoke test, not a proof: the
structural defenses above are the real mitigation; the corpus catches prompt
regressions.

## Known limitations

- **Landlock TCP rules need Linux ≥ 6.7** (ABI ≥ 4); older kernels leave the agent
  process net-unconfined (children stay net-isolated in `strict` via the empty netns).
- **User namespaces must be enabled;** some distros disable them, and agent6
  refuses `strict` there.
- **AppArmor userns (Ubuntu 24.04+)** blocks unprivileged userns without a profile.
    - agent6 ships one scoped to the launcher (`agent6 system apparmor install`);
      with it, per-command jailing is `strict`, without it `hardened`.
    - Caveat: the egress broker needs the *agent process* to make a userns, which
      the launcher-only profile doesn't grant, so a default run downgrades to
      `hardened` (Landlock egress) unless you set the sysctl to 0 host-wide or use
      `agent_network = "open"`.
- **seccomp is required;** kernels that block it from unprivileged callers make the
  jail fail closed.
- **Devcontainers get `hardened`;** the container is the FS blast radius, network
  still Landlock-confined when supported. The XDG state base is ephemeral (lost on
  rebuild), so mount a volume at the state dir or set `[agent6].state_dir` to
  persist runs.
- **Side channels:** no claim about timing/cache/speculative side channels; don't
  co-locate agent6 with secrets if Spectre-class attacks are in your model.
- **Supply chain:** pin your install. Runtime deps `pydantic`, `httpx2`,
  `argcomplete`, the `tree-sitter` pair, `textual`, `ruff`, `ty`; build-dep
  `hatchling`; the jail's Rust crates `nix`, `libc`, `landlock`, `seccompiler`,
  `serde`, `serde_json`.
