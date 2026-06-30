# bench/machines: runnable agent6 state machines

Worked, end-to-end examples of [agent6 state machines](../../docs/state-machines.md):
small operator-authored mini-agents whose control flow is a static graph and
whose work happens inside jailed `agent` and `tool` states. Each directory is a
self-contained bundle (`<name>.asm.toml` plus an optional `scripts/`) with a
`run.sh` that drives it against a throwaway repo, so a run never touches this
checkout.

These were built and run to exercise the feature end to end; the bugs that shook
out are fixed in the same branch (see the `fix(machine)` / `fix(cli)` commits).

## The machines

| machine | demonstrates | run |
|---|---|---|
| [hello](hello/) | the smallest machine: one `agent` state → terminal | `bash hello/run.sh` |
| [multi-model-council](multi-model-council/) | two jurors on different providers, a `branch` consensus, a stronger tiebreaker only on disagreement, `{{ x \| json }}` into a prompt | `bash multi-model-council/run.sh` |
| [code-fixer](code-fixer/) | a real coding loop: a `mode="run"` agent edits to make a test pass, a `tool` verifies, a `branch` loops on an attempt counter | `bash code-fixer/run.sh` |
| [repo-digest](repo-digest/) | `tool` (git log, typed capture) → `agent` (list-field schema) → `tool` (splice the agent's `list[str]` into argv, write to the data dir) | `bash repo-digest/run.sh` |
| [triage-inbox](triage-inbox/) | the canonical poll → classify → act loop: a `wait`, an enum schema, a compound branch predicate, self-terminating when the inbox drains | `bash triage-inbox/run.sh` |
| [wait-clock](wait-clock/) | timing: `until` (deadline), the `--exit-on-wait` persisted-wake driver (pulse), and the v1 cron-reject (cron-demo) | `bash wait-clock/run.sh` |

Observed (one run each, 2026-06-29):

- **hello** — claude-haiku-4-5, 1 transition, ~$0.
- **multi-model-council** — haiku + kimi jurors agreed "91 is not prime" (3
  transitions, $0.0025). A rigged split exercised the sonnet tiebreaker → 4
  transitions, ending `decided`.
- **code-fixer** — claude-sonnet-4-6 fixed `median` in one attempt (3
  transitions); the verify `tool` confirmed; the agent's commits landed.
- **repo-digest** — 5 commits → 4 transitions, digest with 4 highlights written
  to the data dir (claude-haiku-4-5).
- **triage-inbox** — 4 items triaged over 26 transitions (~26s, paced by the
  `wait`); the outage escalated, the rest filed.
- **wait-clock** — deadline fired immediately; pulse persisted its wake and
  resumed to `done`; cron-demo is rejected by `machine check`.

## Sandboxing

All of these run under the **default** sandbox config. Each `tool` state gets its
own network namespace from the jail launcher (the `strict` profile), and each
`agent` state confines its egress to the provider API. That holds even on a host
where the kernel blocks the user namespace the agent egress broker needs (the
agent state falls back to the hardened profile, confining egress with Landlock
instead) — the tool jails keep their per-tool isolation regardless.

On a host that supports **only** the hardened profile (no per-tool network
namespace at all), a `tool` state is refused under the default
`sandbox.tool_network = "block"`. agent6 prints the exact one-line config opt-in
to apply (`sandbox.tool_network = "allow"` + `sandbox.agent_network = "open"`,
letting tools share the host network) and never relaxes the sandbox unattended.
The pure-agent machines (hello, council) and the pure-timer machine (wait-clock)
have no tool states and run confined on any profile.

## Test harnesses

- [_invalid/check_errors.sh](_invalid/check_errors.sh) — 14 intentionally-broken
  machines, asserting `machine check` rejects each at load with a precise
  diagnostic (non-total branch, dangling goto, `len()` of an int, a TOML boolean
  in a predicate, a float `every_secs`, cron, …).
- [_create-bench/run.sh](_create-bench/run.sh) — `agent6 machine create` across
  six providers/models on one poll → classify → act task; see
  [_create-bench/README.md](_create-bench/README.md).
