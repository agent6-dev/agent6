#!/usr/bin/env bash
# Show that `agent6 machine check` rejects malformed machines at LOAD time with a
# precise, fail-loud diagnostic. Each case is an intentionally-broken machine;
# the script asserts `check` exits non-zero and prints the expected phrase. This
# doubles as a regression demo of error quality (the bracketed phrases are the
# load-time guarantees the spec makes).
#
# Usage:  bash bench/machines/_invalid/check_errors.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

HEAD='machine = "bad"
version = 1
initial = "s"
[budget]
max_transitions = 5'

check_case() {
  local name="$1" want="$2" body="$3"
  printf '%s\n%s\n' "$HEAD" "$body" > "$TMP/m.asm.toml"
  local out; out="$("$AGENT6" machine check "$TMP/m.asm.toml" 2>&1)"
  if [ $? -ne 0 ] && grep -qF "$want" <<<"$out"; then
    printf '  ok   %-28s -> %s\n' "$name" "$want"; pass=$((pass+1))
  else
    printf '  FAIL %-28s (wanted %q)\n     got: %s\n' "$name" "$want" "$out"; fail=$((fail+1))
  fi
}

check_case "non-total branch" "not total" '
[vars.code]
n = { type = "int", default = 0 }
[states.s]
kind = "branch"
when = [ { if = "n == 0", goto = "s" } ]'

check_case "dangling goto" "not a declared state" '
[states.s]
kind = "wait"
every_secs = "5"
on = { tick = "ghost", signal = "s" }'

check_case "unreachable state" "unreachable" '
[states.s]
kind = "terminal"
status = "ok"
reason = "x"
[states.orphan]
kind = "terminal"
status = "ok"
reason = "y"'

check_case "tool capture into agent var" "owned by" '
[vars.agent]
v = { type = "str", default = "" }
[states.s]
kind = "tool"
command = ["true"]
timeout_secs = 5
capture = { set = { v = "{{ result }}" } }
on = { ok = "s", nonzero = "s", timeout = "s" }'

check_case "dotting a json var" "cannot navigate into json" '
[vars.code]
blob = { type = "json", default = {} }
[states.s]
kind = "branch"
when = [ { if = "blob.k == 0", goto = "s" }, { else = true, goto = "s" } ]'

check_case "len() of an int" "does not apply to int" '
[vars.code]
n = { type = "int", default = 0 }
[states.s]
kind = "branch"
when = [ { if = "len(n) >= 0", goto = "s" }, { else = true, goto = "s" } ]'

check_case "TOML boolean in predicate" "True/False/None" '
[vars.code]
b = { type = "bool", default = false }
[states.s]
kind = "branch"
when = [ { if = "b == true", goto = "s" }, { else = true, goto = "s" } ]'

check_case "float every_secs ref" "int variable" '
[vars.operator]
secs = { type = "float", value = 1.0 }
[states.s]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "s", signal = "s" }'

check_case "zero every_secs literal" ">= 1" '
[states.s]
kind = "wait"
every_secs = "0"
on = { tick = "s", signal = "s" }'

check_case "garbage until literal" "ISO-8601" '
[states.s]
kind = "wait"
until = "tomorrow"
on = { tick = "s", signal = "s" }'

check_case "cron not implemented" "cron" '
[states.s]
kind = "wait"
cron = "* * * * *"
on = { tick = "s", signal = "s" }'

check_case "two wait timings" "exactly one" '
[states.s]
kind = "wait"
every_secs = "5"
until = "2020-01-01T00:00:00Z"
on = { tick = "s", signal = "s" }'

check_case "duplicate var across owners" "declared in both" '
[vars.code]
x = { type = "int", default = 0 }
[vars.agent]
x = { type = "int", default = 0 }
[states.s]
kind = "terminal"
status = "ok"
reason = "x"'

check_case "enum on non-str field" "enum" '
[schemas.r]
n = { type = "int", enum = [1, 2] }
[states.s]
kind = "terminal"
status = "ok"
reason = "x"'

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]
