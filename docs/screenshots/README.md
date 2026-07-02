# Screenshots

The TUI screenshots and tour video for the docs site (agent6.dev) are generated
from **real, recorded runs** with [vhs](https://github.com/charmbracelet/vhs):
no live LLM calls, no network, no API key. Small, sanitized run fixtures are
committed under `seed/`; vhs replays them through the TUI and captures the images.

We commit the **tape, scripts, and seed fixtures only**. Generated media
(`out/`, `*.png`, `*.webm`) is gitignored and rebuilt on demand.

## Files

- `tour.tape`: vhs script for the PNGs. Drives hub -> config -> search -> help ->
  dashboard -> transcript -> logs and captures one PNG per stop.
- `reel.tape`: vhs script for the `tour.webm` video. One TUI session (hub, into a
  run, transcript, log, config, search, help) with no `Screenshot` commands (they
  stall the video clock). Holds are paced by how dense each screen is.
- `keystroke_overlay.py`: overlays animated keystroke toasts on the recording,
  parsing the tape and scaling the keypress timeline to the actual duration.
- `seed/`: committed, sanitized run fixtures (trimmed `logs.jsonl` + a few
  transcripts). The TUI renders the hub and dashboard entirely from these.
- `seed.py`: installs `seed/` into an isolated `$AGENT6_STATE_HOME` under the demo
  repo's id and writes a demo `config.toml` + `ui.toml` (theme `agent6-dark`).
- `generate.sh`: the orchestrator. Seeds a temp home, runs `tour.tape` for the
  PNGs (1920x1080), then records `reel.tape` and overlays keystroke toasts to make
  `tour.webm`.
- `build_fixtures.py`: dev tool, not run in CI. Rebuilds `seed/` from real runs
  under `$XDG_STATE_HOME/agent6/`, trimming token-delta bloat and scrubbing paths.
- `web_demo.py` + `web_demo.sh`: the web-UI tour (`web-desktop.webm`,
  `web-phone.webm`). Drives `agent6 web` against the same `seed/` fixtures in a
  headless Chromium via Playwright, at desktop (1280x800) and phone (390x844)
  viewports, with an on-page caption banner per step (the browser analogue of
  `keystroke_overlay.py`). Deterministic, no key, no live LLM. Needs a
  Playwright-capable Python; point `$WEB_DEMO_PY` at it (see the script header).

## Demo videos (record/replay)

The two demo videos (`cli-demo.webm`, `machine-demo.webm`) are **real agent6 runs**,
not seeded fixtures: a real loop, real tools, real verify + commit. Determinism
comes from `llm_proxy.py`, a tiny stdlib HTTP server agent6 talks to as a local
model (no agent6 changes, no monkey-patching):

- `record` (live, real key): forwards each LLM call to OpenRouter, relays the
  response, and captures it to a cassette. Run once to capture a real trajectory.
- `replay` (no key): serves the recorded cassette in order, fully deterministic,
  so the same run reproduces exactly. This is what CI renders.

Both demos force streaming (`AGENT6_FORCE_STREAM=1`) so the cassette is SSE and
the model's reasoning streams live in the recording, the same as a real terminal.

- `cli_demo.sh` (`record`|`replay`) + `cli_demo.tape`: a terminal bug-fix run for
  the CLI audience. `agent6 run` (headless) fixes a failing test, then `runs diff`,
  `watch`, `runs show`. Pure typing, so it renders straight with no toast overlay.
  Seed: `seed/cli-repo/` (the buggy stats repo) + `seed/cli-cassette.jsonl`.
- `machine_demo.sh` (`record`|`replay`) + `machine_demo.tape`: the code-fixer
  state machine in the TUI. The Machines page runs the fix-loop (agent edits ->
  tool verifies -> branch loops) and the watch view streams the agent's reasoning.
  Seed: `seed/machine-repo/` (the machine bundle + buggy source) +
  `seed/machine-cassette.jsonl`.

The cassette and its seed repo are committed together: the recorded edits target
that exact source. Re-record (`… record`, needs a key) only when the run itself
should change.

The reel records at 1280x720 / Framerate 8. The live dashboard redraws ~5x/s, and at
full resolution vhs cannot capture that fast (the dashboard records several times too
fast, which desyncs the single-scale toast overlay). At 1280x720 the capture keeps up,
so the whole session records at real time (~97%) and the toasts stay matched to the
screen. The PNGs are captured separately at 1920x1080.

## Run locally

```sh
# needs vhs + ttyd + ffmpeg + agent6 on PATH
bash docs/screenshots/generate.sh        # -> 01-hub.png … tour.webm
bash docs/screenshots/cli_demo.sh        # -> cli-demo.webm (replay, no key)
bash docs/screenshots/machine_demo.sh    # -> machine-demo.webm (replay, no key)
```

vhs renders through a headless Chromium; on Ubuntu set
`kernel.apparmor_restrict_unprivileged_userns=0` first if rendering aborts.

## In CI

The [`pages`](../../.github/workflows/pages.yml) workflow runs `generate.sh` then
the two demo scripts (replay mode, no key) before `mkdocs build` on every release
(or manual dispatch), so the published site's images always match the current UI.
The media ships inside the site artifact (not release assets, not committed), so
it never expires.
