# Screenshots

The TUI screenshots and tour video for the docs site (agent6.dev) are generated
from **real, recorded runs** with [vhs](https://github.com/charmbracelet/vhs):
no live LLM calls, no network, no API key. Small, sanitized run fixtures are
committed under `seed/`; vhs replays them through the TUI and captures the images.

We commit the **tape, scripts, and seed fixtures only**. Generated media
(`out/`, `*.png`, `*.webm`) is gitignored and rebuilt on demand.

Every tape uses the `VHS_THEME_DARK` terminal palette in `src/agent6/ui/tui/theme.py`.
The terminal frame shares the TUI's `agent6-dark` background.
The TUI's ANSI text colours use separate palettes.
Run `uv run python docs/screenshots/sync_theme.py` after editing the terminal palette to update every tape's `Set Theme` line.

The recording scripts convert WebM output to limited-range BT.709 and set its colour-space, primaries and transfer tags.
Each WebM encode checks `color_space=bt709` with `ffprobe` and fails if it is missing.
GIF encoding uses its own palette filter.

## Files

- `tour.tape`: vhs script for the PNGs. Drives hub -> config -> search -> help ->
  log -> conversation -> dashboard and captures one PNG per stop.
- `reel.tape`: vhs script for the TUI video. One TUI session (hub, into a
  run's conversation, Ctrl+D dashboard, back to hub, log, config, search, help)
  with no `Screenshot` commands (they stall the video clock). Holds are paced by
  how dense each screen is. The bare recording is kept as `hero-tui.webm` (the
  hero's TUI segment); `tour.webm` is that recording plus keystroke toasts.
- `webm.py`: converts VHS output in place and checks WebM colour-space tags.
  The browser and keystroke overlay encoders share its conversion and check.
- `keystroke_overlay.py`: overlays animated keystroke toasts on the recording,
  parsing the tape and scaling the keypress timeline to the actual duration.
- `seed/`: committed, sanitized run fixtures (trimmed `logs.jsonl` + a few
  transcripts). The TUI renders the hub and dashboard entirely from these.
- `seed.py`: installs `seed/` into an isolated `$XDG_STATE_HOME/agent6` under the demo
  repo's id and writes a demo `config.toml` + `ui.toml` (theme `agent6-dark`).
- `generate.sh`: the orchestrator. Seeds a temp home, runs `tour.tape` for the
  PNGs (1920x1080), then records `reel.tape` (`hero-tui.webm`) and overlays
  keystroke toasts to make `tour.webm`.
- `build_fixtures.py`: dev tool, not run in CI. Rebuilds `seed/` from real runs
  under `$XDG_STATE_HOME/agent6/`, trimming token-delta bloat and scrubbing paths.
- `tui_snapshot.py`: dev tool, not run in CI. Snapshots a TUI screen (dashboard /
  log / transcript) of any run dir to an SVG (or PNG, if a chromium /
  `rsvg-convert` is found) via Textual's headless Pilot -- to "see" the TUI while
  iterating on it, without a terminal. `uv run python docs/screenshots/tui_snapshot.py
  <session_dir> <out.(svg|png)> [screen]`.
- `hero.sh`: the three per-surface heroes (`hero-tui`, `hero-cli`, `hero-web`,
  each `.webm` for the site plus `.gif` for GitHub, which animates a GIF
  through its image proxy but will not play a webm from another host). Cuts
  windows out of the three BARE source recordings (`hero-src-tui.webm` from
  `hero_run.sh`, `hero-src-cli.webm` from `hero_cli.sh`, `hero-src-web.webm`
  from `web_demo.sh hero` -- no keystroke toasts, no narration banner) and
  retimes each onto its target length in one ffmpeg pass per surface.
  Both formats use a 1440x810 canvas.
- `web_demo.py` + `web_demo.sh`: the web-UI tour (`web-desktop.webm`,
  `web-phone.webm`). Drives `agent6 web` against the same `seed/` fixtures in a
  headless Chromium via Playwright, at desktop (1280x800) and phone (390x844)
  viewports, with an on-page caption banner per step (the browser analogue of
  `keystroke_overlay.py`).
  `web_demo.sh hero` records the desktop tour at 1600x900 with `--mode hero --no-narration` to `hero-src-web.webm`.
  Each video is trimmed
  (ffmpeg) to drop the SPA loading-screen head so it opens on the hub, the
  poster frame shown before play.
  Deterministic, no key, no live LLM. Needs a Playwright-capable Python and
  ffmpeg; point `$WEB_DEMO_PY` at it (see the script header).

## Demo videos (record/replay)

The demo videos (`cli-demo.webm`, `temps-demo.webm`, `rename-demo.webm`, `machine-demo.webm`) are **real agent6 runs**,
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
  the CLI audience. `agent6 run` (headless) fixes a failing test, then `sessions diff`,
  `watch`, `sessions show`. Pure typing, so it renders straight with no toast overlay.
  Seed: `seed/cli-repo/` (the buggy stats repo) + `seed/cli-cassette.jsonl`; the two
  hero sources replay this cassette.
- `temps_demo.sh` (`record`|`replay`) + `temps_demo.tape`: the same shape over
  `seed/temps-repo/` (a temperature-conversion module with one failing test) +
  `seed/temps-cassette.jsonl`; `temps-demo.webm` is the run the Terminal UI page embeds.
- `rename_demo.sh` (`record`|`replay`) + `rename_demo.tape`: a rename task from a
  `TASK.md` (`Cart` to `ShoppingCart` across `shop/`) over `seed/rename-repo/` +
  `seed/rename-cassette.jsonl`; rendered on demand, not in CI.
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
# needs vhs + ttyd + ffmpeg + ffprobe + agent6 on PATH
bash docs/screenshots/generate.sh        # -> 01-hub.png … tour.webm
bash docs/screenshots/cli_demo.sh        # -> cli-demo.webm (replay, no key)
bash docs/screenshots/temps_demo.sh      # -> temps-demo.webm (replay, no key)
bash docs/screenshots/rename_demo.sh     # -> rename-demo.webm (replay, no key)
bash docs/screenshots/machine_demo.sh    # -> machine-demo.webm (replay, no key)
bash docs/screenshots/hero_run.sh        # -> hero-src-tui.webm (replay, no key)
bash docs/screenshots/hero_cli.sh        # -> hero-src-cli.webm (replay, no key)
WEB_DEMO_PY=… bash docs/screenshots/web_demo.sh hero   # -> hero-src-web.webm
bash docs/screenshots/hero.sh            # -> hero-{tui,cli,web}.webm + .gif (needs the three sources)
```

vhs renders through a headless Chromium; on Ubuntu set
`kernel.apparmor_restrict_unprivileged_userns=0` first if rendering aborts.

## In CI

The [`pages`](../../.github/workflows/pages.yml) workflow runs `generate.sh`,
`machine_demo.sh` and `temps_demo.sh` (replay mode, no key), the two hero sources,
the web tours, and `hero.sh` before
`mkdocs build` on every release (or manual dispatch), so the published site's
images -- and the hero the README embeds from it -- always match the current UI.
The media ships inside the site artifact (not release assets, not committed), so
it never expires.
