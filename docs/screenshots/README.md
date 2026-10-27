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

- `cli_demo.sh`: records or replays a terminal bug-fix run with `cli_demo.tape`, `seed/cli-repo/` and `seed/cli-cassette.jsonl`.
- `generate.sh`: seeds a temporary home, captures TUI PNGs with `tour.tape`, then records `reel.tape` and overlays keystroke toasts.
- `hero.sh`: cuts and retimes the three hero source recordings into 1440x810 WebM and GIF clips, one per surface.
- `hero_cli.sh`: replays the CLI cassette through `hero_cli.tape` to record the CLI hero source without overlays.
- `hero_run.sh`: replays the CLI cassette in the run TUI through `hero_run.tape` (or a supplied tape) to record the TUI hero source without overlays.
- `machine_demo.sh`: records or replays the code-fixer machine in the TUI with `machine_demo.tape`, `seed/machine-repo/` and `seed/machine-cassette.jsonl`.
- `rename_demo.sh`: records or replays the `TASK.md` rename from `Cart` to `ShoppingCart` across `shop/` with `rename_demo.tape`, `seed/rename-repo/` and `seed/rename-cassette.jsonl`.
- `temps_demo.sh`: records or replays a temperature-conversion fix with `temps_demo.tape`, `seed/temps-repo/` and `seed/temps-cassette.jsonl`.
- `web_demo.sh`: records seeded desktop and phone web tours, or the web hero source with the `hero` argument, using Playwright.
- `web_machine_demo.sh`: replays the machine cassette and records the code-fixer machine started and watched from the browser.

Supporting files:

- `tour.tape`: captures one PNG per stop through hub, config, search, help, log, findings, receipt, conversation and dashboard.
- `reel.tape`: records one TUI tour without screenshot commands, which stall the video clock.
- `webm.py`: converts VHS output in place and checks WebM colour-space tags for the terminal, browser and overlay encoders.
- `keystroke_overlay.py`: overlays animated keystroke toasts, scaling the tape's keypress timeline to the recording's duration.
- `seed/`: sanitized run fixtures with trimmed `logs.jsonl` files and transcripts for the seeded tours, plus demo repos and cassettes; `stats-demo.bundle` holds the stats repository's history with the two reviewed runs' trees, fetched into the clone of this checkout the tours run in.
- `seed.py`: installs the run fixtures into an isolated `$XDG_STATE_HOME/agent6` under the demo repo's id and writes `config.toml` and `ui.toml` with theme `agent6-dark`.
- `build_fixtures.py`: rebuilds run fixtures from `$XDG_STATE_HOME/agent6/`, trimming token deltas and scrubbing paths; not run in CI.
- `tui_snapshot.py`: captures a run's dashboard, log or transcript through Textual's headless Pilot as SVG, or PNG with Chromium or `rsvg-convert`; not run in CI.
- `web_demo.py`: drives the browser tours with Playwright and trims the loading screen from each recording.
- `llm_proxy.py`: serves recorded LLM responses or forwards live requests to OpenRouter while recording a cassette.
- `make_cover.py`: composes the README cover from TUI and web screenshots and a frame from the temperature demo.
- `sync_theme.py`: copies the terminal palette into each tape's `Set Theme` line.

Run a headless TUI snapshot with `uv run python docs/screenshots/tui_snapshot.py <session_dir> <out.(svg|png)> [screen]`.
Browser recordings need a Playwright-capable Python with Chromium installed; set `$WEB_DEMO_PY` as described in `web_demo.sh`.

## Demo videos (record/replay)

The cassette-backed demos run the real agent6 loop, tools, verification and commits against their seed repos.
`llm_proxy.py` supplies the model responses without changing agent6:

- `record` forwards calls to OpenRouter with a real key and saves the responses to a cassette.
- `replay` serves the committed cassette in order without a key; CI uses this mode.

These demos force streaming with `AGENT6_FORCE_STREAM=1` so the SSE cassette streams the model's reasoning in the recording.
The cassette and its seed repo are committed together because the recorded edits target that exact source.
Re-record with `record` only when the run itself should change.
The seeded TUI and web tours render saved sessions without running a model.

Outputs under `out/` (built by Pages unless marked on demand):

- `cli-demo.webm`: terminal bug-fix run followed by `sessions diff`, `attach --since` and `sessions show`; on demand, not in CI.
- `hero-cli.gif`: CLI hero for GitHub's image proxy, built by `hero.sh`.
- `hero-cli.webm`: CLI hero for the site, built by `hero.sh`.
- `hero-src-cli.webm`: CLI hero source from `hero_cli.sh`.
- `hero-src-tui.webm`: TUI hero source from `hero_run.sh`.
- `hero-src-web.webm`: 1600x900 web hero source without narration from `web_demo.sh hero`.
- `hero-tui.gif`: TUI hero for GitHub's image proxy, built by `hero.sh`.
- `hero-tui.webm`: TUI hero for the site, built by `hero.sh`.
- `hero-web.gif`: web hero for GitHub's image proxy, built by `hero.sh`.
- `hero-web.webm`: web hero for the site, built by `hero.sh`.
- `machine-demo.webm`: code-fixer machine run and watch view in the TUI, with keystroke toasts.
- `rename-demo.webm`: terminal class-rename task from `rename_demo.sh`; on demand, not in CI.
- `temps-demo.webm`: terminal temperature-conversion fix embedded on the Terminal UI page.
- `tour.webm`: seeded TUI tour with keystroke toasts from `generate.sh`.
- `web-desktop.webm`: seeded 1280x800 web tour with captions from `web_demo.sh`.
- `web-machine.webm`: browser code-fixer machine tour from `web_machine_demo.sh`.
- `web-phone.webm`: seeded 390x844 web tour with captions from `web_demo.sh`.

Still images under `out/` (all built by Pages):

- `01-hub.png`: TUI hub from `generate.sh`.
- `02-run-dashboard.png`: TUI run dashboard from `generate.sh`.
- `03-config.png`: TUI configuration from `generate.sh`.
- `04-config-search.png`: TUI configuration search from `generate.sh`.
- `05-transcript.png`: TUI conversation from `generate.sh`.
- `08-help.png`: TUI help from `generate.sh`.
- `09-logs.png`: TUI log from `generate.sh`.
- `cover.png`: 1600x900 README cover from `make_cover.py`.
- `web-shot.png`: web session screenshot from the desktop tour.

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
WEB_DEMO_PY=… bash docs/screenshots/web_demo.sh       # -> web-desktop.webm + web-phone.webm + web-shot.png
WEB_DEMO_PY=… bash docs/screenshots/web_machine_demo.sh # -> web-machine.webm
WEB_DEMO_PY=… bash docs/screenshots/web_demo.sh hero   # -> hero-src-web.webm
bash docs/screenshots/hero.sh            # -> hero-{tui,cli,web}.webm + .gif (needs the three sources)
```

vhs renders through a headless Chromium; on Ubuntu set
`kernel.apparmor_restrict_unprivileged_userns=0` first if rendering aborts.

## In CI

The [`pages`](../../.github/workflows/pages.yml) workflow builds the outputs listed above before `mkdocs build` on each published release or manual dispatch.
It also runs `make_cover.py` to compose the README cover.
The media ships inside the site artifact, not as release assets or committed files, so it never expires.
