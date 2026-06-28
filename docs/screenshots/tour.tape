# agent6 TUI tour — captures the release screenshots, with vhs
# (https://github.com/charmbracelet/vhs).
#
#   bash docs/screenshots/generate.sh        # seeds fixtures, runs this tape
#
# Drives hub -> config -> search -> help -> run dashboard -> transcript -> logs
# and writes one PNG per stop. Requires `vhs`, `ttyd`, `ffmpeg`, and `agent6` on
# PATH. generate.sh exports AGENT6_STATE_HOME / AGENT6_CONFIG_HOME /
# AGENT6_DEMO_REPO (seeded, isolated) before invoking vhs; the tape inherits
# them. Uses only the hub/dashboard's direct key bindings (Enter/l/t/c/?/q and
# /), never the terminal-dependent Alt+<key> menus. The config/search/help
# screens are captured from the freshly mounted hub BEFORE entering the
# dashboard, because hub<->dashboard are separate apps and the app restart drops
# the first keystroke that follows it.
#
# This tape's job is the PNGs. vhs requires an Output, but this recording is a
# throwaway -- the tour.webm video is recorded separately by reel-hub.tape +
# reel-run.tape (no Screenshots, so the toast timeline stays accurate). The wide
# 1920x1080 terminal keeps the dashboard's tool table and diff from clipping.
Output docs/screenshots/out/_capture.webm

Require agent6

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 16
Set Width 1920
Set Height 1080
Set Padding 16
Set Theme "Catppuccin Mocha"

Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "clear" Enter
Show

# --- the hub: run history for the repo -----------------------------------
# Wait for the hub to finish scanning runs + render before the shot.
Type "agent6 tui" Enter
Sleep 5s
Screenshot docs/screenshots/out/01-hub.png

# --- the config page (c): every value + where it came from ---------------
Type "c"
Sleep 2.5s
Screenshot docs/screenshots/out/03-config.png

# --- search the settings (/) ---------------------------------------------
Type "/"
Sleep 700ms
Type "sandbox"
Sleep 1.5s
Screenshot docs/screenshots/out/04-config-search.png
Escape
Sleep 500ms
Escape
Sleep 1s

# --- the keys & actions help (?) -----------------------------------------
Type "?"
Sleep 1.5s
Screenshot docs/screenshots/out/08-help.png
Escape
Sleep 1s

# --- open the top run -> the live dashboard ------------------------------
# (DAG, budget bar, tool table, reasoning stream, log tail, diff)
Enter
Sleep 3.5s
Screenshot docs/screenshots/out/02-run-dashboard.png

# --- the conversation viewer (t): assistant text + every tool call -------
# Jump to the end (G), then back up two pages so an assistant turn with its
# reasoning + tool calls is centered, not the long system prompt.
Type "t"
Sleep 1.5s
Type "G"
Sleep 500ms
PageUp
Sleep 300ms
PageUp
Sleep 1s
Screenshot docs/screenshots/out/05-transcript.png
Escape
Sleep 1s

# --- the full-height log viewer (l) --------------------------------------
Type "l"
Sleep 1.5s
Screenshot docs/screenshots/out/09-logs.png
Escape
Sleep 1s

# --- back to the hub and quit --------------------------------------------
Escape
Sleep 700ms
Type "q"
Sleep 1s
