# The TUI hero source: one real agent6 run in the run TUI, served from the
# committed CLI cassette by llm_proxy.py (replay, no key). hero_run.sh seeds the
# buggy stats repo, starts the proxy, and records this; hero.sh cuts it.
#
# Arc: conversation with detail on (reasoning streams live) -> dashboard ->
# approval modal for the verify command (run_commands=ask; y allows) -> verify
# passes, auto-commit, the diff pane fills -> the run ends and the TUI exits ->
# back in via `agent6 tui`: the hub lists the finished run -> config, filtered
# to sandbox. The approval pauses the run until y, so the beats after it are
# tape-timed, not race-timed.
#
# 1600x900 at FontSize 17: the hero is read at README width, so fewer, larger
# cells beat the docs tapes' 1280x720/13.
Output docs/screenshots/out/hero-src-tui.webm

Require agent6

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 17
Set Width 1600
Set Height 900
Set Padding 24
# agent6-dark, the TUI's own branded palette (src/agent6/ui/tui/theme.py):
# a near-black neutral charcoal, NOT a blue navy, so the recordings look like
# the product rather than like a generic terminal theme.
Set Theme { "name": "agent6-dark", "background": "#161618", "foreground": "#C0CAF5", "cursor": "#06F5F3", "selection": "#2C2C30", "black": "#202023", "red": "#F7768E", "green": "#9ECE6A", "yellow": "#E0AF68", "blue": "#7AA2F7", "magenta": "#BB9AF7", "cyan": "#06F5F3", "white": "#C0CAF5", "brightBlack": "#6E6E78", "brightRed": "#FF8FA3", "brightGreen": "#B4E07F", "brightYellow": "#FFA62B", "brightBlue": "#9CBCFF", "brightMagenta": "#D3B4FF", "brightCyan": "#66FFFD", "brightWhite": "#F2F4FF" }
Set Framerate 8

# The hero opens mid-flight: launch happens hidden, capture starts on the live
# run.
Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "export PYTHONDONTWRITEBYTECODE=1" Enter
Type "clear" Enter
Type `agent6 run --tui "A test in tests/test_stats.py is failing. Find and fix the bug in src/stats.py so the whole suite passes. Keep the change minimal."`
Enter
Sleep 1.5s
Show

Sleep 2s
Ctrl+T
Sleep 10.5s
Ctrl+D
Sleep 10.5s
# The approval modal for run_verify_command pops ~6s after the dashboard flip
# and pauses the run; let it rest well settled, then allow. (Answering the
# instant it mounts has killed the TUI once; settled, it never has.)
Type "y"
Sleep 13s

# The run has ended and the TUI has exited. Come back in through the hub.
Hide
Type "clear" Enter
Type "agent6 tui" Enter
Sleep 2.5s
Show
Sleep 2.5s
Type "c"
Sleep 2s
Type "/"
Sleep 600ms
Type@110ms "sandbox"
Sleep 3s
