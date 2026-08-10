# The CLI hero source: the same real run from the plain terminal, served from
# the committed CLI cassette by llm_proxy.py (replay, no key). hero_cli.sh
# seeds the buggy stats repo, starts the proxy, and records this; hero.sh cuts
# it.
#
# Arc: open on the failing suite -> one command hands it to agent6 (reasoning
# and tool calls stream, verify passes, auto-commit) -> `sessions diff` shows
# the fix -> `sessions show` is the receipt.
#
# 1600x900 at FontSize 17, matching the other hero sources.
Output docs/screenshots/out/hero-src-cli.webm

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

# Open on the failing suite, already printed.
Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "export PS1='$ '" Enter
Type "export PYTHONDONTWRITEBYTECODE=1" Enter
Type "clear" Enter
Type "python3 -m unittest discover -s tests -t . -q"
Enter
Sleep 800ms
Show

Sleep 1.2s
Type@28ms `agent6 run "Find and fix the failing test in tests/test_stats.py"`
Enter
# The replayed run streams at model pace (~20s to the summary), then waits at
# the follow-up prompt ("next (/exit to finish):"); typing earlier would steer
# the run instead of the shell.
Sleep 24s

Type@40ms "/exit"
Enter
Sleep 1.2s

Type@40ms "agent6 sessions diff"
Enter
Sleep 4.5s

Type@40ms "agent6 sessions show"
Enter
Sleep 4.5s
