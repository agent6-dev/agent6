# Machine demo for the docs: a real agent6 state machine running in the TUI.
# Not faked logs -- the Machines page runs the code-fixer fix-loop (agent edits ->
# tool verifies -> branch loops until green), and the watch view streams the
# agent state's reasoning live. The agent's LLM calls come from a recorded
# cassette served by llm_proxy.py (replay, no key); the tool states + branch +
# commits are real. machine_demo.sh starts the proxy, seeds the buggy repo +
# machine, records this, and overlays keystroke toasts with keystroke_overlay.py.
#
# Flow: hub -> Machines page -> Create dialog (how a machine is authored from a
# natural-language description) -> Run the code-fixer (confirm) -> watch it
# execute live (the state loop, the agent fixing the bug, the transitions to
# success) -> back -> quit.
#
# 1280x720 / Framerate 8, matching reel.tape so the overlay timeline lines up.
Output docs/screenshots/out/_machine-raw.webm

Require agent6

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 13
Set Width 1280
Set Height 720
Set Padding 14
# agent6-dark, the TUI's own branded palette (src/agent6/ui/tui/theme.py):
# a near-black teal, the cyan brand's neutral, so the recordings look like
# the product rather than like a generic terminal theme.
Set Theme {"name": "agent6-dark", "background": "#0f1414", "foreground": "#c0caf5", "cursor": "#06F5F3", "selection": "#1e2626", "black": "#161c1c", "red": "#f7768e", "green": "#9ece6a", "yellow": "#e0af68", "blue": "#7aa2f7", "magenta": "#bb9af7", "cyan": "#06f5f3", "white": "#c0caf5", "brightBlack": "#6e6e78", "brightRed": "#ff8fa3", "brightGreen": "#b4e07f", "brightYellow": "#ffa62b", "brightBlue": "#9cbcff", "brightMagenta": "#d3b4ff", "brightCyan": "#66fffd", "brightWhite": "#f2f4ff"}
Set Framerate 8

Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "clear" Enter
Show

Sleep 1.5s
Type@90ms "agent6 tui"
Enter
Sleep 4s

# Machines page: the repo's code-fixer machine.
Type "M"
Sleep 2.5s

# How a machine is authored: the Create dialog drafts a .asm.toml from a plain
# description. Show it, then cancel (Esc) and run the committed code-fixer.
Type "c"
Sleep 1.2s
Type@55ms "loop an agent and the test runner until the failing check passes"
Sleep 2.5s
Escape
Sleep 1.2s

# Run it: confirm the dialog, which spawns the run and opens the live watch view.
Type "R"
Sleep 2s
Type "y"

# Watch the fix-loop execute: the agent state's reasoning streams in, the verify
# tool runs, the branch routes, and the machine reaches success.
Sleep 22s
Escape
Sleep 1.5s
Escape
Sleep 1s
Type "q"
Sleep 1s
