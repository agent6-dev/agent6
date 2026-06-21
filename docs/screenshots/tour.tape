# agent6 TUI — release screenshots, captured with vhs (https://github.com/charmbracelet/vhs).
#
#   vhs docs/screenshots/tour.tape
#
# Requires `vhs` and `agent6` on PATH. Captures PNGs under docs/screenshots/out/
# (gitignored — we commit the scripts, not the media). Point AGENT6_DEMO_REPO at
# a git repo to show a realistic hub/config; defaults to the current directory.
#
# Note: Alt+<key> opens a menu and F1/keys vary by terminal; if a step doesn't
# trigger on your setup, swap the menu open for a mouse step or the command
# palette (Ctrl+P).

Output docs/screenshots/out/tour.gif
Require agent6

Set Shell bash
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 14
Set Theme "Catppuccin Mocha"

Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Show

# --- the hub -------------------------------------------------------------
Type "agent6 tui" Enter
Sleep 2.5s
Screenshot docs/screenshots/out/01-hub.png

# --- the config page (menu bar: File / Config / Help is up top) ----------
Type "c"
Sleep 1.5s
Screenshot docs/screenshots/out/02-config.png

# --- a menu dropdown (Alt+E -> Edit), showing items + their shortcut keys -
Alt+e
Sleep 1s
Screenshot docs/screenshots/out/03-config-edit-menu.png
Escape

# --- search the settings -------------------------------------------------
Type "/"
Sleep 0.5s
Type "sandbox"
Sleep 1s
Screenshot docs/screenshots/out/04-config-search.png
Escape

# --- the Help / keys page (Alt+H -> Keys & actions) ----------------------
Alt+h
Sleep 0.5s
Enter
Sleep 1s
Screenshot docs/screenshots/out/05-help.png
Escape

# --- back to the hub and quit -------------------------------------------
Escape
Sleep 0.5s
Type "q"
Sleep 0.5s
