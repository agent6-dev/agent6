# One agent6 tui session for the tour.webm reel: hub -> open a run (the
# conversation, the run app's primary view) -> dashboard (Ctrl+D) -> back to
# hub -> event log (l) -> config -> search -> help -> quit. The whole tour
# stays inside the TUI (a run is opened with Enter, not a new shell command).
# No Screenshots, which would stall the video clock; generate.sh records this
# and overlays keystroke toasts with keystroke_overlay.py.
#
# Recorded at 1280x720 / Framerate 8: the live dashboard redraws ~5x/s, and at
# this resolution vhs's capture keeps up, so the recording stays at real time
# (~97%). That keeps the overlay's single timeline scale accurate, so the toasts
# match the screen. (The PNGs are captured separately at 1920x1080.)
#
# Holds are paced by how much is on screen: long on the dense conversation,
# dashboard, and config; short on the hub, help, and the transits back. Inside
# the run app the composer bar owns plain letters, so only its priority chords
# (Ctrl+D, Esc) are used. generate.sh exports the seeded AGENT6_* env.
Output docs/screenshots/out/_reel-raw.webm

Require agent6

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 13
Set Width 1280
Set Height 720
Set Padding 14
Set Theme "Catppuccin Mocha"
Set Framerate 8

Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "clear" Enter
Show

Sleep 1.5s
Type@90ms "agent6 tui"
Enter
Sleep 4s
Enter
Sleep 7s
Ctrl+D
Sleep 6s
Escape
Sleep 3s
Type "l"
Sleep 4.5s
Escape
Sleep 1.5s
Type "c"
Sleep 5.5s
Type "/"
Sleep 700ms
Type@110ms "sandbox"
Sleep 3.5s
Escape
Sleep 0.8s
Escape
Sleep 1.5s
Type "?"
Sleep 3.5s
Escape
Sleep 1.2s
Type "q"
Sleep 1s
