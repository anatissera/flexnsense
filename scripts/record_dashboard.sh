#!/usr/bin/env bash
# Record the screen while the dashboard (localhost) is open, into recordings/.
#
# On X11 this grabs the whole screen with ffmpeg directly. On Wayland (this machine's
# session), ffmpeg's x11grab only sees XWayland and returns a black frame for a native
# Wayland window, so there is no reliable headless capture here: GNOME's own screen
# recorder (Ctrl+Shift+Alt+R) is the one that actually works, and this script's job on
# Wayland is just to find what it saved and move it into recordings/ with a session name.
#
#   scripts/record_dashboard.sh start [name]   # X11 only: starts ffmpeg, Ctrl-C to stop
#   scripts/record_dashboard.sh collect [name] # Wayland: moves the newest ~/Videos/Screencast*
#                                               # file (from Ctrl+Shift+Alt+R) into recordings/
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p recordings
stamp="$(date +%Y%m%d-%H%M%S)"
name="${2:-dashboard}"
cmd="${1:-}"

if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
  if [ "$cmd" = "collect" ] || [ -z "$cmd" ]; then
    latest="$(ls -t "$HOME"/Videos/Screencast*.webm 2>/dev/null | head -1)"
    if [ -z "$latest" ]; then
      echo "No ~/Videos/Screencast*.webm found yet."
      echo "Press Ctrl+Shift+Alt+R to start GNOME's screen recorder, open the dashboard,"
      echo "press Ctrl+Shift+Alt+R again to stop, then rerun: $0 collect $name"
      exit 1
    fi
    dest="recordings/${name}-${stamp}.webm"
    mv "$latest" "$dest"
    echo "saved $dest"
    exit 0
  fi
  echo "This session is Wayland: ffmpeg x11grab cannot see it (captures black)."
  echo "Press Ctrl+Shift+Alt+R to start/stop GNOME's own recorder, then run:"
  echo "  $0 collect $name"
  exit 1
fi

# X11 path: grab the whole screen until Ctrl-C.
dest="recordings/${name}-${stamp}.mp4"
echo "recording to $dest (Ctrl-C to stop)"
exec ffmpeg -f x11grab -framerate 30 -i "${DISPLAY:-:0}" -c:v libx264 -preset ultrafast -pix_fmt yuv420p "$dest"
