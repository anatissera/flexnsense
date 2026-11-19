#!/usr/bin/env bash
# Render docs/images/tether.gif from docs/images/src/tether.html.
#
# The page animates on a clock, which a screenshot cannot catch reproducibly, so it takes a
# ?t=<ms> parameter that replays the cycle up to one instant and stops. This walks the cycle
# a frame at a time, which makes the output identical on every run.
#
# Needs google-chrome (or chromium) and ffmpeg. Nothing else, and no hardware.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/docs/images/src/tether.html"
OUT="$REPO/docs/images/tether.gif"

CYCLE_MS=8000          # one full loop of the animation, from the page's own CYCLE
FPS=10
WIDTH=1100             # as rendered in the README
CROP="1830:835:45:148" # title, lead and diagram, without the slide's empty margins

CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)"
[ -n "$CHROME" ] || { echo "error: no chrome or chromium on PATH" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "error: ffmpeg is not on PATH" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

step=$((1000 / FPS))
n=0
for (( t = 0; t < CYCLE_MS; t += step )); do
  "$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
    --window-size=1920,1080 --virtual-time-budget=1500 \
    --screenshot="$(printf '%s/f%04d.png' "$work" "$n")" \
    "file://$SRC?t=$t" >/dev/null 2>&1
  n=$((n + 1))
  printf '\r  frame %d at t=%dms' "$n" "$t"
done
echo

ffmpeg -y -v error -framerate "$FPS" -i "$work/f%04d.png" \
  -vf "crop=$CROP,scale=$WIDTH:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=128:stats_mode=diff[p];\
[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  "$OUT"

echo "  wrote $OUT ($(du -h "$OUT" | cut -f1), $n frames)"
