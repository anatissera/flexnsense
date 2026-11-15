# Flex'N Sense pitch deck, version 3 (HTML)

Version 3 of the pitch deck: one HTML file with real footage from the bench and live captures of the dashboard. Compared with version 2 it has a zig-zag timeline that plays one moment at a time, the team photo on the last slide, and a new capture of the updated dashboard. Version 2 is still at [`docs/pitch-v2/`](../pitch-v2/) and version 1, the original PowerPoint, at [`docs/Pitch.pptx`](../Pitch.pptx).

## Present it

Pull the repo and open `docs/pitch-v3/index.html` in Chrome (double-click, no server or internet needed). Press **F** for full screen. GitHub does not preview HTML files, so it has to be opened from a local copy.

| Key | Does |
|---|---|
| → / Space / click | Next slide |
| ← / click on the left quarter | Previous slide |
| R | Replay the current slide's animation |
| F | Full screen |
| 1 2 3 | On the real-footage slide: open that clip |
| P | On the real-footage slide: pause or play |
| E | Edit text in the browser (Esc to finish) |
| ⌘S / Ctrl+S | Save a copy of the deck with your edits |

A saved copy downloads as `index.html`. Move it into this folder, over the old one, so it still finds `assets/`.

## Slides

1. Title (animated logo)
2. The challenge: 10 million
3. Our objective
4. Our solution (full photo)
5. Three in the loop
6. How it works (animated diagram: the force rises to the limit, the arm gives way)
7. Real footage (three clips, one open at a time)
8. User interface (a recording of the dashboard while the 3D view turns, with labels)
9. Inside the dashboard (the 3D hand, the recovery curve, the force-displacement loop)
10. Team (twelve flags pop in, fly into a ribbon, then the six portraits appear)
11. Timeline (a zig-zag: each moment opens large for 3 seconds, then shrinks into a circle on the line; wins sit high, failures sit low)
12. Thank you (the team photo)

Every animation is CSS or JavaScript in `index.html`. The only pre-recorded motion is the video clips: three from the bench, two screen recordings of the dashboard, and the 3-second clips of the timeline.

## Brand (deck only)

The repo logo in `docs/images/` is unchanged. This deck uses its own:

- **Mark** (`assets/brand/mark.svg`): a finger, or a gauge needle, sweeping to the limit, with the red marker at the tip.
- **Wordmark**: "Flex'N Sense" with the apostrophe drawn as the red marker of the PESOLA scale.
- **Colours**, from the bench: ink `#0E141F` (the arm), paper `#F4F7FB` and drape `#DCE7F3` (the surgical drape), red `#E63B2E` (the marker and the force limit, nothing else), green `#1B9A66` (only when the arm gives way).
- **Type**: Avenir Next for text, DIN Alternate for numbers and labels (the typeface of instrument scales). Both ship with macOS. On Windows the deck falls back to Segoe UI and Bahnschrift.
- **Progress bar**: the footer is the scale, the red marker is the current slide.

Colours and fonts are CSS variables at the top of `index.html` (`:root`).

## Swap content

- **Text**: press E in the browser, or edit `index.html`.
- **Videos**: `assets/video/*.mp4` (720×1280 portrait, H.264, no audio) with a `.jpg` poster of the same name. To add a new clip from a phone video:
  `ffmpeg -i in.mp4 -an -vf "scale=720:1280,fps=30" -c:v libx264 -crf 21 -pix_fmt yuv420p -movflags +faststart assets/video/name.mp4`
- **Dashboard media**: `assets/dash/`. Captured on 15 Nov 2026 from the app in `dashboard/` (main at `4b5547f`) running locally (`pnpm install`, `pnpm dev`, then http://localhost:3000/?grasp) in a 1600×1000 window:
  - `run.mp4` (1920×1200): the whole dashboard while the view cube turns the 3D hand.
  - `hand.mp4` (720×1280): the same recording, cropped to the 3D hand.
  - `recovery.png`, `loop.png`: the "Recovery index vs week" and "Force-displacement loop" charts (index finger, week 24).
  The webcam was not opened, so the rep machine is shown idle. The data is the dashboard's simulated patient, and slide 9 says so.
- **Timeline moments** (slide 11): one `<figure class="ev">` per moment inside `#evs` in `index.html`, in order. The line and the spacing are computed from the list, so add, remove or reorder figures freely.
  - `data-dir="up"` is a win (caption above the circle), `data-dir="down"` is a failure (red ring, caption below).
  - `data-y` is the height of the circle on the 1080 px stage (about 490 to 590 for a win, 770 for a failure).
  - Media goes in `assets/timeline/`: an image, or a landscape clip of 3 seconds with a `.jpg` poster:
    `ffmpeg -ss 4 -t 3 -i in.mp4 -an -vf "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720" -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart assets/timeline/name.mp4`
  - `HOLD` in `startJourney` is how long each moment stays large. Pressing next while it plays shows the finished line; pressing next again moves on.
  - The times and several captions are placeholders until the team fills them in.
- **Portraits**: `assets/people/`, flags: `assets/flags/`.
