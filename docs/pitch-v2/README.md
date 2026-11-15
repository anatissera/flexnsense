# Flex'N Sense pitch deck, version 2 (HTML)

Version 2 of the pitch deck: one HTML file with real footage from the bench and live captures of the dashboard. Version 1, the original PowerPoint, is still at [`docs/Pitch.pptx`](../Pitch.pptx).

## Present it

Pull the repo and open `docs/pitch-v2/index.html` in Chrome (double-click, no server or internet needed). Press **F** for full screen. GitHub does not preview HTML files, so it has to be opened from a local copy.

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
8. User interface (a recording of the dashboard running an index-finger curl, with labels)
9. Inside the dashboard (the 3D hand curling, the recovery curve, the force-displacement loop)
10. Team (twelve flags pop in, fly into a ribbon, then the six portraits appear)
11. Timeline
12. Thank you

Every animation is CSS or JavaScript in `index.html`. The only pre-recorded motion is the video clips: three from the bench and two screen recordings of the dashboard.

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
- **Dashboard media**: `assets/dash/`. Captured on 15 Nov 2026 from the app in `dashboard/` running locally (`pnpm install`, `pnpm dev`, then http://localhost:3000) in a 1280×800 window at 2x:
  - `run.mp4` (1600×1000): "Run index curl" on the synthetic patient, week 24.
  - `hand.mp4` (540×960): the same run, cropped to the 3D hand.
  - `recovery.png`, `loop.png`: the "Recovery index vs week" and "Force-displacement loop" charts (week 4).
  The data is the dashboard's simulated patient, and slide 9 says so.
- **Portraits**: `assets/people/`, flags: `assets/flags/`.
