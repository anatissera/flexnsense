# Flex'N Sense pitch deck, version 6: storytelling, with an intro film (HTML)

Version 6 is version 5 (`../pitch-v5-storytelling/`, kept as it was) with an intro film in front. Like version 4 and 5, it tells the pitch as one patient's story in four acts, with real footage from the bench and live captures of the dashboard. It replaces the earlier HTML versions (2 and 3), which are no longer in the repo. The original PowerPoint is still at [`docs/Pitch.pptx`](../Pitch.pptx).

## Present it

Online: **https://flexnsense.vercel.app** (this folder, deployed to Vercel as the project `flexnsense`; run `vercel deploy --prod` from this folder to update it).

Offline: pull the repo and open `docs/pitch-v6-storytelling/index.html` in Chrome (double-click, no server or internet needed). Press **F** for full screen. GitHub does not preview HTML files, so it has to be opened from a local copy.

| Key | Does |
|---|---|
| → / Space / click | Next slide. On slide 1: start the intro film (press again to skip it) |
| ← / click on the left quarter | Previous slide |
| R | Replay the current slide's animation |
| F | Full screen |
| 1 2 3 | On the real-footage slide: open that clip |
| P | On the real-footage slide: pause or play |
| S | On the timeline slide: skip, show every moment at once |
| C | Hide or show the text over the intro film |
| E | Edit text in the browser (Esc to finish) |
| ⌘S / Ctrl+S | Save a copy of the deck with your edits |

A saved copy downloads as `index.html`. Move it into this folder, over the old one, so it still finds `assets/`.

## Slides

**Before the story**

1. The intro film (25 s, with sound). The slide is black with "Press space to start, full screen": browsers do not play sound until a key is pressed. One press starts the film and takes the deck full screen (it stays full screen for the slides after; F or Esc leaves it), and when it ends the deck moves to slide 2 by itself. A second press during the film skips it. The words to say over it are in [INTRO-SCRIPT.md](./INTRO-SCRIPT.md). (new in version 6)

**Act 1, a person**

2. The question: "What could a robot do in a mountain village?", where the specialist is often far away (new)
3. The patient: Jordan, 58, the same synthetic patient the dashboard shows (new)
4. The gap: his hand therapist can't be at every session, and may be in another city or country (new)
5. And Jordan is not alone: 10 million

**Act 2, the answer**

6. Flex'N Sense (the title, as the answer to the question; a portrait clip of our bench, restyled as a clinic, plays beside it)
7. The promise: hand rehabilitation at home, after surgery
8. Our solution: "Safety from physics, not only from software."
9. Who gains: three in the loop

**Act 3, the proof**

10. How it works (animated diagram)
11. Real footage (three clips). A good moment to switch to the live demo.
12. User interface (dashboard recording, with labels). Two buttons at the top right open a pop-up: "Replay the demo" plays the recorded voice-assistant demo with sound, "Medical report" shows the report the dashboard exports
13. Inside the dashboard

**Act 4, the people and what is next**

14. How we got here (the zig-zag timeline, one moment per press)
15. Our goals at Track 4 (Elderly Care & Rehabilitation, Rehab coach): three numbered goals that match the track's expectations. Give a robot arm the capacity to sense force with a low-tech proprioceptive ruler, use a standard tool from the physiotherapist's lab, prove that a robot arm can be used safely with patients (new)
16. The team
17. What comes next: built this weekend, then one hand clinic, then orthopedic rehabilitation for other joints (new)
18. A future approach: two looping tiles, Isaac Sim (the bench rebuilt) and Isaac Lab (the pull policy in training). A click on a tile opens the whole clip in a pop-up at 80 % of the screen; Esc or Close shuts it. Two links at the bottom right open the policies and the dataset on Hugging Face (new)
19. Thank you (team photo, then the card)
20. Find it all: six link cards (the GitHub repo, the public Hugging Face models and datasets, the deck) and a QR code of the deck. A click on a card opens it in a new tab (new)

To check before presenting:

- Slides 3 and 4 use images generated with FLUX (Black Forest Labs), in `assets/story/`. Each slide says so on screen. Jordan is not a real patient.
- Slide 9 ("Who gains"): the "Rehab AI assistant" disc shows a picture of a small white robot supplied by the team (`assets/story/little-robot.jpg`), not the team's arm. Its source and licence are not known: check them before the deck is published anywhere. It is not labelled on screen, so say aloud that it is a concept: our real arm is on slides 8, 10 and 11.
- Slide 17 uses a concept image supplied by the team (`assets/story/next.jpg`), labelled "Concept image": the robot in it is not something we built.
- The wording follows the team's review of 15 Nov: the patient had planned orthopedic surgery (not an injury), the opening is about a village far from the specialist (not about the spa), and the plan stays on the hand first because hand surgery is its own field, with the rest of orthopedic rehabilitation after.
- The dashboard recording on slide 12 still shows the app's own patient card ("Flexor tendon repair"), which the app sets, not the deck.
- Slide 17 lists what is built and what comes next. These lines are a first draft: correct them to what is true on Monday (press E to edit). The failures are told on the timeline, slide 14.

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
- **Keep it light**: every clip in `assets/` is H.264 at `-crf 28 -preset veryslow` with no audio, and photos are JPEG, at most 1920 px on the long side. Encode new media the same way before committing, so a pull stays small (the whole folder is about 9 MB):
  `ffmpeg -i in.mp4 -an -c:v libx264 -preset veryslow -crf 28 -pix_fmt yuv420p -movflags +faststart out.mp4`
- **Dashboard media**: `assets/dash/`. Captured on 15 Nov 2026 from the app in `dashboard/` (main at `4b5547f`) running locally (`pnpm install`, `pnpm dev`, then http://localhost:3000/?grasp) in a 1600×1000 window:
  - `run.mp4` (1920×1200): the whole dashboard while the view cube turns the 3D hand.
  - `hand.mp4` (720×1280): the same recording, cropped to the 3D hand.
  - `recovery.png`, `loop.png`: the "Recovery index vs week" and "Force-displacement loop" charts (index finger, week 24).
  The webcam was not opened, so the rep machine is shown idle. The data is the dashboard's simulated patient, and slide 13 says so.
- **Pop-ups on slide 12**: Esc, "Close" or a click outside closes one; Space pauses the demo. While one is open the arrow keys do not change slide.
  - `assets/dash/assistant-demo.mp4` (1600×994, 56 s, with sound): a screen recording of the dashboard's voice assistant made at the bench on 15 Nov 2026, cut from 2 min 18 s to four exchanges: which muscle is most at risk, which dashboard section shows its metrics, a better view of the muscle, and how to export for the physician. The chatter before, two failed camera angles and the Spanish test at the end are cut. The patient data in it is simulated.
  - `assets/dash/report.html`: the page behind the dashboard's "Export Medical Report" button, saved as it came out on 15 Nov 2026 after one simulated index curl (patient P1, week 24, 4 repetitions). To get there without a camera the simulated run controls were switched on locally (`SHOW_SIM_RUN_CONTROLS` in `dashboard/app/components/SimApp.tsx`); that change was not kept. Every number in it is simulated, and the report says so itself.
- **Intro film** (slide 1): `assets/video/intro.mp4` (1920×1080, 25 s, with sound, 10 MB). It opens on a pile of pictures and clips from the hackathon thrown onto the screen over deep drums, with one larger clip in the middle (people dancing in VR headsets, then the mixed-reality view they saw), switches off to black, then shows the valley, a village, the team and two generated scenes under soft music, with no voice: the presenter speaks over it. Two of its shots are generated clips (a robot walking with villagers, an old man and a robot in a village clinic) and carry no label, so say aloud that they are imagined scenes if asked. The drums and the music were generated with ElevenLabs. To change the film, replace the file: the deck moves on when whatever is there ends.
- **Simulation videos** (slide 18, shown as a future approach), both from the team's internal simulation notes of 15 Nov 2026, re-encoded smaller, no audio:
  - `assets/video/sim-isaac-sim.mp4` (1920×658, 12.5 s): three views of the simulated bench with the physical grip. The motion is scripted. The tile shows only the bench view.
  - `assets/video/sim-isaac-lab.mp4` (1280×830, 42 s): both arms in training on shared tables, from 1 table to 1,024 (the training used 4,096 tables). One arm limits the force, the other moves the scale. The tile shows the picture without its header. Two seconds of it (64 tables) are the last moment of the timeline, `assets/timeline/sim.mp4`.
  - Nothing here has run on the real arm, and the slide says so. `sim-real-grip.mp4` is the earlier clip and is no longer used.
- **Title slide video** (slide 6): `assets/video/tether-clinic.mp4` (720×1112 portrait, 9 s, no audio) with its poster `tether-clinic.jpg`. It is the bench clip `tether.mp4` with its background restyled by a generative video model into a clinic (blue drape, white room), then cut to the 9 seconds where the arm is at the hand and cropped below the hands that hold the upper arm, so only the two arms, the scale and the forearm show. The model also changed details of the people and the arm, so the caption says "restyled": the unaltered footage is on slide 11.
- **Timeline moments** (slide 14): one `<figure class="ev">` per moment inside `#evs` in `index.html`, in order. The line and the spacing are computed from the list, so add, remove or reorder figures freely.
  - `data-dir="up"` is a win (caption above the circle), `data-dir="down"` is a failure (red ring, caption below).
  - `data-y` is the height of the circle on the 1080 px stage (about 490 to 590 for a win, 770 for a failure).
  - Media goes in `assets/timeline/`: an image, or a landscape clip of 3 seconds with a `.jpg` poster:
    `ffmpeg -ss 4 -t 3 -i in.mp4 -an -vf "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720" -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart assets/timeline/name.mp4`
  - The first moment opens by itself when the slide opens. Each press of next (→, Space or a click) opens the following one. A moment stays large for `HOLD` milliseconds (3000, in `startJourney`) and then closes by itself, so only one is ever open; pressing next while it is open closes it early. After the last moment, next goes to the next slide. R starts the slide again.
  - The Skip button (top right of the slide) or the S key shows every moment at once, and next then goes straight to the next slide. A circle already on the line can be clicked to open its moment again: it closes by itself after `HOLD`, or earlier on a click or on next.
  - The times and several captions are placeholders until the team fills them in.
- **Portraits**: `assets/people/`, flags: `assets/flags/`.

## Browsers

The deck is checked in Chrome and in WebKit, the engine of Safari (every slide screenshotted in both and compared, 16 Nov 2026). Firefox could not be run on the test machine, so check it by hand before presenting from Firefox.
The QR code on the last slide (`assets/brand/qr.svg`) encodes `https://flexnsense.vercel.app`; make a new one if the address changes.
