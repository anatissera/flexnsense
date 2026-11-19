# Scale Reader

Reads a **Pesola Medio-Line 600 g** spring scale through a camera and shows the weight (or pulling force) in grams, live, and logs it to a file. It works while the scale is held in the hands and moving.

## What you need

- The Pesola 600 g scale (0–600 g, 5 g per line). Datasheet: [pesola-medio-line-40600-datasheet.pdf](pesola-medio-line-40600-datasheet.pdf) (item 40600: capacity 600 g, division 5 g, precision ±0.3 %, scale length 100 mm, tube 12 mm).
- **Two red dots** drawn or stuck on the scale's tube (see below).
- A camera: the laptop camera, a USB webcam, or (on a Mac) an iPhone via Continuity Camera (better picture).
- Python 3.12 and this repository.
- macOS or Linux. The operating system is detected automatically; Linux needs a few system packages (below).

## Set up the scale: two red dots

The scale already has a red **marker** that slides along the slot as the load changes. Add two red dots that never move:

```
  [label] ●  ●marker ───────────────────────── 600   ●
          ↑  ↑                                       ↑
     dot 1   marker (moves with the weight)     dot 2
```

- **Dot 1:** about 1 cm before the **0** line, where the marker rests when nothing hangs on the scale.
- **Dot 2:** about 1 cm after the **600** line.
- Both on the tube's side facing the camera, at the same height, on the line the marker travels.
- About 4–6 mm wide. Do not let a dot touch the marker.
- Hold the scale by the ring and the hook, never over the dots.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

On Linux (Ubuntu), also install the system packages and give your user access to the cameras:

```bash
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0 v4l-utils
sudo usermod -aG video $USER      # then log out and in again
```

Details, and what each package is for: [SETUP.md](../SETUP.md), section 2.

| | macOS | Linux |
|---|---|---|
| Camera backend | AVFoundation | V4L2 |
| Camera number `N` | OpenCV's own order (sorted by unique ID) | `/dev/videoN` |
| Where the names come from | `system_profiler` (fallback: `ffmpeg`) | `/sys/class/video4linux/videoN/name` |
| iPhone by name (`--camera iphone`) | yes (Continuity Camera) | no: use a number or the webcam's name, e.g. `--camera rog` |

## Use it

Open the control panel:

```bash
.venv/bin/python -m arm1.gauge_reader.launcher
```

1. Pick the **camera** (cameras appear by name; on a Mac an iPhone is preselected) and keep **Mode = Stickers**.
2. Press **1 · Calibrate**. With **nothing hanging on the scale**, in the camera window:
   - **SPACE** freezes the picture.
   - Click the **0** line, then the **600** line.
   - The two dots get circled automatically. Press **ENTER**.
   - Press **z** to set zero, **s** to save, **q** to quit.
3. Press **2 · Run**. The weight appears on the video:
   - **green** = steady reading,
   - **amber** = the load is changing,
   - a message such as **"sticker hidden"** = a dot is covered, so no number is shown (it never guesses).

Each session is saved as a CSV file in `recordings/` (time and grams; no images).

Optional: to double-check, hang a known weight during calibration, press **w** in the camera window, type the grams in the control panel and press **Send**.

### Command line

The control panel runs these commands; they also work on their own:

```bash
.venv/bin/python -m arm1.gauge_reader cameras                       # list cameras and their numbers (macOS and Linux)
.venv/bin/python -m arm1.gauge_reader calibrate --camera iphone     # sticker mode (default); --mode fixed for a clamped scale
.venv/bin/python -m arm1.gauge_reader calibrate --resume            # reload the saved calibration: re-zero, add weights
.venv/bin/python -m arm1.gauge_reader run --camera iphone           # live reading + CSV in recordings/
.venv/bin/python -m arm1.gauge_reader run --camera 2               # Linux: /dev/video2 (or a name fragment, or the path itself)
.venv/bin/python -m arm1.gauge_reader run --headless --video clip.mp4 --csv out.csv

# tests without hardware: synthetic hand-held video with exact ground truth, then score it
.venv/bin/python -m arm1.gauge_reader.synthetic.moving recordings/moving.mp4 --speed 2 --occlude-every 40
.venv/bin/python -m arm1.gauge_reader evaluate --video recordings/moving.mp4
```

Keys in the camera window - calibrate: `SPACE` freeze, click 0 then 600, `ENTER` accept, `c` click the dots by hand, `z` zero, `w` known weight, `s` save, `m` mask, `q` quit. Run: `z` zero, `s` save, `SPACE` pause, `m` mask, `q` quit.

CSV columns: `time_s, frame, found, t_px, raw_g, grams, stable, std_g, confidence, u, sticker_dist_px, reason`.

## How it works (simply)

1. Every video frame, the program looks for **red spots**.
2. It finds the **three spots in a row**: the two outer ones are the dots, the one between them is the marker.
3. It measures **where the marker sits between the two dots**, as a fraction (for example 40 % of the way).
4. Calibration told it which fraction means 0 g and which means 600 g, so the fraction becomes **grams**.
5. A smoothing filter removes jitter and marks the reading as steady when it stops changing.

Because everything is measured *relative to the dots*, it does not matter if the scale moves, tilts, or gets closer to or further from the camera: the fraction stays the same.

### Why moving the scale does not change the reading

The program never keeps pixel positions from the calibration picture. It keeps **fractions**.

**At calibration (once)** it stores where the 0 line and the 600 line fall *between the two dots*, as fractions of the dot-to-dot distance:

```
dot 1 ●───┃0───────────────────────────┃600───● dot 2
      0   0.124                       0.918   1
```

(Real values from our iPhone calibration: the 0 line sits at 0.124, the 600 line at 0.918.)

**In every frame** it finds dot 1 (D1), dot 2 (D2) and the marker (M) again, in that frame's pixels, and computes:

```
u      = ((M − D1) · (D2 − D1)) / |D2 − D1|²     how far along the dots the marker is, 0 to 1
grams  = 600 × (u − 0.124) / (0.918 − 0.124)     then corrected by the zero set with z
```

When the scale moves closer or further, **every distance in the picture grows or shrinks by the same factor**, so the fraction does not change:

| | Scale close to the camera | Scale further away |
|---|---|---|
| Dot 1 to dot 2 | 1340 px | 900 px |
| Dot 1 to marker | 697 px | 468 px |
| **u = marker ÷ dots** | **0.520** | **0.520** |
| Reading | 300 g | 300 g |

The same holds when the scale slides or turns in the picture. In our two real calibrations the dot distance changed by 14 % (about 1340 px vs 1160 px), but the 0-line fraction changed by less than 0.1 % (0.1239 vs 0.1231).

What *does* affect it:

- **Tilting the scale towards or away from the camera**: the near end looks bigger, so the fraction shifts a little (about 1–2 g for a small tilt). Keep the scale roughly face-on.
- **Dots that can move against the printed lines**: draw them on the same rigid tube as the numbers, not on the hook or the moving part.
- **A dot covered or outside the picture**: no reading is shown until both dots are visible again.

### Smoothing: the Kalman filter

Every frame gives a raw reading with a little noise. Before it is shown, it goes through a small **Kalman filter** (`arm1/lib/filters.py`; the method is explained step by step at [kalmanfilter.net](https://kalmanfilter.net/)). It tracks two numbers, the weight in grams and how fast it is changing in grams per second, and blends:

- a **prediction**: "it was 250 g and rising 20 g/s, so now it should be about 251 g", and
- the **new measurement**: "the camera says 252 g",

weighing each by how much it can be trusted. The result is smoother than the raw value without lagging as much as a plain average.

What it does for the scale:

| Job | Without it | With it |
|---|---|---|
| Jitter at rest | The number flickers a gram or two | Steady |
| Steady flag (green / amber) | Hard to decide | Steady when the rate is under 5 g/s and the spread is small |
| One bad frame (blur, glare) | A spike in the log | Dropped as an outlier |
| A real, fast pull | – | Recognised; the filter restarts instead of trailing behind |
| Uneven frame timing (iPhone 30 fps, Mac 15 fps) | Breaks a fixed-window average | Handled: each step uses the real time between frames |

What it cannot do:

- **Fix systematic errors.** A wrong calibration, the scale tilted towards the camera or a misplaced dot shifts every reading the same way; the filter just smooths the wrong value. Those are fixed with **z**, known weights and a good setup.
- **Avoid a little lag on fast pulls**: up to about 25 g during a quick ramp in our tests, catching up once the load settles. The raw value is always in the CSV too.
- **Guess while a dot is covered.** It could keep predicting, but those guesses were up to 155 g off in tests, so the app shows "sticker hidden" instead of a number.

Tuning: the filter's noise settings are currently set from test videos. To match a real camera, hold the scale still for 10 seconds, look at the spread of `raw_g` in the CSV, and set the filter's measurement noise (`measurement_std_g`) to that value.

A bonus for rehab: the filter also estimates the **rate of pull** (grams per second), which shows how quickly a patient builds force. It is available in the code (`velocity_g_s`) and could be added to the CSV and the therapist view.

### Other red marks as references

Two matching round dots are the most reliable references, and the only kind found automatically. If the scale has other fixed red marks instead (for example the printed red band next to the label at one end and a small red mark at the other), click them by hand at calibration: after the 0 and 600 clicks the window asks for sticker 1, then sticker 2. Each click snaps to the centre of the red mark under it, and the calibration file stores what the two marks look like (`ref_profile`: their size ratio, shape and redness). From then on the reader accepts only a pair that looks like those two, in that order, so a band and a small mark work even though they are not matching dots.

A worked example with a USB webcam (camera settings, a calibration file, what was measured) is in [config/cameras/rog-eye-s](../config/cameras/rog-eye-s/README.md).

Limits: each mark must be fully visible, clearly red in the picture and fixed to the tube. A mark half covered by a clip changes size from frame to frame and the reading drops out ("sticker hidden"). Recalibrate if the marks or the lighting change a lot.

## How precise is it?

- The scale itself is accurate to about **±1.8 g** (±0.3 % of 600 g, from the [datasheet](pesola-medio-line-40600-datasheet.pdf)).
- On test videos of a moving, tilting, blurred scale, the camera adds less than **0.5 g** of error.
- Expect about **±3 g** in real use with a steady light. Fast movement or bad light makes it worse; a covered dot stops the reading instead of giving a wrong one.

## Troubleshooting

| Problem | What to do |
|---|---|
| Camera does not open (macOS) | System Settings → Privacy & Security → Camera → allow your terminal or IDE. |
| Camera does not open (Linux) | Camera `N` is `/dev/videoN`: list them with `python -m arm1.gauge_reader cameras`. Check `ls -l /dev/video*` and that you are in the `video` group (`sudo usermod -aG video $USER`, then log out and in). Close any other program using the camera. |
| No camera listed (Linux) | Plug the camera in and check `ls /dev/video*`. `v4l2-ctl --list-devices` (package `v4l-utils`) shows what the kernel sees. |
| `cv2.imshow` error "The function is not implemented" (Linux) | That Python has `opencv-python-headless` (LeRobot installs it), which has no windows. Use this repo's `.venv`. |
| iPhone picture is black | Lock the iPhone, keep it still in landscape near the Mac (Wi-Fi and Bluetooth on, same Apple ID). It should say "Connected to Mac". Opening Photo Booth and choosing the iPhone forces the connection. |
| "Stickers not found" at calibration | Both dots and the marker must be visible. Check there are **two** dots plus the marker. You can also click dot 1, then dot 2, by hand (see "Other red marks as references" below). |
| Reads a few grams at 0 g | Press **z** with nothing on the scale (zero), then **s** to save. |
| "Sticker hidden" often | Fingers cover a dot, or a dot leaves the picture. Hold the scale by its ends and keep it fully in view. On a USB webcam under dim light, press **m** for the mask view: if the marks themselves look faint (small, grey, barely-there blobs) rather than covered, the light is too dim or the fixed exposure is too low for this camera - raise `CAMERA_V4L2_EXPOSURE` (Linux) rather than moving anything. |
| A hand near the scale gets picked up as a dot | Some cameras render skin red enough to cross the mask under some lighting, even though it normally does not. Set `GAUGE_SAT_STRONG` in `.env` to a saturation skin does not reach but the marks do (`arm1/gauge_reader/detector.py`'s own numbers: stickers ~220, the marker ~200, skin ~130, wood ~75 - try 150). |
| Jumpy numbers | Use steady, even light; avoid moving shadows and strong reflections on the metal tube. |

## Where the code is

| File | Job |
|---|---|
| `arm1/gauge_reader/stickers.py` | Finds dot 1, dot 2 and the marker in each frame; returns the fraction u or a reason |
| `arm1/gauge_reader/calibration.py` | Turns u (or pixels, in fixed mode) into grams; saves `config/scale_calib.json` |
| `arm1/gauge_reader/detector.py` | Red mask and sub-pixel centres; the fixed-mode marker detector |
| `arm1/gauge_reader/cli.py` | `python -m arm1.gauge_reader`: cameras, calibrate, run, evaluate |
| `arm1/gauge_reader/launcher.py` | The control panel |
| `arm1/gauge_reader/overlay.py` | Drawing on the video |
| `gauge_reader/synthetic/` | Test videos with exact ground truth (still and hand-held) |
| `arm1/lib/camera.py` | Shared: opens cameras (by number or name, e.g. `iphone`) and video files |
| `arm1/lib/filters.py` | Shared: the Kalman filter and steady flag |
| `arm1/lib/geometry.py`, `arm1/lib/env.py` | Shared: line maths; settings from `.env` |

The `lib/` package knows nothing about the scale, so robot and calibration scripts can use the same camera and filter code.

## Privacy

Frames are processed in memory and never saved. The CSV holds only times and grams. The optional `--record` mode saves only a narrow strip around the scale, never the full picture (the camera may also see the person using it).

## Showing the force in the dashboard

The dashboard can use the scale as its force input, in place of the constant 20 N it otherwise assumes.

1. Calibrate as usual (`python -m arm1.gauge_reader calibrate`), so `config/scale_calib.json` exists.
2. Start the server: `python -m arm1.gauge_reader.server` (listens on `ws://127.0.0.1:8766/force`; needs `fastapi` and `uvicorn`).
3. In the dashboard, under the camera controls, pick the camera that sees the scale in **Force camera** and press **Start force**. The browser owns the camera, so a machine with several can dedicate one to the scale and another to the hand. **Zero (unloaded)** tares the reading.

Each JPEG frame the browser sends is measured by the same pipeline as `run` and discarded (nothing is written to disk). The reply is `{found, newtons, grams, stable, confidence, reason, inferenceMs}`. `newtons` and `grams` are `null` when there is no recent measurement. The dashboard treats a null or a reading older than 0.5 s as no force: the fingertip arrow disappears and the analysis records 0 N for that sample. It never holds the last value.

The hand camera must also be on, since the live analysis is driven by the tracked finger. Starting the force camera while the hand camera is already on works; stopping it returns the analysis to the 20 N assumption.
