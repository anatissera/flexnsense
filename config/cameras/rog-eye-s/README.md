# ROG EYE S (USB-C webcam) for the scale reader

Everything needed to read the Pesola scale with the ASUS ROG EYE S webcam on the MacBook Air. The MacBook camera and the iPhone (Continuity Camera) keep working as before: the code is shared and their behaviour did not change. This folder only holds what is specific to the ROG.

| File | What it is |
|---|---|
| `camera.json` | The camera's identity, the settings the reader opens it with, and what we measured through it |
| `scale_calib.json` | A working calibration made with this camera on 2026-11-15 (zeroed, scale unloaded) |
| `README.md` | This page |

## Use it

```bash
.venv/bin/python -m gauge_reader cameras                 # the ROG must be in the list
cp config/cameras/rog-eye-s/scale_calib.json config/scale_calib.json   # optional: start from our calibration
.venv/bin/python -m gauge_reader run --camera rog        # live reading
.venv/bin/python -m gauge_reader calibrate --camera rog  # or calibrate again
```

`config/scale_calib.json` is the file the reader and the dashboard force server load. It is per machine and not in git, so copy ours there or make your own.

Always choose the camera by name (`--camera rog`, or **ROG EYE S** in the control panel), not by number: the number changes when cameras are plugged in or out.

## Camera settings

- 1920 x 1080 at 30 fps, opened through OpenCV's AVFoundation backend.
- Auto exposure and auto white balance cannot be locked on this camera from OpenCV. Keep the light steady and even.
- The picture is softer and noisier than the iPhone's. The scale should fill as much of the picture as possible: in our setup it spans about 730 of 1920 pixels, which is the minimum we would use.

## Two things that were wrong, and the fixes

**1. The wrong camera opened.** Choosing "ROG EYE S" opened the MacBook camera. The names came from ffmpeg's device order, but OpenCV numbers the cameras sorted by their unique ID. With the ROG plugged in the two orders differ:

| Camera | ffmpeg number | OpenCV number |
|---|---|---|
| ROG EYE S | 1 | 0 |
| MacBook Air Camera | 0 | 1 |
| iPhone Camera | 2 | 2 |

Fix: `lib/camera.py` (`list_cameras`) now takes names and unique IDs from `system_profiler` and sorts them the way OpenCV does. ffmpeg is only the fallback.

**2. "Sticker hidden" on every frame.** This scale does not have the two matching red dots the reader expects. Its fixed red marks are the printed red band next to the label and a small red mark at the far end, by the clip. The generic rules reject that pair: the band is a stripe (not a round dot), the two marks differ a lot in size, and through this camera the far mark is duller than the redness cut-off.

Fix: when the references are clicked by hand at calibration, the reader now learns what they look like and stores it in the calibration file as `ref_profile` (size ratio, shape and redness of the two marks). While running it accepts only a pair that looks like those two, in that order. Scales with two normal dots are read exactly as before. Code: `gauge_reader/stickers.py` (`StickerProfile`, `measure_profile`), `gauge_reader/calibration.py`, `gauge_reader/cli.py`.

## Calibrate with this camera

1. Lay the scale flat, unloaded, fully in view, with the far red mark not covered by the clip.
2. `calibrate --camera rog` (or **1 · Calibrate** in the control panel), then **SPACE** to freeze.
3. Click the **0** line, then the **600** line.
4. The window says the stickers were not found. Click the **red band by the label**, then the **red mark at the far end**. Each click snaps to the centre of the mark.
5. **ENTER**, then **z** to zero, **s** to save, **q** to quit. Without **s** nothing is saved.

## What we measured (2026-11-15)

- 0 g line at u = 0.1246 and 600 g line at u = 0.9203 between the two marks (754 g per unit u).
- Scale at rest: a reading on 92 % of frames, noise 0.85 g (standard deviation), worst frame about 4 g off.
- The far mark decides everything. Half covered by the clip it flickered between about 5 and 226 pixels and the reading dropped out on two thirds of the frames. Fully visible it is about 350 pixels and stable.

## Limits

- The saved calibration was zeroed with nothing on the scale but not checked against a known weight. Hang a known weight and use **w** before trusting absolute values.
- A proper red dot (4 to 6 mm) past the 600 line, clear of the clip, would be more reliable than the small far mark.
- Recalibrate after changing the marks, the lighting or the camera distance by a lot.
