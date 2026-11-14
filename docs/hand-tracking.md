# Hand tracking: finger flexion from a camera

Reads a hand through a camera with MediaPipe's Hand Landmarker and reports one flexion angle
per finger (degrees: 0 = fully extended, larger = more bent). This is roadmap item 5, independent
of the force-sensing work: it measures range of motion during a session, it does not drive the
controller.

## Why MediaPipe (not WiLoR)

MediaPipe Hand Landmarker is the safe default: pretrained, runs on CPU in real time, no GPU needed.
WiLoR gives 3D MANO joint rotations (useful later, see ROADMAP.md item 5), but needs a GPU and is
unverified on a 4 GB card. Start here; add WiLoR alongside once it is verified with `nvidia-smi`,
not instead of this.

## Install

```bash
uv pip install --python .venv/bin/python -r requirements.txt   # includes mediapipe
```

The model (`hand_landmarker.task`, a few MB) downloads once, automatically, to `models/` on first
run. It is not committed (`.gitignore`); if the machine has no network access, download it by hand:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Run it

```bash
python -m hand_tracking run                                 # live window, one hand
python -m hand_tracking run --csv recordings/session.csv     # + log one row per frame
python -m hand_tracking run --headless --video clip.mp4 --csv out.csv   # no camera needed
```

`q` quits the window. CSV columns: `time_s, frame, found, thumb, index, middle, ring, pinky`
(the last five in degrees, blank when no hand is found that frame).

## The angle

For each of index/middle/ring/pinky, the angle is at the **PIP** joint (between the MCP and DIP
landmarks); for the thumb, which has no DIP, it is at the **IP** joint (between MCP and TIP). This
is the same joint and the same convention (`180 - angle between the two segments`) used in published
single-camera ROM measurement for finger joints, so the numbers mean something a clinician recognizes.

The angle is computed from `hand_world_landmarks` (metres, origin at the hand's centre), not the
pixel landmarks used only to draw the overlay: pixel x/y are normalized to the image's width and
height separately and z is a rough relative depth, so an angle computed from them is distorted by
the image's aspect ratio and by how far the hand is from the camera. World landmarks do not have
that problem. See `hand_tracking/landmarks.py` for the geometry (no mediapipe or cv2 import there,
so it is unit-tested directly with synthetic points, no camera needed).

## Known limits

- One hand at a time by default (`--max-hands` to change it); with two hands in frame, only the
  first detection's angles are used.
- A bare angle at one joint, not a full ROM protocol: it does not know the finger's end-of-range,
  does not distinguish active from passive motion, and is not itself the stiffness curve (that
  is roadmap item 6, which combines this with the force reading).
- Lighting and occlusion affect MediaPipe's detection same as any camera-based method; when no
  hand is found, the CSV row is blank rather than guessing.
