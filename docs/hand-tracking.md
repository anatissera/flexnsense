# Hand tracking: finger flexion from a camera

Reads a hand through a camera with MediaPipe's Hand Landmarker and reports one flexion angle
per finger (degrees: 0 = fully extended, larger = more bent). This is roadmap item 5, independent
of the force-sensing work: it measures range of motion during a session, it does not drive the
controller.

## Why MediaPipe (not WiLoR)

MediaPipe Hand Landmarker is the safe default: pretrained, runs on CPU in real time, no GPU needed.
WiLoR gives 3D MANO joint rotations (useful later, see ROADMAP.md item 5), but needs a GPU.

WiLoR has now been measured on the 4 GB RTX 3050, in fp16 with its `--fast` path, on a 640x480
frame with two hands in it:

| stage | per frame |
| --- | --- |
| YOLO hand detector | 19.2 ms |
| WiLoR mesh inference | 70.4 ms |
| pyrender 3D overlay | 247.6 ms |
| total | 337.2 ms (3.0 FPS) |
| without the pyrender overlay | 89.6 ms (11.2 FPS) |

Peak VRAM was 1.69 GB, so the 4 GB card does hold it, which was the open question in the roadmap.
Two things follow. First, pyrender is 73% of the cost and the dashboard does not need it: the
dashboard already draws its own 3D hand from joint angles (MyoViewer, three.js and MuJoCo), so the
figure that matters is 11 FPS, not 3. Second, 11 FPS is still well short of MediaPipe in-browser,
and WiLoR cannot run in a browser at all. It is PyTorch and CUDA behind a 2.5 GB checkpoint, so it
needs a server process sitting next to the camera and the GPU.

The two are therefore not substitutes, they do different jobs:

- **Live loop** (the 3D hand, the live analyzer, later the controller): MediaPipe. Low latency, runs
  wherever the clinician is, no backend, and camera frames never leave the browser.
- **Precision capture** (the report, and the musculoskeletal model in roadmap item 6): WiLoR, over
  recorded segments or at a low rate. Its MANO rotations are full 3 DOF per joint, so they can also
  fill the `*_abduction_r` joints of the MyoHand model that a single flexion angle leaves at zero.

Both emit the same `HandFrame` (below), so nothing downstream has to know which one ran.

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

## HandFrame: the shared output shape

Everything a tracker knows about one frame goes out in a single shape, so that swapping MediaPipe
for WiLoR changes nothing downstream. It is declared twice, and the two have to agree:

- `hand_tracking/frame.py`, used by anything in Python (a WiLoR backend emits this).
- `dashboard/app/lib/handFrame.ts`, used by the in-browser MediaPipe tracker.

`tests/hand_tracking/test_frame.py` reads the TypeScript file and fails if the two drift apart, and
also checks every joint name against the committed MuJoCo model.

```jsonc
{
  "schema": 1,                       // bumped when a field changes meaning
  "source": "mediapipe",             // or "wilor"
  "t": 1234.57,                      // ms since tracking started
  "handedness": "right",             // or "left", or null
  "landmarks": [{"x": 0.51, "y": 0.42, "z": 0.0}, ...],   // 21, normalized image coords, for drawing
  "world":     [{"x": 0.03, "y": -0.02, "z": 0.01}, ...], // 21, metres, hand centred, angles come from these
  "angles":    {"mcp2_flexion_r": 0.4123, ...},           // radians, MuJoCo joint names
  "tipWristMm": 87.65,               // index fingertip to wrist
  "mano": {                          // only when source is "wilor"
    "handPose":     [[0.1, 0.2, 0.3], ...],  // 15 x 3 axis-angle, radians, full 3 DOF per joint
    "globalOrient": [0.4, 0.5, 0.6],         // hand orientation in camera space
    "betas":        [0.0, ...]               // 10 MANO shape coefficients
  }
}
```

Angles here are in **radians**, not the degrees the CLI's CSV uses: a CSV is read by a person,
a HandFrame is fed to MuJoCo and to MANO, and both of those want radians.

The 15 joint names are the MyoHand flexion joints (`mcp2..5`, `pm2..5`, `md2..5`, plus the thumb's
`cmc`, `mp`, `ip`, all suffixed `_flexion_r`). The model also has abduction joints, which one
flexion angle per joint cannot fill; that is what `mano.handPose` is reserved for.

In the dashboard, pass `onFrame` to `<HandTracker>` to receive one of these per tracked frame. The
five fingertips are drawn larger and in amber over the other landmarks, so what the angles are
derived from is visible while tracking.

## Known limits

- One hand at a time by default (`--max-hands` to change it); with two hands in frame, only the
  first detection's angles are used.
- A bare angle at one joint, not a full ROM protocol: it does not know the finger's end-of-range,
  does not distinguish active from passive motion, and is not itself the stiffness curve (that
  is roadmap item 6, which combines this with the force reading).
- Lighting and occlusion affect MediaPipe's detection same as any camera-based method; when no
  hand is found, the CSV row is blank rather than guessing.
