# Leukerbad Hackathon 2026 — Rehab & Elderly Care

[![Leukerbad Robotics Hackathon 2026](https://img.shields.io/badge/Leukerbad_Robotics_Hackathon-2026-0b7285)](#)
[![Track: Elderly Care & Rehabilitation](https://img.shields.io/badge/track-Elderly_Care_%26_Rehabilitation-2f9e44)](#the-idea)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](./SETUP.md)
[![LeRobot SO-101](https://img.shields.io/badge/LeRobot-SO--101-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/lerobot/so101)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](./gauge_reader)
[![Tests: 168 passing](https://img.shields.io/badge/tests-168_passing-success)](./tests)
[![Status: prototype](https://img.shields.io/badge/status-prototype-orange)](./ROADMAP.md)

Project for the **Leukerbad Robotics Hackathon 2026** (Nov 14–16, 2026, Leukerbad, Switzerland), **Elderly Care & Rehabilitation** track.

## The idea

A **force-limited tether** for rehabilitating fingers after hand surgery, built from two LeRobot SO-101 arms, with a clinician supervising remotely.

- **Arm A** holds and stabilizes the patient's hand in a fixed pose. No policy — a hardcoded position.
- **Arm B** is connected to one of the patient's fingers through a **PESOLA spring scale** (600 g range, 5 g graduations, red indicator sliding along a linear printed scale).

The patient moves the finger. Arm B stays put, acting as an anchor, so the spring stretches and the force rises. Once the force reaches that patient's prescribed threshold (e.g. 200 g), arm B **yields**: it moves slowly in the direction of the finger so the spring stretches no further, capping the force at the threshold.

The clinician sets the threshold and watches the session remotely.

![Force-limited finger tether: arm A holds the hand, the finger pulls on the PESOLA scale held by arm B, a camera reads grams, the clinician sets the cap remotely](docs/images/concept.svg)

Dynamic finger splints in hand therapy work in the 0–450 g range, so a 200 g cap sits in a clinically sensible band. The device category this belongs to is **CPM (continuous passive motion)**, used after hand surgery to prevent capsular and tendon adhesions.

## How force is measured without a force sensor

The kit has no force/torque sensor, but there are two independent signals:

1. **Camera + PESOLA** — a vision model locates the red indicator against the printed scale and converts its position to grams. Physical ground truth, ~30 Hz, but fragile (specular glare off the aluminium tube, occlusion, motion blur).
2. **Servo current** — the Feetech STS3215 motors expose `Present_Load` (register 60) and `Present_Current` (register 69, ×6.5 → mA), readable through LeRobot's motor bus. Fast and robust, but unitless.

The plan is to **calibrate (2) against (1)**, run the control loop on servo current for speed and stability, and keep the camera as independent verification and as the gram readout shown to the clinician. See [docs/force-calibration.md](./docs/force-calibration.md).

This has a real risk attached: the follower servos are geared 1/345, and friction in a highly geared drivetrain produces a dead zone and hysteresis that may swamp a 200 g signal. The calibration experiment is designed to find that out on day one, before anything is built on top of it.

## Control

Admittance control with a deadband — clinically, *assist-as-needed*:

```
F = estimate_force()            # calibrated servo current, filtered
if F < F_threshold:
    hold_position()             # the arm is the anchor
else:
    v = k * (F - F_threshold)   # yield proportionally to the excess
    v = clip(v, 0, v_max)       # always slow
    move(v, direction_that_reduces_F)
```

This is deliberately **not** a learned policy. ACT and Diffusion Policy execute action chunks open-loop and cannot react to contact forces mid-chunk, which is the exact failure mode a device pulling on a post-surgical finger must not have. They also need 50–200 demonstrations and ~16 GB of VRAM to train, against a 4 GB laptop GPU and two days. RL from scratch is out for the same reason.

![Force over time: it rises while the finger moves, stays at the 200 g cap while arm B yields, and drops when the finger relaxes](docs/images/force-cap.svg)

## Where the learning actually is

- The **vision model** that reads the PESOLA scale.
- **Hand keypoint tracking** (MediaPipe or WiLoR) to measure finger range of motion during the session.
- **Per-patient threshold adaptation** from the patient's response within a session — a lightweight online estimator, not deep RL.
- The **force–displacement curve** recorded across the session, which yields joint stiffness. Stiffness derived from force–displacement curves is a published quantitative biomarker of post-surgical tendon adhesion, so the device measures recovery rather than only exercising it. This is the clinical deliverable the remote clinician watches evolve between sessions.

## Safety

Patient-contact force limits are enforced **independently of the primary control loop**:

- Torque/current cap written to the servo registers, in force even if the control process dies.
- Software position limits bounding how far arm B can travel.
- Vision watchdog — on lost detection, low confidence, or dropped frames, **release tension and stop**, never hold.
- Low maximum velocity throughout.
- Physical e-stop within reach of both the patient and the operator.
- The spring itself acts as a mechanical fuse.

## Code layout

```
lib/              shared helpers for any script (robot arms, force calibration, gauge reader)
  camera.py       open a camera by number or name ("iphone") or a video file; warm-up; python -m lib.camera lists cameras
  env.py          read settings from the environment or .env (CAMERA_INDEX, ROBOT_PORT, ...)
  filters.py      1D Kalman filter with outlier gate and "steady" flag (any noisy signal, e.g. servo current)
  geometry.py     a line in an image: project points, distances along it
gauge_reader/     roadmap item 2: read the PESOLA scale through a camera (python -m gauge_reader)
scripts/          robot and calibration scripts (force calibration)
tests/lib/, tests/gauge_reader/
docs/             guides; docs/scale-reader.md for the gauge reader
```

```python
from lib.camera import open_source
from lib.filters import ReadingFilter

src, filt = open_source(camera="iphone"), ReadingFilter(measurement_std_g=2.0)
ok, frame, ts, idx = src.read()
smooth = filt.update(reading, ts)            # .grams (smoothed value), .stable, .velocity_g_s
```

## Gauge reader (camera → grams)

Roadmap item 2, in `gauge_reader/`, built on `lib/`. Two red dots on the PESOLA tube act as fixed references; the red indicator's position between them (a fraction) is converted to grams, so the reading survives the scale moving, turning or changing distance. Each frame yields grams, a steady flag and a reason when no reading is possible (dot covered, scale not found) — the confidence signal for the watchdog.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m gauge_reader.launcher      # control panel: camera, calibrate, run
.venv/bin/python -m gauge_reader --help         # command line: cameras, calibrate, run, evaluate
.venv/bin/python -m pytest -q                   # 168 tests, no camera needed
```

![Gauge reader: find the red spots, the outer two are the dots and the middle one the marker, the marker's fraction between the dots becomes grams](docs/images/gauge-reader.svg)

Setup, calibration, how it works and precision: [docs/scale-reader.md](./docs/scale-reader.md).

## Dashboard (Next.js app)

A browser dashboard in `app/`: a 3D MyoHand (MuJoCo, in the browser) with a finger-curl stress test, recovery metrics and charts, and a text/voice assistant (OpenAI) that answers from the simulated results. All data is simulated.

```bash
pnpm install                    # Node 20+, pnpm 9
cp .env.example .env            # then set OPENAI_API_KEY in .env
pnpm dev                        # http://localhost:3000
```

- `OPENAI_API_KEY` is read server-side only (`app/api/chat`, `app/api/realtime`); the browser never sees it. Without it the simulator still runs, but the chat and voice panel return an error.
- Optional overrides in `.env`: `OPENAI_MODEL` (chat) and `OPENAI_REALTIME_MODEL` (voice).
- The MyoHand model files are served from `public/models/myohand`.
- Voice needs microphone permission and network access to `api.openai.com`.
- `pnpm build && pnpm start` runs the production build; `pnpm lint` runs ESLint.

## Where to start

[ROADMAP.md](./ROADMAP.md) lists the subtasks and the branch for each.
[SETUP.md](./SETUP.md) covers environment setup, [AGENTS.md](./AGENTS.md) the git workflow.
