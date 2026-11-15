# Workflow

`main` is protected: no direct pushes. All changes go through a branch and a Pull Request.

1. Create a branch from `main`:
   ```
   git checkout main
   git pull
   git checkout -b your-branch-name
   ```
2. Commit and push the branch:
   ```
   git push -u origin your-branch-name
   ```
3. Open a Pull Request into `main` (GitHub web or `gh pr create`).
4. Merge the PR from GitHub once reviewed/approved. Direct pushes to `main` are not allowed.

## Branch naming

One branch per subtask in [ROADMAP.md](./ROADMAP.md), using the name given there - `feature/force-calibration`, `feature/gauge-reader`, and so on. For anything outside the roadmap: `fix/...` or `docs/...`.

## Scope

Keep a branch to one roadmap item so PRs stay reviewable and two people can work in parallel without colliding.

## No AI attribution

Commits, pull requests, issues, code comments and docs must **not** credit an AI tool. In particular:

- no `Co-Authored-By:` lines for an AI (Claude, Copilot, ChatGPT, …) in commit messages;
- no "Generated with …" / "🤖" footers in PR descriptions or comments;
- no mention of an AI assistant as an author anywhere in the repo.

Write commit messages and PR descriptions as the human author would.

## No em dashes

No em dash (the long dash, Unicode U+2014) anywhere in this repo: not in the README, not in any doc, not in code, comments, commit messages or PR descriptions. Use a period, a comma, or a plain hyphen instead.

This is enforced in CI (`.github/workflows/no-em-dash.yml`). A PR containing one will not pass.

## Units

Grams-force (gf) below the boundary, newtons (N, SI) above it:

- The PESOLA is graduated in grams, that's what a human reads off it, what the gauge reader calibrates
  against, and what hand-therapy literature and dynamometers already use. Keep the instrument-facing
  numbers (the gauge reader's `grams` field, the CSV's `grams`/`raw_g` columns, the calibration rig's
  known weights) in grams-force. Do not rename these or change their meaning.
- Everything past that boundary, control math, thresholds, any value the admittance controller or the
  safety layer reads, any force or torque equation, is in newtons. `FilteredReading.newtons` and
  `FrameResult.newtons` / `StickerFrameResult.newtons` (and the CSV's `newtons` column) already do the
  conversion; read those, not `grams`, from control code.
- The conversion lives in exactly one place, `lib/units.py` (`grams_force_to_newtons`,
  `newtons_to_grams_force`, `STANDARD_GRAVITY_M_S2`). Never hardcode `9.80665`, `9.81` or `/ 1000`
  anywhere else; import from there.
- When writing a new doc or script that states a force value, give the newton figure first (it's what
  the system runs on) with the gram figure alongside for intuition, e.g. "1.96 N (200 gf)".

# Guide for coding agents

Read [README.md](./README.md) (the idea, safety rules, code layout) and [ROADMAP.md](./ROADMAP.md) (the subtasks) first.

## Repository layout

```
lib/              shared helpers with no knowledge of any one device; use them from any script
  camera.py       cameras by number or name ("iphone") or video files, warm-up, frame source; macOS and Linux (host_os)
  env.py          settings from the environment or .env (CAMERA_INDEX, STATION_HOST, FOLLOWER_BUS_SERIAL)
  filters.py      1D Kalman filter with outlier gate and "steady" flag
  geometry.py     line maths (Axis)
  station.py      the NormaCore Station client: connect, follow motor state, send position commands
gauge_reader/     roadmap item 2: read the PESOLA spring scale through a camera (python -m gauge_reader)
hand_tracking/    roadmap item 5: finger flexion angles through a camera (python -m hand_tracking)
  landmarks.py    pure geometry on MediaPipe's 21 landmarks; no mediapipe/cv2 import, so it is unit-testable
control/          roadmap item 3: the admittance controller
  safety.py       the gate: one force reading -> HOLD / YIELD / RELEASE. Pure, exhaustively tested
  admittance.py   the law: overshoot -> how far to move this cycle. Pure
  loop.py         the only part that touches hardware: camera reading -> gate -> law -> Station
  simulate.py     closed-loop simulation, for choosing B and friends without a finger
dashboard/        Next.js web dashboard (own package.json; run pnpm from here)
scripts/          robot and calibration scripts (e.g. force calibration), against Station
scripts/lerobot/  NOT our hardware: original SO-101-via-LeRobot scripts, parked in case we ever port
tests/lib/, tests/gauge_reader/, tests/hand_tracking/, tests/control/
docs/             user-facing guides (docs/elrobot.md, docs/scale-reader.md, docs/force-calibration.md, docs/hand-tracking.md, docs/admittance-design.md)
```

- Code that any script could reuse (camera, filtering, settings, maths) goes in `lib/`. Keep `lib/` free of device-specific logic, with one deliberate exception: `lib/station.py` is where the Station client lives, so no script reimplements the wire protocol.
- Each roadmap item gets its own package or script folder, named after it, with its tests under `tests/<name>/`.

## Setup and commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                 # all tests; must pass before every commit
.venv/bin/python -m gauge_reader --help        # gauge reader CLI (cameras, calibrate, run, evaluate)
.venv/bin/python -m gauge_reader.launcher      # gauge reader control panel
.venv/bin/python -m hand_tracking run           # finger flexion angles, live window
.venv/bin/python -m lib.camera                 # list cameras (macOS and Linux)
```

## The arm: NormaCore Station, not LeRobot

Our hardware is the **NormaCore ElRobot** (7 joints + gripper, 8 Feetech ST3215). **There is
no LeRobot support for it** - no robot type, no driver, confirmed against upstream. Every
script that talks to the arm goes through NormaCore's own binary, Station, over the protocol
in `lib/station.py`. Full setup: [SETUP.md](./SETUP.md). Background, what's verified and
what isn't: [docs/elrobot.md](./docs/elrobot.md).

- Never write a new path to the arm outside `lib/station.py`. If a script needs to read motor
  state or send a position, it imports from there.
- `scripts/lerobot/` holds the original SO-101-via-LeRobot scripts, unused, kept only in case
  the project ever ports to a LeRobot-supported arm. Don't extend them; don't assume they run.
- Station enforces its own current limits in `station.yaml`, independent of whatever Python
  we run - that's the register-level safety cap, not something to reimplement in our code.

## Rules

- **Tests:** add or update tests with every change. Tests must run without a camera or robot: use the synthetic videos (`gauge_reader/synthetic/`) or fakes.
- **Hardware:** never open a camera or move an arm from an automated step without the person at the keyboard knowing. Arm commands must respect the safety layer in the README (torque cap, position limits, watchdog, low velocity).
- **Safety first:** a missing or low-confidence force reading must lead to "release and stop", never to a guess. Do not weaken safety limits to make a test pass.
- **Never commit** `.env`, `config/scale_calib.json`, `recordings/` or `internal-docs/` (all git-ignored): they are per machine or private.
- **Do commit** `config/calibration/*.json`: the arm's joint-range calibration lives in the servo firmware and the bench constants describe the physical rig, so neither is per machine and the team shares both. Never hand-edit the numbers; re-measure and commit the result. The procedures are in [docs/bench-measurements.md](./docs/bench-measurements.md).
- **Privacy:** never save full camera frames of a patient; log numbers, or crops of the device only.
- **Docs:** when behaviour or commands change, update the README and the guide in `docs/` in the same PR.

<!-- BEGIN:nextjs-agent-rules -->

## This is NOT the Next.js you know

This version has breaking changes - APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `dashboard/node_modules/next/dist/docs/` (the Next.js app lives in `dashboard/`; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` - verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
