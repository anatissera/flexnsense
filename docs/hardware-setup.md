# Physical rig: which arm does what

Four ElRobot units on the bench, two leader/follower pairs. This records which bus serial
is which arm and what each one is for, so a bus serial in a config file or a log line maps
back to a physical arm without asking.

![The bench from above: Leader 2 top left, Follower 2 left of the hand, Follower 1 bottom
centre with the PESOLA running from its gripper up to a fingertip, Leader 1 bottom right. The
hand rests palm-up on the cardboard work surface and the dark boom over the wrist carries the
camera.](images/setup.jpg)

What the photo shows, beyond the four labels:

- **The tether axis is roughly in the plane of the table**, running from the fingertip down to
  Follower 1's gripper. That is the direction `excursion_m` has to be projected onto, and the
  direction the `ticks_per_metre` measurement has to move the arm along.
- **The hand rests on the cardboard; no arm is holding it.** See the note below.
- The two leaders sit idle at the table edge. They are for teleoperation during setup and for
  recording demonstrations, not part of a running session.
- The dark boom over the wrist carries the camera, which is the only force sensor in the system
  (see [docs/force-calibration-result.md](./force-calibration-result.md) for why).

## There is no "Arm A" yet

The README's description of the solution opens with "**Arm A** holds the patient's hand still
in a fixed pose". The rig in the photo does not do that: the hand lies on the work surface, and
the two followers are the tether (Follower 1) and the hook mover (Follower 2). Neither is
holding the hand, and there is no fifth arm.

This matters for two things rather than being a nitpick. The pose the hand happens to rest in
is what every pose-dependent number is measured against, so "the session pose" currently means
"however the hand was placed that day" unless something fixes it. And the excursion numbers in
`config/calibration/bench.md` were taken from a hand that was free to shift while flexing,
which is part of why they are recorded as an upper bound.

Either the README should describe the rig as built, or a hand rest of some kind should fix the
pose. Worth deciding before the pose-dependent measurements are taken, since they are only
valid for the pose they were measured in.

## Pair 1: the admittance controller (this repo's `control/`)

- **Follower 1** - bus serial `5B8E114979`. This is "arm B" in the README's diagram: the
  PESOLA dynamometer is hooked to its gripper, pulling along the therapy axis. It is the
  arm `arm1/control/loop.py` drives: camera reading -> `arm1/control/safety.py` gate ->
  `arm1/control/admittance.py` law -> `arm1/lib/station.py.set_position` on this bus. Joint range
  calibration confirmed and recorded in
  [`config/calibration/follower-5B8E114979.json`](../config/calibration/follower-5B8E114979.json).
- **Leader 1** - bus serial not yet logged here. Used to teleoperate Follower 1 by hand
  (setup, slack-taking, finding a pose) before the controller takes over; not part of the
  closed loop itself.

Follower 1 is **not** running a learned policy. The admittance law is a fixed control
equation (see [docs/admittance-design.md](./admittance-design.md)); nothing about it is
trained, and nothing here depends on recorded demonstrations.

## Pair 2: re-hooking the dynamometer between fingers (Justo's task)

- **Follower 2** - bus serial `5B3D050113`, runs at 12 V. Its task is moving the
  dynamometer's hook from one finger to the next between sets, a manipulation problem rather
  than a 1-DOF force-tracking one. Justo is training this with imitation learning: teleoperate
  the move with Leader 2, record it through Station, train a policy on the recordings.
- **Leader 2** - bus serial `5B61034827`, runs at about 7.4 V. The teleoperation side for
  Follower 2's demonstrations.

Both serials come from `arm2/data_collection/README.md` on the `data_collection` branch (not yet
merged), which is the recording pipeline for this pair. The demonstrations go to the Hugging
Face dataset `JustoDario/flexnsense_skill1`, one Parquet file per episode in Station's own
schema (**not** a `LeRobotDataset`), with the task stored as a plain language string, for
example `"move device from right index to right middle finger"`. That README is explicit that
**training and running a policy on the arm are not covered by it**: recording works, the
policy does not exist yet. Joint range calibration for this pair was done in the web UI but
is not yet recorded under `config/calibration/`.

## Why this split matters

The two followers must not be confused in scripts or config. Anything that reads
`FOLLOWER_BUS_SERIAL` from `arm1/lib/env.py` or a `config/calibration/*.json` file should resolve
to **Follower 1** (`5B8E114979`) when the work is the admittance controller - that is the
only bus this repo's `control/` package is meant to drive. Follower 2 belongs to the
separate imitation-learning task and has no code in this repo yet.

## Summary table

| Arm | Bus serial | Voltage | Role |
|---|---|---|---|
| Follower 1 | `5B8E114979` | 12 V | Runs the admittance controller, PESOLA hooked to the gripper |
| Leader 1 | not yet logged | ~7.4 V | Hand teleoperation of Follower 1, setup only |
| Follower 2 | `5B3D050113` | 12 V | Moves the hook between fingers, learned policy (Justo) |
| Leader 2 | `5B61034827` | ~7.4 V | Teleoperation for Follower 2's demonstrations |

## Open items

- Record Leader 1's bus serial here once read off Station's web UI.
- Record Follower 2's joint range calibration under `config/calibration/`, same format as
  `follower-5B8E114979.json`/`.md`, so it is recoverable the same way.
- How the two arms hand control to each other during a session is
  [docs/session-orchestration.md](./session-orchestration.md); that interlock is the reason
  these roles must not be confused.
