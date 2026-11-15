# Physical rig: which arm does what

Four ElRobot units on the bench, two leader/follower pairs. This records which bus serial
is which arm and what each one is for, so a bus serial in a config file or a log line maps
back to a physical arm without asking.

## Pair 1: the admittance controller (this repo's `control/`)

- **Follower 1** - bus serial `5B8E114979`. This is "arm B" in the README's diagram: the
  PESOLA dynamometer is hooked to its gripper, pulling along the therapy axis. It is the
  arm `control/loop.py` drives: camera reading -> `control/safety.py` gate ->
  `control/admittance.py` law -> `lib/station.py.set_position` on this bus. Joint range
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

Both serials come from `data_collection/README.md` on the `data_collection` branch (not yet
merged), which is the recording pipeline for this pair. The demonstrations go to the Hugging
Face dataset `JustoDario/flexnsense_skill1`, one Parquet file per episode in Station's own
schema (**not** a `LeRobotDataset`), with the task stored as a plain language string, for
example `"move device from right index to right middle finger"`. That README is explicit that
**training and running a policy on the arm are not covered by it**: recording works, the
policy does not exist yet. Joint range calibration for this pair was done in the web UI but
is not yet recorded under `config/calibration/`.

## Why this split matters

The two followers must not be confused in scripts or config. Anything that reads
`FOLLOWER_BUS_SERIAL` from `lib/env.py` or a `config/calibration/*.json` file should resolve
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
