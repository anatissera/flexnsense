# ElRobot: NormaCore Station, not LeRobot

Our arm is the **NormaCore ElRobot**: 7 revolute joints plus a gripper, 8 Feetech ST3215
servos on one serial bus, ~430 mm reach.

**There is no LeRobot support for this arm.** Confirmed directly against the upstream
`lerobot` source (checked against 0.7.1): no `elrobot_follower` robot type, no driver. A
community repo surfaced during early research (`dreamerarun/elrobot_teleoperation`) looked
like a LeRobot integration, but it is not what this hardware runs, and chasing it wasted
real time. It is not used anywhere in this repo. If you find it referenced in an old commit,
that was a mistake, now corrected.

**What actually operates the arm is NormaCore's own software stack: Station.** A
self-contained binary (no Python packages, no system libraries) that owns servo IDs,
calibration, recording, and inference, with a web UI. LeRobot only enters later, if at all,
to train on data Station recorded, after converting Station's own Parquet format.

## The real sources, read directly, not researched secondhand

- Main repo: [norma-core/norma-core](https://github.com/norma-core/norma-core)
- Station Python client (verified against the actual source):
  [`software/station/shared/station_py/`](https://github.com/norma-core/norma-core/tree/main/software/station/shared/station_py)
  - `client.py`, `example_follow.py`, `example_commands.py` is where
  [`lib/station.py`](../lib/station.py) comes from, line for line on the protocol parts.
- ElRobot hardware (BOM, STL/STEP, manuals, URDF):
  [`hardware/elrobot/`](https://github.com/norma-core/norma-core/tree/main/hardware/elrobot)
- Releases (the `station` binary): [releases page](https://github.com/norma-core/norma-core/releases)

## How Station exposes the servos

Everything flows over one TCP connection to the `station` binary (default port 8888),
protobuf-encoded, through named queues:

- **`st3215/inference`** - a continuous stream of every bus and every motor's raw state.
  Subscribe with `client.follow("st3215/inference", queue)`.
- **`commands`** - write a `DriverCommand` here to move a motor.

A motor's state is a raw byte blob at fixed ST3215 register offsets (from NormaCore's own
example, which cites `protocol/memory.rs`):

| Field | Offset | Size | Notes |
|---|---|---|---|
| Position | `0x38` | 2 bytes | sign bit flags direction; `lib/station.py` unwraps it |
| Current | `0x45` | 2 bytes | **milliamps directly** - Station has already decoded it, no `x6.5` scaling needed |
| Temperature | `0x3F` | 1 byte | degrees C |
| Voltage | `0x3E` | 1 byte | 0.1 V units |

`Present_Load` is not read anywhere in NormaCore's own examples; current is the force proxy
they use and the one we use too, in `scripts/log_force_calibration.py`.

Each motor also carries `range_min` / `range_max`, the calibrated position range, set once in
the web UI and otherwise read-only from Python.

## Setting up the Python client

Full steps in [SETUP.md](../SETUP.md). Summary: sparse-clone `norma-core/norma-core` (~35 MB,
not the full repo with STL files), generate the protobufs with the bundled `gremlin.py`
codegen (pure Python, no Go toolchain), and add a `.pth` file so the clone is importable
without `pip install`. It has zero external dependencies and works in whatever Python
environment is already active - reuse an existing env, no new one needed just for this.

## Motor map

Servo ids 1 to 8, base to gripper, from the official BOM and the URDF (`rev_motor_01`
through `rev_motor_08`):

| Id | URDF joint | |
|---|---|---|
| 1-7 | `rev_motor_01` .. `rev_motor_07` | the 7 arm joints |
| 8 | `rev_motor_08` | gripper |

NormaCore's hardware docs don't name the joints (unlike the SO-101's `shoulder_pan` /
`shoulder_lift` / etc convention); the follower manual's assembly order is the only source
for which physical joint is which id, and that hasn't been cross-checked against the shipped
units yet. Confirm by moving one motor at a time in Station's web UI and watching which
joint turns, not by assuming an order from the SO-101.

**Follower: confirmed from the servo label, code `ST-3215-C018`, 345:1 gear ratio** - the same
part as the SO-101 follower. Stall torque 30 kg·cm (12 V), but **rated (continuous) torque is
only 10 kg·cm**; the force-calibration estimate in
[force-calibration.md](./force-calibration.md) should use the rated figure, not stall.
Backlash 0.870°, which is exactly what robonine.com's independent test measured - that test
is this servo, not just a similar one, so its dead-zone finding (~30-50 g at their ~15 cm
lever arm) applies directly, not as a prior to recheck.

Leader: 8x ST3215, 7.4 V, two different gear variants in circulation (C044 1:191 or C046
1:147) - read the label, the BOM alone doesn't tell you which one shipped.

## Redundancy: 7 joints for a 1-DOF task

A tether pulling along one axis is a 1-DOF task; 7 joints leaves a 6-dimensional null space.
**Lock the joints the task does not need and hold the rest rigid at a fixed pose.** Fewer
moving parts beside a healing finger, and the force calibration (pose-dependent, see
[force-calibration.md](./force-calibration.md)) only has to hold over a small region instead
of the whole workspace.

## Calibration has two unrelated meanings here, don't conflate them

1. **Joint range calibration** (`range_min`/`range_max` per motor) - done once per physical
   arm, by hand, in Station's web UI. No script for this exists on either side (NormaCore
   doesn't publish one, and we don't write one over a manual-only UI flow). Committed to the
   repo once done: [`config/calibration/README.md`](../config/calibration/README.md).
   `scripts/calibrate.py` only checks that this has already happened; it does not perform it.
2. **Force calibration** (servo current -> grams) - our own experiment, run through
   `scripts/log_force_calibration.py`, described in
   [force-calibration.md](./force-calibration.md). Needs (1) done first, since an
   uncalibrated motor's position readings aren't meaningful, but is otherwise unrelated.

## Safety: Station has its own layer, use it

`station.yaml` configures servo-level current limits that Station enforces independently of
whatever Python we run:

```yaml
drivers:
  st3215:
    enabled: true
    current-threshold: 100     # mA, global current limit
    deadband: 20
    motor-current-thresholds:  # per-motor overrides
      8: 40
```

This is the register-level cap our admittance-control design calls for ("in force even if
the control process dies") - configure it here, in Station, not by trying to replicate it in
our own Python, which would not survive our process dying.

## What we have not yet verified

- **Which servo id is which physical joint**, beyond "ids 1-7 are the arm joints in the
  manual's assembly order, 8 is the gripper." Confirm per unit, don't assume.
- **The leader's exact gear ratio** (C044 1:191 vs C046 1:147) - the follower's is confirmed
  (see above), the leader's is not, and it doesn't carry the tether load so it matters less,
  but check the label if the leader is ever used for anything force-sensitive.
- **Teleoperation across two Stations on different machines** is documented; a single
  Station driving both the leader and follower buses is not verified by NormaCore's own
  README, only assumed to work.
