# Arm motor calibration: committed on purpose

This is the one calibration file in the repo that **is** committed. It differs from the
gauge reader's `config/scale_calib.json` (gitignored) in what it depends on: the scale
calibration depends on where *this laptop's camera* sits, so it is per machine; the motor
calibration (joint min/max, homing offset, drive mode) depends on the physical servos of a
specific ElRobot unit, not on which laptop happens to be plugged into it. Whoever connects to
the same physical arm, from any machine, wants the same numbers.

## Where it comes from

LeRobot writes/reads this file itself, from a fixed location outside the repo:

```
~/.cache/huggingface/lerobot/calibration/robots/elrobot_follower/<robot.id>.json
~/.cache/huggingface/lerobot/calibration/teleoperators/elrobot_leader/<teleop.id>.json
```

(`$HF_LEROBOT_CALIBRATION` if set, else `$HF_HOME/lerobot/calibration`.) `<robot.id>` /
`<teleop.id>` is whatever you passed to `--robot.id=` / `--teleop.id=` when you ran
`lerobot-calibrate`.

## Naming, since we have two pairs

Use `pair<N>_follower` / `pair<N>_leader` as the id while calibrating, so the files don't
collide:

```bash
lerobot-calibrate --robot.type=elrobot_follower --robot.port=/dev/ttyACM0 --robot.id=pair1_follower
lerobot-calibrate --teleop.type=elrobot_leader  --teleop.port=/dev/ttyACM1 --teleop.id=pair1_leader
# repeat with pair2_follower / pair2_leader for the second pair
```

## Committing it

From the machine that actually ran the calibration:

```bash
mkdir -p config/calibration
cp ~/.cache/huggingface/lerobot/calibration/robots/elrobot_follower/pair1_follower.json \
   config/calibration/
cp ~/.cache/huggingface/lerobot/calibration/teleoperators/elrobot_leader/pair1_leader.json \
   config/calibration/
git add config/calibration/pair1_follower.json config/calibration/pair1_leader.json
```

Repeat for `pair2_*` once that pair is calibrated.

## Using a committed file on another machine

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/elrobot_follower
cp config/calibration/pair1_follower.json \
   ~/.cache/huggingface/lerobot/calibration/robots/elrobot_follower/
# then: --robot.id=pair1_follower skips recalibrating and loads this file
```

## Recalibrate, do not hand-edit

If a joint's range looks wrong, run `lerobot-calibrate` again and commit the new file. Do not
edit the numbers by hand: `range_min`/`range_max`/`homing_offset` come from physically moving
the joint through its range during calibration, and a wrong hand-edited value is a silent way
to command a joint past where it can actually go.

## Physical units, once we know which pair does what

Track bus serial -> pair -> role (Arm A / Arm B) here once roles are assigned, so a laptop that
sees two serial numbers on the USB bus knows which is which without guessing:

| Bus serial | Pair | Role | Project role |
|---|---|---|---|
| | 1 | follower | |
| | 1 | leader | |
| | 2 | follower | |
| | 2 | leader | |
