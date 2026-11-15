# Follower bus 5B8E114979 - joint range calibration

Calibrated by hand in Station's web UI (ST3215 bus calibration page) on 2026-11-15, then
confirmed by reading it back through `scripts/calibrate.py` (PR #17, Station client).

This range lives on the servos themselves (firmware), not on any computer - if it is ever
reset (a factory reset, a firmware reflash, replacing a servo), **this is recoverable without
redoing the manual move-every-joint flow**: `follower-5B8E114979.json` has the same numbers in
a machine-readable form, and `scripts/restore_calibration.py` writes them straight back to the
servos. See that script's docstring before using it - it skips the manual calibration step on
purpose, which only belongs the first time a joint's real range is unknown.

| Motor | range_min | range_max |
|---|---|---|
| 1 | 1085 | 3011 |
| 2 | 996 | 3100 |
| 3 | 891 | 3205 |
| 4 | 898 | 3198 |
| 5 | 139 | 3957 |
| 6 | 1024 | 3072 |
| 7 | 109 | 3987 |
| 8 | 1242 | 2854 |

Motor-to-joint mapping (from the BOM, not yet confirmed per-motor on this physical unit - see
docs/elrobot.md "What we have not yet verified"): 1-7 are the arm joints `rev_motor_01`
through `rev_motor_07`, 8 is the gripper.
