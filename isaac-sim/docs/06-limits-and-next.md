# Limits, and what to do next

## What is assumed (not measured)

| Thing | Value used | How to measure it |
|---|---|---|
| Servo stiffness and damping | 40 N·m/rad, 2 N·m·s/rad | Step a joint by a known amount and record how it settles; or push with a known force and read the deflection |
| Scale spring | 5.886 N over 0.10 m | Hang known weights, measure the marker's travel |
| Apparent mass of a jaw | 0.3 kg | From the gearbox ratio and rotor inertia, or by timing a jaw closing |
| Tube mass | 50 g | A kitchen scale |
| Friction, tube on jaws | 1.2 | Tilt test |

## What is not done

- The pull policy has not run on the real arm. The bridge script is untested.
- The mapping from servo ticks to the URDF's joint angles (zero and sign per motor) has not been measured.
- The move policy works only with a rule-based grip, moves at full motor speed, and needs positions that no camera provides today.
- No simulated episodes were recorded in NormaCore's Parquet format.
- No camera images are used by these policies. The team's result that servo current carries no usable force signal
  (`docs/force-calibration-result.md`) is why the force comes from the scale, not from the motors.

## Next, in order of value

1. Measure the three numbers that matter for the pull task (servo stiffness, spring constant, camera delay), put them
   in `sim/rl/flex_tasks/pull_env_cfg.py`, retrain (8 minutes), re-measure.
2. Measure the tick-to-angle mapping and do the dry run in `policies/README.md`, section 3.
3. Retrain the move task with a speed limit, then on the physical scale of `08` instead of rules.
4. Record simulated episodes in NormaCore's Parquet format if imitation learning with images is wanted.
