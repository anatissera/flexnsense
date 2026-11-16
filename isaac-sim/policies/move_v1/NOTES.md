# Move-arm policy, version 1: where it ended

Checkpoint: iteration 4,750 of the last run (`checkpoint.pt`; exported as `policy.onnx` and `policy.pt`).

Measured on 1,024 simulated arms at the real placing tolerance of 8 mm (`summary.json`):

| | |
|---|---|
| Loop placed on the target finger | 100 % (99.9 % at the checkpoint 250 iterations earlier) |
| Dropped | 0 % |
| Hit a finger on the way | 0 % |
| Highest lift during the carry, average | 46 mm |
| Time from start to placed, average / slowest | 0.40 s / 0.78 s |

How it got there: six training runs, about 80 minutes of training in total, about 470 million simulated steps. The first
five runs each exposed a flaw in the reward or in the task's rules; the record is in `../../docs/03-training.md`
(in the working folder: `process/03-training.md`).

**Do not run this policy on the robot.** It is being redone on the physical scene, starting from the resting posture.

Read this before using it:

- The grip is a rule in this task, not physics: jaws closing within 15 mm of the grasp point take the tube; opening
  within 8 mm of the target finger places the loop. The physical pinch is shown in `sim/08_physical_scale.py`.
- The policy is fast: 0.4 s for the whole move. It uses the full motor speed the URDF allows (4.71 rad/s). A real
  session next to a patient's hand would want a speed limit, which means retraining with one.
- It is told where the scale, the loop and the fingers are. Nothing measures those on the real bench today.
