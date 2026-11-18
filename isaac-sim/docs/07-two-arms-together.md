# Both arms on one table, learning together (16 November 2026)

This is the record of the night both arms were trained at the same time, each with its own policy, on the bench as it
is measured. Everything is simulation. It includes what failed, because three of the five runs looked excellent by
their success rate and were not.

## The task (`Flex-Team-v0`)

One wooden table per bench, both follower arms on it, two agents:

- **pull**: holds the scale's ring; gives way when the force passes the limit; holds still while the scale is moved.
- **move**: starts asleep (its rest posture, turned away); when the pull ends it wakes up, takes the scale at the top
  of the tube, lifts the loop over the finger, sets it on the other finger, lets go and goes back to rest.

One episode is 24 s at 10 decisions a second and starts in one of two ways: the loop is on a finger (that finger
pulls, the loop is moved, the other finger pulls), or the loop lies on the table beside the hand (the move arm picks it
up and puts it on a finger, which then pulls).

Measured on the real bench and used here: the two bases 28 cm apart, the move arm taking the scale about 7 cm from the
pull arm's gripper, the camera that reads the scale 50 cm above the table. The scale is the PESOLA Medio-Line 40600
of the datasheet in `docs/`: 600 g over a 100 mm scale (58.86 N/m), tube 12 mm across, 225 mm without load.

Each arm has its own network, observations and reward (IPPO, `skrl`), and they share the outcome: both are rewarded
when the loop is placed and both are penalised when the move fails.

## What is real, what is a rule

| Simulated | A rule or a model |
|---|---|
| Both arms (NormaCore's ElRobot model), the table they can touch, servos with stiffness, damping and torque limits | **The grip.** Jaws closing within 15 mm of the grasp point take the tube; the loop then follows the gripper; opening within 8 mm of the target finger places it; opening elsewhere drops it |
| The scale's force on the pull arm (the datasheet's spring) | **Fingers as obstacles.** A loop carried lower than 4 cm above its resting height between the two fingers has hit one |
| | **The camera reading.** The marker's place between the two reference stickers in whole camera pixels, with noise, and no reading while the move arm's gripper covers the marker |

The team's own scale reader (`arm1/gauge_reader/stickers.py`) was run on the rendered overhead camera in the films: it found
the two stickers and the marker in 87 to 95 % of pictures and its reading was within 2.5 to 3.9 gf (one standard
deviation) of the true force, worst case about 17 gf. It has no reading while the move arm covers a sticker, as on the
real bench. That check is what the camera model in training is based on; the reader itself does not run during
training (rendering 4,096 cameras each step would take days).

## The five runs

| Run | What it was | Result in simulation (1,024 unseen tables) | What was wrong |
|---|---|---|---|
| 1 | Both arms from scratch, 78 min | Loop placed 100 %, none dropped, none hitting a finger, 3.4 s per move | (a) pull policy failed the real-robot runner test: 4.0 N against a 1.96 N limit; (b) the rule allowed 2 cm of lift where a real finger needs 3; (c) move gripper up to 0.43 m/s |
| 2 | From scratch with varied servos, lag, 4 cm clearance and a speed limit, 32 min | Never took the scale | Too hard from scratch. Stopped |
| 3 | Run 1's arms continued under run 2's conditions, 43 min | Placed 99.8 %, lift 7.8 cm | Runner test still failed (3.5 N). Cause found: both policies parked a servo goal at the end of its range, so the servo pushed with constant torque |
| 4 | Run 3 continued with a limit on how far a goal may lead the joint (0.12 / 0.15 rad), 36 min | Placed 99.9 % | Runner test worse (19.9 N). Cause found: the pull arm pressed its gripper on the table with full torque and braced on it |
| 5 | Run 4 continued with tighter limits (0.08 / 0.06 rad) and a cost on holding a goal away from the joint, 36 min | Placed 99.8 %, none dropped, none hitting a finger, lift 6.3 cm, 3.5 s per move | Runner test still fails (14.8 N). Not explained. **Final run** |

Total: about 3 h 45 min of training.

## Final numbers (run 5, `policies/team_v3`)

| | |
|---|---|
| Loop placed on the target finger (8 mm), loop started on a finger | 99.7 % |
| Loop placed, loop started on the table | 100 % |
| Dropped / hit a finger / too late | 0 % / 0 % / 0 % |
| Time from the end of the pull to placed | 3.5 s on average, 8.1 s at most |
| Highest lift during the carry | 6.3 cm on average (4 cm required) |
| Move arm away from its rest posture at the end | 0.12 rad on average |
| Peak force, first pull / second pull | 1.61 N / 1.58 N on average (limit 1.96 N) |
| Worst single pull | 3.9 N |
| Tables with a pull more than 10 % over the limit | 9.2 % |

## What is not good enough, plainly

- **The pull policy from the joint training must not go on the robot.** It fails the no-robot test of the real-robot
  runner in every run. For the robot the policy to use is `policies/pull_bench_v1`, which passes it.
- **Force.** 9.2 % of tables have a pull more than 10 % over the limit, and the worst pull reaches 3.9 N. The bench
  policy does better (2.5 %).
- **Speed of the move arm is not verified.** The training log says its gripper peaks near 0.27 m/s; the separate
  evaluation measured 0.68 m/s on average. The two disagree and the reason was not found.
- **The grip is a rule.** Real pinching with contact is only in the scripted scene `sim/08_physical_scale.py`.
- **The move policy is told where the scale and the fingers are.** Nothing measures that on the real bench today.

## The lesson worth keeping

A policy maximises the reward as written. Five times tonight a high success rate hid a cheat: swiping the loop through
a finger, grabbing off-centre to exploit a flaw in a rule, hovering where the reward was highest, torque through a
parked goal, bracing on the table. Each was found by looking at something other than the success rate: the lift
height, the time to place, one action pinned at its maximum, and above all the same policy run against a different
model of the arm.

## Files

- `sim/rl/flex_tasks/team_env.py`, `team_env_cfg.py`, `team_ippo.yaml`: the task and its settings.
- `sim/rl/flex_tasks/duo_env.py`: the same scene with a fixed pull policy (shares the geometry and the rules).
- `sim/rl/flex_tasks/bench_draw.py`: the drawn scale and hand.
- `sim/rl/eval_team.py`, `export_team.py`, `record_team.py`: numbers, stand-alone policy files, recorded episodes.
- `sim/rl/film_team.py`, `film_learning.sh`, `film_team_movie.py`: the films.
- `sim/rl/team_deliver.sh`, `team_finish.sh`: everything above in one go.
- `policies/team_v3/`: the two exported policies (TorchScript) and their numbers.

## The recorded episodes

`sim/rl/record_team.py` ran the final policies on 1,024 tables and kept only clean episodes: 937 kept (408 starting
with the loop on the table, 529 with it on a finger), 86 left out because a pull went more than 10 % over the limit,
1 because the loop was not placed. Each kept episode is 23.8 s at 10 rows a second, one Parquet file per arm in
NormaCore's six-column format (joint numbers only, no images). Average peak force in the kept episodes: 1.59 N.
