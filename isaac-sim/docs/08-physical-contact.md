# The move arm learning by contact alone (16 November 2026, morning)

The two-arm task of the night before ([07-two-arms-together.md](07-two-arms-together.md)) attached the scale to the
gripper by a rule. This one has no such rule. It is the task as the team described it:

1. The scale's loop lies on the table. The **move arm** wakes from rest, pinches the tube, puts the loop round the
   **index finger** and lets go.
2. The **pull arm** pulls back until the scale shows 200 g, then returns to zero.
3. The move arm takes the tube again and moves the loop to the **middle finger**.
4. The pull arm pulls again. The move arm goes back to rest.

The move arm is the trained policy. The pull arm follows a fixed routine. Everything is simulation.

## What is physical

| Part | How it is simulated |
|---|---|
| The scale | Real bodies on the pull arm's gripper: a swivel (two free hinges), the tube (12 mm across, 28 g), and a sliding rod with the wire loop on a spring of 58.86 N/m (the datasheet's 600 g over 100 mm). Written into a copy of NormaCore's URDF by `tools/make_phys_urdfs.py` |
| The force | The stretch of that spring. Nothing else produces it |
| The grip | The move arm's jaws, each with a flat pad, close on the tube and hold it by friction (6 N per jaw, friction 1.0) |
| The loop on a finger | A ring of eight small bars (inner radius 15 mm) that has to go over a solid post. It counts as "on the finger" only when the ring is round the post |
| The fingers | Four solid posts, 16 mm across and 54 mm tall, 4 cm apart. The hand lies palm up, thumb on the right pointing at the move arm, so the index finger is the post nearest the thumb |
| The table | Solid. The loop, the tube and both arms rest on it or collide with it |

Layout as measured on the bench: bases 28 cm apart, the tube taken 7 cm from the pull arm's gripper.

## What is not physical, or simplified

- The pull arm holds the scale's top with a swivel joint, not by pinching it.
- The ring is larger and the fingers are further apart than on a real hand, so that the ring fits between two posts.
- The policy reads numbers (its joints, where the tube, the loop and the target finger are, the scale's reading, the
  stage of the sequence). It does not read pictures.
- The scale has no stickers or marker in this version, so the team's gauge reader cannot read it from the overhead
  picture. The reading comes from the spring.

## The reward, one part per stage

| For | Reward |
|---|---|
| Every stage already finished | 5 per second, so being further along always pays more than lingering |
| Reaching the tube, jaws open far away and closing near it | up to 2 per second |
| The loop's remaining way: up over the fingertip, across to the post, down round it | up to 2 per second |
| The ring round the right finger and the jaws opening | 1 per second |
| Ring round the finger, jaws open, gripper clear | 10, once (index), 10, once (middle) |
| Both pulls done | 5, once |
| While the pull arm pulls: resting, and keeping away from the scale | 1.5 per second, minus 3 per second if closer than 6 cm |
| Gripper faster than 0.25 m/s, jerky actions, yanking the scale, touching the table | penalties |
| Stage not finished in 14 s | minus 3 and the episode ends |

## Training

PPO (`rsl_rl`), 4,096 tables, 120 physics steps a second, 10 decisions a second. 62 minutes in three parts:

| Part | What happened |
|---|---|
| 26 min (iterations 0 to 453) | The arm learned to reach the tube and then stalled 3.5 cm above it. The reward pointed at the middle of the jaw pads, and with the tube lying low the fingertips touch the table before that point can reach it |
| 24 min (from iteration 400) | Reference point moved to the front of the pads. First grips within 5 minutes; the whole sequence in every logged episode by iteration 680. The gripper got faster and faster (0.8 m/s) |
| 12 min (from iteration 900) | Speed penalty raised from 6 to 40 per m/s. Still every episode; speed down to 0.6 m/s |

## Result (`policies/phys_v1`), on 1,024 unseen tables, every episode from the table

| | |
|---|---|
| Loop put on the index finger | 100 % |
| Loop then moved to the middle finger | 100 % |
| Whole sequence with both pulls | 100 % |
| Ran out of time | 0 % |
| Until the loop is on the index finger | 4.6 s |
| Whole sequence | 20.7 s |
| Top reading in a pull | 2.13 N (217 g) on average, 2.31 N at most (target 1.96 N, 200 g) |
| Loop's highest point | 6.9 cm |
| Move arm's fastest gripper speed | 0.53 m/s on average |

## What is not good enough

- **The pull overshoots 200 g** by about 17 g on average (36 g at most): the pull arm reverses only after the reading
  has passed the target.
- **The move arm is too fast**: 0.53 m/s at its peak, against the 0.25 m/s wanted.
- **Simulation only.** This policy has not been through the real-robot runner, and it is told where the tube, the
  loop and the fingers are. Nothing measures that on the real bench today.

## The recorded episodes

`film_phys.py --record` saved 16 complete episodes with **two pictures in every frame**: the overhead camera and the
camera on the move arm (640 x 540 JPEG). `phys_to_parquet.py` wrote them in NormaCore's six-column format, one file
per arm, with the same episodes and pictures in both. Those are the files to train an image policy on.

## Files

- `tools/make_phys_urdfs.py`, `sim/11_import_phys.sh`: the two arm models (pull arm with the scale, move arm with pads).
- `sim/rl/flex_tasks/phys_env.py`, `phys_env_cfg.py`: the task (`Flex-Phys-v0`).
- `sim/rl/look_phys.py`, `dbg_phys.py`, `dbg_phys_policy.py`: checks used on the way.
- `sim/rl/eval_phys.py`, `film_phys.py`, `phys_learning.sh`, `phys_to_parquet.py`, `phys_deliver.sh`: numbers, films,
  the learning record, the dataset.
- Videos: `22` (after 30 min), `23` (learning record), `24` (final, two cameras).
