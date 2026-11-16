# How the training works

## The idea in four lines

1. Put 4,096 copies of the arm in the simulator, each facing a slightly different situation.
2. A small neural network (the policy) looks at some numbers and decides how to move the motors, 60 times a second.
3. A formula (the reward) scores every instant: good behaviour scores high, bad behaviour scores low.
4. An algorithm (PPO) nudges the network towards whatever scored higher. Repeat a few hundred times.

One round ("iteration") is 24 decisions by each of the 4,096 arms: 98,304 simulated steps. The workstation does about
110,000 to 145,000 steps a second, so an iteration takes under a second.

## The pull arm (`Flex-Pull-v0`): trained, works in simulation

**The situation.** The arm holds the top of the scale. A finger pulls on the other end. In the simulation the scale
is a spring between the gripper and a finger point that moves away, holds, and comes back.

**What changes from arm to arm** (so the policy does not learn one single pull): how far the finger pulls (3 to 10 cm),
how fast (0.8 to 3 s), how long it holds (1 to 3 s), when it starts, the direction (up to 10 degrees off line), the
spring's stiffness (plus or minus 30 %), the arm's starting posture, and noise on the force reading.

**What the policy sees (29 numbers).** The seven arm joints (angle and speed), how far the driven joints are from their
targets, the force reading now and one step earlier, where the gripper is relative to where it started (along the
scale, across it, up), and its own last action.

**What it does (5 numbers).** Small changes to the targets of motors 1, 2, 3, 4 and 6. The two roll motors (5 and 7)
and the gripper are held.

**The reward, per second.**

| Term | Value | Why |
|---|---|---|
| Alive | +1 | A baseline |
| Force above the limit | -6 x (excess + excess squared), in newtons | The whole point: do not exceed 1.96 N |
| Away from its place while the force is low | -8 per metre | Come back when nobody pulls; do not give way for nothing |
| Off the scale's line, or up or down | -8 per metre | Give way along the scale, not sideways |
| Pointing away from the scale's line | -1 x (1 - cosine) | Keep the gripper in line |
| Jerky actions | -0.02 x squared change | Smooth motion |

**The run.** 600 iterations, 59 million steps, 7.7 minutes. Network: three layers of 128, 128 and 64 units.

**The result** on 1,024 pulls it had not seen: average peak force 1.60 N with the policy, 3.61 N with the arm held
still; 1.1 % of pulls go more than 10 % over the limit, against 85 %. Full table in `../policies/README.md`.

**How it learned** (video 10): untrained, the arm flails (4.7 N peaks). After 100 iterations it gives way far too
easily (0.7 N peaks: safe, but no resistance for the exercise). By 599 it resists up to about 1.6 N and then yields.

## The move arm (`Flex-Move-v0`): trained after six attempts

**The situation.** The scale lies between the pull arm's jaws (the ring end) and a finger (the loop end). The move arm
must take the tube near the loop, lift it, carry the loop to the neighbouring finger, set it down, let go.

**A simplification.** Here the scale is not a physical body. Rules decide what happens: jaws that close within 15 mm
of the grasp point take the tube; the loop then follows the gripper; letting go within 8 mm of the target places it;
letting go elsewhere drops it; a loop that is lower than 2 cm above its rest height while between two fingers has hit a
finger. Rules are fast (4,096 arms) and the physical pinch was already shown to work separately (`08`).

**What happened, attempt by attempt.** This is the honest record; each attempt is 10 to 20 minutes of training, about 80 minutes in all.

| Attempt | What the policy did | Cause | Change |
|---|---|---|---|
| 1 | Hovered near the tube, never closed the jaws | The reward punished closing until within 12 mm: a cliff | Closing rewarded gradually over the last 3 cm |
| 2 | Took the tube, lifted it, held it forever | Letting go in the wrong place ended the episode with a penalty; and hovering beside the target paid more than descending onto it | Placing tolerance starts generous and tightens; smooth carry reward |
| 3 | "Placed" in 99.5 % of episodes, in 0.2 s, lifting 5 mm | It swiped the loop through the finger: the penalty for that was tiny | Hitting a finger ends the episode as a failure |
| 4 | "Placed" 100 % with a 1 mm lift | A flaw in the rule: at the moment of taking, the loop jumped onto the gripper's line, so grabbing off-centre teleported it to the next finger | The tube keeps the offset it was taken with; fast flicks through a finger are caught |
| 5 | Took the tube, lifted it 2 to 3 cm, would not carry it sideways; later carried it to 9 mm from the target and held it there, jaws shut | Crossing between fingers had become risky and the reward gave too little reason to do it; then opening over the target paid too little | The carry reward now follows the remaining route (up, across, down); opening over the target pays clearly |
| 6 | Takes, lifts about 5 cm, sets the loop on the target finger, lets go | | 100 % placed on 1,024 arms at 8 mm, none dropped, none hit a finger (`../policies/move_v1/NOTES.md`) |

Attempts 3 and 4 are worth remembering: a policy maximises the reward as written, not the task as intended. A
success rate alone would have said "done". The lift height and the time to place showed it was cheating.
