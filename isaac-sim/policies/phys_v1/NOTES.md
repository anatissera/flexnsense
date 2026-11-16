# Move-arm policy trained by contact alone (16 November 2026)

`policy.pt` is the move arm's policy of `Flex-Phys-v0` as a stand-alone TorchScript file: 52 numbers in, 6 out
(motors 1, 2, 3, 4, 6 and open/close), 10 times a second. `summary.json` has its numbers on 1,024 unseen tables:
the whole sequence (table to index finger, pull, index to middle finger, pull) in every one.

**Simulation only. Do not run it on the robot.** It is told where the tube, the loop and the fingers are, which
nothing measures on the real bench, and it has not been through the real-robot runner. Its gripper also moves faster
(0.53 m/s at peak) than is wanted next to a hand.

The record of how it was trained and what is simplified: [../../docs/08-physical-contact.md](../../docs/08-physical-contact.md).
