# Joint policies, both arms trained together (run 5 of 16 November 2026)

`pull/policy.pt` and `move/policy.pt` are the two arms' policies as stand-alone TorchScript files (observation scaling
included). They were trained at the same time in `Flex-Team-v0` and checked to give the same actions as the trained
agent. `summary.json` has the numbers on 1,024 unseen tables; `params/` the training settings.

**Simulation only. Do not run either on the robot.**

- The pull policy fails the no-robot test of the real-robot runner (`selftest_pull.py`): 14.8 N on average against a
  1.96 N limit, with the arm drifting at rest. For the robot use `../pull_bench_v1`, which passes that test.
- The move policy uses a rule for the grip and is told where the scale and the fingers are.

The full record, including the four earlier runs and what was wrong with each, is in
[../../docs/07-two-arms-together.md](../../docs/07-two-arms-together.md).
