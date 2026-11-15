# Files and folders

```
isaac-sim/            
├── README.md                       state of the work, findings, how to watch the live stream
├── .env / .env.example             Nebius project, network and subnet ids
├── process/                        this folder
├── policies/
│   ├── README.md                   get the policies, upload them, put the pull policy on the robot
│   ├── run_pull_on_robot.py        reference bridge policy -> Station (never run on hardware; dry run by default)
│   ├── pull_v1/                    policy.onnx, policy.pt, checkpoint, settings, summary.json, trace.csv
│   └── move_v1/                    the move arm's latest state and NOTES.md
├── videos/                         every video shown, numbered 01 to 10+
├── assets/elrobot/                 NormaCore's URDF and meshes (Apache-2.0), SOURCE.txt
├── assets/backdrop/                the panorama used for one title render
├── reference/bench-photos/         photos of the real bench
├── cloud/nebius/                   create / start / stop / status / ssh / sync / setup_vm for the workstation
├── tools/
│   ├── check_urdf.py               checks the URDF's joints, limits, masses
│   ├── seed_pose.py                forward kinematics and a solver for reachable start postures
│   └── plot_phase2.py              the force chart of the first loop
├── runs/                           frames, logs and summaries of each run
└── sim/                            runs on the workstation
    ├── 01_import_elrobot.sh        URDF -> simulator model (simple collision shapes)
    ├── 01b_import_decomposed.sh    URDF -> simulator model (finer collision shapes)
    ├── 02_move_joints.py           every joint moves; first render
    ├── 03_yield_loop.py            first force loop: hold vs give way
    ├── 04_show_loop.py, 05_bench_scene.py, 06_move_scale.py, 07_two_arms.py     earlier scenes
    ├── 08_physical_scale.py        two arms, physical scale, real pinch and carry (scripted)
    ├── 09_fix_gripper_usd.py       writes the corrected arm model
    ├── dbg_gripper.py              the wrist-turn experiment
    ├── run_grip_tests.sh, record.sh    run tests / record a video of 08
    └── rl/
        ├── launch.py               registers the tasks, then Isaac Lab's own train / play
        ├── train.sh, stop.sh       start a training in the background / stop it
        ├── eval_pull.py, eval_move.py     measure a policy, film it, export it
        ├── film_training.sh        the policy at several moments of its training, as one video
        ├── push_hf.py              upload to a private Hugging Face repo
        └── flex_tasks/
            ├── __init__.py         the task ids: Flex-Pull-v0, Flex-Move-v0
            ├── pull_env.py, pull_env_cfg.py    the pull task and its settings
            ├── move_env.py, move_env_cfg.py    the move task and its settings
            └── agents.py           PPO settings
```

On the workstation: `~/IsaacLab` (Isaac Lab), `~/flexnsense_sim` (the copy of `sim/`), `~/flexnsense-sim`
(models, runs), `~/IsaacLab/logs/rsl_rl/flex_pull` and `flex_move` (checkpoints), `~/train_<task>.log`.
