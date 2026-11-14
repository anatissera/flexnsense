# ElRobot (NormaCore) — Setup Guide

How to install and use the **ElRobot** arm by NormaCore on any computer: servo IDs, calibration, teleoperation, dataset recording, SmolVLA training and on-robot deployment.

> ElRobot is **not** an SO-101. It is a 7+1 DoF arm (7 joints + gripper) with **8 ST3215 servos per arm**. LeLab and upstream `lerobot` only support the 6-motor SO-101, so this guide uses NormaCore's own software stack (**Station**).

## Official NormaCore resources

| Resource | Link |
|---|---|
| Main repository | https://github.com/norma-core/norma-core |
| ElRobot hardware (BOM, STL/STEP, print files) | https://github.com/norma-core/norma-core/tree/main/hardware/elrobot |
| Follower assembly manual (PDF) | https://github.com/norma-core/norma-core/blob/main/hardware/elrobot/manual-follower.pdf |
| Leader assembly manual (PDF) | https://github.com/norma-core/norma-core/blob/main/hardware/elrobot/manual-leader.pdf |
| URDF / simulation | https://github.com/norma-core/norma-core/tree/main/hardware/elrobot/simulation |
| URDF web playground | https://normacore.dev/elrobot-urdf/ |
| Station (README) | https://github.com/norma-core/norma-core/tree/main/software/station/bin/station |
| Station releases | https://github.com/norma-core/norma-core/releases |
| Dataset recording & export guide | https://github.com/norma-core/norma-core/blob/main/docs/datasets/export-parquet/datasets-export.md |
| Remote teleop example (leader → follower) | https://github.com/norma-core/norma-core/tree/main/software/station/examples/st3215-remote-teleop-py |
| SmolVLA fine-tune (train + deploy) | https://github.com/norma-core/norma-core/tree/main/software/ai/smolvla_py |
| Nebius cloud training recipe | https://github.com/norma-core/norma-core/blob/main/software/ai/smolvla_py/nebius.md |
| Website | https://normacore.dev |
| Discord | https://discord.gg/Z4Ytw3QfHP |
| Intro video | https://youtu.be/WXRG1KnzKv4 |

## Hardware summary

From the official BOM:

| | Follower | Leader |
|---|---|---|
| Servos | 8× ST3215, 12 V / 30 kg | 8× ST3215, 7.4 V (C044 1:191 or C046 1:147) |
| Power supply | 12 V / 4 A | 5 V / 3 A |
| Servo driver | Waveshare Bus Servo Adapter A, or Seeed XIAO bus servo board | same |

URDF joints: `rev_motor_01` … `rev_motor_07` (arm) and `rev_motor_08` (gripper). Servo IDs are **1 to 8**, from base to gripper.

## 1. Install Station

Station is a single self-contained binary: no Python packages and no system libraries to install. It handles servo IDs, calibration, recording and inference, and has a web UI.

Download the archive for your platform from the [releases page](https://github.com/norma-core/norma-core/releases). Current version at the time of writing: `v0.1.0-beta.10`.

| Platform | Asset |
|---|---|
| Linux x86_64 | `station-linux-amd64.tar.gz` |
| Linux ARM64 (Raspberry Pi, Jetson) | `station-linux-arm64.tar.gz` |
| macOS Apple Silicon | `station-macos-arm64.dmg` (desktop app) or `station-macos-arm64.zip` (CLI) |

> The Station README lists `station-linux-x86_64.tar.gz`, but the actual asset is named `station-linux-amd64.tar.gz`.

### Linux

```bash
mkdir -p ~/norma-station && cd ~/norma-station
curl -LO https://github.com/norma-core/norma-core/releases/download/v0.1.0-beta.10/station-linux-amd64.tar.gz
tar -xzf station-linux-amd64.tar.gz      # → station, dataset-generator, dataset-mp4

# USB serial access without sudo (log out and back in afterwards)
sudo usermod -a -G dialout $USER
```

### macOS

Open the `.dmg` and drag **NormaCore Station** to Applications; the daemon runs in the background. For the CLI tools, use the `.zip` instead.

## 2. Run Station and open the web UI

```bash
cd ~/norma-station
./station --web 127.0.0.1:8889 --tcp 127.0.0.1:8888
```

Open http://localhost:8889.

- `--web` / `--tcp` with no value listen on `0.0.0.0`, which exposes them to the whole network. Bind to `127.0.0.1` unless another machine needs to reach this Station.
- On first run Station creates `station.yaml` (drivers, cameras, inference) and a `station_data/` folder. Station stores a full history of everything it records, so watch your disk space (`--max-queue-disk-size 5G`).
- Each servo bus is identified by a **bus serial**, shown on the bus card in the web UI. You will need it for the scripts below.

### Troubleshooting: `Failed to render ST3215: Error creating WebGL context`

This means the browser cannot use WebGL for the 3D URDF view. The fault is in the browser, not in Station.

1. Open the UI in Google Chrome or Chromium.
2. Check `chrome://gpu` → *WebGL*. If it is disabled, enable *Settings → System → Use hardware acceleration when available* and restart the browser.
3. Or launch Chrome with `google-chrome --ignore-gpu-blocklist --enable-webgl http://localhost:8889`.
4. On hybrid NVIDIA laptops, force the dGPU: `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia google-chrome --ignore-gpu-blocklist http://localhost:8889`.

## 3. Servo IDs and calibration

1. **Set servo IDs before assembly.** This is step 4 of the follower manual. Label the servos 1 to 8, then connect **one servo at a time** to the driver board and set its ID in the Station web UI (*ST3215 motor config* page). Repeat 8 times per arm.
2. **Calibrate** each arm in the web UI (*ST3215 bus calibration* page). Station stores the calibrated min/max range of each motor and uses it to normalize positions (`position_norm`).
3. Optional safety settings in `station.yaml`:

```yaml
drivers:
  st3215:
    enabled: true
    current-threshold: 100     # mA, global current limit
    deadband: 20
    motor-current-thresholds:  # per-motor overrides
      8: 40
```

> NormaCore does not publish a written step-by-step calibration procedure; it lives in the web UI. Ask on [Discord](https://discord.gg/Z4Ytw3QfHP) if something is unclear.

## 4. Python client and examples (optional, needed for teleop and training)

The Python client has **no external dependencies**. It only needs the repository on `PYTHONPATH` and generated protobuf files. Any Python ≥ 3.10 environment works, so you can reuse an existing `lerobot` conda env or venv.

```bash
# Sparse clone (~35 MB instead of the full repo with STL files)
git clone --depth 1 --filter=blob:none --sparse https://github.com/norma-core/norma-core.git
cd norma-core
git sparse-checkout set software/station/shared software/station/examples \
    software/ai/smolvla_py shared/gremlin_py protobufs \
    hardware/elrobot/simulation docs/datasets

# Generate the Python protobufs (pure Python, no Go toolchain needed)
for d in protobufs/*/; do
  n=$(basename "$d")
  python shared/gremlin_py/gremlin.py --proto-root "$d" \
    --target-root "target/gen_python/protobuf/$n" \
    --project-root . --gremlin-import-path shared.gremlin_py.gremlin
done

# Make the repo importable from the active environment (no pip install)
echo "$PWD" > "$(python -c 'import site; print(site.getsitepackages()[0])')/norma_core.pth"

# Check
python -c "from software.station.shared.station_py import new_station_client; \
from target.gen_python.protobuf.drivers.st3215 import st3215; print('OK')"
```

Run these commands with the environment you want to use already activated (`conda activate lerobot` or `source .venv/bin/activate`).

## 5. Teleoperation (leader → follower)

Connect both arms. The example mirrors the leader bus onto the follower bus:

```bash
cd norma-core/software/station/examples/st3215-remote-teleop-py
python main.py \
  --leader-server   localhost --leader-bus   <LEADER_BUS_SERIAL> \
  --follower-server localhost --follower-bus <FOLLOWER_BUS_SERIAL>
```

The arms can also be on two different machines, each running Station; pass each machine's hostname. Press Ctrl+C to stop. On exit the follower's torque is disabled and the arm goes limp, so support it.

> Not verified: the README only shows the two-machine case. Using both buses on a single Station should work but has not been tested.

## 6. Record datasets

Station records continuously. You mark the recording window with tags ([official guide](https://github.com/norma-core/norma-core/blob/main/docs/datasets/export-parquet/datasets-export.md)):

1. In the web UI press **Tag** to create the start tag.
2. Record episodes while teleoperating. Default episode length is 45 s. Leave at least 5 s between episodes (8 s recommended) to reset the scene.
3. Press **Tag** to create the end tag.
4. Go to **History** and copy the frame numbers of both tags.

Cameras are captured through the `usb-video` driver in `station.yaml`:

```yaml
drivers:
  usb-video:
    enabled: true
    resolution: auto
    resize_target: 224
    frame-skip: 0
```

## 7. Export to Parquet

```bash
cd ~/norma-station
./dataset-generator \
  -episode.duration 35 \
  -from <START_FRAME> -to <END_FRAME> \
  -robot 127.0.0.1 \
  -output ~/datasets/my-task \
  -task "put the cube inside the box"

# Optional: render to video to inspect it
./dataset-mp4 --input ~/datasets/my-task.parquet --output ~/datasets/my-task.mp4
```

If episodes are discarded with `insufficient commands (93 < 100)`, add `-episode.min-commands 50` and re-export.

The output is NormaCore's own Parquet schema (one row = one frame, with per-joint `position_norm` / `goal_norm` and embedded JPEGs). It is **not** a `LeRobotDataset`.

## 8. Train SmolVLA

NormaCore ships a self-contained SmolVLA port (`software/ai/smolvla_py`) that trains directly on those Parquet files.

**Required change for ElRobot.** The scripts are written for the 6-motor SO-101:

| File | Change |
|---|---|
| `scripts/train.py` lines 31–32 | `STATE_DIM = 6` / `ACTION_DIM = 6` → `8` |
| `scripts/train.py:30`, `scripts/run_policy.py:60` | Expect exactly **2 cameras** (`cam0`, `cam1`). Record with 2 cameras or edit `IMAGE_KEYS`. |

Train locally:

```bash
cd norma-core/software/ai/smolvla_py
uv sync
uv run python scripts/train.py \
    --steps 30000 --batch-size 64 \
    --parquets ~/datasets/my-task.parquet \
    --output checkpoints/run
```

- To merge several recordings first: `uv run python scripts/merge_datasets.py --inputs ~/datasets/*.parquet --output merged.parquet`.
- SmolVLA fine-tuning needs a GPU with plenty of VRAM. Reduce `--batch-size` on smaller cards, or train in the cloud with the [Nebius recipe](https://github.com/norma-core/norma-core/blob/main/software/ai/smolvla_py/nebius.md) (`./scripts/build_bundle.sh my-task.parquet`).

## 9. Run the policy on the arm

Enable the inference queue in `station.yaml` and restart Station:

```yaml
inference:
  - queue-id: "inference/normvla"
    shm: "/tmp/normvla"
    shm-size-mb: 12
    format: "normvla"
    st3215-bus: "auto"        # or the follower bus serial
    update-interval: "100ms"
```

Then:

```bash
cd norma-core/software/ai/smolvla_py
uv run python scripts/run_policy.py \
    --checkpoint checkpoints/run/final \
    --task "put the cube inside the box" \
    --bus-serial <FOLLOWER_BUS_SERIAL> \
    --motor-ids 1,2,3,4,5,6,7,8 \
    --max-delta-ticks 512            # interactive: SPACE per step, q to quit
# add --auto --max-ticks 200 for continuous execution
```

There is no offline evaluation; validation happens on the real arm. Keep a hand near the power switch.

## Alternative: upstream LeRobot

Upstream `lerobot` has **no ElRobot robot type**. To use `lerobot-record`, `lerobot-train`, ACT or π0, or to push datasets to the Hugging Face Hub, you would have to:

- add `elrobot_follower` and `elrobot_leader` robot classes, adapted from `so_follower` and `so_leader`, with 8 `sts3215` motors on `FeetechMotorsBus`;
- or write a converter from NormaCore Parquet to `LeRobotDataset`.

Neither exists yet. LeLab will still not work, because it is hard-wired to SO-101.
