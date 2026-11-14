# Environment setup

## 1. Python environment

LeRobot requires **Python >= 3.12**. Using `conda` is recommended:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

(`uv venv --python 3.12` works too if you prefer `uv`.)

## 2. System dependencies

Install `ffmpeg` (needed for video decoding):

```bash
conda install ffmpeg -c conda-forge
```

On Linux, if you hit build errors, also install:

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

## 3. Install LeRobot + Feetech motor support (SO-101 uses Feetech motors)

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[core_scripts,feetech]"
```

`core_scripts` gives you `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-calibrate`, `lerobot-record`, `lerobot-replay`.

## 4. Connect and identify the arm(s)

```bash
lerobot-find-port
```

On Linux you may need to grant USB access:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

## 5. Set motor IDs (once per arm) and calibrate

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=rehab_follower
```

Repeat with `--teleop.type=so101_leader` for the leader arm if used for teleoperation.

Full reference: [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101).

## 6. Environment variables

Copy `.env.example` to `.env` and fill in what you actually use:

```bash
cp .env.example .env
```

- `HF_TOKEN` - only needed if pushing/pulling datasets or models to/from the Hugging Face Hub (`hf auth login` also works instead of a `.env` entry).
- `ROBOT_PORT` / camera and doctor-dashboard settings - local, machine-specific; do not commit real values.

`.env` is git-ignored - never commit it.
