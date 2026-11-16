# ElRobot teleoperation data collection

Voice-guided pipeline to record teleoperated episodes with a leader/follower pair of
**NormaCore ElRobot** arms (8 ST3215 servos each) and upload them to a Hugging Face dataset.

It runs on top of **NormaCore Station**, not on LeRobot: ElRobot has 8 motors, and LeRobot /
LeLab only support the 6-motor SO-101.

Everything here was used to record the `JustoDario/flexnsense_skill1` dataset on
Ubuntu 24.04 (x86_64). Steps marked **(not verified)** were not run as written.

## Files

| File | Purpose |
|---|---|
| `record_dataset.py` | Recorder: voice prompts, keyboard control, background upload to the Hub |
| `camera_setup.sh` | Sets manual exposure on the cameras (run before starting Station) |
| `station.yaml` | Station config used for recording |
| `70-normacore-uvc.rules` | udev rule so Station can open USB cameras without root |
| `parquet_to_mp4.py` | Renders a recorded episode to an `.mp4` to check it (draws the hand keypoints) |
| `hand_keypoints.py` | Hand keypoints from the overhead camera with WiLoR; shared by the two scripts below |
| `add_hand_keypoints.py` | Adds the hand keypoints to episodes that are already on the Hub |

## Dependencies that are not in this repo

### 1. System packages and groups

```bash
sudo apt install espeak v4l-utils ffmpeg
sudo usermod -a -G dialout,plugdev $USER     # then log out and back in
```

- `espeak`: voice prompts. `v4l-utils`: camera exposure. `ffmpeg`: episode preview.
- `dialout`: serial access to the servo boards. `plugdev`: used by the udev rule below.

### 2. udev rule for the cameras

Station opens cameras through libuvc, which needs write access to `/dev/bus/usb/...`.
Station's README only mentions `dialout`; without this rule Station records no video.

```bash
sudo cp 70-normacore-uvc.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb
ls -l /dev/bus/usb/*/ | grep plugdev        # one line per external camera
```

The rule skips USB vendor `30c9` (the built-in webcam of the laptop we used) so Station
ignores it. Edit or remove that condition for your machine.

### 3. NormaCore Station (binary)

Download `station-linux-amd64.tar.gz` (v0.1.0-beta.10 was used) from the
[norma-core releases](https://github.com/norma-core/norma-core/releases) and extract it
**into this folder**, next to `station.yaml`:

```bash
cd data_collection
curl -LO https://github.com/norma-core/norma-core/releases/download/v0.1.0-beta.10/station-linux-amd64.tar.gz
tar -xzf station-linux-amd64.tar.gz          # -> station, dataset-generator, dataset-mp4
```

The binaries and Station's data folder are git-ignored.

### 4. norma-core clone (Python client + teleop example)

Clone it at the **repository root** (`record_dataset.py` looks for `../norma-core`) and
generate the Python protobufs:

```bash
cd ..                                         # repository root
git clone --depth 1 --filter=blob:none --sparse https://github.com/norma-core/norma-core.git
cd norma-core
git sparse-checkout set software/station/shared software/station/examples shared/gremlin_py protobufs
for d in protobufs/*/; do
  n=$(basename "$d")
  python shared/gremlin_py/gremlin.py --proto-root "$d" \
    --target-root "target/gen_python/protobuf/$n" \
    --project-root . --gremlin-import-path shared.gremlin_py.gremlin
done
```

### 5. Python packages

Any Python >= 3.10 environment. Versions used: `pyarrow 26.0.1`, `huggingface_hub 2.4.0`,
`numpy 2.2.6`, `pillow 12.0.0`.

```bash
pip install pyarrow huggingface_hub numpy pillow
hf auth login                                 # account with write access to the dataset
```

The Station Python client itself has no dependencies.

### 6. Robot setup (once)

- Servo IDs 1..8 and calibration of both arms are done in the Station web UI
  (http://localhost:8889). Use Chrome: the 3D view needs WebGL.
- Find the two **bus serials** in the web UI (bus cards). The follower runs at 12 V, the
  leader at about 7.4 V. Ours: follower `5B3D050113`, leader `5B61034827`.
- Put the follower serial in `station.yaml` (`inference[0].st3215-bus`).
- `camera_setup.sh` identifies cameras by USB port (`/dev/v4l/by-path/...`). Update the three
  paths for your machine (`ls -l /dev/v4l/by-path/`), and tune `TABLE_EXPOSURE` /
  `GRIPPER_EXPOSURE` for your lighting.

### 7. WiLoR, for the hand keypoints (optional)

The recorder runs [WiLoR](https://github.com/rolpotamias/WiLoR) on the overhead camera and
stores the hand keypoints with every frame. It needs an NVIDIA GPU with CUDA (measured on an
RTX 3050 Ti, 4 GB: 47 ms per frame, 1.6 GB of VRAM). Without one, record with `--no-hands`.

Clone WiLoR at the **repository root** and download its weights:

```bash
cd ..                                         # repository root
git clone --depth 1 https://github.com/rolpotamias/WiLoR.git
cd WiLoR
H=https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main
curl -L -o pretrained_models/detector.pt     $H/pretrained_models/detector.pt
curl -L -o pretrained_models/wilor_final.ckpt $H/pretrained_models/wilor_final.ckpt   # 2.5 GB
curl -L -o mano_data/MANO_RIGHT.pkl          $H/mano_data/mano/MANO_RIGHT.pkl
git clone --depth 1 https://github.com/mattloper/chumpy /tmp/chumpy && cp -r /tmp/chumpy/chumpy .
```

`MANO_RIGHT.pkl` is under the [MANO license](https://mano.is.tue.mpg.de) (non-commercial
research; it asks you to register there). `chumpy` is copied in as source because it no longer
builds with pip on Python 3.12; it is only needed to read that file.

Python packages, in the same environment, with a CUDA build of PyTorch already installed:

```bash
pip install ultralytics pytorch-lightning smplx==0.1.28 yacs timm einops pyrender scikit-image opencv-python tqdm six lazy_loader
```

Versions used: `torch 2.15.0+cu132`, `ultralytics 8.4.213`, `pytorch-lightning 2.6.6`,
`timm 1.0.31`. WiLoR's own `requirements.txt` was **not** installed as written; the list above
is what inference turned out to need.

The scripts restart themselves with `python -s`, so packages in `~/.local` are ignored (a
stale `transformers` there broke the WiLoR import on our laptop). Everything has to be
installed in the environment itself.

## Recording

Three terminals. Start them in this order.

**Terminal 1: cameras + Station**

```bash
cd data_collection
./camera_setup.sh
./station --web 127.0.0.1:8889 --tcp 127.0.0.1:8888 --max-queue-disk-size 3G
```

`camera_setup.sh` must run while Station is stopped: Station takes the cameras and
`/dev/videoN` disappear while it runs.

**Terminal 2: teleoperation (leader -> follower)**

```bash
cd norma-core/software/station/examples/st3215-remote-teleop-py
python main.py --leader-server localhost --leader-bus <LEADER_SERIAL> \
               --follower-server localhost --follower-bus <FOLLOWER_SERIAL>
```

Put the leader in a pose close to the follower's before starting. On Ctrl+C the follower
loses torque and drops, so hold it.

**Terminal 3: recorder**

```bash
cd data_collection
python record_dataset.py "move device from right index to right middle finger" 20
```

Arguments: the task description (stored as the language instruction) and how many episodes
to record in this run. Options: `--repo` (default `JustoDario/flexnsense_skill1`),
`--cameras` (default 3), `--joints` (default 8), `--no-hands` (record without hand keypoints),
`--hand-camera` (default `auto`), `--wilor-root`, `--upload-timeout` (default 300 s).

`python record_dataset.py --upload-only` records nothing: it uploads whatever in `episodes/`
is not on the Hub yet and exits.

### Workflow

1. The laptop says "Recording episode N. Start." and recording begins.
2. Teleoperate the task.
3. Press a key:
   - `n` (next): save the episode. The laptop asks you to reset the environment.
   - `c` (cancel): discard it. After the reset, the same episode number is recorded again.
4. Reset the scene and press `r` (ready) to record the next one.
5. After the last episode the program waits for the uploads still in the queue and exits.
   Ctrl+C there is safe: the rest is uploaded on the next run.

Details:

- N continues from the episodes already on the Hub for that task.
- Each saved episode is written to `episodes/data/<task_slug>/episode_NNNN.parquet` and kept
  there (never deleted), then put on a background upload queue. A separate thread uploads it,
  one commit per episode, to the same path in the dataset repo while you keep recording; the
  status line shows how many are waiting. A slow or dropped connection never blocks the
  recording: an attempt that does not finish in `--upload-timeout` seconds is killed and
  retried with a growing pause (up to 2 min).
- `episodes/uploaded.json` lists the files that are on the Hub, with their sha256. On every
  run, local episodes not in it are queued again (including leftovers of the old
  `pending_uploads/` folder, which are moved into `episodes/`).
- No double uploads: before sending a file the uploader asks the Hub for it. Same sha256:
  marked as uploaded, nothing sent. A different file at that path (e.g. another machine used
  the same episode number): not overwritten, reported as a conflict and kept on disk.
- If the Hub does not answer at start, episode numbers continue from the local copies.
- Episodes shorter than 2 s are treated as cancelled.
- If no data arrives for 3 s the laptop warns you: either teleoperation is not running
  (Station only emits frames while the follower has torque) or the number of cameras or
  joints does not match `--cameras` / `--joints`.
- The hand tracker runs in a second thread while you record. The status line shows
  `hand found/processed`; if no hand is found in the first 20 frames the laptop says so.
  The overhead camera is found automatically on the first frame (`[hands] overhead camera =
  images[N]` in the terminal). This relies on our setup: the overhead image has black left and
  right edges. If it cannot tell, the recorder stops and asks for `--hand-camera <index>`.
- The first frame of each camera is saved to `cam_preview/cam0.jpg`, `cam1.jpg`, ... Check at
  the start of every session that the camera order has not changed: frames do not carry
  camera names, only their position in the list.

## Dataset format

One Parquet file per episode, one row per frame, about 10 Hz. Same schema as NormaCore's
`dataset-generator`. It is **not** a `LeRobotDataset`.

| Column | Type | Meaning |
|---|---|---|
| `episode_start_ns` | uint64 | Wall-clock time of the first frame |
| `global_frame_id` | binary | Station frame id |
| `timestamp_ns_since_episode_start` | uint64 | Time within the episode |
| `joints` | list of 8 structs | `range_min`, `range_max`, `position`, `position_norm`, `goal`, `goal_norm`, `current_ma`, `velocity` |
| `images` | list of structs `{jpeg}` | One 224x224 JPEG per camera |
| `task` | string | Task description |

NormaCore's convention: state = `position_norm`, action = `goal_norm`.

Hand keypoints (absent in episodes recorded with `--no-hands`):

| Column | Type | Meaning |
|---|---|---|
| `hand_camera` | uint8 | Index of the overhead camera in `images` for this episode |
| `hand_detected` | bool | WiLoR found a hand in this frame |
| `hand_is_right` | bool | Handedness reported by WiLoR |
| `hand_keypoints_2d` | 42 x float32 | 21 points, `x, y` as fractions (0..1) of the overhead image, origin top-left |
| `hand_fingertips_2d` | 10 x float32 | Thumb, index, middle, ring, pinky tips: points 4, 8, 12, 16, 20 of the above |
| `hand_keypoints_3d` | 63 x float32 | 21 points, `x, y, z` in metres, hand centred: hand shape, not hand position |

The 21 points follow the MediaPipe / OpenPose layout (`hand_tracking/pose.py`). Frames with no
detection repeat the nearest detection of the episode and have `hand_detected = false`.

**The order of the cameras in `images` is not the same in every session.** Station lists
cameras in the order it finds them. In `JustoDario/flexnsense_skill1`, episodes 1-20 are
`[overhead, gripper, side]` and episodes 21-43 are `[side, overhead, gripper]`. Use
`hand_camera` to find the overhead image; the other two have to be told apart by the training
code (or fixed in the dataset) before training on images.

### Adding the keypoints to episodes already on the Hub

```bash
python add_hand_keypoints.py --dry-run       # process and report only
python add_hand_keypoints.py                 # process and replace the files on the Hub
```

It skips episodes that already have the columns (`--force` recomputes them), writes one
overlay image per episode to `hand_preview/`, and leaves every other column unchanged. The
previous version of each file stays in the dataset's git history on the Hub.

## Checking an episode

```bash
T=$(python -c "from huggingface_hub import get_token; print(get_token())")
curl -L -H "Authorization: Bearer $T" -o episode_0001.parquet \
  "https://huggingface.co/datasets/JustoDario/flexnsense_skill1/resolve/main/data/<task_slug>/episode_0001.parquet"
python parquet_to_mp4.py episode_0001.parquet      # -> episode_0001.mp4, cameras side by side
```

NormaCore's own `dataset-mp4` failed on these files, hence `parquet_to_mp4.py`.

## Disk usage

Station records everything, all the time, into `station_data/` (about 2 GB per hour with
3 cameras), including while you calibrate. It stalled when it reached
`--max-queue-disk-size`. The dataset does not need that history. After each session, with
Station **stopped**:

```bash
rm -rf station_data/*/{usbvideo,inference}
```

Setting `frame-skip: 2` in `station.yaml` should cut the video history to a third
**(not verified)**.

## Known limits

- In `station.yaml`, keep `formats: []`. A list of strings such as `"640x480@30fps,mjpeg"`
  makes Station fail at startup.
- Camera exposure is not configurable in Station; `camera_setup.sh` sets it on the camera
  before Station starts, and Station keeps it.
- Hand keypoints: one hand only. Their accuracy was checked by eye on overlays, not measured
  against ground truth. The overhead image is 224x224, so one pixel is about 0.45 % of the image.
- Training and running a policy on the arm are not covered here.
