# Environment setup

Our arm is the **NormaCore ElRobot**. There is no LeRobot support for it (no robot type, no
driver, confirmed against upstream) - everything that talks to the arm goes through
NormaCore's own software, **Station**. Background: [docs/elrobot.md](./docs/elrobot.md).

The original SO-101-via-LeRobot scripts are kept, unused, in `scripts/lerobot/`, in case we
ever port to a LeRobot-supported arm. They are not part of this setup.

**One environment for everything.** The camera packages (`gauge_reader`, `hand_tracking`) and
the Station Python client both run from the repo's `.venv`. Station's client has zero external
dependencies, so this isn't adding weight - it's one less environment to keep straight.

## 1. Python environment

```bash
cd ~/Documents/leukerbad-hackathon
uv venv --python 3.12 .venv
```

Every command below assumes `.venv/bin/python`. No need to activate anything, but if you'd
rather: `source .venv/bin/activate`.

## 2. System dependencies

`ffmpeg`, for video decoding (gauge reader, hand tracking):

```bash
sudo apt-get install -y ffmpeg
```

## 3. Project packages (camera, filters, hand tracking)

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q      # should pass with no camera, robot or Station running
```

## 4. Station: the binary that runs the arm

Single self-contained executable, no Python packages, no system libraries.

```bash
mkdir -p ~/norma-station && cd ~/norma-station
curl -LO https://github.com/norma-core/norma-core/releases/download/v0.1.0-beta.10/station-linux-amd64.tar.gz
tar -xzf station-linux-amd64.tar.gz      # -> station, dataset-generator, dataset-mp4

sudo usermod -aG dialout $USER           # USB serial access without sudo; log out and back in
```

> Check the [releases page](https://github.com/norma-core/norma-core/releases) for a newer
> version than `v0.1.0-beta.10`. The asset is `station-linux-amd64.tar.gz` even though
> Station's own README says `station-linux-x86_64.tar.gz` - that filename in the README is
> wrong, the amd64 one is what's actually published.

Run it, binding to localhost only unless another machine needs to reach it:

```bash
cd ~/norma-station
./station --web 127.0.0.1:8889 --tcp 127.0.0.1:8888
```

Open http://localhost:8889. First run creates `station.yaml` and a `station_data/` folder
next to the binary.

**Servo IDs and joint-range calibration are done once per physical arm, by hand, in this web
UI** (*ST3215 motor config* and *ST3215 bus calibration* pages) - NormaCore does not publish a
scripted procedure for either. Each bus shows a serial number on its card; note it down, you
need it for everything below. Our `scripts/calibrate.py` only checks that this has already
been done - run it after, not instead of, the web UI step.

Once calibrated, commit the result so the team doesn't redo it per laptop: a
`config/calibration/<bus serial>.json` + `.md` pair (see the existing ones for the format). If
a calibration is ever lost (servo reset, reflash, replaced servo), restore it from that file
without redoing the manual web UI step:

```bash
python scripts/restore_calibration.py config/calibration/<bus serial>.json
```

## 5. The Python client (for our scripts)

Zero external dependencies - it only needs the repo on `PYTHONPATH` and generated protobufs.
Works in the `.venv` from step 1, or any other Python >= 3.10 environment already in use.

```bash
cd ~/Documents/leukerbad-hackathon   # or wherever, this clone is independent of the repo

git clone --depth 1 --filter=blob:none --sparse https://github.com/norma-core/norma-core.git
cd norma-core
git sparse-checkout set software/station/shared software/station/examples \
    software/ai/smolvla_py shared/gremlin_py protobufs \
    hardware/elrobot/simulation docs/datasets

# Generate the Python protobufs (pure Python, no Go toolchain needed)
for d in protobufs/*/; do
  n=$(basename "$d")
  .venv/bin/python shared/gremlin_py/gremlin.py --proto-root "$d" \
    --target-root "target/gen_python/protobuf/$n" \
    --project-root . --gremlin-import-path shared.gremlin_py.gremlin
done

# Make the repo importable from our .venv, no pip install
echo "$PWD" > "$(../.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/norma_core.pth"

# Check
cd ..
.venv/bin/python -c "from station_py import new_station_client; print('OK')"
```

(The check import path differs slightly from NormaCore's own README example - theirs checks
`software.station.shared.station_py`; `lib/station.py` in this repo imports from `station_py`
directly, since that's what ends up on `sys.path` after the sparse checkout. If the check
fails, try both.)

## 6. Environment variables

```bash
cp .env.example .env
```

- `STATION_HOST` - where Station is listening, e.g. `localhost` or `localhost:8888`. Default: `localhost`.
- `FOLLOWER_BUS_SERIAL` - the bus serial for arm B (the one pulling against the PESOLA), from Station's web UI.
- `LEADER_BUS_SERIAL` - if a leader arm is used, same idea.
- `CAMERA_INDEX` - camera number, or a name fragment such as `iphone`.

`.env` is git-ignored. Never commit it.

## 7. Check it end to end

```bash
.venv/bin/python -m pytest -q                         # software, no hardware, no Station needed
.venv/bin/python scripts/calibrate.py                   # Station: is the bus calibrated, do 8 motors answer
.venv/bin/python scripts/log_force_calibration.py --probe   # current readings, unloaded
```

`scripts/calibrate.py` is the first thing to run against the real arm: it connects to
Station, watches the bus, and tells you which motor ids respond and whether each one already
has a calibrated range - before you trust any position command or force reading. Force
calibration procedure: [docs/force-calibration.md](./docs/force-calibration.md).
