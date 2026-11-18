#!/usr/bin/env python
"""Voice-guided episode recorder for ElRobot + NormaCore Station.

Records episodes from Station's `inference/normvla` queue (follower joints + all cameras)
and guides the operator by voice. Every saved episode is kept on disk in `episodes/` (never
deleted) and put on a background upload queue: a separate thread uploads it to the Hugging
Face Hub while you keep recording, so a slow or dropped connection never blocks the
recording. `episodes/uploaded.json` records which files are on the Hub (with their sha256);
episodes not in it are queued again on the next run.

Each upload runs in a child process with a time limit (a hung connection is killed and
retried with a growing pause) and first asks the Hub whether the file is already there:
  - same sha256 on the Hub -> marked as uploaded, nothing sent (no double uploads, also when
    a previous attempt timed out after its commit went through);
  - a different file at that path -> NOT overwritten: reported as a conflict, kept on disk.

Usage:
    python record_dataset.py "<task description>" <num_episodes>
    python record_dataset.py --upload-only        # just upload what is left in episodes/

Keys:
    n  next   -> close the episode and save it
    c  cancel -> discard the episode and record the same number again
    r  ready  -> environment is reset, start the next recording
    Ctrl+C    -> quit (the episode being recorded is discarded)

Needs: Station running (with --tcp) and the teleop script running, because Station only
produces frames while the follower has torque enabled.

Hand keypoints: unless --no-hands is given, WiLoR runs on the overhead camera in a second
thread while the episode is recorded, and every frame gets the `hand_*` columns described in
hand_keypoints.py. This needs a CUDA GPU and a WiLoR checkout (see README).

Output: data/<task_slug>/episode_NNNN.parquet in the dataset repo, in the same Parquet
schema as NormaCore's `dataset-generator` (so `smolvla_py` can train on it), plus the hand
columns.
"""

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from hand_keypoints import find_overhead_camera, rerun_without_user_site, with_hand_columns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "norma-core"))
from software.station.shared.station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.inferences import normvla  # noqa: E402

QUEUE_ID = "inference/normvla"
MIN_FRAMES = 20  # ~2 s at Station's 10 Hz; shorter episodes are treated as cancelled
PREVIEW_DIR = Path(__file__).resolve().parent / "cam_preview"
EPISODES_DIR = Path(__file__).resolve().parent / "episodes"       # local copy of every episode, kept
LEDGER = EPISODES_DIR / "uploaded.json"                             # {path in repo: sha256} on the Hub
OLD_PENDING_DIR = Path(__file__).resolve().parent / "pending_uploads"  # previous layout, moved on start

JOINT_T = pa.struct(
    [
        ("range_min", pa.uint32()),
        ("range_max", pa.uint32()),
        ("position", pa.uint32()),
        ("position_norm", pa.float32()),
        ("goal", pa.uint32()),
        ("goal_norm", pa.float32()),
        ("current_ma", pa.uint32()),
        ("velocity", pa.uint32()),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("episode_start_ns", pa.uint64(), nullable=False),
        pa.field("global_frame_id", pa.binary(), nullable=False),
        pa.field("timestamp_ns_since_episode_start", pa.uint64(), nullable=False),
        pa.field("joints", pa.list_(JOINT_T)),
        pa.field("images", pa.list_(pa.struct([("jpeg", pa.binary())]))),
        pa.field("task", pa.string()),
    ]
)


async def say(text: str, wait: bool = True) -> None:
    print(f"\n[voice] {text}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        "espeak", "-v", "en-us", "-s", "150", text,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    if wait:
        await proc.wait()


class HandWorker:
    """Runs the hand tracker on recorded frames in its own thread, in the order they arrive.

    Each row gets a "hand" entry (the detection, or None). `camera` is the index of the overhead
    camera in the frame's images; None means: find it on the first frame and keep it.
    """

    def __init__(self, tracker, camera: int | None):
        self.tracker, self.camera = tracker, camera
        self.done = self.found = 0
        self.error: Exception | None = None
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def new_episode(self) -> None:
        self._queue.put(None)

    def submit(self, row: dict) -> None:
        self._queue.put(row)

    def wait(self) -> None:
        """Block until every submitted frame has been processed."""
        self._queue.join()

    def _run(self) -> None:
        while True:
            row = self._queue.get()
            try:
                if row is None:
                    self.tracker.reset()
                    self.done = self.found = 0
                elif self.error is None:
                    jpegs = [im["jpeg"] for im in row["images"]]
                    if self.camera is None:
                        self.camera = find_overhead_camera(jpegs)
                        print(f"\n[hands] overhead camera = images[{self.camera}]", flush=True)
                    row["hand"] = self.tracker.detect(jpegs[self.camera])
                    self.done += 1
                    self.found += row["hand"] is not None
            except Exception as e:  # noqa: BLE001
                self.error = e
            finally:
                self._queue.task_done()


class Recorder:
    """Follows the normvla queue; keeps frames only while `recording` is True."""

    def __init__(self, client, n_joints: int, n_cameras: int, hands: HandWorker | None = None):
        self.n_joints, self.n_cameras, self.hands = n_joints, n_cameras, hands
        self.recording = False
        self.rows: list[dict] = []
        self.bad_frames = 0
        self._last_id = b""
        self._preview_done = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._errors = client.follow(QUEUE_ID, self._queue)

    async def pump(self) -> None:
        while True:
            if not self._errors.empty():
                raise RuntimeError(f"{QUEUE_ID} stream error: {self._errors.get_nowait()}")
            entry = await self._queue.get()
            if entry is None:
                raise RuntimeError(f"{QUEUE_ID} stream closed")
            if self.recording:
                self._add(entry)

    def _add(self, entry) -> None:
        frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
        fid = bytes(frame.get_global_frame_id())
        if fid == self._last_id:
            return
        self._last_id = fid
        joints = frame.get_joints() or []
        images = frame.get_images() or []
        if len(joints) != self.n_joints or len(images) != self.n_cameras:
            self.bad_frames += 1
            return
        jpegs = [bytes(im.get_jpeg()) for im in images]
        if not self._preview_done:
            PREVIEW_DIR.mkdir(exist_ok=True)
            for i, jpeg in enumerate(jpegs):
                (PREVIEW_DIR / f"cam{i}.jpg").write_bytes(jpeg)
            self._preview_done = True
        self.rows.append(
            {
                "wall_ns": time.time_ns(),
                "stamp_ns": frame.get_monotonic_stamp_ns(),
                "global_frame_id": fid,
                "joints": [
                    {
                        "range_min": j.get_range_min(),
                        "range_max": j.get_range_max(),
                        "position": j.get_position(),
                        "position_norm": j.get_position_norm(),
                        "goal": j.get_goal(),
                        "goal_norm": j.get_goal_norm(),
                        "current_ma": j.get_current_ma(),
                        "velocity": j.get_velocity(),
                    }
                    for j in joints
                ],
                "images": [{"jpeg": jpeg} for jpeg in jpegs],
            }
        )
        if self.hands is not None:
            self.hands.submit(self.rows[-1])

    def start(self) -> None:
        if self.hands is not None:
            self.hands.new_episode()
        self.rows, self.bad_frames, self.recording = [], 0, True

    def stop(self) -> list[dict]:
        self.recording = False
        rows, self.rows = self.rows, []
        return rows


def to_parquet(rows: list[dict], task: str, hand_camera: int | None = None) -> bytes:
    """Rows of one episode to Parquet bytes. With `hand_camera`, the hand columns are added."""
    start_wall, start_stamp = rows[0]["wall_ns"], rows[0]["stamp_ns"]
    table = pa.Table.from_pylist(
        [
            {
                "episode_start_ns": start_wall,
                "global_frame_id": r["global_frame_id"],
                "timestamp_ns_since_episode_start": max(0, r["stamp_ns"] - start_stamp),
                "joints": r["joints"],
                "images": r["images"],
                "task": task,
            }
            for r in rows
        ],
        schema=SCHEMA,
    )
    if hand_camera is not None:
        table = with_hand_columns(table, [r.get("hand") for r in rows], hand_camera)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def local_episodes() -> list[Path]:
    return sorted(EPISODES_DIR.glob("data/*/episode_*.parquet"))


def repo_path(f: Path) -> str:
    return f.relative_to(EPISODES_DIR).as_posix()


def sha256(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def move_old_pending() -> None:
    """Episodes left in pending_uploads/ by the previous version of this script go to episodes/."""
    for f in sorted(OLD_PENDING_DIR.glob("data/*/episode_*.parquet")):
        dst = EPISODES_DIR / f.relative_to(OLD_PENDING_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256(dst) != sha256(f):
                print(f"[hub] {f} differs from {dst}: left where it is, check it by hand")
            continue
        shutil.move(str(f), str(dst))
        print(f"[hub] moved {f.relative_to(OLD_PENDING_DIR.parent)} -> {dst.relative_to(EPISODES_DIR.parent)}")


def upload_one(repo: str, local: str, path_in_repo: str, digest: str) -> int:
    """Child process: upload one file unless the Hub already has it. Exit code says what happened."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi()

    def remote_sha() -> str | None:
        try:
            info = api.get_paths_info(repo, [path_in_repo], repo_type="dataset")
        except EntryNotFoundError:
            return None
        if not info:
            return None
        lfs = getattr(info[0], "lfs", None)
        return lfs.sha256 if lfs else "unknown (not an LFS file)"

    before = remote_sha()
    if before == digest:
        print("exists")
        return 0
    if before is not None:
        print(f"conflict: the Hub has a different file there (sha256 {before})")
        return 3
    api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo, repo_type="dataset",
                    commit_message=f"Add {path_in_repo.rsplit('/', 1)[-1]}")
    after = remote_sha()
    if after != digest:
        print(f"uploaded but the Hub reports sha256 {after}, expected {digest}")
        return 4
    print("uploaded")
    return 0


class Uploader:
    """Background upload queue: one thread, one episode at a time, retried until it gets through.

    Uploads go through `upload_one` in a child process so that a connection that hangs can be
    killed (timeout) instead of blocking forever. Nothing is ever deleted locally.
    """

    def __init__(self, repo: str, timeout: float):
        self.repo, self.timeout = repo, timeout
        self.q: queue.Queue[Path] = queue.Queue()
        self.lock = threading.Lock()
        self.queued: set[str] = set()
        self.done: dict[str, str] = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        self.conflicts: list[str] = []
        self.current: str | None = None
        self.last_error = ""
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, f: Path) -> None:
        key = repo_path(f)
        with self.lock:
            if key in self.done or key in self.queued:
                return
            self.queued.add(key)
        self.q.put(f)

    def add_all_local(self) -> int:
        before = self.pending()
        for f in local_episodes():
            self.add(f)
        return self.pending() - before

    def pending(self) -> int:
        with self.lock:
            return len(self.queued)

    def status(self) -> str:
        n = self.pending()
        if not n:
            return "uploads: all done"
        err = f", retrying: {self.last_error}" if self.last_error else ""
        return f"uploads: {n} waiting{err}"

    def _mark_done(self, key: str, digest: str) -> None:
        with self.lock:
            self.done[key] = digest
            self.queued.discard(key)
            tmp = LEDGER.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.done, indent=1, sort_keys=True))
            os.replace(tmp, LEDGER)

    def _run(self) -> None:
        while True:
            f = self.q.get()
            key, digest, pause = repo_path(f), sha256(f), 5.0
            self.current = key
            while True:
                cmd = [sys.executable, "-s", str(Path(__file__).resolve()), "--upload-one",
                       self.repo, str(f), key, digest]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                    out = (r.stdout.strip().splitlines() or [""])[-1]
                    if r.returncode == 0:
                        self._mark_done(key, digest)
                        self.last_error = ""
                        print(f"\n[hub] {key}: {out}", flush=True)
                        break
                    if r.returncode == 3:
                        with self.lock:
                            self.queued.discard(key)
                        self.conflicts.append(key)
                        print(f"\n[hub] {key}: {out}. NOT uploaded, kept in {f}", flush=True)
                        break
                    err = (r.stderr.strip().splitlines() or [out or f"exit {r.returncode}"])[-1]
                except subprocess.TimeoutExpired:
                    err = f"no answer in {self.timeout:.0f} s"
                self.last_error = f"{key}: {err[:120]}"
                time.sleep(pause)
                pause = min(pause * 2, 120.0)
            self.current = None

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while self.pending():
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.5)
        return True


def hub_episode_names(repo: str, timeout: float) -> list[str] | None:
    """File list of the dataset repo, or None if the Hub does not answer in time."""
    result: list = []

    def run() -> None:
        try:
            result.append(list(HfApi().list_repo_files(repo, repo_type="dataset")))
        except Exception as e:  # noqa: BLE001
            result.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if not result or isinstance(result[0], Exception):
        return None
    return result[0]


def next_episode_index(repo: str, slug: str, uploader: Uploader) -> int:
    pattern = re.compile(rf"^data/{re.escape(slug)}/episode_(\d+)\.parquet$")
    names = [repo_path(f) for f in local_episodes()] + list(uploader.done)
    hub = hub_episode_names(repo, timeout=30)
    if hub is None:
        print("WARNING: the Hub did not answer; episode numbers continue from the local copies only. "
              "If another machine uploaded episodes of this task, they are not overwritten: those "
              "uploads stop with 'conflict' and stay on disk.")
    else:
        names += hub
    found = [int(m.group(1)) for f in names if (m := pattern.match(f))]
    return max(found, default=0) + 1


async def main_async(args, uploader: Uploader) -> None:
    slug = re.sub(r"[^a-z0-9]+", "_", args.task.lower()).strip("_")
    first = await asyncio.to_thread(next_episode_index, args.repo, slug, uploader)
    last = first + args.num_episodes - 1
    print(f"Task: {args.task!r} -> {args.repo}/data/{slug}/  | episodes {first}..{last}"
          f"  | each episode is uploaded in the background")
    if uploader.pending():
        print(f"{uploader.pending()} episodes from previous runs are not on the Hub yet; uploading them now.")
    print("Keys: [n] next (save)   [c] cancel (discard)   [r] ready   Ctrl+C quit")

    hands = None
    if not args.no_hands:
        print("Loading the hand tracker (WiLoR) ...", flush=True)
        from hand_keypoints import HandKeypointTracker

        camera = None if args.hand_camera == "auto" else int(args.hand_camera)
        hands = HandWorker(HandKeypointTracker(args.wilor_root), camera)

    client = await new_station_client(args.server, logging.getLogger("station"))
    rec = Recorder(client, args.joints, args.cameras, hands)
    pump = asyncio.create_task(rec.pump())

    keys: asyncio.Queue = asyncio.Queue()
    asyncio.get_running_loop().add_reader(sys.stdin, lambda: keys.put_nowait(sys.stdin.read(1).lower()))

    async def wait_key(allowed: str, status=None) -> str:
        while not keys.empty():
            keys.get_nowait()
        while True:
            if pump.done():
                raise RuntimeError(f"Lost connection to Station: {pump.exception()}")
            try:
                key = await asyncio.wait_for(keys.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if status:
                    await status()
                continue
            if key in allowed:
                return key

    ep = first
    while ep <= last:
        await say(f"Recording episode {ep}. Start.")
        rec.start()
        t0 = time.monotonic()
        warned = warned_hand = False

        async def status() -> None:
            nonlocal warned, warned_hand
            elapsed = time.monotonic() - t0
            hand = f"  hand {hands.found}/{hands.done}" if hands else ""
            print(f"\r  REC episode {ep}: {elapsed:5.1f}s  {len(rec.rows)} frames  "
                  f"({rec.bad_frames} skipped){hand}  | {uploader.status()}   ", end="", flush=True)
            if hands and hands.error:
                raise RuntimeError(f"Hand tracker failed: {hands.error}")
            if hands and hands.done >= 20 and not hands.found and not warned_hand:
                warned_hand = True
                await say("Warning. No hand detected.", wait=False)
            if elapsed > 3 and not rec.rows and not warned:
                warned = True
                problem = ("Wrong number of cameras or joints." if rec.bad_frames
                           else "No data from the robot. Is teleoperation running?")
                await say(f"Warning. {problem}", wait=False)

        key = await wait_key("nc", status)
        rows = rec.stop()

        if key == "c" or len(rows) < MIN_FRAMES:
            reason = "cancelled" if key == "c" else "too short, discarded"
            await say(f"Episode {ep} {reason}. Reset the environment and press R to record it again.", wait=False)
            await wait_key("r")
            continue

        if hands:
            await asyncio.to_thread(hands.wait)  # the tracker may be a few frames behind
            if hands.error:
                raise RuntimeError(f"Hand tracker failed: {hands.error}")
            hand_note = f", hand in {sum(r.get('hand') is not None for r in rows)}/{len(rows)}"
        data = to_parquet(rows, args.task, hands.camera if hands else None)
        path = EPISODES_DIR / "data" / slug / f"episode_{ep:04d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)          # never a half-written file under the final name
        uploader.add(path)
        print(f"\n  episode {ep}: {len(rows)} frames{hand_note if hands else ''}, {len(data) / 1e6:.1f} MB saved "
              f"to {path.relative_to(EPISODES_DIR.parent)}  | {uploader.status()}")
        del rows, data

        if ep < last:
            await say(f"Episode {ep} saved. Reset the environment and press R when ready.", wait=False)
            await wait_key("r")
        ep += 1

    if uploader.pending():
        await say("Last episode saved. Waiting for the uploads to finish.", wait=False)
        await wait_uploads(uploader)
    if uploader.conflicts:
        await say("Finished, but some episodes were not uploaded. Check the terminal.")
    else:
        await say("All episodes recorded and uploaded. Goodbye.")


async def wait_uploads(uploader: Uploader) -> None:
    """Show progress until the queue is empty. Ctrl+C leaves the rest for the next run."""
    while uploader.pending():
        print(f"\r  {uploader.status()}  (Ctrl+C to stop: the rest is uploaded on the next run)   ",
              end="", flush=True)
        await asyncio.sleep(1)
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if len(sys.argv) == 6 and sys.argv[1] == "--upload-one":      # child process of Uploader
        sys.exit(upload_one(*sys.argv[2:]))
    p.add_argument("task", nargs="?", help='Task description in English, e.g. "pick the cube and put it in the box"')
    p.add_argument("num_episodes", type=int, nargs="?", help="How many episodes to record in this run")
    p.add_argument("--repo", default="JustoDario/flexnsense_skill1")
    p.add_argument("--upload-only", action="store_true", help="Do not record: upload what is left in episodes/ and exit")
    p.add_argument("--upload-timeout", type=float, default=300,
                   help="Seconds before an upload attempt that does not finish is killed and retried")
    p.add_argument("--server", default="localhost", help="Station host")
    p.add_argument("--cameras", type=int, default=3, help="Expected images per frame")
    p.add_argument("--joints", type=int, default=8, help="Expected joints per frame")
    p.add_argument("--no-hands", action="store_true", help="Record without hand keypoints (no GPU needed)")
    p.add_argument("--hand-camera", default="auto",
                   help="Index of the overhead camera in the frame's images, or 'auto' (default)")
    p.add_argument("--wilor-root", help="WiLoR checkout (default: WILOR_ROOT, else <repo>/WiLoR)")
    args = p.parse_args()
    if not args.upload_only and (args.task is None or args.num_episodes is None):
        p.error("task and num_episodes are required (or use --upload-only)")
    if not args.no_hands and not args.upload_only:
        rerun_without_user_site()

    logging.basicConfig(level=logging.WARNING)
    EPISODES_DIR.mkdir(exist_ok=True)
    move_old_pending()
    uploader = Uploader(args.repo, args.upload_timeout)
    uploader.add_all_local()
    if args.upload_only:
        print(f"{uploader.pending()} episodes to upload from {EPISODES_DIR}")
        try:
            asyncio.run(wait_uploads(uploader))
            print("Conflicts (not uploaded): " + ", ".join(uploader.conflicts) if uploader.conflicts else "Done.")
        except KeyboardInterrupt:
            print(f"\nStopped. {uploader.pending()} episodes are still to upload; run it again.")
        return

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    try:
        asyncio.run(main_async(args, uploader))
    except KeyboardInterrupt:
        print("\nStopped. The episode being recorded (if any) was discarded.")
        if uploader.pending():
            print(f"{uploader.pending()} saved episodes are not on the Hub yet. They are kept in {EPISODES_DIR} "
                  "and uploaded on the next run (or with --upload-only).")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
