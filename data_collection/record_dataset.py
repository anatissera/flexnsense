#!/usr/bin/env python
"""Voice-guided episode recorder for ElRobot + NormaCore Station.

Records episodes from Station's `inference/normvla` queue (follower joints + all cameras)
and guides the operator by voice. Episodes are kept in `pending_uploads/` and uploaded to
the Hugging Face Hub in batches (every 10 episodes by default, one commit per batch); the
local copies are deleted once the upload succeeds. Anything left over (end of the run, or
a previous run that was interrupted) is uploaded too.

Usage:
    python record_dataset.py "<task description>" <num_episodes>

Keys:
    n  next   -> close the episode and save it
    c  cancel -> discard the episode and record the same number again
    r  ready  -> environment is reset, start the next recording
    Ctrl+C    -> quit (the episode being recorded is discarded)

Needs: Station running (with --tcp) and the teleop script running, because Station only
produces frames while the follower has torque enabled.

Output: data/<task_slug>/episode_NNNN.parquet in the dataset repo, in the same Parquet
schema as NormaCore's `dataset-generator` (so `smolvla_py` can train on it).
"""

import argparse
import asyncio
import io
import logging
import re
import sys
import termios
import time
import tty
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "norma-core"))
from software.station.shared.station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.inferences import normvla  # noqa: E402

QUEUE_ID = "inference/normvla"
MIN_FRAMES = 20  # ~2 s at Station's 10 Hz; shorter episodes are treated as cancelled
PREVIEW_DIR = Path(__file__).resolve().parent / "cam_preview"
PENDING_DIR = Path(__file__).resolve().parent / "pending_uploads"

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


class Recorder:
    """Follows the normvla queue; keeps frames only while `recording` is True."""

    def __init__(self, client, n_joints: int, n_cameras: int):
        self.n_joints, self.n_cameras = n_joints, n_cameras
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

    def start(self) -> None:
        self.rows, self.bad_frames, self.recording = [], 0, True

    def stop(self) -> list[dict]:
        self.recording = False
        rows, self.rows = self.rows, []
        return rows


def to_parquet(rows: list[dict], task: str) -> bytes:
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
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def pending_files() -> list[Path]:
    return sorted(PENDING_DIR.glob("data/*/episode_*.parquet"))


def flush(api: HfApi, repo: str) -> bool:
    """Upload every pending episode in one commit, then delete the local copies.

    Only the files present when the upload starts are uploaded and deleted, so an episode
    saved while the upload is running is left for the next batch. On failure the files stay
    in pending_uploads/ and are retried with the next batch or the next run.
    """
    files = pending_files()
    if not files:
        return True
    ops = [CommitOperationAdd(path_in_repo=f.relative_to(PENDING_DIR).as_posix(), path_or_fileobj=str(f))
           for f in files]
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    for attempt in range(1, 4):
        try:
            api.create_commit(
                repo_id=repo, repo_type="dataset", operations=ops,
                commit_message=f"Add {len(files)} episodes ({files[0].stem}..{files[-1].stem})",
            )
            for f in files:
                f.unlink()
            print(f"\n[hub] uploaded {len(files)} episodes ({size_mb:.1f} MB), local copies deleted", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"\n[hub] upload attempt {attempt}/3 failed: {e}", flush=True)
            time.sleep(3)
    print(f"\n[hub] UPLOAD FAILED. {len(files)} episodes kept in {PENDING_DIR}", flush=True)
    return False


def next_episode_index(api: HfApi, repo: str, slug: str) -> int:
    pattern = re.compile(rf"^data/{re.escape(slug)}/episode_(\d+)\.parquet$")
    names = list(api.list_repo_files(repo, repo_type="dataset"))
    names += [f.relative_to(PENDING_DIR).as_posix() for f in pending_files()]
    found = [int(m.group(1)) for f in names if (m := pattern.match(f))]
    return max(found, default=0) + 1


async def main_async(args) -> None:
    api = HfApi()
    slug = re.sub(r"[^a-z0-9]+", "_", args.task.lower()).strip("_")
    first = next_episode_index(api, args.repo, slug)
    last = first + args.num_episodes - 1
    print(f"Task: {args.task!r} -> {args.repo}/data/{slug}/  | episodes {first}..{last}"
          f"  | upload every {args.batch} episodes")
    if pending_files():
        print(f"Found {len(pending_files())} episodes from a previous run in {PENDING_DIR}; they go in the first upload.")
    print("Keys: [n] next (save)   [c] cancel (discard)   [r] ready   Ctrl+C quit")

    client = await new_station_client(args.server, logging.getLogger("station"))
    rec = Recorder(client, args.joints, args.cameras)
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

    pending: asyncio.Task | None = None
    failed = 0
    ep = first
    while ep <= last:
        await say(f"Recording episode {ep}. Start.")
        rec.start()
        t0 = time.monotonic()
        warned = False

        async def status() -> None:
            nonlocal warned
            elapsed = time.monotonic() - t0
            print(f"\r  REC episode {ep}: {elapsed:5.1f}s  {len(rec.rows)} frames  "
                  f"({rec.bad_frames} skipped)   ", end="", flush=True)
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

        data = to_parquet(rows, args.task)
        path = PENDING_DIR / "data" / slug / f"episode_{ep:04d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        n_pending = len(pending_files())
        print(f"\n  episode {ep}: {len(rows)} frames, {len(data) / 1e6:.1f} MB saved "
              f"({n_pending}/{args.batch} waiting for upload)")
        del rows, data
        if n_pending >= args.batch and ep < last:
            if pending is not None and not await pending:  # previous upload must finish first
                failed += 1
            pending = asyncio.create_task(asyncio.to_thread(flush, api, args.repo))

        if ep < last:
            await say(f"Episode {ep} saved. Reset the environment and press R when ready.", wait=False)
            await wait_key("r")
        ep += 1

    await say("Last episode saved. Uploading.", wait=False)
    if pending is not None and not await pending:
        failed += 1
    if not await asyncio.to_thread(flush, api, args.repo):
        failed += 1
    if failed and pending_files():
        await say("Finished, but some uploads failed. Check the terminal.")
    else:
        await say("All episodes recorded and uploaded. Goodbye.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", help='Task description in English, e.g. "pick the cube and put it in the box"')
    p.add_argument("num_episodes", type=int, help="How many episodes to record in this run")
    p.add_argument("--repo", default="JustoDario/flexnsense_skill1")
    p.add_argument("--batch", type=int, default=10, help="Upload to the Hub every N episodes")
    p.add_argument("--server", default="localhost", help="Station host")
    p.add_argument("--cameras", type=int, default=3, help="Expected images per frame")
    p.add_argument("--joints", type=int, default=8, help="Expected joints per frame")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopped. The episode being recorded (if any) was discarded.")
        if pending_files():
            print(f"{len(pending_files())} saved episodes not uploaded yet are in {PENDING_DIR}; "
                  "they will be uploaded on the next run.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
