"""Does opening the camera stop Station's position stream? Measures it, three ways.

    .venv/bin/python scripts/measure/check_camera_vs_station.py

**Nothing moves.** This opens the camera and reads Station, both read-only, and commands no
motor at any point. It exists because a session can look completely normal - force climbing,
phases advancing, the printed step column tracking the admittance law - while the arm never
receives a single command, and the reason is not in any of that output.

The question it answers: `scripts/arm/go_home.py --check` reads every joint instantly, and
`scripts/session/run_session.py` with a fake camera reads them fine too, but the same session with the
real camera open received **zero** position updates in 22 seconds on this bench. Those two
facts together point at the camera and the arm competing rather than at the control code, and
`lsusb -t` on this machine shows why that is plausible: the camera (1920x1080 MJPG, 480M) and
the arm (cdc_acm, 12M) are on the same USB 2.0 root hub, where an isochronous video stream
takes reserved bandwidth ahead of the arm's bulk transfers.

Plausible is not measured, so this measures it:

  1. Station alone, camera closed. The baseline: how many position updates per second arrive
     when nothing competes.
  2. Station with the camera open and streaming frames, which is exactly the state a session
     runs in.
  3. Station alone again, after closing the camera. Rules out "Station simply died partway
     through the test" as the explanation for a low number in phase 2.

A large drop in phase 2 that recovers in phase 3 is the contention, and the fix is physical
(move the camera to a different USB controller, or lower its resolution), not a code change.
Phase 2 matching phase 1 means the camera is innocent and the problem is elsewhere.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm1.lib.env import env_get  # noqa: E402


async def count_updates(client, bus_serial: str, seconds: float) -> tuple[int, set[int]]:
    """Position updates received in `seconds`, and which motor ids they covered."""
    from arm1.lib.station import follow_motor_state

    count = 0
    seen: set[int] = set()

    async def collect():
        nonlocal count
        async for bus, mid, motor in follow_motor_state(client):
            if bus == bus_serial and motor.position is not None:
                count += 1
                seen.add(mid)

    try:
        await asyncio.wait_for(collect(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    return count, seen


async def run(args) -> int:
    from arm1.lib.station import connect

    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    print("Read-only on both devices. The camera opens; no motor is ever commanded.\n")
    client = await connect(host=host)

    print(f"1. Station alone, camera closed ({args.seconds:.0f} s) ...")
    alone, seen_alone = await count_updates(client, bus_serial, args.seconds)
    print(f"   {alone} updates, motors {sorted(seen_alone)}"
          f"   -> {alone / args.seconds:.0f} per second\n")

    print(f"2. Station with the camera open and streaming ({args.seconds:.0f} s) ...")
    from arm1.lib.camera import open_source

    source = open_source(camera=args.camera)
    frames = 0

    async def pump_frames():
        """Read frames the way a session does, so the camera is genuinely streaming.

        `source.read()` is synchronous and blocks the event loop while it waits, exactly as it
        does in `scripts/session/run_session.py`; the `sleep(0)` between reads is what lets the
        Station subscription get its turn, also exactly as it does there. Reproducing that
        shape matters: a version of this check that read frames in a thread would measure a
        situation the session never actually runs in.
        """
        nonlocal frames
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(0)
            ok, _frame, _ts, _ = source.read()
            if ok:
                frames += 1

    with_camera, seen_with = await asyncio.gather(
        count_updates(client, bus_serial, args.seconds), pump_frames()
    )
    with_camera, seen_with = with_camera
    print(f"   {with_camera} updates, motors {sorted(seen_with)}"
          f"   -> {with_camera / args.seconds:.0f} per second")
    print(f"   ({frames} camera frames read, {frames / args.seconds:.0f} fps)\n")

    source.release()
    await asyncio.sleep(1.0)   # let the camera fully let go of the bus

    print(f"3. Station alone again, camera closed ({args.seconds:.0f} s) ...")
    after, seen_after = await count_updates(client, bus_serial, args.seconds)
    print(f"   {after} updates, motors {sorted(seen_after)}"
          f"   -> {after / args.seconds:.0f} per second\n")

    print("-" * 72)
    if alone == 0 and after == 0:
        print("Station sent nothing even with the camera closed, so this test cannot say "
              "anything about the camera. Fix the Station connection first "
              "(scripts/arm/go_home.py --check).")
        return 1
    share = with_camera / alone if alone else 0.0
    print(f"with the camera open, Station delivered {share * 100:.0f} percent of its "
          f"camera-closed rate")
    if share < 0.2:
        print("\nThat is the contention. The camera and the arm are competing for the same "
              "USB controller, and the arm loses. Nothing in the control code can fix it:\n"
              "  - move the camera to a port on a DIFFERENT controller (`lsusb -t`: it and "
              "the arm must not share a root hub), or\n"
              "  - lower the camera's resolution so it needs less of the bus.")
        return 1
    if share < 0.8:
        print("\nA real but partial slowdown. Position updates still arrive, just less often, "
              "so a session will work and feel laggy rather than not move at all.")
        return 0
    print("\nThe camera is not the problem: Station keeps up with it open. Look elsewhere.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=5.0, help="how long each phase runs")
    p.add_argument("--camera", default=None)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
