"""Send one hook request to a remote actuator and print its reported outcome.

    .venv/bin/python scripts/try_remote_hook.py \
        --url ws://justo.local:8770/arm2 --finger index

This probes one RemoteHookMover request over the real WebSocket. It does not run the session
orchestrator. With the actuator's default manual mode, a person on that machine moves the hook
by hand and confirms whether it seated.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session.link import connect_to_actuator  # noqa: E402
from session.remote_arm2 import RemoteHookMover  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    try:
        link, task = await connect_to_actuator(args.url)
    except Exception as exc:
        print(f"could not connect to {args.url}: {exc}", file=sys.stderr)
        return 2

    mover = RemoteHookMover(link, move_timeout_s=args.timeout)
    mover.request(args.finger)
    print(f"connected to {args.url}; requested the {args.finger} finger")
    print(f"waiting up to {args.timeout:g}s for the actuator's report ...")
    try:
        loop = asyncio.get_running_loop()
        while True:
            outcome = mover.poll(now=loop.time())
            if outcome is not None:
                print(f"\nresult: {outcome.result.value}; detail: {outcome.detail!r}")
                # Give the queued abort from a timeout a chance to reach the actuator.
                if not outcome.confirmed:
                    await asyncio.sleep(0.1)
                return 0 if outcome.confirmed else 1

            if task.done():
                try:
                    task.result()
                except Exception as exc:
                    print(f"actuator connection ended before a report: {exc}", file=sys.stderr)
                else:
                    print("actuator connection closed before a report", file=sys.stderr)
                return 2

            if mover.rejected:
                for reason in mover.rejected:
                    print(f"rejected message: {reason}", file=sys.stderr)
                mover.rejected.clear()
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        mover.abort()
        await asyncio.sleep(0.1)
        raise
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True,
        help="ws://host:port/arm2 for the other machine's session.link serve",
    )
    parser.add_argument("--finger", default="index", help="finger to request, default: index")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="seconds to wait for a report, default: 90")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
