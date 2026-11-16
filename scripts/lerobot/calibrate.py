"""NOT our hardware. Kept in case we ever port to a LeRobot-supported arm (SO-101 etc).

Our arm is the ElRobot, which has no LeRobot support (confirmed upstream: no robot type, no
driver) and is controlled through NormaCore Station instead - see scripts/calibrate.py (the
default, one level up) and docs/elrobot.md. This file is the original SO-101-via-LeRobot
script, unmodified, parked here rather than deleted.

Calibrate the SO-101 follower and/or leader arm using ports from .env.

Usage:
    python scripts/lerobot/calibrate.py follower
    python scripts/lerobot/calibrate.py leader
    python scripts/lerobot/calibrate.py both
"""

import os
import sys


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def calibrate_follower(port, arm_id):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(port=port, id=arm_id)
    follower = SO101Follower(config)
    follower.connect(calibrate=False)
    follower.calibrate()
    follower.disconnect()


def calibrate_leader(port, arm_id):
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    config = SO101LeaderConfig(port=port, id=arm_id)
    leader = SO101Leader(config)
    leader.connect(calibrate=False)
    leader.calibrate()
    leader.disconnect()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("follower", "leader", "both"):
        print(__doc__)
        sys.exit(1)

    load_env()
    target = sys.argv[1]

    if target in ("follower", "both"):
        port = os.environ.get("ROBOT_PORT")
        if not port:
            sys.exit("ROBOT_PORT is not set in .env")
        calibrate_follower(port, arm_id="rehab_follower")

    if target in ("leader", "both"):
        port = os.environ.get("LEADER_PORT")
        if not port:
            sys.exit("LEADER_PORT is not set in .env")
        calibrate_leader(port, arm_id="rehab_leader")


if __name__ == "__main__":
    main()
