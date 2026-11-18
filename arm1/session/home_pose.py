"""The pose a session starts from and returns to.

A straight pull is only as good as the pose it starts in. The same three joints that track a
line to half a millimetre from one configuration wander 8.6 mm from another, and worse, a pose
can start well conditioned and drive itself towards a singularity partway through a 90 mm pull
(`lib.kinematics.Chain.can_pull_straight` is what catches that). Leaving the starting pose to
however the arm happened to be left makes every session's control quality a matter of luck.

So the arm has a **home pose**: one configuration, chosen once by searching the workspace for
the best worst-case, which every session begins from and returns to. That makes the control
behaviour repeatable, and it means the pose-dependent bench numbers (`ticks_per_metre`, the
yield sign, the force calibration) are measured in the pose they are actually used in, which
`docs/force-calibration.md` says they have to be.

How this one was chosen: 6000 random poses inside the calibrated joint ranges, keeping clear
of the limits, scored by the smallest singular value of the jacobian, which is the arm's gain
in its *worst* direction. The best candidates were then simulated through full 90 mm pulls in
28 directions, out and back, and ranked by the worst deviation any of them produced. The
winner holds every direction to 0.79 mm, against 8.6 mm for the all-centred pose. The search
is in `scripts/arm/find_home_pose.py` so it can be re-run if the arm or the rig changes.
"""
from __future__ import annotations

import json
from pathlib import Path

HOME_POSE_PATH = "config/calibration/home-pose.json"

#: Where `scripts/session/run_cpm.py`'s stroke begins and ends: tether already taken up, finger
#: attached. Not a parking pose, and deliberately not the same file as `HOME_POSE_PATH`,
#: because a pose with the tether slack and nothing hooked cannot be where a stroke that
#: needs tension already present starts from. Lives next to `HOME_POSE_PATH` rather than in
#: `scripts/session/run_cpm.py` so `scripts/arm/go_home.py` can walk to either one without importing a
#: script from a script, which `scripts/` being unpackaged rules out anyway.
CPM_START_POSE_PATH = "config/calibration/cpm-start-pose.json"

#: Every pose `scripts/arm/go_home.py` is willing to walk to, by name. Closed on purpose: an open
#: `--path` would let a typo or an unrelated file become a destination the arm actually
#: drives to, and the whole point of a named pose is that it was placed and measured with the
#: real rig in front of someone, not that it is any JSON file with a `ticks` key.
NAMED_POSES: dict[str, str] = {
    "home": HOME_POSE_PATH,
    "cpm": CPM_START_POSE_PATH,
}


def load_home_pose(path: str | Path = HOME_POSE_PATH) -> dict[int, int]:
    """The home pose as motor id to tick value. Raises with what to run if it is missing."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"no home pose at {p}.\n\n"
            "Find one with:\n\n"
            "    .venv/bin/python scripts/arm/find_home_pose.py --save\n\n"
            "It searches the calibrated joint ranges for the configuration that pulls "
            "straightest in the worst direction. See session/home_pose.py."
        )
    raw = json.loads(p.read_text())
    return {int(k): int(v) for k, v in raw["ticks"].items()}


def home_pose_metadata(path: str | Path = HOME_POSE_PATH) -> dict:
    """Everything recorded about the home pose, including how good it measured."""
    return json.loads(Path(path).read_text())


def distance_from_home(pose: dict[int, float], home: dict[int, int] | None = None) -> int:
    """Largest tick difference between `pose` and home, over the joints home specifies.

    What "the arm is home" means in practice, and what `scripts/arm/go_home.py` reports. A single
    number rather than a per-joint dict because the question it answers is binary: close
    enough to start, or not.
    """
    if home is None:
        home = load_home_pose()
    return max((abs(int(pose.get(m, home[m])) - home[m]) for m in home), default=0)


def is_home(pose: dict[int, float], tolerance_ticks: int = 20,
            home: dict[int, int] | None = None) -> bool:
    """Whether the arm is close enough to home to begin a session.

    The tolerance is not zero because a servo holding position sits within a few ticks of its
    goal, and demanding exactness would mean never being home. 20 ticks is under 2 degrees.
    """
    return distance_from_home(pose, home) <= tolerance_ticks


def home_waypoints(start: dict[int, float], home: dict[int, int] | None = None,
                   max_step_ticks: int = 80) -> list[dict[int, int]]:
    """Intermediate poses from `start` to `home`, each joint no more than `max_step_ticks`
    further along than the last waypoint.

    Pure: no Station, no sleeping, no confirmation prompt. This exists so `scripts/arm/go_home.py`
    and `scripts/session/run_session.py`'s own return-to-home share one answer to "how do I get from
    here to home without asking any servo to cross its whole range in one jump" - a servo
    given a goal far from its current position moves as fast as it can get there, and that is
    the one thing this arm must never do near a hand. Each caller still owns the pacing
    (`settle` between steps) and whatever it does with a `KeyboardInterrupt` midway.

    Returns `[]` if `start` is already at `home` (within rounding): "zero waypoints" and "the
    last waypoint happens to equal home" would otherwise be the same string of calls with
    nothing to tell them apart.
    """
    if home is None:
        home = load_home_pose()
    start_ticks = {m: int(start.get(m, home[m])) for m in home}
    if all(start_ticks[m] == home[m] for m in home):
        return []
    gap = distance_from_home(start_ticks, home)
    steps = max(1, int(gap / max_step_ticks) + 1)
    return [
        {m: int(round(start_ticks[m] + (home[m] - start_ticks[m]) * (i / steps))) for m in home}
        for i in range(1, steps + 1)
    ]
