"""This machine's end of the two-machine session: a hook_request runs the skill, and only a
finished, confirmed skill is reported as confirmed. Fake link, fake clocks, fake skill."""

import pytest

import hook_actuator
from hook_actuator import HookActuator, LocalClockLink
from session.arm2 import HookResult
from session.remote_arm2 import ActuatorSide, RemoteHookMover


class Pipe:
    """One direction of a link: what one end sends, the other drains."""

    def __init__(self):
        self.out, self.peer = [], None

    def send(self, message):
        self.peer.out.append(message)

    def drain(self):
        got, self.out = self.out, []
        return got


class FakeJob:
    def __init__(self, finger, extra):
        self.finger, self.code, self.keys, self.stopped = finger, None, [], False

    def key(self, key):
        self.keys.append(key)
        self.code = 0 if key == "d" else 2

    def poll(self):
        return self.code

    def stop(self):
        self.stopped = True


class Rig:
    """A brain and this actuator on two machines whose clocks are `skew` seconds apart."""

    def __init__(self, skew=5000.0):
        brain_end, here = Pipe(), Pipe()
        brain_end.peer, here.peer = here, brain_end
        self.t, self.skew, self.jobs, self.log = 0.0, skew, [], []
        self.brain = RemoteHookMover(brain_end, move_timeout_s=60.0, lease_period_s=0.5)
        side = ActuatorSide(LocalClockLink(here, lambda: self.t))
        self.actuator = HookActuator(side, start=self._start, log=self.log.append)

    def _start(self, finger, extra):
        self.jobs.append(FakeJob(finger, extra))
        return self.jobs[-1]

    def tick(self, keys="", brain=True, dt=0.1):
        """One cycle on both machines. `brain=False` is the brain going quiet."""
        self.t += dt
        got = self.brain.poll(self.t + self.skew) if brain else None
        self.actuator.step(self.t, keys)
        return got

    def run_until_report(self, keys=""):
        self.tick(keys)
        for _ in range(5):
            got = self.tick()
            if got is not None:
                return got
        return None


def test_every_trained_task_is_reachable_by_its_destination_finger():
    assert hook_actuator.TASKS == {
        "middle": "move device from right index to right middle finger",
        "ring": "move device from right middle to right ring finger",
        "little": "move device from right ring to right pinky finger",
    }


@pytest.mark.parametrize("skew", [5000.0, -5000.0, 0.0])
def test_a_request_runs_the_skill_and_done_is_reported_as_confirmed(skew):
    rig = Rig(skew)
    rig.brain.request("middle")
    for _ in range(20):                       # two seconds: many lease periods, the skill still running
        assert rig.tick() is None
    assert [j.finger for j in rig.jobs] == ["middle"] and not rig.jobs[0].stopped
    got = rig.run_until_report("d")
    assert got.result is HookResult.CONFIRMED and got.finger == "middle"
    assert not rig.actuator.busy


def test_nothing_is_reported_while_the_skill_is_still_running():
    rig = Rig()
    rig.brain.request("ring")
    assert all(rig.tick() is None for _ in range(50))
    assert rig.actuator.busy


def test_giving_up_is_reported_as_failed_never_confirmed():
    rig = Rig()
    rig.brain.request("ring")
    rig.tick()
    got = rig.run_until_report("f")
    assert got.result is HookResult.FAILED and "gave up" in got.detail


def test_a_skill_that_dies_or_misses_home_is_failed():
    rig = Rig()
    rig.brain.request("middle")
    rig.tick(), rig.tick()
    rig.jobs[0].code = 1
    got = rig.run_until_report()
    assert got.result is HookResult.FAILED and "code 1" in got.detail


def test_pinky_is_the_little_finger():
    rig = Rig()
    rig.brain.request("pinky")
    rig.tick(), rig.tick()
    assert rig.jobs[0].finger == "little"


def test_a_finger_without_a_policy_is_placed_by_hand_and_moves_no_arm():
    side = ActuatorSide(Pipe())
    assert isinstance(hook_actuator.start_job("index", []), hook_actuator.ManualJob)
    result, detail = hook_actuator.outcome("index", 0, manual=True)
    assert result is HookResult.CONFIRMED and "by hand" in detail
    assert not side.rejected


def test_the_skill_is_stopped_when_the_brain_goes_quiet_and_nothing_is_confirmed():
    rig = Rig()
    rig.brain.request("middle")
    rig.tick(), rig.tick()
    for _ in range(8):                        # 0.8 s without a heartbeat: the 0.5 s lease is gone
        rig.tick(brain=False)
    assert rig.jobs[0].stopped and not rig.actuator.busy
    rig.tick("d", brain=False)                # a late key press confirms nothing
    assert rig.brain.link.drain() == []


def test_an_abort_stops_the_skill():
    rig = Rig()
    rig.brain.request("middle")
    rig.tick(), rig.tick()
    rig.brain.abort()
    rig.tick(brain=False)
    assert rig.jobs[0].stopped and not rig.actuator.busy


def test_a_key_pressed_with_no_request_in_hand_does_nothing():
    rig = Rig()
    rig.tick("d")
    assert rig.jobs == [] and rig.brain.link.drain() == []


def test_the_next_request_after_a_confirmed_one_runs_a_new_skill():
    rig = Rig()
    rig.brain.request("middle")
    rig.tick()
    assert rig.run_until_report("d").confirmed
    rig.brain.request("ring")
    rig.tick()
    got = rig.run_until_report("d")
    assert got.confirmed and got.finger == "ring" and [j.finger for j in rig.jobs] == ["middle", "ring"]
