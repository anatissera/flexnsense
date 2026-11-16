"""A whole `cpm` set, closed-loop, through every real layer at once.

The unit tests either side of this check one thing each: `tests/control/test_loop.py` that the
pull law computes the right step, `tests/session/test_reps.py` that the rep machine advances
through the phases, `tests/scripts/test_run_session.py` that the script turns a `RepStep` into
a `PullPlan`. All three can pass while the assembled thing does not move, because the bugs
that live between them are sign and bookkeeping bugs, and a sign error is invisible until the
pieces are wired together and something has to actually arrive somewhere.

So this runs the real `RepMachine`, the real `ControlLoop`, the real `SafetyGate`, the real
`pull_plan_for` out of the script, and the real `CoordinatedAxis` solved from NormaCore's
URDF at the pose the arm was last standing in. The only thing simulated is the physics: the
arm reaches the position it is commanded, and the tether is a spring of the bench-measured
stiffness. That is the same honesty boundary `control.simulate` draws, and it is the right
one here, because what is being checked is whether the control path is self-consistent, not
whether the servos are quick.

**This is not a substitute for a bench run.** It cannot see backlash, it cannot see the camera
dropping frames in bursts, and it assumes a commanded position is a reached position. What it
can prove is the thing worth proving before a finger is anywhere near the rig: that the arm
goes out to the prescribed distance, comes back to where it started, and does it again, with
the force never passing the cap.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

from control.admittance import AdmittanceParams
from control.loop import AxisCalibration, ControlLoop
from control.safety import Action, SafetyGate, SafetyLimits
from lib.units import grams_force_to_newtons
from session.protocol import Block
from session.reps import RepInput, RepLimits, RepMachine

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

#: The pose the arm was actually standing in on 2026-11-15, read from Station.
POSE = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275, 8: 1301}

#: Bench-measured tether stiffness, config/calibration/bench.json.
TETHER_N_PER_M = 56.2

#: The example protocol's index block, which `Block.check_against` confirms is self-consistent:
#: the cap has to be reachable *after* ARMED has already taken the tether to a quarter of it,
#: or the stretch stops at the cap every repetition and never reaches the target.
CAP_GF = 110.0
ROM_M = 0.012

#: Largest force the stretch may overshoot the cap by, and it is not zero: the cap stops the
#: *next* step, not the one in flight, so the overshoot is bounded by one step's worth of
#: stretch against the tether's stiffness. `pull_max_step_mm` is the knob that bounds it, and
#: the hard ceiling (well above here) is what makes the bound safe rather than merely small.
ONE_STEP_OVERSHOOT_N = 0.004 * TETHER_N_PER_M


@pytest.fixture(scope="module")
def run_session():
    """The script, for its real `pull_plan_for`. A copy of that logic here would prove
    nothing: the function under test is the one the arm actually runs."""
    spec = importlib.util.spec_from_file_location(
        "run_session_cpm", REPO / "scripts" / "run_session.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pull_arg_namespace():
    """Only the fields `pull_plan_for` reads, so this test does not depend on the whole CLI."""
    return types.SimpleNamespace(
        pull_mm_s=20.0, pull_max_step_mm=4.0, arm_mm_s=8.0, max_travel_mm=80.0
    )


class Rig:
    """The simulated physics: a commanded position is reached, and the tether is a spring.

    `arm_m` is in the same signed quantity the pull law uses: positive means the arm has given
    way (paid line out, lowering tension), negative means it has taken line up and stretched
    the finger. The finger itself never moves in `cpm`, which is the entire premise of the
    mode, so there is nothing to track for it.
    """

    def __init__(self, resting_gf=35.0, ticks_per_metre=1.0, sign=1):
        self.arm_m = 0.0
        self.resting_n = grams_force_to_newtons(resting_gf)
        self.ticks_per_metre = ticks_per_metre
        self.sign = sign

    @property
    def force_n(self) -> float:
        # Stretch is the arm having taken line up, which is negative travel in the loop's
        # sign convention. Paying line out past the rig point just goes slack, it does not
        # push, so the resting tension is a floor.
        stretch_m = max(0.0, -self.arm_m)
        return self.resting_n + TETHER_N_PER_M * stretch_m

    @property
    def ticks(self) -> int:
        return int(round(self.arm_m * self.sign * self.ticks_per_metre))


def run_set(run_session, reps=2, hold_s=0.3, cap_gf=CAP_GF, rom_m=ROM_M,
            resting_gf=35.0, dt=1 / 30.0, max_seconds=120.0):
    """One block, start to finish, with every decision coming from the real code."""
    block = Block.from_grams("index", "cpm", reps=reps, cap_gf=cap_gf, hold_s=hold_s,
                             rom_target_m=rom_m)
    machine = RepMachine(block, RepLimits())

    ticks_per_metre = 6566.0        # the coordinated value measured at this pose
    limits = SafetyLimits(
        f_threshold_n=block.cap_n,
        f_max_n=grams_force_to_newtons(590.0),
        max_age_s=0.5,
        max_dropouts=45,
        max_travel_m=0.080,
        anticipation_n=grams_force_to_newtons(5.0),
    )
    params = AdmittanceParams(damping_n_s_per_m=10.0, v_max_m_s=0.035, deadband_n=0.05,
                              max_step_m=0.004)
    axis = AxisCalibration(motor_id=4, ticks_per_metre=ticks_per_metre, sign=-1)
    loop = ControlLoop(limits, params, axis, gate=SafetyGate(limits),
                       yield_check_window_s=None)
    rig = Rig(resting_gf=resting_gf, ticks_per_metre=ticks_per_metre, sign=axis.sign)
    args = pull_arg_namespace()

    trace = []
    t = 0.0
    for _ in range(int(max_seconds / dt)):
        t += dt
        force = rig.force_n
        step = machine.step(RepInput(force_n=force, reading_age_s=0.01,
                                     travel_m=loop.state.travel_m, now=t, confident=True))
        plan = run_session.pull_plan_for(step, machine, args)
        decision, step_m = loop.decide(force, now=t, frame_t=t, position_ticks=rig.ticks,
                                       pull=plan)
        # travel * axis.sign, not raw travel: the raw register direction is a property of
        # the rig, while "how far has this arm given way" is the quantity the pull law and the
        # protocol are both written in. Recording the raw value made this test pass or fail on
        # the tick polarity rather than on the control path.
        yielded_m = loop.state.travel_m * axis.sign
        trace.append((t, step.phase.value, force, yielded_m, decision.action))
        if decision.action is Action.RELEASE:
            break
        # The arm reaches what it is told, which is this test's stated simplification.
        if plan is not None:
            rig.arm_m += step_m
        loop.state.last_t = t
        if step.finished:
            break
    return machine, loop, rig, trace


# ----------------------------------------------------------------------------- the whole set
def test_a_cpm_set_completes_with_nothing_but_the_real_control_code(run_session):
    """The assembled claim: the arm takes the finger out and brings it back, twice, on its own."""
    machine, loop, rig, trace = run_set(run_session, reps=2)
    phases = [p for i, (_, p, *_) in enumerate(trace) if i == 0 or trace[i - 1][1] != p]
    assert machine.reps_completed == 2, f"phases: {phases}"
    assert machine.attempts_abandoned == 0, f"phases: {phases}"


def test_the_arm_actually_reaches_the_prescribed_stretch(run_session):
    """A sign error anywhere in the chain shows up here and essentially nowhere else: the arm
    would travel the wrong way, the stretch would never arrive, and every unit test either
    side of this would still pass."""
    _, _, _, trace = run_set(run_session, reps=1)
    furthest = min(yielded for _, _, _, yielded, _ in trace)
    assert furthest == pytest.approx(-ROM_M, abs=0.0005), (
        f"the stretch should reach {-ROM_M * 1000:.0f} mm, got {furthest * 1000:.1f} mm"
    )


def test_the_arm_comes_back_to_where_the_rep_started(run_session):
    """The half that has no equivalent in `passive`, where the patient does it."""
    _, loop, _, _ = run_set(run_session, reps=1)
    assert abs(loop.state.travel_m) <= RepLimits().home_m


def test_the_stretch_stays_within_one_step_of_the_cap(run_session):
    """The cap stops the next step, not the one already in flight, so a single step's worth of
    stretch can land past it. That residue is what `--pull-max-step-mm` bounds, and the point
    of checking it here is that it stays a residue rather than growing into the gap between the
    cap and the hard ceiling."""
    machine, _, _, trace = run_set(run_session, reps=2)
    peak = max(force for _, _, force, _, _ in trace)
    cap = machine.block.cap_n
    assert peak <= cap + ONE_STEP_OVERSHOOT_N, (
        f"peak {peak:.3f} N is more than one step past the {cap:.3f} N cap"
    )


def test_the_stretch_stays_far_below_the_hard_ceiling(run_session):
    """The cap is where the stroke stops; the ceiling is where safety takes over. In a mode
    bounded by position, the two should never come close to meeting."""
    _, loop, _, trace = run_set(run_session, reps=2)
    peak = max(force for _, _, force, _, _ in trace)
    assert peak < loop.limits.f_max_n / 2, (
        f"peak {peak:.2f} N is within half the {loop.limits.f_max_n:.2f} N hard ceiling"
    )


def test_nothing_in_the_set_releases(run_session):
    _, _, _, trace = run_set(run_session, reps=2)
    assert all(action is not Action.RELEASE for *_, action in trace)


def test_every_repetition_is_the_same_motion(run_session):
    """The property that distinguishes this mode: nothing in a repetition waits on a person,
    so two reps should travel the same distance rather than however far a patient went."""
    _, _, _, trace = run_set(run_session, reps=3)
    per_rep: dict[int, float] = {}
    for _, phase, _, yielded, _ in trace:
        if phase in ("ramp", "work", "hold"):
            pass
    # Furthest point reached in each repetition, keyed by how many were done before it.
    machine_reps = []
    deepest = 0.0
    seen_reset = False
    for _, phase, _, yielded, _ in trace:
        deepest = min(deepest, yielded)
        if phase == "reset":
            seen_reset = True
        elif seen_reset and phase in ("cueing", "ramp"):
            machine_reps.append(deepest)
            deepest, seen_reset = 0.0, False
    machine_reps.append(deepest)
    assert len(machine_reps) >= 2
    assert max(machine_reps) == pytest.approx(min(machine_reps), abs=0.001), (
        f"repetitions travelled differently: {[round(v * 1000, 1) for v in machine_reps]} mm"
    )
    del per_rep


# ------------------------------------------------------------------- what stops it, and why
def test_a_stiffer_finger_stops_the_stretch_at_the_cap_instead(run_session):
    """A clinical finding, not a fault: the finger is stiffer than the prescription assumed,
    so the stroke ends short of the target. The repetition still counts, because the arm did
    everything it was allowed to."""
    # A cap only just above the resting tension, so it arrives after a few millimetres: the
    # arm stops there, well short of the 12 mm the prescription asked for.
    machine, _, _, trace = run_set(run_session, reps=1, cap_gf=45.0, resting_gf=35.0)
    furthest = min(yielded for _, _, _, yielded, _ in trace)
    assert abs(furthest) < ROM_M, "it should have stopped short of the target"
    assert machine.reps_completed == 1


def test_the_geometry_at_the_real_pose_can_follow_the_stroke():
    """The one thing in this file that is not a simulation: whether the joints chosen for the
    live pose can actually trace the stretch and the return without wandering off the tether's
    line. A pull the pose cannot follow is refused before anything moves, and `cpm` travels
    the distance twice per repetition, out and back."""
    pytest.importorskip("numpy")
    if not URDF.is_file():
        pytest.skip("needs the norma-core clone (hardware/elrobot/simulation); see SETUP.md")
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain(URDF)
    direction = tether_tension_direction(chain, POSE)
    motors = chain.choose_pull_motors(POSE, direction)
    # The stretch raises tension, so it travels along `direction` itself, which is the stroke
    # the yielding controller never takes and therefore the one never checked before.
    ok, why = chain.can_pull_straight(motors, POSE, direction, ROM_M)
    assert ok, f"this pose cannot take the cpm stretch with joints {motors}: {why}"
