"""The TypeScript ports in the dashboard have to agree with the Python originals.

Python is the tested side and the source of truth; `dashboard/app/lib/session/*.ts`,
`excursion.ts`, `handGeometry.ts` and `units.ts` are ports. Nothing else type-checks the two
against each other, so this does it by *differential testing*: build the TypeScript to
CommonJS, hand it the same randomised inputs the Python code sees, and compare every output.

For the rep machine the inputs come from a closed-loop simulation driven by the Python machine,
so the traces actually reach cueing, ramp, work, hold, reset, counted reps, abandoned attempts
and the deadman. The TypeScript machine then replays the recorded inputs.

Skipped when `node` or the dashboard's TypeScript is not installed (`pnpm install` in
`dashboard/`), so the suite still runs on a machine with no JS toolchain.

Error cases compare only that both sides refuse, not the wording: Python formats floats as
`1.0` and JavaScript as `1`, so the numbers inside a message legitimately differ.
"""
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from arm1.hand_tracking.excursion import fit_excursion, tip_minus_wrist
from arm1.hand_tracking.landmarks import all_flexion_deg
from arm1.lib.units import STANDARD_GRAVITY_M_S2, grams_force_to_newtons, newtons_to_grams_force
from arm1.session.fingers import normalise
from arm1.session.instructions import instruction_for, instruction_key
from arm1.session.modes import MODES, Phase, plan_for
from arm1.session.protocol import Block, Protocol
from arm1.session.reps import RepInput, RepMachine

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "dashboard"
TSC = DASHBOARD / "node_modules" / "typescript" / "bin" / "tsc"
RUNNER = Path(__file__).with_name("ts_port_runner.js")
SOURCES = [
    "app/lib/session/fingers.ts",
    "app/lib/session/format.ts",
    "app/lib/session/instructions.ts",
    "app/lib/session/modes.ts",
    "app/lib/session/protocol.ts",
    "app/lib/session/reps.ts",
    "app/lib/session/simulation.ts",
    "app/lib/excursion.ts",
    "app/lib/handGeometry.ts",
    "app/lib/units.ts",
]

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(
    NODE is None or not TSC.exists(),
    reason="needs node and dashboard/node_modules (run `pnpm install` in dashboard/)",
)


# ------------------------------------------------------------------ scenario generation
def _rep_scenario(rng: random.Random) -> dict:
    """One block and a recorded input trace, driven by the Python machine so it goes somewhere."""
    mode = rng.choice(["active", "passive"])
    cap_gf = rng.choice([80, 120, 200, 300, 450])
    block = Block.from_grams(
        rng.choice(["thumb", "index", "middle", "ring", "little"]), mode,
        reps=rng.randint(1, 3), cap_gf=cap_gf, hold_s=rng.choice([0.2, 0.5, 1.0]),
        rom_target_m=rng.choice([0.01, 0.02, 0.03]) if mode == "passive" else None,
    )
    machine = RepMachine(block)
    force, travel, now = 0.0, 0.0, 0.0
    stall = rng.random() < 0.2  # a patient who never starts, to hit the timeouts
    inputs: list[dict] = []
    last = None
    for _ in range(900):
        now += rng.choice([0.033] * 20 + [0.5, 5.0, 31.0])
        phase = last.phase if last else Phase.IDLE
        pre, cap = machine.pretension_n, block.cap_n
        if phase in (Phase.IDLE, Phase.ARMED):
            force = min(force + 0.08, pre * 1.2)
        elif phase is Phase.CUEING:
            if block.mode_spec.work.driver.value == "patient" and not stall and rng.random() < 0.1:
                force = pre + 0.2
        elif phase in (Phase.RAMP, Phase.WORK):
            if not stall:
                force = min(force + 0.1, cap * 1.05)
            if force >= cap:
                travel += 0.0006 if block.mode == "passive" else 0.0004
        elif phase is Phase.HOLD:
            force = cap
        elif phase is Phase.RESET:
            force = max(pre, force - 0.1)
            travel = travel * 0.9 if abs(travel) > 0.0005 else 0.0
        if phase is Phase.CUEING and last is not None and last.note.startswith("repetition"):
            travel = 0.0
        inp = {
            "force_n": None if rng.random() < 0.04 else force + rng.uniform(-0.01, 0.01),
            "reading_age_s": rng.choice([0.01] * 30 + [0.5]),
            "travel_m": travel,
            "now": now,
            "confident": rng.random() > 0.03,
        }
        if rng.random() < 0.002:
            inputs.append({"abandon": "stopped by test"})
            machine.abandon("stopped by test")
            break
        inputs.append(inp)
        last = machine.step(RepInput(
            force_n=inp["force_n"], reading_age_s=inp["reading_age_s"], travel_m=inp["travel_m"],
            now=inp["now"], confident=inp["confident"],
        ))
        if last.finished:
            inputs.append({"force_n": force, "reading_age_s": 0.0, "travel_m": 0.0,
                           "now": now + 1, "confident": True})
            machine.step(RepInput(force, 0.0, 0.0, now + 1))
            break
    return {
        "block": {"finger": block.finger, "mode": block.mode, "reps": block.reps,
                  "cap_n": block.cap_n, "hold_s": block.hold_s, "rom_target_m": block.rom_target_m},
        "inputs": inputs,
    }


def _replay_in_python(scenario: dict) -> list[dict]:
    b = scenario["block"]
    machine = RepMachine(Block(finger=b["finger"], mode=b["mode"], reps=b["reps"],
                               cap_n=b["cap_n"], hold_s=b["hold_s"], rom_target_m=b["rom_target_m"]))
    out = []
    for inp in scenario["inputs"]:
        if "abandon" in inp:
            s = machine.abandon(inp["abandon"])
        else:
            s = machine.step(RepInput(force_n=inp["force_n"], reading_age_s=inp["reading_age_s"],
                                      travel_m=inp["travel_m"], now=inp["now"],
                                      confident=inp["confident"]))
        out.append({
            "phase": s.phase.value, "law": s.law.value, "target": s.target.value,
            "driver": s.driver.value,
            "instruction": s.instruction, "rep_index": s.rep_index,
            "reps_completed": s.reps_completed, "note": s.note, "finished": s.finished,
            "attempts_abandoned": s.attempts_abandoned,
        })
    return out


def _points(rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(21)]


def _excursion_trajectory(rng: random.Random) -> list[list[float]]:
    n = rng.randint(2, 300)
    d = [rng.uniform(-1, 1) for _ in range(3)]
    noise = rng.choice([0.0, 0.0005, 0.005])
    return [[d[i] * t + rng.uniform(-noise, noise) for i in range(3)]
            for t in (rng.uniform(0, 0.08) for _ in range(n))]


def _scenarios() -> dict:
    rng = random.Random(20261115)
    flexion = [_points(rng) for _ in range(40)]
    degenerate = _points(rng)
    degenerate[6] = list(degenerate[5])  # MCP == PIP: a zero-length bone, so NaN
    flexion.append(degenerate)

    fingers = ["thumb", "Index", " MIDDLE ", "ring", "little", "pinky", "fifth", "little finger", "toe", ""]
    ties = [(x, d) for x in (0.5, 1.5, 2.5, 0.125, 0.375, 2.675, -0.5, -2.5, 0.25, 1e-7)
            for d in (0, 1, 2)]
    formats = ties + [(rng.uniform(-500, 500), rng.randint(0, 3)) for _ in range(300)]

    good = {"finger": "index", "mode": "active", "reps": 3, "cap_gf": 200}
    blocks = [
        good,
        {**good, "finger": "pinky"},
        {**good, "mode": "passive", "rom_target_m": 0.02},
        {**good, "mode": "passive"},               # needs a target
        {**good, "rom_target_m": 0.02},            # has no use for one
        {**good, "reps": 0},
        {**good, "cap_gf": -5},
        {**good, "hold_s": -1},
        {**good, "mode": "sideways"},
        {**good, "finger": "toe"},
        {**good, "bogus": 1},
        {"finger": "index", "mode": "active", "reps": 3},
        {**good, "cap_n": 1.0},                    # both caps given
        {**good, "mode": "passive", "rom_target_m": -0.01},
    ]
    protocols = [
        {"ceiling_gf": 600, "blocks": [good]},
        {"ceiling_gf": 100, "blocks": [good]},     # cap at or above the ceiling
        {"ceiling_n": 5.0, "blocks": []},
        {"ceiling_n": 5.0, "ceiling_gf": 500, "blocks": [good]},
        {"ceiling_n": 5.0, "blocks": [good, {**good, "finger": "ring"}], "name": "x"},
        {"ceiling_n": 5.0, "blocks": [good], "extra": 1},
    ]
    return {
        "reps": [_rep_scenario(rng) for _ in range(120)],
        "simulation": [
            {"mode": mode, "reps": reps, "capGf": cap, "dropout": dropout}
            for mode in ("active", "passive") for reps in (1, 3) for cap in (100, 300)
            for dropout in (False, True)
        ],
        "excursion": [_excursion_trajectory(rng) for _ in range(40)],
        "tipMinusWrist": [{"points": _points(rng), "finger": f}
                          for f in ("thumb", "index", "middle", "ring", "pinky", "little")],
        "flexion": flexion,
        "modes": [{"mode": m, "phase": p.value} for m in MODES for p in Phase],
        "fingers": fingers,
        "format": formats,
        "units": [0.0, 1.0, 200.0, 450.0, 1.96, 0.001],
        "blocks": blocks,
        "protocols": protocols,
    }


# ------------------------------------------------------------------ the run
@pytest.fixture(scope="module")
def ts(tmp_path_factory) -> dict:
    """Build the ports to CommonJS, run the runner on the scenarios, return both sides' data."""
    build = tmp_path_factory.mktemp("ts_build")
    compiled = subprocess.run(
        [NODE, str(TSC), "--outDir", str(build), "--rootDir", "app/lib", "--module", "commonjs",
         "--target", "ES2021", "--strict", "--skipLibCheck", *SOURCES],
        cwd=DASHBOARD, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, f"tsc failed:\n{compiled.stdout}\n{compiled.stderr}"

    scenarios = _scenarios()
    path = build / "scenarios.json"
    path.write_text(json.dumps(scenarios))
    ran = subprocess.run([NODE, str(RUNNER), str(build), str(path)],
                         capture_output=True, text=True)
    assert ran.returncode == 0, f"runner failed:\n{ran.stderr}"
    return {"scenarios": scenarios, "out": json.loads(ran.stdout)}


def _close(a, b, tol=1e-9) -> bool:
    if a == "NaN" or b == "NaN":
        return a == b
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_the_rep_machine_agrees_step_for_step(ts):
    traces = ts["scenarios"]["reps"]
    assert any(len(t["inputs"]) > 50 for t in traces)
    seen_phases: set[str] = set()
    seen_notes: set[str] = set()
    for scenario, got in zip(traces, ts["out"]["reps"]):
        want = _replay_in_python(scenario)
        assert len(got) == len(want)
        for i, (g, w) in enumerate(zip(got, want)):
            assert g == w, f"step {i} of {scenario['block']}: ts={g} py={w}"
            seen_phases.add(w["phase"])
            seen_notes.add(w["note"].split(",")[0])
    # The traces must really exercise the machine, or agreement proves nothing.
    assert {"armed", "cueing", "ramp", "work", "hold", "reset", "set_done", "aborted"} <= seen_phases
    assert any("repetitions done" in n for n in seen_notes)
    assert any("holding still rather than pulling blind" in n for n in seen_notes)


def test_excursion_fits_agree(ts):
    for vectors, got in zip(ts["scenarios"]["excursion"], ts["out"]["excursion"]):
        r = fit_excursion([tuple(v) for v in vectors])
        want = [r.excursion_m, *r.axis, r.variance_explained, r.perpendicular_rms_m, r.n_samples]
        assert all(_close(g, w) for g, w in zip(got, want)), f"ts={got} py={want}"


def test_tip_minus_wrist_agrees(ts):
    for case, got in zip(ts["scenarios"]["tipMinusWrist"], ts["out"]["tipMinusWrist"]):
        finger = case["finger"]
        if finger == "little":  # the tracker's name is "pinky"; both sides must refuse this one
            with pytest.raises(ValueError):
                tip_minus_wrist([tuple(p) for p in case["points"]], finger)
            assert "error" in got
            continue
        want = tip_minus_wrist([tuple(p) for p in case["points"]], finger)
        assert all(_close(g, w) for g, w in zip(got["ok"], want))


def test_flexion_angles_agree_including_degenerate_bones(ts):
    for points, got in zip(ts["scenarios"]["flexion"], ts["out"]["flexion"]):
        want = all_flexion_deg([tuple(p) for p in points])
        assert set(got) == set(want)
        for name, value in want.items():
            assert _close(got[name], "NaN" if math.isnan(value) else value, tol=1e-9), name
    assert any(v == "NaN" for g in ts["out"]["flexion"] for v in g.values())


def test_modes_and_instructions_agree(ts):
    for case, got in zip(ts["scenarios"]["modes"], ts["out"]["modes"]):
        mode, phase = case["mode"], Phase(case["phase"])
        plan = plan_for(MODES[mode], phase)
        assert got == {
            "driver": plan.driver.value, "law": plan.law.value, "target": plan.target.value,
            "device_driven": plan.device_driven,
            "key": instruction_key(mode, phase),
            "text": instruction_for(mode, phase, "index"),
        }, (mode, phase)


def test_finger_names_agree(ts):
    for name, got in zip(ts["scenarios"]["fingers"], ts["out"]["fingers"]):
        try:
            assert got == {"ok": normalise(name)}
        except ValueError:
            assert "error" in got, name


def test_fixed_point_formatting_matches_python_including_ties(ts):
    for (x, places), got in zip(ts["scenarios"]["format"], ts["out"]["format"]):
        assert got == f"{x:.{places}f}", (x, places)


def test_units_agree_and_the_constant_is_the_same(ts):
    for x, (n, gf) in zip(ts["scenarios"]["units"], ts["out"]["units"]):
        assert _close(n, grams_force_to_newtons(x))
        assert _close(gf, newtons_to_grams_force(x))
    source = (DASHBOARD / "app" / "lib" / "units.ts").read_text()
    assert f"STANDARD_GRAVITY_M_S2 = {STANDARD_GRAVITY_M_S2};" in source


def test_blocks_and_protocols_accept_and_refuse_the_same_things(ts):
    for raw, got in zip(ts["scenarios"]["blocks"], ts["out"]["blocks"]):
        try:
            want = Block.from_dict(raw).to_dict()
        except ValueError:
            assert "error" in got, raw
            continue
        assert "ok" in got, (raw, got)
        want.setdefault("rom_target_m", None)
        assert got["ok"].keys() == want.keys()
        assert all(_close(got["ok"][k], want[k]) if isinstance(want[k], float) else got["ok"][k] == want[k]
                   for k in want), (raw, got, want)
    for raw, got in zip(ts["scenarios"]["protocols"], ts["out"]["protocols"]):
        try:
            p = Protocol.from_dict(raw)
        except ValueError:
            assert "error" in got, raw
            continue
        assert "ok" in got, (raw, got)
        assert got["ok"]["blocks"] == len(p) and _close(got["ok"]["ceiling_n"], p.ceiling_n)


def test_the_simulated_patient_completes_sets_and_the_deadman_stops_a_blind_pull(ts):
    """What the dashboard's rep machine panel runs: whole sets, and a lost reading mid-pull."""
    for case, got in zip(ts["scenarios"]["simulation"], ts["out"]["simulation"]):
        assert got["finished"], case
        assert 0.0 <= got["flexMin"] <= got["flexMax"] <= 1.0, case
        if not case["dropout"]:
            assert got["phase"] == "set_done", (case, got)
            assert got["repsCompleted"] == case["reps"], (case, got)
            assert got["attemptsAbandoned"] == 0, (case, got)
        else:
            # Every mode has a device-driven beat, so losing the force there must stop the block
            # rather than let the arm keep pulling with no idea how hard.
            assert got["phase"] == "aborted", (case, got)
            assert "while the arm was moving the finger" in got["note"], (case, got)
