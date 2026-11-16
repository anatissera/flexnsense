// Runs the TypeScript ports on the scenarios tests/session/test_ts_port.py generated, and prints
// what they produced as JSON. Usage: node ts_port_runner.js <compiled-dir> <scenarios.json>
//
// The compiled dir is `dashboard/app/lib` built to CommonJS by the test. NaN does not survive
// JSON, so it is written as the string "NaN" here and the Python side does the same.
const path = require("path");
const fs = require("fs");

const lib = path.resolve(process.argv[2]);
const scenarios = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

const { RepMachine } = require(path.join(lib, "session/reps"));
const { blockFromGrams, blockFromJSON, protocolFromJSON } = require(path.join(lib, "session/protocol"));
const { RepSimulation, DEMO_PASSIVE_TARGET_M } = require(path.join(lib, "session/simulation"));
const { MODES, planFor, deviceDriven, modeByName } = require(path.join(lib, "session/modes"));
const { instructionFor, instructionKey } = require(path.join(lib, "session/instructions"));
const { normaliseFinger } = require(path.join(lib, "session/fingers"));
const { formatFixed } = require(path.join(lib, "session/format"));
const { fitExcursion, tipMinusWrist } = require(path.join(lib, "excursion"));
const { allFlexionDeg } = require(path.join(lib, "handGeometry"));
const units = require(path.join(lib, "units"));

const nan = (x) => (typeof x === "number" && Number.isNaN(x) ? "NaN" : x);
const attempt = (fn) => {
  try {
    return { ok: fn() };
  } catch (e) {
    return { error: String(e && e.message ? e.message : e) };
  }
};

const out = {};

out.reps = scenarios.reps.map((sc) => {
  const machine = new RepMachine(
    {
      finger: sc.block.finger,
      mode: sc.block.mode,
      reps: sc.block.reps,
      capN: sc.block.cap_n,
      holdS: sc.block.hold_s,
      romTargetM: sc.block.rom_target_m,
    },
  );
  return sc.inputs.map((inp) => {
    const s = inp.abandon !== undefined
      ? machine.abandon(inp.abandon)
      : machine.step({
          forceN: inp.force_n,
          readingAgeS: inp.reading_age_s,
          travelM: inp.travel_m,
          now: inp.now,
          confident: inp.confident,
        });
    return {
      phase: s.phase, law: s.law, target: s.target, driver: s.driver, instruction: s.instruction,
      rep_index: s.repIndex, reps_completed: s.repsCompleted, note: s.note,
      finished: s.finished, attempts_abandoned: s.attemptsAbandoned,
    };
  });
});

// Whole sets against the simulated patient and arm, the way the dashboard's rep machine panel runs
// them. With `dropout`, the force reading is lost whenever the arm is the one moving the finger.
out.simulation = scenarios.simulation.map(({ mode, reps, capGf, dropout }) => {
  const block = blockFromGrams({
    finger: "index", mode, reps, capGf, holdS: 1.0,
    romTargetM: mode === "passive" ? DEMO_PASSIVE_TARGET_M : null,
  });
  const sim = new RepSimulation(block, 0);
  let now = 0;
  let last = null;
  let flexMin = Infinity;
  let flexMax = -Infinity;
  for (let i = 0; i < 20000; i++) {
    now += 0.033;
    const blind = dropout && last !== null && ["work", "reset"].includes(last.phase) && last.driver === "device";
    const r = sim.tick(now, blind);
    last = r.step;
    flexMin = Math.min(flexMin, r.flex);
    flexMax = Math.max(flexMax, r.flex);
    if (r.step.finished) break;
  }
  return {
    finished: last.finished, phase: last.phase, repsCompleted: last.repsCompleted,
    attemptsAbandoned: last.attemptsAbandoned, note: last.note, flexMin, flexMax,
  };
});

out.excursion = scenarios.excursion.map((vectors) => {
  const r = fitExcursion(vectors);
  return [r.excursionM, ...r.axis, r.varianceExplained, r.perpendicularRmsM, r.nSamples];
});

out.tipMinusWrist = scenarios.tipMinusWrist.map(({ points, finger }) =>
  attempt(() => tipMinusWrist(points, finger)));
out.flexion = scenarios.flexion.map((points) => {
  const r = allFlexionDeg(points);
  return Object.fromEntries(Object.entries(r).map(([k, v]) => [k, nan(v)]));
});

out.modes = scenarios.modes.map(({ mode, phase }) => {
  const plan = planFor(modeByName(mode), phase);
  return {
    driver: plan.driver, law: plan.law, target: plan.target, device_driven: deviceDriven(plan),
    key: instructionKey(mode, phase),
    text: instructionFor(mode, phase, "index"),
  };
});

out.fingers = scenarios.fingers.map((name) => attempt(() => normaliseFinger(name)));
out.format = scenarios.format.map(([x, places]) => formatFixed(x, places));
out.units = scenarios.units.map((x) => [units.gramsForceToNewtons(x), units.newtonsToGramsForce(x)]);

out.blocks = scenarios.blocks.map((raw) => attempt(() => {
  const b = blockFromJSON(raw);
  return { finger: b.finger, mode: b.mode, reps: b.reps, cap_n: b.capN, hold_s: b.holdS, rom_target_m: b.romTargetM };
}));
out.protocols = scenarios.protocols.map((raw) => attempt(() => {
  const p = protocolFromJSON(raw);
  return { ceiling_n: p.ceilingN, blocks: p.blocks.length };
}));

process.stdout.write(JSON.stringify(out));
