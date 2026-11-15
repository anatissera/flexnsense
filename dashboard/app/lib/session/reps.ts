/**
 * The per-repetition machine: cue the patient, and count what they actually did.
 *
 * Mirror of `session/reps.py`, which is the tested side and carries the full reasoning (read its
 * docstrings before changing a threshold or a transition). `tests/session/test_ts_port.py` feeds
 * both machines the same randomised input traces and fails if they ever disagree.
 *
 * One skeleton for both modes. The mode fills in which law runs in the work and reset beats and
 * who drives each (`modes.ts`), and nothing else differs.
 *
 * Pure: no I/O, no hardware, and the clock arrives as a number. It reports which `Law` the arm
 * should be running and leaves commanding it to the caller.
 *
 * IMPORTANT for whoever wires this into the UI: the browser has no force gauge and no arm, so
 * this is for display, rehearsal and simulation. The machine that decides when a real arm
 * pulls a real finger is the Python one, behind `control.safety.SafetyGate`, and the dashboard
 * is a view of it (see `sessionSnapshot.ts`). Do not drive hardware from this copy.
 *
 * The safety contribution of the machine is the deadman on a device-driven phase: while the arm
 * is pulling the finger, advancing requires a fresh, confident reading on the current cycle, so
 * a single missing frame stops the advance.
 */
import { formatFixed } from "./format";
import { instructionFor } from "./instructions";
import {
  deviceDriven,
  planFor,
  type Driver,
  type Law,
  type Phase,
  type PhasePlan,
} from "./modes";
import { blockCapGf, blockModeSpec, type Block } from "./protocol";

/** What counts as started, finished, and still. Timings a clinician may want to change. */
export type RepLimits = {
  /** Where ARMED leaves the tether between repetitions, as a fraction of the block's cap. */
  pretensionFraction: number;
  /** Force rise above the rep's baseline that counts as the patient having started, newtons. */
  startDeltaN: number;
  /** Minimum time an instruction is shown before a device-driven phase starts moving, seconds. */
  cueDwellS: number;
  /** How long to wait for a patient-driven phase to start before giving up on the rep, seconds. */
  cueTimeoutS: number;
  /** Movement smaller than this does not count as advancing, metres. */
  stillM: number;
  /** How long travel has to stay within `stillM` before the patient is at end of range, seconds. */
  settleS: number;
  /** How close to the rep's start position counts as back, metres. */
  homeM: number;
  /** How long a device-driven phase may go without a usable reading before abandoning, seconds. */
  blindGraceS: number;
  /** Longest a single phase may last before the repetition is abandoned, seconds. */
  phaseTimeoutS: number;
};

export const DEFAULT_REP_LIMITS: RepLimits = {
  pretensionFraction: 0.25,
  startDeltaN: 0.15,
  cueDwellS: 1.5,
  cueTimeoutS: 20.0,
  stillM: 0.002,
  settleS: 0.7,
  homeM: 0.004,
  blindGraceS: 0.3,
  phaseTimeoutS: 60.0,
};

const LIMIT_KEYS: [keyof RepLimits, string][] = [
  ["pretensionFraction", "pretension_fraction"],
  ["startDeltaN", "start_delta_n"],
  ["cueDwellS", "cue_dwell_s"],
  ["cueTimeoutS", "cue_timeout_s"],
  ["stillM", "still_m"],
  ["settleS", "settle_s"],
  ["homeM", "home_m"],
  ["blindGraceS", "blind_grace_s"],
  ["phaseTimeoutS", "phase_timeout_s"],
];

/** Fill in defaults and validate, with the same checks as `RepLimits.__post_init__`. */
export function makeRepLimits(overrides: Partial<RepLimits> = {}): RepLimits {
  const limits = { ...DEFAULT_REP_LIMITS, ...overrides };
  for (const [key, pyName] of LIMIT_KEYS) {
    if (!(limits[key] > 0)) throw new Error(`${pyName} must be positive, got ${limits[key]}`);
  }
  if (limits.pretensionFraction >= 1.0) {
    throw new Error(
      `pretension_fraction must be below 1.0, got ${limits.pretensionFraction}: resting at or ` +
        "above the cap would hold the finger at the therapeutic force between repetitions",
    );
  }
  return limits;
}

/**
 * One cycle's worth of what the machine can observe.
 *
 * `forceN` is null when the gauge reader did not see the marker. It is never zero in that case,
 * and this module never treats it as zero.
 */
export type RepInput = {
  /** Newtons, or null when the force cannot be seen this cycle. */
  forceN: number | null;
  /** Seconds since the frame this reading came from. */
  readingAgeS: number;
  /** Displacement since the repetition began, metres. Signed, positive in the slack direction. */
  travelM: number;
  /** Monotonic seconds. Passed in rather than read, so the machine stays pure. */
  now: number;
  /** Whether the reading is trustworthy. False is treated exactly like a missing reading. */
  confident?: boolean;
};

const usable = (inp: RepInput) => inp.forceN !== null && (inp.confident ?? true);

/** What the machine decided this cycle. */
export type RepStep = {
  phase: Phase;
  /** What the arm should be doing now. */
  law: Law;
  driver: Driver;
  instruction: string;
  /** 1-based index of the repetition in progress, or the total once the set is done. */
  repIndex: number;
  repsCompleted: number;
  /** Why this cycle looks like it does. For the session log. */
  note: string;
  /** Set once the block is over, whether every repetition was completed or not. */
  finished: boolean;
  /** Attempts that started but did not complete. */
  attemptsAbandoned: number;
};

/** Whether the arm should be moving at all this cycle. */
export const mayAdvance = (step: RepStep) => step.law === "yield" || step.law === "pull";

type PhaseState = {
  enteredAt: number;
  baselineN: number | null;
  /** Furthest travel reached in this phase, for deciding the patient has stopped advancing. */
  peakTravelM: number;
  lastMovedAt: number;
  blindSince: number | null;
};

const fixed = formatFixed;

export class RepMachine {
  readonly block: Block;
  readonly limits: RepLimits;
  readonly pretensionN: number;
  phase: Phase = "idle";
  repsCompleted = 0;
  attemptsAbandoned = 0;

  private st: PhaseState = { enteredAt: 0, baselineN: null, peakTravelM: 0, lastMovedAt: 0, blindSince: null };
  /** Whether the work beat of the current attempt actually reached the cap. */
  private reachedWork = false;

  constructor(block: Block, limits?: Partial<RepLimits>) {
    this.block = block;
    this.limits = makeRepLimits(limits);
    this.pretensionN = block.capN * this.limits.pretensionFraction;
    const headroom = block.capN - this.pretensionN;
    if (headroom <= this.limits.startDeltaN) {
      throw new Error(
        `a ${fixed(blockCapGf(block), 0)} gf cap rests at ${fixed(this.pretensionN, 2)} N and ` +
          `leaves only ${fixed(headroom, 2)} N below the cap, which is not more than the ` +
          `${fixed(this.limits.startDeltaN, 2)} N rise that counts as the patient starting: ` +
          "the repetition would reach the cap at the same instant it began, so RAMP would have " +
          "no duration and the patient would be at the therapeutic force before being cued. " +
          "The cap is too close to the gauge reader's noise floor for this exercise to be " +
          "measurable; raise the cap or lower start_delta_n only if the reader is actually " +
          "quieter than it assumes.",
      );
    }
  }

  // ------------------------------------------------------------------ state
  /** 1-based index of the repetition in progress, capped at the block's total. */
  get repIndex(): number {
    return Math.min(this.repsCompleted + 1, this.block.reps);
  }

  get finished(): boolean {
    return this.phase === "set_done" || this.phase === "aborted";
  }

  get plan(): PhasePlan {
    return planFor(blockModeSpec(this.block), this.phase);
  }

  // ------------------------------------------------------------------ the one entry point
  /** Advance one cycle. */
  step(inp: RepInput): RepStep {
    if (this.finished) return this.emit("the block is over", undefined, true);

    if (this.phase === "idle") return this.enter("armed", inp, "taking up the slack");

    // A device-driven phase may not advance on absent evidence. Checked before anything else,
    // because every other decision below could otherwise command a pull.
    const deadman = this.deadman(inp);
    if (deadman !== null) return deadman;

    switch (this.phase) {
      case "armed":
        return this.armed(inp);
      case "cueing":
        return this.cueing(inp);
      case "ramp":
        return this.ramp(inp);
      case "work":
        return this.work(inp);
      case "hold":
        return this.hold(inp);
      case "reset":
        return this.reset(inp);
      default:
        return this.emit("waiting");
    }
  }

  /**
   * Stop the block. Terminal, and never refused. A latched RELEASE from the safety gate, a human
   * stopping, and the orchestrator stopping the session all arrive here. The repetition in
   * progress is counted as an abandoned attempt rather than silently dropped.
   */
  abandon(reason: string): RepStep {
    if (!this.finished && this.phase !== "idle" && this.phase !== "armed") {
      this.attemptsAbandoned += 1;
    }
    this.phase = "aborted";
    return this.emit(reason, undefined, true);
  }

  // ------------------------------------------------------------------ the deadman
  /**
   * Stop a device-driven advance the moment the force stops being visible. A patient-driven
   * phase tolerates a dropout, because a lost reading there leaves the arm static. In a
   * device-driven phase the arm is pulling the finger, so a lost reading would leave it pulling
   * with no idea how hard. Holds still, and abandons the attempt if the reading does not return.
   */
  private deadman(inp: RepInput): RepStep | null {
    if (!deviceDriven(this.plan)) {
      this.st.blindSince = null;
      return null;
    }
    const stale = inp.readingAgeS > this.limits.blindGraceS;
    if (usable(inp) && !stale) {
      this.st.blindSince = null;
      return null;
    }

    const why = !usable(inp)
      ? "no usable force reading"
      : `force reading ${fixed(inp.readingAgeS * 1000, 0)} ms old`;
    if (this.st.blindSince === null) {
      this.st.blindSince = inp.now;
    } else if (inp.now - this.st.blindSince > this.limits.blindGraceS) {
      return this.abandon(
        `${why} for more than ${fixed(this.limits.blindGraceS * 1000, 0)} ms while the arm ` +
          "was moving the finger",
      );
    }
    // Hold still, not release: the gate owns releasing, and this is a pause of the advance.
    return this.emit(`${why}; holding still rather than pulling blind`, "still");
  }

  // ------------------------------------------------------------------ phases
  /** Take the slack up to just under the cap, so the patient's range goes into the exercise. */
  private armed(inp: RepInput): RepStep {
    if (this.timedOut(inp)) return this.abandon("could not take up the slack in time");
    if (!usable(inp)) {
      return this.emit("waiting for a force reading before taking up slack", "still");
    }
    const force = inp.forceN as number;
    if (force >= this.pretensionN) return this.enter("cueing", inp, "pre-tensioned, cueing the patient");
    return this.emit(`taking up slack, ${fixed(force, 2)} of ${fixed(this.pretensionN, 2)} N`);
  }

  /** Show the instruction and wait for the phase's driver to act. */
  private cueing(inp: RepInput): RepStep {
    const elapsed = inp.now - this.st.enteredAt;
    if (blockModeSpec(this.block).work.driver === "device") {
      if (elapsed < this.limits.cueDwellS) {
        return this.emit(
          `instruction shown for ${fixed(elapsed, 1)}s of ${fixed(this.limits.cueDwellS, 1)}s ` +
            "before the arm moves",
        );
      }
      return this.enter("ramp", inp, "the arm begins to move the finger");
    }

    if (elapsed > this.limits.cueTimeoutS) {
      // Not an abort. A patient who cannot move the finger is a clinical finding.
      return this.abandonAttempt(
        inp,
        `the patient did not start within ${fixed(this.limits.cueTimeoutS, 0)}s`,
      );
    }
    if (usable(inp) && this.st.baselineN !== null) {
      const force = inp.forceN as number;
      if (force - this.st.baselineN >= this.limits.startDeltaN) {
        return this.enter("ramp", inp, "the patient started moving");
      }
    }
    return this.emit("waiting for the patient to start");
  }

  /** Force rising toward the cap. The run-up, kept separate from the exercise. */
  private ramp(inp: RepInput): RepStep {
    if (this.timedOut(inp)) return this.abandonAttempt(inp, "the force never reached the cap");
    if (usable(inp)) {
      if ((inp.forceN as number) >= this.block.capN) {
        this.reachedWork = true;
        return this.enter("work", inp, "at the cap");
      }
    }
    return this.emit("force rising toward the cap");
  }

  /** The exercise. Ends when the mover has gone as far as it is going to. */
  private work(inp: RepInput): RepStep {
    if (this.timedOut(inp)) return this.abandonAttempt(inp, "the work phase did not end in time");
    this.trackMovement(inp);

    if (this.plan.target === "rom") {
      const target = this.block.romTargetM;
      if (target === null) throw new Error("a ROM-targeted phase requires rom_target_m");
      if (Math.abs(inp.travelM) >= target) {
        return this.enter("hold", inp, `reached the ${fixed(target * 1000, 0)} mm target`);
      }
      return this.emit(
        `stretching, ${fixed(Math.abs(inp.travelM) * 1000, 0)} of ${fixed(target * 1000, 0)} mm`,
      );
    }

    // Patient-driven: they are done when they stop advancing, which is velocity falling away
    // rather than a clock running out.
    if (inp.now - this.st.lastMovedAt >= this.limits.settleS) {
      return this.enter("hold", inp, "the patient stopped advancing");
    }
    return this.emit(`following the patient, ${fixed(Math.abs(inp.travelM) * 1000, 0)} mm`);
  }

  private hold(inp: RepInput): RepStep {
    if (inp.now - this.st.enteredAt >= this.block.holdS) {
      return this.enter("reset", inp, "hold done, returning");
    }
    const held = inp.now - this.st.enteredAt;
    return this.emit(`holding, ${fixed(held, 1)}s of ${fixed(this.block.holdS, 1)}s`);
  }

  /** Back to where the repetition started, under whichever law the mode's reset beat uses. */
  private reset(inp: RepInput): RepStep {
    if (this.timedOut(inp)) {
      return this.abandonAttempt(inp, "did not return to the start position in time");
    }
    if (Math.abs(inp.travelM) <= this.limits.homeM) return this.completeRep(inp);
    return this.emit(`returning, ${fixed(Math.abs(inp.travelM) * 1000, 0)} mm from the start`);
  }

  // ------------------------------------------------------------------ counting
  /**
   * Count a repetition. Only ever called once the cap was actually reached: `reset` is reachable
   * only through `hold`, which is reachable only through `work`, and `work` is entered only when
   * the force reaches the cap. The guarantee is a property of the state graph, so it is asserted.
   */
  private completeRep(inp: RepInput): RepStep {
    if (!this.reachedWork) {
      throw new Error(
        "reached REP_DONE without passing through WORK; the phase graph is supposed to make " +
          "this unreachable, so a new transition has broken the counting guarantee",
      );
    }
    this.repsCompleted += 1;
    this.reachedWork = false;
    if (this.repsCompleted >= this.block.reps) {
      this.phase = "set_done";
      return this.emit(`${this.repsCompleted} of ${this.block.reps} repetitions done`, undefined, true);
    }
    return this.enter(
      "cueing",
      inp,
      `repetition ${this.repsCompleted} of ${this.block.reps} done`,
    );
  }

  /**
   * Give up on this repetition and move on, without stopping the block. Distinct from
   * `abandon`: nothing unsafe happened, the patient just did not complete the movement.
   */
  private abandonAttempt(inp: RepInput, why: string): RepStep {
    this.attemptsAbandoned += 1;
    this.reachedWork = false;
    const attempted = this.repsCompleted + this.attemptsAbandoned;
    if (attempted >= this.block.reps) {
      this.phase = "set_done";
      return this.emit(`${why}; ending the set`, undefined, true);
    }
    return this.enter("cueing", inp, `${why}; moving to the next attempt`);
  }

  // ------------------------------------------------------------------ helpers
  private enter(phase: Phase, inp: RepInput, note: string): RepStep {
    this.phase = phase;
    this.st = {
      enteredAt: inp.now,
      baselineN: usable(inp) ? inp.forceN : null,
      peakTravelM: Math.abs(inp.travelM),
      lastMovedAt: inp.now,
      blindSince: null,
    };
    return this.emit(note);
  }

  private trackMovement(inp: RepInput): void {
    const travel = Math.abs(inp.travelM);
    if (travel > this.st.peakTravelM + this.limits.stillM) {
      this.st.peakTravelM = travel;
      this.st.lastMovedAt = inp.now;
    }
  }

  private timedOut(inp: RepInput): boolean {
    return inp.now - this.st.enteredAt > this.limits.phaseTimeoutS;
  }

  private emit(note: string, law?: Law, finished = false): RepStep {
    const plan = this.plan;
    return {
      phase: this.phase,
      law: law ?? plan.law,
      driver: plan.driver,
      instruction: instructionFor(this.block.mode, this.phase, this.block.finger),
      repIndex: this.repIndex,
      repsCompleted: this.repsCompleted,
      note,
      finished: finished || this.finished,
      attemptsAbandoned: this.attemptsAbandoned,
    };
  }
}
