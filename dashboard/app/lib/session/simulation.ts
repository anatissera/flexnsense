/**
 * A simulated patient and arm for rehearsing the rep machine with no hardware.
 *
 * The plant only reacts to what the machine decides, the way a real arm and finger would: it
 * takes up slack when told to, flexes when it is the patient's move, pulls when it is the arm's,
 * and stops moving when the machine pauses a device-driven phase. Nothing here is measured, the
 * rates are chosen to make a rep take a few seconds, and nothing here can move a real arm.
 *
 * Pure: the clock arrives as a number, so it runs the same under a test as in the browser.
 */
import type { Block } from "./protocol";
import { RepMachine, type RepStep } from "./reps";

/** Rates, in newtons or metres per second. Made up for the demo, not taken from a rig. */
export const FORCE_RATE_N_S = 1.0;
export const ACTIVE_FLEX_M_S = 0.012;
export const ACTIVE_FLEX_MAX_M = 0.025;
export const PASSIVE_PULL_M_S = 0.015;
export const RETURN_M_S = 0.02;
/** How long the simulated patient takes to react to "start flexing". */
export const PATIENT_REACTION_S = 1.0;
/** A passive block's stretch target, for building a demo block. */
export const DEMO_PASSIVE_TARGET_M = 0.02;

export type SimReading = {
  step: RepStep;
  /** Null while a gauge dropout is being simulated. */
  forceN: number | null;
  travelM: number;
  /** 0 = relaxed, 1 = fully flexed, for driving a hand model. */
  flex: number;
};

export class RepSimulation {
  readonly machine: RepMachine;
  private force = 0;
  private travel = 0;
  private phase = "idle";
  /** The force when the current cue began: the patient's push is measured from here. */
  private cueBase = 0;
  private since: number;
  private last: number;
  private readonly fullTravelM: number;

  constructor(block: Block, now: number) {
    this.machine = new RepMachine(block);
    this.since = now;
    this.last = now;
    this.fullTravelM = block.mode === "active" ? ACTIVE_FLEX_MAX_M : (block.romTargetM ?? DEMO_PASSIVE_TARGET_M);
  }

  /** Advance to `now` (seconds). With `dropout`, the machine is told the force cannot be seen. */
  tick(now: number, dropout = false): SimReading {
    const m = this.machine;
    const dt = Math.min(Math.max(now - this.last, 0), 0.1);
    this.last = now;
    const cap = m.block.capN;

    const step = m.step({ forceN: dropout ? null : this.force, readingAgeS: 0.01, travelM: this.travel, now });

    if (step.phase !== this.phase) {
      this.phase = step.phase;
      this.since = now;
      if (step.phase === "cueing") {
        this.travel = 0; // a new repetition starts from the rep's origin
        this.cueBase = this.force;
      }
    }
    const t = now - this.since;
    // A device-driven phase the machine has paused (no usable reading) must not move the finger.
    const paused = step.driver === "device" && step.law === "still";
    // Never lowers the force: a target below the current force leaves it where it is.
    const rise = (to: number) => { this.force = Math.max(this.force, Math.min(this.force + FORCE_RATE_N_S * dt, to)); };
    const fall = (to: number) => { this.force = Math.max(this.force - FORCE_RATE_N_S * dt, to); };

    switch (step.phase) {
      case "armed":
        if (!paused) rise(m.pretensionN * 1.3);
        break;
      case "cueing":
        if (step.driver === "patient" && t > PATIENT_REACTION_S) rise(this.cueBase + 1);
        break;
      case "ramp":
        rise(cap * 1.02);
        break;
      case "work":
        this.force = cap;
        if (!paused) {
          if (m.block.mode === "active") this.travel = Math.min(this.travel + ACTIVE_FLEX_M_S * dt, ACTIVE_FLEX_MAX_M);
          else this.travel += PASSIVE_PULL_M_S * dt;
        }
        break;
      case "hold":
        this.force = cap;
        break;
      case "reset":
        if (!paused) this.travel = Math.max(this.travel - RETURN_M_S * dt, 0);
        fall(m.pretensionN);
        break;
      default:
        this.force = Math.max(this.force - 2 * FORCE_RATE_N_S * dt, 0);
    }

    return {
      step,
      forceN: dropout ? null : this.force,
      travelM: this.travel,
      flex: Math.min(Math.max(this.travel / this.fullTravelM, 0), 1),
    };
  }
}
