/**
 * SessionSnapshot: the one shape the session sends the dashboard, identical for every mode.
 *
 * `session/snapshot.py` is the Python mirror of this file and the one that is unit tested
 * (this repo has no JS test runner). Change both together, and keep `SCHEMA_VERSION` in step.
 *
 * The dashboard is a view and a command surface, not the brain. Session state lives in Python:
 * the browser can be closed, and the patient-facing screen must not be the thing deciding when
 * to stop pulling on a finger. So nothing here is computed locally from a timer.
 *
 * `driver` is the field that does the work. It is the only thing the UI needs to decide whether
 * the guide hand is a DEMAND ("do this now") or a PREVIEW ("this is about to happen to you"),
 * which is a different animation and a different colour but the same panel. Everything else
 * renders the same way in both modes, so a third exercise mode changes the Python instruction
 * table and nothing in here.
 *
 * `driver` sits next to `phase` rather than inside `block`, because it flips within a
 * repetition: in `active` the patient drives the work beat and the device drives the reset; in
 * `passive` it is the other way round.
 *
 * Units: `forceN` is newtons (what the control math runs on), `forceGf` is grams-force (what the
 * PESOLA is graduated in and what the clinician's cap is set in, so it is the number a human
 * compares against). Distances arrive in millimetres, already converted for display.
 *
 * What this replaces: GuideHand.tsx currently runs its own `performance.now()` timeline over
 * hardcoded REPS, HOLD_S and RELEASE_S. That is why what the patient hears cannot be aligned
 * with what the arm is doing, by construction. Rendering this snapshot instead makes the
 * instruction, the animation and the force the same state by definition.
 */

export const SESSION_SCHEMA_VERSION = 1;

/** Who is the source of motion right now. See the note above: this drives the whole display. */
export type Driver = "patient" | "device";

/**
 * What the arm is doing in this phase.
 *
 * `yield` moves only once the force is over the cap, in the direction that reduces it.
 * `pull` advances while the force is UNDER the cap, toward a position target, and stops on
 * reaching the cap. They are mirror images and are not interchangeable: a `pull` phase is the
 * one where the arm can load the finger.
 */
export type Law = "yield" | "pull" | "still" | "release";

/**
 * One repetition, start to finish.
 *
 * `ramp` and `work` run the same law; they are separate because `ramp` is the stretch it takes
 * to reach the cap (tens of millimetres on a soft tether) and `work` is the part that is
 * actually the exercise. The force-against-displacement curve wants them apart.
 */
export type Phase =
  | "idle"
  | "armed"
  | "cueing"
  | "ramp"
  | "work"
  | "hold"
  | "reset"
  | "rep_done"
  | "set_done"
  | "aborted";

export type ExerciseMode = "active" | "passive" | "cpm";

/** The block being exercised: one finger, one mode, a number of repetitions. */
export type SnapshotBlock = {
  finger: "thumb" | "index" | "middle" | "ring" | "little";
  mode: ExerciseMode;
  reps: number;
  /** The clinician's force cap, newtons. */
  capN: number;
  /** The same cap in grams-force, for display next to the PESOLA. */
  capGf: number;
  holdS: number;
  /** How far a device-driven pull may stretch the finger. Null for a mode that never pulls. */
  romTargetMm: number | null;
};

export type SessionSnapshot = {
  schema: number;
  phase: Phase;
  driver: Driver;
  law: Law;
  /** True when the arm, not the patient, is moving the finger. Render a preview, not a demand. */
  deviceDriven: boolean;
  /**
   * What to show and speak. `key` is stable per (mode, phase) so a client can memoise or
   * localise without parsing the text.
   */
  instruction: { key: string; text: string };
  rep: { index: number; of: number };
  blockIndex: { index: number; of: number };
  /** Null before the session starts. */
  block: SnapshotBlock | null;
  /**
   * Latest force in newtons, or null when the marker was not seen.
   *
   * Null is NOT zero. Zero means "no tension measured"; null means "we cannot see the force".
   * Never substitute one for the other, and never hold a last-known value forward: a stale
   * reading is what the safety gate exists to refuse.
   */
  forceN: number | null;
  /** The same reading in grams-force, or null. */
  forceGf: number | null;
  /** Arm travel since this repetition began. Signed, positive in the slack direction. */
  travelMm: number;
  /** 0 to 1 through the current phase where meaningful, else null. For a bar, not a decision. */
  progress: number | null;
  /** Set once the session has stopped, with why. Drives the clinician's banner. */
  stoppedReason: string | null;
};

/** Phases where a repetition is running, so where a cue is on screen. */
export const RUNNING_PHASES: readonly Phase[] = [
  "armed",
  "cueing",
  "ramp",
  "work",
  "hold",
  "reset",
] as const;

export const isRunning = (s: SessionSnapshot) => RUNNING_PHASES.includes(s.phase);

/** True when the session has stopped and needs a human before anything moves again. */
export const isAborted = (s: SessionSnapshot) => s.phase === "aborted";

/**
 * How the guide hand should present this phase.
 *
 * "demand" asks the patient to move. "preview" shows what the arm is about to do to them, which
 * must look different: a patient who reads a preview as a demand will push against the arm, and
 * a force cap cannot tell that apart from a stiff joint.
 */
export const guideIntent = (s: SessionSnapshot): "demand" | "preview" | "none" => {
  if (!isRunning(s)) return "none";
  return s.driver === "patient" ? "demand" : "preview";
};
