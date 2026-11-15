/**
 * The two exercise modes, as data rather than as branches.
 *
 * Mirror of `session/modes.py`, which is the tested side. Both modes are the same two-beat
 * repetition and differ only in who is the source of motion in each beat:
 *
 *   active    work: the patient flexes, the arm yields at constant force
 *             reset: the arm walks back to where the rep began, patient relaxed
 *   passive   work: the arm pulls the finger toward extension at constant force
 *             reset: the patient flexes back, the arm releases while they do
 *
 * The `Driver`, `Law`, `Phase` and `ExerciseMode` types are the ones `sessionSnapshot.ts`
 * already declares, so a machine here and a snapshot streamed from Python share one vocabulary.
 */
import type { Driver, ExerciseMode, Law, Phase } from "../sessionSnapshot";

export type { Driver, ExerciseMode, Law, Phase };

/** What bounds how far the arm travels in a phase, when the arm is the one moving. */
export type Target = "none" | "rep_start" | "rom";

export type PhasePlan = {
  driver: Driver;
  law: Law;
  target: Target;
};

export type Mode = {
  name: ExerciseMode;
  work: PhasePlan;
  reset: PhasePlan;
};

/**
 * True when the arm is the source of motion in this phase. A device-driven phase has to advance
 * while the force is UNDER the cap and stop on reaching it, which is the opposite of the yield
 * law, so it needs a fresh, confident reading every cycle (see `RepMachine`'s deadman).
 */
export const deviceDriven = (plan: PhasePlan) => plan.driver === "device" && plan.law === "pull";

/** Whether a prescription for this mode has to carry a stretch distance. */
export const needsRomTarget = (mode: Mode) =>
  [mode.work, mode.reset].some((p) => p.target === "rom");

export const ACTIVE: Mode = {
  name: "active",
  work: { driver: "patient", law: "yield", target: "none" },
  reset: { driver: "device", law: "pull", target: "rep_start" },
};

export const PASSIVE: Mode = {
  name: "passive",
  work: { driver: "device", law: "pull", target: "rom" },
  reset: { driver: "patient", law: "yield", target: "none" },
};

/** Every mode a prescription may name. */
export const MODES: Record<ExerciseMode, Mode> = { active: ACTIVE, passive: PASSIVE };

export function modeByName(name: string): Mode {
  if (name in MODES) return MODES[name as ExerciseMode];
  throw new Error(
    `unknown exercise mode ${JSON.stringify(name)}; known modes are ${JSON.stringify(Object.keys(MODES).sort())}`,
  );
}

/**
 * What happens in `phase` under `mode`. ARMED is the device taking up slack, CUEING reports the
 * driver of the work phase about to start (that is who is being asked to act), and RAMP returns
 * the same plan as WORK on purpose: they differ only in what they mean to the log.
 */
export function planFor(mode: Mode, phase: Phase): PhasePlan {
  if (phase === "armed") return { driver: "device", law: "pull", target: "none" };
  if (phase === "cueing") return { driver: mode.work.driver, law: "still", target: "none" };
  if (phase === "ramp" || phase === "work") return mode.work;
  if (phase === "hold") return { driver: mode.work.driver, law: "still", target: "none" };
  if (phase === "reset") return mode.reset;
  if (phase === "aborted") return { driver: "device", law: "release", target: "none" };
  return { driver: "device", law: "still", target: "none" };
}
