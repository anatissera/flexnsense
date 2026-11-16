/**
 * What the patient is told, in one table.
 *
 * Mirror of `session/instructions.py`, which is the tested side and holds the reasoning behind
 * each wording (every device-driven row says "relax", both RESET rows announce a change of
 * driver). The wording is copied verbatim and `tests/session/test_ts_port.py` fails if it drifts.
 */
import type { ExerciseMode, Phase } from "./modes";

/** `${mode}.${phase}` to the sentence the patient hears and reads. `{finger}` is substituted. */
export const INSTRUCTIONS: Record<string, string> = {
  "active.armed": "Relax your hand while the arm takes up the slack",
  "active.cueing": "Start flexing your {finger} finger, slowly",
  "active.ramp": "Keep flexing",
  "active.work": "Keep going, the arm is following you",
  "active.hold": "Hold it there",
  "active.reset": "Relax. The arm will bring your finger back",
  "passive.armed": "Relax your hand while the arm takes up the slack",
  "passive.cueing": "Relax your {finger} finger and let the arm move it",
  "passive.ramp": "Stay relaxed",
  "passive.work": "Stay relaxed, let it stretch",
  "passive.hold": "Hold here, stay relaxed",
  "passive.reset": "Now flex your {finger} finger back to the start",
  // cpm asks nothing of the patient in either beat. The reset row is the one that differs
  // most from the other modes: there is no change of driver to announce, because the arm had
  // it all along, so telling a patient to "now flex back" here would invite exactly the
  // fighting a force cap cannot distinguish from a stiff joint.
  "cpm.armed": "Relax your hand while the arm takes up the slack",
  "cpm.cueing": "Relax your {finger} finger. The arm will do the work",
  "cpm.ramp": "Stay relaxed",
  "cpm.work": "Stay relaxed, let it stretch",
  "cpm.hold": "Hold here, stay relaxed",
  "cpm.reset": "Stay relaxed. The arm is bringing your finger back",
};

/** Phases whose wording does not depend on the mode. */
export const SHARED: Partial<Record<Phase, string>> = {
  idle: "Ready when you are",
  rep_done: "Good. Next one",
  set_done: "That set is done",
  aborted: "Stop and relax your hand",
};

/** A stable identifier for one row. Shared phases key on the phase alone. */
export const instructionKey = (mode: ExerciseMode, phase: Phase) =>
  phase in SHARED ? phase : `${mode}.${phase}`;

/** The sentence to show and speak. Throws if there is no row, rather than going quiet. */
export function instructionFor(mode: ExerciseMode, phase: Phase, finger: string): string {
  const shared = SHARED[phase];
  if (shared !== undefined) return shared;
  const template = INSTRUCTIONS[`${mode}.${phase}`];
  if (template === undefined) {
    throw new Error(
      `no instruction for mode ${JSON.stringify(mode)} in phase ${JSON.stringify(phase)}. ` +
        "Every mode needs a row for every non-terminal phase.",
    );
  }
  return template.replaceAll("{finger}", finger);
}
