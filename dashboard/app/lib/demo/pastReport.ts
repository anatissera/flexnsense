/**
 * One past session, written the way the clinic would read it back.
 *
 * Deliberately the same shape as `medicalReportMd`: what was prescribed, what the patient
 * actually did, and the data-quality facts kept separate from the clinical ones. A dropout is
 * the camera's problem and never the patient's, and a report that mixes the two invites a
 * clinician to read a lost marker as a worse week.
 */

import { gramsForceToNewtons, newtonsToGramsForce } from "../units";
import { blockCapGf } from "../session/protocol";
import { planOf, type DemoPatient, type PastSession } from "./patients";

const pct = (v: number | null) => (v === null ? "not determined" : `${(v * 100).toFixed(0)}%`);

export function pastSessionMd(patient: DemoPatient, s: PastSession): string {
  const completion = s.repsPrescribed === 0 ? 0 : s.repsCounted / s.repsPrescribed;
  let ceiling = "";
  try {
    ceiling = `${newtonsToGramsForce(planOf(patient).ceilingN).toFixed(0)} gf`;
  } catch {
    ceiling = "not available";
  }
  const lines = [
    "# Session report",
    "",
    `- **Patient:** ${patient.name} (${patient.id})`,
    `- **Date:** ${s.date}`,
    `- **Week:** ${s.week}`,
    `- **Blocks:** ${s.blocks}`,
    `- **Force cap set by the clinician:** ${s.capGf} gf (${gramsForceToNewtons(s.capGf).toFixed(2)} N)`,
    `- **Session ceiling:** ${ceiling}`,
    "",
    "## What the patient did",
    "",
    `**${s.repsCounted} of ${s.repsPrescribed} prescribed repetitions counted (${(completion * 100).toFixed(0)}%).** `
      + `${s.repsAbandoned} abandoned.`,
    "",
    "A repetition is counted only when the force actually reached the cap. One that stopped short is",
    "recorded as abandoned, which is a finding about the finger and not a failure of the session.",
    "",
    `- Peak force reached: ${s.peakGf} gf`,
    `- Median travel per repetition: ${s.romMm.toFixed(1)} mm`,
    `- Recovery index against the other hand: ${pct(s.recoveryIndex)}`,
    "",
    "## Data quality",
    "",
    s.dropoutS > 0
      ? `The force reading was missing for ${s.dropoutS.toFixed(1)} s in total. While there is no reading the `
        + "arm does not pull and nothing is counted: the last value is never held. Time lost to this is a "
        + "property of the camera's view of the scale, not of the patient."
      : "No gaps in the force reading.",
    "",
    "## Clinician note",
    "",
    s.note,
    "",
    "---",
    "",
    "Demonstration data. This is an invented patient and an invented session, produced by the Flex 'N Sense",
    "dashboard with no hardware attached. Not a record, and not for clinical decisions.",
  ];
  return lines.join("\n");
}

/** The session the plan would produce if it ran today, for comparing intent against history. */
export function plannedSummary(patient: DemoPatient): string {
  try {
    const plan = planOf(patient);
    const caps = plan.blocks.map((b) => Math.round(blockCapGf(b)));
    return `${plan.blocks.length} blocks, caps ${Math.min(...caps)} to ${Math.max(...caps)} gf`;
  } catch {
    return "plan not available";
  }
}
