"use client";

/**
 * Two made-up patients, complete enough to show what the product is for.
 *
 * A dashboard with one anonymous synthetic patient shows the maths. It does not show the thing
 * the system is actually about: a clinician writes a prescription, the machine runs it, and the
 * next clinician reads what happened. So there are two patients here, chosen because they are the
 * two different reasons to own this machine:
 *
 *   Jordan Woods, five weeks after a flexor tendon repair. ACTIVE mode: the patient does the
 *   work and the arm gives way at the cap, because after a tendon repair the tendon has to glide
 *   under load the patient generates themselves.
 *
 *   Margrit Imboden, eighty-one, no surgery, a weak hand losing range. CPM: the arm takes the
 *   finger out and brings it back and the patient does nothing, because what helps here is many
 *   identical repetitions rather than effort.
 *
 * The prescriptions are the two files in config/protocols/, copied verbatim, in the exact JSON
 * shape `session/protocol.py` writes and reads. They are parsed through `protocolFromJSON`, the
 * same validation Python applies, so a plan that would be rejected on the rig is rejected here.
 * The reasoning behind every number in them is in config/protocols/README.md.
 *
 * Everything else, the people, the dates, the session history, is invented. Nobody's record.
 */

import { protocolFromJSON, type Protocol } from "../session/protocol";

/** The day the demo is written against, so "week 5" and the dates below agree. */
export const DEMO_TODAY = "2026-11-16";

export type PastSession = {
  /** ISO date. */
  date: string;
  /** Weeks since surgery, or since the programme began for a patient who had none. */
  week: number;
  blocks: string;
  repsPrescribed: number;
  repsCounted: number;
  repsAbandoned: number;
  /** The cap the clinician set for the session, grams-force. */
  capGf: number;
  /** Highest force actually reached, grams-force. */
  peakGf: number;
  /** Median travel per repetition, millimetres. */
  romMm: number;
  /** Against the other hand, where there was one to compare with. */
  recoveryIndex: number | null;
  /** Seconds the force reading was missing, which is a data-quality fact, not a patient one. */
  dropoutS: number;
  note: string;
};

export type DemoPatient = {
  id: string;
  name: string;
  /** The demographic block on the card. Written without pronouns, the way a note is. */
  lines: string[];
  photo: string | null;
  summary: string;
  plan: {
    uploadedBy: string;
    uploadedOn: string;
    note: string;
    /** Verbatim from config/protocols/. Parsed, not trusted. */
    json: Record<string, unknown>;
  };
  history: PastSession[];
};

/** config/protocols/tendon-repair-active.json, verbatim. */
const TENDON_REPAIR_ACTIVE = {
  name: "EXAMPLE, not a prescription: active mode, flexor tendon repair of the hand, week 5",
  ceiling_gf: 250,
  blocks: [
    { finger: "index", mode: "active", reps: 5, cap_gf: 120, hold_s: 3.0 },
    { finger: "index", mode: "active", reps: 5, cap_gf: 150, hold_s: 3.0 },
    { finger: "middle", mode: "active", reps: 5, cap_gf: 120, hold_s: 3.0 },
    { finger: "middle", mode: "active", reps: 5, cap_gf: 150, hold_s: 3.0 },
    { finger: "ring", mode: "active", reps: 4, cap_gf: 100, hold_s: 2.0 },
    { finger: "little", mode: "active", reps: 4, cap_gf: 100, hold_s: 2.0 },
  ],
};

/** config/protocols/age-related-weakness-cpm.json, verbatim. */
const AGE_RELATED_WEAKNESS_CPM = {
  name: "EXAMPLE, not a prescription: cpm mode, age-related hand weakness, maintaining range",
  ceiling_gf: 200,
  blocks: [
    { finger: "index", mode: "cpm", reps: 10, cap_gf: 110, hold_s: 1.0, rom_target_m: 0.012 },
    { finger: "middle", mode: "cpm", reps: 10, cap_gf: 110, hold_s: 1.0, rom_target_m: 0.012 },
    { finger: "ring", mode: "cpm", reps: 10, cap_gf: 95, hold_s: 1.0, rom_target_m: 0.01 },
    { finger: "little", mode: "cpm", reps: 10, cap_gf: 95, hold_s: 1.0, rom_target_m: 0.01 },
  ],
};

export const DEMO_PATIENTS: readonly DemoPatient[] = [
  {
    id: "P1",
    name: "Jordan Woods",
    lines: [
      "58 y | MRN 004-71829",
      "DOB 1967-03-14 | Right-hand dominant",
      "Dx: Flexor tendon repair, zone II (digit 2 and 3, R)",
      "Surgery 2026-10-12 | Referred by Dr. A. Meier",
    ],
    photo: "/profile_photo.png",
    summary:
      "Five weeks after a zone II repair of the index and middle flexors. The tendon has to glide "
      + "under load the patient generates, so the arm is an anchor that gives way, never a mover. "
      + "Caps step up within a session and the hold is long, which is where the gliding happens.",
    plan: {
      uploadedBy: "Dr. A. Meier, hand surgery",
      uploadedOn: "2026-11-13",
      note:
        "Week 5. Index and middle step 120 to 150 gf within the session; ring and little stay at 100 gf "
        + "and shorter holds. Stop the set if the patient reports pain at the repair site rather than "
        + "stretch, and send me the session report either way.",
      json: TENDON_REPAIR_ACTIVE,
    },
    history: [
      {
        date: "2026-10-26", week: 2, blocks: "index, middle: active",
        repsPrescribed: 12, repsCounted: 8, repsAbandoned: 4, capGf: 80, peakGf: 78, romMm: 9.4,
        recoveryIndex: 0.31, dropoutS: 2.1,
        note: "First session on the machine. Four repetitions abandoned: the patient could not reach the cap. Expected at this stage.",
      },
      {
        date: "2026-10-29", week: 2, blocks: "index, middle: active",
        repsPrescribed: 12, repsCounted: 11, repsAbandoned: 1, capGf: 80, peakGf: 80, romMm: 12.8,
        recoveryIndex: 0.36, dropoutS: 0.4,
        note: "Reached the cap on nearly every repetition. Travel up 3 mm on the index.",
      },
      {
        date: "2026-11-02", week: 3, blocks: "index, middle: active",
        repsPrescribed: 16, repsCounted: 15, repsAbandoned: 1, capGf: 100, peakGf: 101, romMm: 15.1,
        recoveryIndex: 0.44, dropoutS: 0.0,
        note: "Cap raised to 100 gf. No complaints of pain at the repair site.",
      },
      {
        date: "2026-11-05", week: 3, blocks: "index, middle, ring: active",
        repsPrescribed: 20, repsCounted: 18, repsAbandoned: 2, capGf: 100, peakGf: 103, romMm: 16.0,
        recoveryIndex: 0.48, dropoutS: 6.8,
        note: "Ring finger added. Seven seconds without a force reading: the hook shadowed the marker. Those repetitions were not counted.",
      },
      {
        date: "2026-11-09", week: 4, blocks: "index, middle, ring, little: active",
        repsPrescribed: 24, repsCounted: 23, repsAbandoned: 1, capGf: 120, peakGf: 121, romMm: 18.3,
        recoveryIndex: 0.55, dropoutS: 0.9,
        note: "All four fingers. Index travel now 18 mm, against 114 mm of healthy excursion.",
      },
      {
        date: "2026-11-12", week: 4, blocks: "index, middle, ring, little: active",
        repsPrescribed: 24, repsCounted: 24, repsAbandoned: 0, capGf: 120, peakGf: 124, romMm: 19.6,
        recoveryIndex: 0.59, dropoutS: 0.0,
        note: "Full set completed for the first time. Hold steadiness improving.",
      },
      {
        date: "2026-11-14", week: 4, blocks: "index, middle, ring, little: active",
        repsPrescribed: 24, repsCounted: 22, repsAbandoned: 2, capGf: 150, peakGf: 148, romMm: 20.1,
        recoveryIndex: 0.62, dropoutS: 1.3,
        note: "Trial of the 150 gf step before this week's plan. Two repetitions stopped short of the cap on the ring finger.",
      },
    ],
  },
  {
    id: "P2",
    name: "Margrit Imboden",
    lines: [
      "81 y | MRN 004-83104",
      "DOB 1945-01-28 | Right-hand dominant",
      "Dx: Age-related hand weakness, losing range, no surgery",
      "Programme began 2026-09-29 | Referred by geriatric rehabilitation",
    ],
    photo: null,
    summary:
      "No injury and nothing to protect: the problem is a hand that is slowly losing range and "
      + "a patient for whom effort is the scarce thing. CPM, so the arm takes each finger out to a "
      + "small stretch target and brings it back, ten times, with nothing in the repetition waiting "
      + "on a person. The cap is a ceiling rather than the exercise: a stroke that ends at the cap "
      + "means the finger was stiffer than prescribed, which is a finding and not a failed rep.",
    plan: {
      uploadedBy: "M. Brunner, geriatric rehabilitation",
      uploadedOn: "2026-11-10",
      note:
        "Twice weekly, all four fingers, 10 repetitions each. Targets stay at 10 to 12 mm: the aim is "
        + "range maintained, not gained. If a finger reaches the cap before the target on three strokes "
        + "in a row, note which finger and leave the target where it is.",
      json: AGE_RELATED_WEAKNESS_CPM,
    },
    history: [
      {
        date: "2026-10-01", week: 0, blocks: "index, middle: cpm",
        repsPrescribed: 20, repsCounted: 20, repsAbandoned: 0, capGf: 90, peakGf: 71, romMm: 8.2,
        recoveryIndex: null, dropoutS: 0.0,
        note: "Familiarisation. Two fingers only. Tolerated the arm moving the finger without resisting it, which is the thing that had to be true.",
      },
      {
        date: "2026-10-06", week: 1, blocks: "index, middle, ring, little: cpm",
        repsPrescribed: 40, repsCounted: 37, repsAbandoned: 3, capGf: 90, peakGf: 90, romMm: 8.6,
        recoveryIndex: null, dropoutS: 3.4,
        note: "Little finger reached the cap before the target on three strokes. Target left at 10 mm per the plan.",
      },
      {
        date: "2026-10-13", week: 2, blocks: "index, middle, ring, little: cpm",
        repsPrescribed: 40, repsCounted: 40, repsAbandoned: 0, capGf: 90, peakGf: 86, romMm: 9.1,
        recoveryIndex: null, dropoutS: 0.0,
        note: "All forty strokes reached the target. Less force needed than last week for the same travel.",
      },
      {
        date: "2026-10-20", week: 3, blocks: "index, middle, ring, little: cpm",
        repsPrescribed: 40, repsCounted: 38, repsAbandoned: 2, capGf: 95, peakGf: 94, romMm: 9.8,
        recoveryIndex: null, dropoutS: 11.2,
        note: "Eleven seconds without a force reading, the longest so far. Glare on the scale after the bench was moved. Two strokes not counted.",
      },
      {
        date: "2026-10-27", week: 4, blocks: "index, middle, ring, little: cpm",
        repsPrescribed: 40, repsCounted: 40, repsAbandoned: 0, capGf: 95, peakGf: 88, romMm: 10.4,
        recoveryIndex: null, dropoutS: 0.0,
        note: "Index and middle now reach 12 mm inside the cap. Ring and little unchanged.",
      },
      {
        date: "2026-11-03", week: 5, blocks: "index, middle, ring, little: cpm",
        repsPrescribed: 40, repsCounted: 39, repsAbandoned: 1, capGf: 95, peakGf: 91, romMm: 10.6,
        recoveryIndex: null, dropoutS: 1.1,
        note: "Steady. Patient reports the hand is easier in the morning.",
      },
      {
        date: "2026-11-10", week: 6, blocks: "index, middle, ring, little: cpm",
        repsPrescribed: 40, repsCounted: 40, repsAbandoned: 0, capGf: 95, peakGf: 89, romMm: 10.9,
        recoveryIndex: null, dropoutS: 0.0,
        note: "Six weeks of maintained range. Plan renewed unchanged.",
      },
    ],
  },
];

/**
 * The patient's prescription, validated. Parsed on demand rather than at module load so that a
 * mistake in one patient's plan surfaces as that patient's panel failing, not a blank page.
 */
export const planOf = (patient: DemoPatient): Protocol => protocolFromJSON(patient.plan.json);

/** Total prescribed repetitions across every block, which is what a session is measured against. */
export const plannedReps = (plan: Protocol) => plan.blocks.reduce((n, b) => n + b.reps, 0);

/** Adherence over the recorded history: counted repetitions against prescribed ones. */
export function adherence(history: readonly PastSession[]): number | null {
  const prescribed = history.reduce((n, s) => n + s.repsPrescribed, 0);
  if (prescribed === 0) return null;
  return history.reduce((n, s) => n + s.repsCounted, 0) / prescribed;
}
