/**
 * What is real on this page and what is not, in one list.
 *
 * The banner and the opening overlay both render this, so there is exactly one answer to the
 * question and it cannot drift between the two places a visitor might read it. The split is
 * deliberately three ways rather than two: "simulated" is not the opposite of "real" here. The
 * repetition machine's decisions are the real ones, running against a simulated finger, and
 * saying so is more honest (and more interesting) than calling the whole page a mock-up.
 */

export type Reality =
  /** Genuinely running in front of you, and the same code the rig runs. */
  | "real"
  /** A number on screen that came from a model, not from a measurement. */
  | "simulated"
  /** Exists only at the bench, with the hardware attached. */
  | "bench";

export type InventoryItem = {
  reality: Reality;
  what: string;
  detail: string;
};

export const REALITY_LABEL: Record<Reality, string> = {
  real: "Real, right here",
  simulated: "Simulated",
  bench: "Only at the bench",
};

export const INVENTORY: readonly InventoryItem[] = [
  {
    reality: "real",
    what: "The 3D hand",
    detail:
      "The MyoSim MyoHand model, stepped by MuJoCo compiled to WebAssembly in your browser. The tendons are the real ones.",
  },
  {
    reality: "real",
    what: "Hand tracking, from your laptop camera",
    detail:
      "MediaPipe runs in the page: your frames are measured and discarded, never uploaded and never stored. At the bench the same code reads the camera on the boom over the patient's wrist.",
  },
  {
    reality: "real",
    what: "The repetition machine",
    detail:
      "The cues, the phase changes, the counting and the stop conditions are the state machine that drives the arm, ported from session/reps.py. Its decisions are real; the finger and the arm it is deciding about are not.",
  },
  {
    reality: "real",
    what: "The analysis and the reports",
    detail:
      "Range of motion, rate of force development, hysteresis, the recovery index and the exported report are the real calculations, run over simulated signals.",
  },
  {
    reality: "real",
    what: "The assistant",
    detail:
      "A real model, answering about what is on screen and able to move the camera, highlight a tendon and walk you through the panels.",
  },
  {
    reality: "simulated",
    what: "Every force reading",
    detail:
      "No gauge is attached, so the newtons come from a simulated finger and arm. Nothing on this page is a measurement of anything.",
  },
  {
    reality: "simulated",
    what: "The patient",
    detail:
      "Jordan Woods, the weekly curves and the per tendon fits are one synthetic patient from a seeded generator. Not a person, not a record.",
  },
  {
    reality: "simulated",
    what: "The arm",
    detail:
      "Nothing in this page can move a robot, here or anywhere. The only path to an arm is the Python one, behind the safety gate.",
  },
  {
    reality: "bench",
    what: "The PESOLA gauge camera",
    detail:
      "A camera reading the red marker on a 600 gf spring scale at about 30 Hz. It is the only force sensor in the system: the servo current turned out to be unusable through a 345:1 gearbox.",
  },
  {
    reality: "bench",
    what: "The two ElRobot arms",
    detail:
      "One carries the tether to the finger and runs the admittance loop, the other moves the hook from one finger to the next.",
  },
  {
    reality: "bench",
    what: "WiLoR hand tracking",
    detail:
      "The precision hand model, 11 FPS on a local GPU behind its own server. MediaPipe is the live path and the one you are using.",
  },
];

export const inventoryFor = (reality: Reality) => INVENTORY.filter((i) => i.reality === reality);
