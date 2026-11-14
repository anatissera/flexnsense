export type Finger = "thumb" | "index" | "middle" | "ring" | "little";
export const FINGERS: Finger[] = ["thumb", "index", "middle", "ring", "little"];
export type Hand = "injured" | "contralateral";
/** full = whole-finger curl; fds = DIP-blocked curl isolating FDS (PIP flexion). */
export type TestKind = "full" | "fds";

export interface Trial {
  patientId: string;
  finger: Finger;
  hand: Hand;
  session: number; // weeks since surgery
  trialIndex: number;
  test: TestKind;
  fs: number;
  t: Float64Array;
  x: Float64Array; // mm
  F: Float64Array; // N
}

/** MyoHand actuator names for the flexor tendons driving each finger. */
export const FLEXOR_MUSCLES: Record<Finger, { fdp: string | null; fds: string | null; ext: string[]; label: string }> = {
  thumb: { fdp: "FPL_r", fds: null, ext: ["EPL_r", "EPB_r", "APL_r"], label: "FPL" },
  index: { fdp: "FDP2_r", fds: "FDS2_r", ext: ["EDC2_r", "EIP_r"], label: "FDP/FDS" },
  middle: { fdp: "FDP3_r", fds: "FDS3_r", ext: ["EDC3_r"], label: "FDP/FDS" },
  ring: { fdp: "FDP4_r", fds: "FDS4_r", ext: ["EDC4_r"], label: "FDP/FDS" },
  little: { fdp: "FDP5_r", fds: "FDS5_r", ext: ["EDC5_r", "EDM_r"], label: "FDP/FDS" },
};

export const SESSIONS = [2, 4, 6, 8, 12, 16, 24];
export const HAND_LENGTH_MM = 190;
