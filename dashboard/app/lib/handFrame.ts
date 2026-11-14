/**
 * HandFrame: the one shape every hand tracker emits, whatever model produced it.
 *
 * MediaPipe (in-browser, see HandTracker.tsx) fills the base fields. A WiLoR backend fills the
 * same base fields plus `mano`, so anything reading only the base fields works with either
 * source and the two stay swappable.
 *
 * `hand_tracking/frame.py` is the Python mirror of this file, emitting the same JSON keys, and
 * is the one that is unit tested (this repo has no JS test runner). Change both together.
 *
 * Units: angles in radians (what both the MyoHand MuJoCo joints and MANO use), `world` in metres,
 * `tipWristMm` in millimetres, `t` in milliseconds since tracking started.
 */

export type HandSource = "mediapipe" | "wilor";

export type Vec3 = { x: number; y: number; z: number };

/** Fingertip indices in the shared 21-point layout: thumb, index, middle, ring, pinky. */
export const FINGERTIPS = [4, 8, 12, 16, 20] as const;

/**
 * The 15 MyoHand flexion joints a tracked hand drives, in MuJoCo's naming. Every one of these
 * exists in public/models/myohand/myohand_r.xml.
 *
 * The model also carries `cmc_abduction_r` and `mcp2..5_abduction_r`, which a single flexion
 * angle per joint cannot fill. A MANO source can, because its rotations are full 3 DOF: that is
 * what `ManoParams.handPose` is reserved for.
 */
export const FLEXION_JOINTS = [
  "mcp2_flexion_r", "pm2_flexion_r", "md2_flexion_r",
  "mcp3_flexion_r", "pm3_flexion_r", "md3_flexion_r",
  "mcp4_flexion_r", "pm4_flexion_r", "md4_flexion_r",
  "mcp5_flexion_r", "pm5_flexion_r", "md5_flexion_r",
  "cmc_flexion_r", "mp_flexion_r", "ip_flexion_r",
] as const;

/** Rotations and shape from a MANO-based model. Only a MANO source (WiLoR) sets this. */
export type ManoParams = {
  /**
   * 15 x 3 axis-angle rotations in radians, one per MANO hand joint, full 3 DOF each.
   *
   * Careful: these are in MANO's own joint order, which is NOT the 21-point order the rest of a
   * HandFrame uses. MANO goes index, middle, pinky, ring, thumb, three joints each, base to tip.
   * Only `landmarks` and `world` are remapped to the shared layout.
   */
  handPose: number[][];
  /** Hand orientation in camera space, axis-angle, radians. */
  globalOrient: number[];
  /** The 10 MANO shape coefficients, i.e. this subject's hand proportions. */
  betas: number[];
};

export type HandFrame = {
  /** Bumped when a field changes meaning, so a reader can reject what it cannot parse. */
  schema: 1;
  source: HandSource;
  /** Milliseconds since tracking started. */
  t: number;
  handedness: "left" | "right" | null;
  /**
   * 21 points in normalized image coordinates (0..1 of width and height). For drawing only.
   * Only x and y are contractual: z is a relative depth from MediaPipe and a camera-space depth
   * in metres from WiLoR. Read `world` for anything quantitative.
   */
  landmarks: Vec3[];
  /** 21 points in metres, hand centred. Angles come from these, never from `landmarks`. */
  world: Vec3[];
  /** Joint name to flexion angle in radians. Keys are a subset of FLEXION_JOINTS. */
  angles: Record<string, number>;
  /** Index fingertip to wrist distance, millimetres. */
  tipWristMm: number;
  mano?: ManoParams;
};

// Enough precision to be lossless in practice, short enough to log or stream: 4 places on a
// normalized coordinate is a tenth of a pixel at 1000 px, 5 places on metres is 0.01 mm, and
// 4 places on radians is 0.006 degrees.
const LANDMARK_PLACES = 4;
const WORLD_PLACES = 5;
const ANGLE_PLACES = 4;
const MM_PLACES = 2;

const round = (v: number, places: number) => {
  const f = 10 ** places;
  return Math.round(v * f) / f;
};

const roundPoints = (points: Vec3[], places: number): Vec3[] =>
  points.map((p) => ({ x: round(p.x, places), y: round(p.y, places), z: round(p.z, places) }));

/** Assemble a HandFrame, rounding every number to the precision the schema documents. */
export function buildHandFrame(input: {
  source: HandSource;
  t: number;
  handedness: "left" | "right" | null;
  landmarks: Vec3[];
  world: Vec3[];
  angles: Record<string, number>;
  tipWristMm: number;
  mano?: ManoParams;
}): HandFrame {
  const angles: Record<string, number> = {};
  for (const [name, value] of Object.entries(input.angles)) angles[name] = round(value, ANGLE_PLACES);
  const frame: HandFrame = {
    schema: 1,
    source: input.source,
    t: round(input.t, MM_PLACES),
    handedness: input.handedness,
    landmarks: roundPoints(input.landmarks, LANDMARK_PLACES),
    world: roundPoints(input.world, WORLD_PLACES),
    angles,
    tipWristMm: round(input.tipWristMm, MM_PLACES),
  };
  if (input.mano) frame.mano = input.mano;
  return frame;
}
