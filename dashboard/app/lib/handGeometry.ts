/**
 * One flexion angle per finger, in degrees, from MediaPipe world landmarks.
 *
 * Mirror of `hand_tracking/landmarks.py`. This is the clinician-facing convention used in
 * published single-camera ROM work: 0 = fully extended, larger = more bent, taken at the PIP
 * joint of index/middle/ring/pinky and at the IP joint of the thumb (which has no DIP).
 *
 * It is the same joint and the same maths as the `pm{2..5}_flexion_r` and `ip_flexion_r` entries
 * `landmarksToPose` already computes for the 3D hand (in radians, thumb sign flipped), so this
 * is for anything that wants plain degrees per finger, such as a rep counter or a ROM readout.
 *
 * Pass world landmarks (metres, hand centred), never the normalized image ones: those distort
 * angles by the image's aspect ratio and by how far the hand is from the camera.
 *
 * 21-point layout, same indices for image and world landmarks:
 *   0 WRIST, 1-4 THUMB (CMC MCP IP TIP), 5-8 INDEX, 9-12 MIDDLE, 13-16 RING, 17-20 PINKY
 *   (each finger MCP PIP DIP TIP)
 */
import type { Point } from "./excursion";

export type FingerIndices = readonly [number, number, number, number];

export const THUMB: FingerIndices = [1, 2, 3, 4];
export const INDEX: FingerIndices = [5, 6, 7, 8];
export const MIDDLE: FingerIndices = [9, 10, 11, 12];
export const RING: FingerIndices = [13, 14, 15, 16];
export const PINKY: FingerIndices = [17, 18, 19, 20];
export const FINGERS = { thumb: THUMB, index: INDEX, middle: MIDDLE, ring: RING, pinky: PINKY } as const;
export type FingerName = keyof typeof FINGERS;

const sub = (a: Point, b: Point): Point => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (v: Point) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

/** Angle a-b-c at vertex b, degrees: 180 = perfectly straight, 0 = folded flat onto itself. NaN if degenerate. */
export function jointAngleDeg(a: Point, b: Point, c: Point): number {
  const u = sub(a, b);
  const v = sub(c, b);
  const nu = norm(u);
  const nv = norm(v);
  if (nu === 0 || nv === 0) return NaN;
  const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
  return (Math.acos(Math.max(-1.0, Math.min(1.0, cos))) * 180) / Math.PI;
}

/** Flexion at joint b: 0 = fully extended (straight), larger = more bent, up to ~180. */
export const flexionDeg = (a: Point, b: Point, c: Point) => 180.0 - jointAngleDeg(a, b, c);

/** PIP-joint flexion for one of INDEX/MIDDLE/RING/PINKY: angle at PIP, between MCP and DIP. */
export function fingerFlexionDeg(points: readonly Point[], finger: FingerIndices): number {
  const [mcp, pip, dip] = finger;
  return flexionDeg(points[mcp], points[pip], points[dip]);
}

/** IP-joint flexion for the thumb (it has no DIP): angle at IP, between MCP and TIP. */
export function thumbFlexionDeg(points: readonly Point[]): number {
  const [, mcp, ip, tip] = THUMB;
  return flexionDeg(points[mcp], points[ip], points[tip]);
}

/** One representative flexion angle per finger: PIP for the four fingers, IP for the thumb. */
export function allFlexionDeg(points: readonly Point[]): Record<FingerName, number> {
  const out = {} as Record<FingerName, number>;
  for (const name of Object.keys(FINGERS) as FingerName[]) {
    out[name] = name === "thumb" ? thumbFlexionDeg(points) : fingerFlexionDeg(points, FINGERS[name]);
  }
  return out;
}
