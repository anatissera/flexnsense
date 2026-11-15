/**
 * How far a fingertip travels along its own pull direction, from a sequence of tracked frames.
 *
 * Mirror of `hand_tracking/excursion.py`, which is the tested side and carries the full
 * reasoning. `tests/session/test_ts_port.py` runs both on the same trajectories and fails if
 * they disagree. The operations are in the same order on purpose, so the two agree to rounding.
 *
 * The use: move one finger slowly from full extension to full flexion while the tracker runs,
 * collect `tipMinusWrist` for every frame, then `fitExcursion` the lot. The result is how far
 * the tip moved along the direction it actually moved in.
 *
 * Two things rule out differencing the first and last frame:
 *  - MediaPipe's `world` landmarks are hand-centred, so their origin drifts a little as the
 *    hand's pose changes. Taking tip-minus-wrist first removes it, because the drift cancels
 *    between the two points of each frame.
 *  - The two endpoints alone throw away every other frame, including whichever one was the most
 *    extended or most flexed.
 * So this fits a single axis through the whole trajectory (its direction of greatest spread)
 * and reports the spread of all frames projected onto that axis.
 */

export type Point = readonly [number, number, number];

/** The tracker's vocabulary for the fifth digit is "pinky", as in `hand_tracking/landmarks.py`. */
export const FINGERTIP_BY_NAME = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 } as const;
export type TrackedFinger = keyof typeof FINGERTIP_BY_NAME;

const WRIST = 0;

const sub = (a: Point, b: Point): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (v: Point) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

/** One frame's fingertip position relative to the wrist, which cancels the world origin drift. */
export function tipMinusWrist(points: readonly Point[], finger: string): [number, number, number] {
  if (!(finger in FINGERTIP_BY_NAME)) {
    throw new Error(
      `unknown finger ${JSON.stringify(finger)}; known: ${JSON.stringify(Object.keys(FINGERTIP_BY_NAME).sort())}`,
    );
  }
  if (points.length < 21) throw new Error(`need 21 landmarks, got ${points.length}`);
  return sub(points[FINGERTIP_BY_NAME[finger as TrackedFinger]], points[WRIST]);
}

/** The fit through one finger's tip-minus-wrist trajectory across a recording. */
export type ExcursionResult = {
  /** Spread of the trajectory along its own dominant direction, metres. The headline number. */
  excursionM: number;
  /** Unit vector, the fitted axis. Sign is arbitrary; only the line it defines is meaningful. */
  axis: [number, number, number];
  /**
   * Fraction of the trajectory's total variance this one axis accounts for, 0..1. Low means the
   * motion was not close to a single straight line (a flinch, a rotated wrist, a second finger
   * moving), so `excursionM` is a line fitted through something that was not really 1-D motion.
   */
  varianceExplained: number;
  /** RMS distance of the trajectory from the fitted axis, metres. Compare it to `excursionM`. */
  perpendicularRmsM: number;
  nSamples: number;
};

/** Fit the dominant axis through tip-minus-wrist vectors, and the excursion along it. */
export function fitExcursion(vectors: readonly Point[]): ExcursionResult {
  const n = vectors.length;
  if (n < 2) throw new Error(`need at least 2 samples, got ${n}`);

  const mean: [number, number, number] = [0, 1, 2].map(
    (i) => vectors.reduce((s, v) => s + v[i], 0) / n,
  ) as [number, number, number];
  const centered = vectors.map((v) => sub(v, mean));

  // Power iteration on the 3x3 covariance matrix for the dominant eigenvector.
  const cov = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const v of centered) {
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += v[i] * v[j];
  }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] /= n;

  const axis = dominantEigenvector(cov);
  const projections = centered.map((v) => v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]);
  const excursionM = Math.max(...projections) - Math.min(...projections);

  const totalVar = cov[0][0] + cov[1][1] + cov[2][2];
  const alongVar = projections.reduce((s, p) => s + p * p, 0) / n;
  const varianceExplained = totalVar > 0 ? alongVar / totalVar : 1.0;

  let perpSq = 0;
  centered.forEach((v, k) => {
    const p = projections[k];
    perpSq += norm(sub(v, [p * axis[0], p * axis[1], p * axis[2]])) ** 2;
  });

  return {
    excursionM,
    axis,
    varianceExplained,
    perpendicularRmsM: Math.sqrt(perpSq / n),
    nSamples: n,
  };
}

/**
 * Power iteration: the eigenvector for the largest eigenvalue of a 3x3 symmetric matrix. Good
 * enough because the matrix is tiny and the trajectory is expected to have one axis that
 * dominates; a case where it would converge slowly (near-equal top eigenvalues) is already a
 * case `varianceExplained` flags as untrustworthy.
 */
function dominantEigenvector(cov: number[][], iterations = 200): [number, number, number] {
  let x = 1.0;
  let y = 1.0;
  let z = 1.0;
  for (let k = 0; k < iterations; k++) {
    const nx = cov[0][0] * x + cov[0][1] * y + cov[0][2] * z;
    const ny = cov[1][0] * x + cov[1][1] * y + cov[1][2] * z;
    const nz = cov[2][0] * x + cov[2][1] * y + cov[2][2] * z;
    const length = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (length < 1e-18) return [0, 0, 0];
    x = nx / length;
    y = ny / length;
    z = nz / length;
  }
  return [x, y, z];
}

/**
 * Collects a recording frame by frame, so a "calibrate range of motion" button can feed it from
 * the tracker's `onFrame` and read the excursion when the sweep ends.
 */
export class ExcursionRecorder {
  private readonly vectors: [number, number, number][] = [];

  constructor(readonly finger: TrackedFinger) {}

  /** Add one frame's world landmarks (metres, hand centred). */
  add(world: readonly Point[]): void {
    this.vectors.push(tipMinusWrist(world, this.finger));
  }

  get length(): number {
    return this.vectors.length;
  }

  clear(): void {
    this.vectors.length = 0;
  }

  /** The fit so far. Throws until there are at least 2 frames. */
  result(): ExcursionResult {
    return fitExcursion(this.vectors);
  }
}
