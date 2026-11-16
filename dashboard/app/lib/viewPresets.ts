/**
 * The named camera views of the 3D scene. The view cube (top right) and the chat agent's camera tools
 * both read this table, so a view means the same thing whoever asks for it.
 *
 * World axes, Z up, forearm along Y. `dir` is the side the camera moves to and `up` is what points up on
 * screen in that view. The palm faces +Z, so TOP is the palm side and BOTTOM is the back of the hand.
 */
export type V3 = readonly [number, number, number];

export type ViewFace = { id: string; label: string; lines: string[]; dir: V3; up: V3 };

/** The six faces, with the anatomical names of the planes. `lines` is how the cube labels the face. */
export const FACES: ViewFace[] = [
  { id: "sagittal_right", label: "Sagittal, right side (the starting side profile)", lines: ["SAGITTAL", "RIGHT"], dir: [1, 0, 0], up: [0, 0, 1] },
  { id: "sagittal_left", label: "Sagittal, left side", lines: ["SAGITTAL", "LEFT"], dir: [-1, 0, 0], up: [0, 0, 1] },
  { id: "frontal_back", label: "Frontal (coronal), from the back", lines: ["FRONTAL", "BACK"], dir: [0, 1, 0], up: [0, 0, 1] },
  { id: "frontal_front", label: "Frontal (coronal), from the front", lines: ["FRONTAL"], dir: [0, -1, 0], up: [0, 0, 1] },
  { id: "transverse_top", label: "Transverse (axial), from above: the palm side", lines: ["TRANSVERSE", "TOP"], dir: [0, 0, 1], up: [0, 1, 0] },
  { id: "transverse_bottom", label: "Transverse (axial), from below: the back of the hand", lines: ["TRANSVERSE", "BOTTOM"], dir: [0, 0, -1], up: [0, -1, 0] },
];

/**
 * The eight corners: a three-quarter view from along the cube's diagonal, 45 degrees round from each of
 * the two faces that meet there and tipped up or down. `up` stays world up, so the horizon stays level.
 */
export const CORNERS: { id: string; name: string; dir: V3; up: V3 }[] = [-1, 1].flatMap((sz) =>
  [-1, 1].flatMap((sy) =>
    [-1, 1].map((sx) => {
      const name = `${sz > 0 ? "top" : "bottom"} ${sx > 0 ? "right" : "left"} ${sy > 0 ? "back" : "front"}`;
      return { id: name.replace(/ /g, "_"), name, dir: [sx, sy, sz] as V3, up: [0, 0, 1] as V3 };
    }),
  ),
);

/** Ids the chat tool accepts for a camera view: the faces, the corners, and "home". */
export const VIEW_IDS = [...FACES.map((f) => f.id), ...CORNERS.map((c) => c.id), "home"] as const;

/** The cube's turn buttons, by the name the chat tool uses: which camera axis, and which sign (90 degrees is one step). */
export const TURNS = {
  up: ["pitch", 1],
  down: ["pitch", -1],
  left: ["yaw", 1],
  right: ["yaw", -1],
  roll_clockwise: ["roll", 1],
  roll_counterclockwise: ["roll", -1],
} as const satisfies Record<string, readonly ["yaw" | "pitch" | "roll", number]>;
