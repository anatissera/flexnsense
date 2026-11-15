/**
 * The exercise prescription, as data.
 *
 * Mirror of `session/protocol.py`, which is the tested side. Validation is the same, with the
 * same messages, so a prescription that loads in Python loads here and vice versa.
 *
 * Not ported: `check_against(calib)`. It needs the bench constants in `session/calibration.py`,
 * which are measured on the rig and live on the Python side.
 *
 * Units, per AGENTS.md: the clinician sets a cap in grams because that is what the PESOLA reads,
 * and `capN` stores it in newtons because that is what the control math runs on. Build a block
 * from a clinician's number with `blockFromGrams`, so the conversion happens in one place.
 */
import { gramsForceToNewtons, newtonsToGramsForce } from "../units";
import { FINGERS, normaliseFinger, type SessionFinger } from "./fingers";
import { formatFixed } from "./format";
import { modeByName, needsRomTarget, type ExerciseMode, type Mode } from "./modes";

/** One finger, one mode, a number of repetitions. */
export type Block = {
  finger: SessionFinger;
  mode: ExerciseMode;
  reps: number;
  /** The clinician's force cap, newtons. */
  capN: number;
  /** Seconds to hold at end of range before the reset beat. */
  holdS: number;
  /** How far a device-driven pull may stretch the finger, metres. Null for a mode that never pulls. */
  romTargetM: number | null;
};

export type BlockInput = {
  finger: string;
  mode: string;
  reps: number;
  capN: number;
  holdS?: number;
  romTargetM?: number | null;
};

/** Validate and build a block. Throws with the same wording as `Block.__post_init__`. */
export function makeBlock(input: BlockInput): Block {
  const { finger, mode, reps, capN } = input;
  const holdS = input.holdS ?? 3.0;
  const romTargetM = input.romTargetM ?? null;

  if (!(FINGERS as readonly string[]).includes(finger)) {
    let canonical: string | null = null;
    try {
      canonical = normaliseFinger(finger);
    } catch {
      canonical = null;
    }
    throw new Error(
      canonical !== null && canonical !== finger
        ? `block names the finger ${JSON.stringify(finger)}, which is not canonical. Use ${JSON.stringify(canonical)}.`
        : `unknown finger ${JSON.stringify(finger)}; known fingers are ${JSON.stringify(FINGERS)}`,
    );
  }
  const spec = modeByName(mode); // throws with the list of known modes
  if (reps < 1) throw new Error(`reps must be at least 1, got ${reps}`);
  if (capN <= 0) throw new Error(`cap_n must be positive, got ${capN}`);
  if (holdS < 0) throw new Error(`hold_s cannot be negative, got ${holdS}`);
  if (needsRomTarget(spec) && romTargetM === null) {
    throw new Error(
      `mode ${JSON.stringify(mode)} pulls the finger, so it needs rom_target_m: a device-driven ` +
        "pull is not bounded by force alone, because under the cap nothing tells it to stop",
    );
  }
  if (!needsRomTarget(spec) && romTargetM !== null) {
    throw new Error(
      `mode ${JSON.stringify(mode)} never pulls toward a target, so rom_target_m has no meaning ` +
        "here; the patient's own range is the limit",
    );
  }
  if (romTargetM !== null && romTargetM <= 0) {
    throw new Error(`rom_target_m must be positive, got ${romTargetM}`);
  }
  return {
    finger: finger as SessionFinger,
    mode: mode as ExerciseMode,
    reps,
    capN,
    holdS,
    romTargetM,
  };
}

/** Build a block from a cap in grams-force, which is the unit a clinician sets. */
export const blockFromGrams = (input: Omit<BlockInput, "capN"> & { capGf: number }): Block =>
  makeBlock({ ...input, capN: gramsForceToNewtons(input.capGf) });

/** The cap back in grams, for anything a human reads. */
export const blockCapGf = (block: Block) => newtonsToGramsForce(block.capN);

export const blockModeSpec = (block: Block): Mode => modeByName(block.mode);

/** An ordered list of blocks: the whole session. */
export type Protocol = {
  blocks: readonly Block[];
  /** Absolute force ceiling for the session, newtons. No block's cap may reach it. */
  ceilingN: number;
  name: string | null;
};

export function makeProtocol(input: {
  blocks: readonly Block[];
  ceilingN: number;
  name?: string | null;
}): Protocol {
  if (input.blocks.length === 0) throw new Error("a protocol needs at least one block");
  if (input.ceilingN <= 0) throw new Error(`ceiling_n must be positive, got ${input.ceilingN}`);
  input.blocks.forEach((b, i) => {
    if (b.capN >= input.ceilingN) {
      throw new Error(
        `block ${i} (${b.finger}, ${b.mode}) has cap ${formatFixed(blockCapGf(b), 0)} gf at or ` +
          `above the session ceiling of ${formatFixed(newtonsToGramsForce(input.ceilingN), 0)} gf; the cap ` +
          "has to leave the controller headroom to give way in before the gate releases",
      );
    }
  });
  return { blocks: input.blocks, ceilingN: input.ceilingN, name: input.name ?? null };
}

export const protocolTotalReps = (p: Protocol) => p.blocks.reduce((n, b) => n + b.reps, 0);

/** Each finger the hook has to be moved to, in order, collapsing consecutive repeats. */
export function fingersInOrder(p: Protocol): SessionFinger[] {
  const out: SessionFinger[] = [];
  for (const b of p.blocks) if (out.length === 0 || out[out.length - 1] !== b.finger) out.push(b.finger);
  return out;
}

type RawBlock = Record<string, unknown>;

/** Read a block from the JSON shape `session/protocol.py` writes (`cap_n` or `cap_gf`, snake case). */
export function blockFromJSON(raw: RawBlock): Block {
  const known = new Set(["finger", "mode", "reps", "cap_n", "cap_gf", "hold_s", "rom_target_m"]);
  const unknown = Object.keys(raw).filter((k) => !known.has(k)).sort();
  if (unknown.length > 0) throw new Error(`unknown block keys: ${JSON.stringify(unknown)}`);
  if (("cap_n" in raw) === ("cap_gf" in raw)) {
    throw new Error("a block needs exactly one of cap_n or cap_gf, not both or neither");
  }
  const capN = "cap_n" in raw ? Number(raw.cap_n) : gramsForceToNewtons(Number(raw.cap_gf));
  const rom = raw.rom_target_m;
  return makeBlock({
    // Normalised on the way in, so a file written against the tracker's vocabulary loads.
    finger: normaliseFinger(String(raw.finger)),
    mode: String(raw.mode),
    reps: Math.trunc(Number(raw.reps)),
    capN,
    holdS: raw.hold_s === undefined ? 3.0 : Number(raw.hold_s),
    romTargetM: rom === undefined || rom === null ? null : Number(rom),
  });
}

/** Read a protocol from the JSON shape `session/protocol.py` writes. */
export function protocolFromJSON(raw: Record<string, unknown>): Protocol {
  const known = new Set(["blocks", "ceiling_n", "ceiling_gf", "name"]);
  const unknown = Object.keys(raw).filter((k) => !known.has(k)).sort();
  if (unknown.length > 0) throw new Error(`unknown protocol keys: ${JSON.stringify(unknown)}`);
  if (("ceiling_n" in raw) === ("ceiling_gf" in raw)) {
    throw new Error("a protocol needs exactly one of ceiling_n or ceiling_gf");
  }
  const ceilingN =
    "ceiling_n" in raw ? Number(raw.ceiling_n) : gramsForceToNewtons(Number(raw.ceiling_gf));
  if (!Array.isArray(raw.blocks)) throw new Error("blocks must be a list");
  return makeProtocol({
    blocks: (raw.blocks as RawBlock[]).map(blockFromJSON),
    ceilingN,
    name: raw.name === undefined || raw.name === null ? null : String(raw.name),
  });
}
