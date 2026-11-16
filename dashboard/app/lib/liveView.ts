/**
 * A live `SessionSnapshot` in the shape the rep-machine panel already renders.
 *
 * The panel was built around `RepStep`, which is what the browser's own simulation produces.
 * A snapshot from a real session carries the same decisions, made by the same machine, but it
 * is not the same object: a `RepStep` is one cycle's output on this machine, and a snapshot is
 * what a session chose to publish.
 *
 * So this converts rather than casts, and the difference matters in one place. The snapshot
 * does not carry `note`, the rep machine's per-cycle "why this cycle looks like it does",
 * because that is a log line and the snapshot is a view. Rather than invent one, `note` here
 * is empty while a session runs normally and carries `stoppedReason` once it has stopped,
 * which is the only part of it a clinician's screen was showing anyway.
 *
 * `law` and `target` are not produced at all: the panel does not render them, and a value that
 * nothing reads is a value that can silently go wrong. If the panel ever needs them, they
 * belong in the snapshot (and its schema) rather than being reconstructed here.
 */

import type { Driver, Phase, SessionSnapshot } from "./sessionSnapshot";

/** Exactly the fields the rep-machine panel reads, and nothing it does not. */
export type LiveStep = {
  phase: Phase;
  driver: Driver;
  instruction: string;
  repIndex: number;
  repsCompleted: number;
  attemptsAbandoned: number;
  finished: boolean;
  note: string;
};

export type LiveView = {
  step: LiveStep;
  forceN: number | null;
  travelMm: number;
  capN: number;
  reps: number;
};

/**
 * `null` until a session has a block, which is until it has actually started.
 *
 * A snapshot with no block is a session that has connected but not begun. Rendering it would
 * mean showing a cap and a rep count of zero as though they were a prescription.
 */
export function liveView(snapshot: SessionSnapshot | null): LiveView | null {
  if (snapshot === null || snapshot.block === null) return null;
  return {
    step: {
      phase: snapshot.phase,
      driver: snapshot.driver,
      instruction: snapshot.instruction.text,
      repIndex: snapshot.rep.index,
      // Straight from the machine, never inferred from the index: a repetition counts only
      // once the force reached the cap, so a patient who moved a little and came back is an
      // abandoned attempt. Deriving "completed" from "index" would report attempts as reps.
      repsCompleted: snapshot.rep.completed,
      attemptsAbandoned: snapshot.rep.abandoned,
      finished: snapshot.phase === "set_done" || snapshot.phase === "aborted",
      note: snapshot.stoppedReason ?? "",
    },
    // Passed through, including null. Null means the camera could not see the marker, which
    // the panel already renders as "no reading"; substituting zero would show a slack tether
    // for a session that has gone blind.
    forceN: snapshot.forceN,
    travelMm: snapshot.travelMm,
    capN: snapshot.block.capN,
    reps: snapshot.block.reps,
  };
}
