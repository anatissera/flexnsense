"use client";

/**
 * The stop that is always reachable.
 *
 * A clinician watching a session must not have to find the right panel first, so the control
 * lives in the bar at the top and anything that can be running listens for it here. It is a
 * plain subscriber list rather than React state on purpose: a stop must not wait for a render,
 * and the things that need stopping (an animation loop, a camera, a state machine) are not
 * rendered values.
 *
 * What it does NOT do: reach an arm. Nothing in this browser can. On the rig the equivalent is
 * `session/interlock.py` plus Station's own current limits, and the control loop's gate releases
 * on its own if the force reading goes away. This button stops what this page is driving and
 * says so.
 */

const subscribers = new Set<(reason: string) => void>();

/** Listen for a stop. Returns the unsubscribe, for an effect's cleanup. */
export function onEmergencyStop(fn: (reason: string) => void): () => void {
  subscribers.add(fn);
  return () => { subscribers.delete(fn); };
}

/** Stop everything this page is driving. Safe to call when nothing is running. */
export function fireEmergencyStop(reason = "emergency stop"): void {
  subscribers.forEach((fn) => {
    try {
      fn(reason);
    } catch {
      // One listener failing must never stop the others from stopping.
    }
  });
}
