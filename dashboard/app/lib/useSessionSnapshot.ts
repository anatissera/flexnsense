"use client";

/**
 * Subscribe to the session running on this machine, so the dashboard renders the arm rather
 * than its own clock.
 *
 * `GuideHand.tsx` counts repetitions off `performance.now()` against hardcoded constants, so
 * what the patient hears and what the arm is doing are two independent stories that happen to
 * start together. A snapshot stream makes them the same state by definition: every field here
 * is produced by the same `RepMachine` that decides what the arm does.
 *
 * ## Null is a reading, not a gap
 *
 * `forceN: null` means the camera could not see the marker. It is NOT zero and it is NOT "keep
 * showing the last number". This hook passes it straight through, and anything rendering it
 * must show "no reading" rather than a stale value: a dashboard displaying a plausible force
 * for a session that has gone blind is exactly what the safety layer exists to refuse.
 *
 * ## It never commands anything
 *
 * One-way by construction: this opens a socket, reads, and never writes. Stopping a session is
 * not done from here (docs/two-machine-session.md). A view that could also command would be a
 * second place where "who may move the arm" is decided, and there is only one.
 *
 * ## It is absent, not fake, when nothing is running
 *
 * With no session on the other end, `snapshot` stays null and `status` says so. There is no
 * demo data path in this file: the demonstration build gets its simulation from `lib/demo.ts`,
 * which labels it, and mixing the two here would make "is this real?" unanswerable from the
 * code that renders it.
 */

import { useEffect, useRef, useState } from "react";

import type { SessionSnapshot } from "./sessionSnapshot";

/** Loopback for the same reason the gauge and tracker servers are: the session runs beside the screen. */
export const SESSION_URL = "ws://127.0.0.1:8771/session";

/** The schema this client was written against. A mismatch is reported, never guessed at. */
export const EXPECTED_SCHEMA = 1;

export type SessionStatus =
  | "connecting"
  | "live"
  /** Connected before, gone now. The last snapshot is kept so the screen does not blank. */
  | "disconnected"
  /** Nothing is listening: no session is running on this machine. */
  | "absent"
  /** The far end speaks a different snapshot schema. */
  | "mismatch";

export type SessionStream = {
  snapshot: SessionSnapshot | null;
  status: SessionStatus;
  /** Human-readable, for a status line. Says what to run when nothing is listening. */
  detail: string;
};

const ABSENT_DETAIL =
  "No session on 127.0.0.1:8771. Start one, or run: python -m session.snapshot_server --demo";

/**
 * Watch the session stream.
 *
 * @param enabled pass false in the demonstration build, where probing loopback would be both
 *   pointless and misleading: there is no session, and the page already says so.
 */
export function useSessionSnapshot(enabled = true, url = SESSION_URL): SessionStream {
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [status, setStatus] = useState<SessionStatus>(enabled ? "connecting" : "absent");
  const [detail, setDetail] = useState(enabled ? "Connecting to the session…" : ABSENT_DETAIL);
  // Whether this connection ever delivered a frame, so a close can tell "nothing was there"
  // from "the session ended", which are different things to put on a clinician's screen.
  const everOpened = useRef(false);

  useEffect(() => {
    if (!enabled) {
      setStatus("absent");
      setDetail(ABSENT_DETAIL);
      return;
    }
    let cancelled = false;
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      ws = new WebSocket(url);

      ws.onopen = () => {
        if (cancelled) return;
        everOpened.current = true;
        setStatus("connecting");
        setDetail("Waiting for the first snapshot…");
      };

      ws.onmessage = (event) => {
        if (cancelled) return;
        let next: SessionSnapshot;
        try {
          next = JSON.parse(event.data as string) as SessionSnapshot;
        } catch {
          return; // a frame we cannot parse is dropped, never half-applied
        }
        if (next.schema !== EXPECTED_SCHEMA) {
          // Refused rather than rendered. A snapshot whose meaning is uncertain is worse than
          // none: the fields may have moved under the same names.
          setStatus("mismatch");
          setDetail(
            `The session speaks snapshot schema ${next.schema}, this dashboard speaks ${EXPECTED_SCHEMA}. Update one of them.`,
          );
          ws?.close();
          return;
        }
        setSnapshot(next);
        setStatus("live");
        setDetail(next.stoppedReason ? `Session stopped: ${next.stoppedReason}` : "Live");
      };

      ws.onerror = () => {
        if (cancelled) return;
        if (!everOpened.current) {
          setStatus("absent");
          setDetail(ABSENT_DETAIL);
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        setStatus((previous) => (previous === "mismatch" ? previous : everOpened.current ? "disconnected" : "absent"));
        setDetail((previous) =>
          previous.startsWith("The session speaks")
            ? previous
            : everOpened.current
              ? "The session stream closed. Retrying…"
              : ABSENT_DETAIL,
        );
        // Keep trying: a session started after the page was opened should appear on its own,
        // and a session that ends should be picked up again when the next one starts.
        retry = setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [enabled, url]);

  return { snapshot, status, detail };
}
